"""fast_load TP slicing (moe_ep_size 1): CPU check on synthetic shards with the checkpoint's expert shapes.

Routed experts use the real shapes and dtypes (w1/w3 I8 [2304, 2560] + F8_E8M0 scale [2304, 160],
w2 I8 [5120, 1152] + scale [5120, 72]; 4 target experts, 2 draft experts), next to a shared expert,
non-expert tensors of every other dtype and an Engram table. For every rank at EP1 (moe_tp 4) and
EP2 (moe_tp 2), in target and draft phase, the tensors come out of the wrapped ``safe_open`` the way
``buffered_multi_thread_safetensors_weights_iterator`` takes them, go through FusedMoE's own
``_load_w13`` / ``_load_w2`` when sglang is importable (else a mirror of their narrowing), and the
bytes that land in the expert parameter must equal the stock ``safe_open`` path. Also: EP2 gets the
exact legacy read list (byte-identical behaviour), out-of-slice access raises, the pacing budget
counts the resident slice, and bounce buffers are bounded and released.
"""
import concurrent.futures
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import inspect
import types

os.environ["DSV41_FAST_LOAD"] = "1"
os.environ.pop("DSV41_FAST_LOAD_TP_SLICE", None)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapter"))
import torch  # noqa: E402
import safetensors  # noqa: E402
import safetensors.torch as st  # noqa: E402

import fast_load as fl  # noqa: E402

if not hasattr(os, "posix_fadvise"):  # macOS dev runs; the image is Linux
    os.posix_fadvise = lambda *a: None
    os.POSIX_FADV_DONTNEED = 4

RAW_OPEN = safetensors.safe_open
E8M0 = torch.float8_e8m0fnu
EXPERT = {"w1.weight": ((2304, 2560), torch.int8), "w1.scale": ((2304, 160), E8M0),
          "w3.weight": ((2304, 2560), torch.int8), "w3.scale": ((2304, 160), E8M0),
          "w2.weight": ((5120, 1152), torch.int8), "w2.scale": ((5120, 72), E8M0)}
N_TARGET, N_DRAFT = 4, 2
g = torch.Generator().manual_seed(0)


def rnd(shape, dtype):
    if dtype.is_floating_point and torch.finfo(dtype).bits == 8:
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g).view(dtype)
    if dtype in (torch.int8,):
        return torch.randint(-128, 128, shape, dtype=dtype, generator=g)
    return torch.randn(shape, generator=g).to(dtype)


def experts(prefix, ids):
    return {f"{prefix}.ffn.experts.{e}.{k}": rnd(*v) for e in ids for k, v in EXPERT.items()}


def build(d):
    s1 = {"layers.0.attn.wq_a.weight": rnd((96, 128), torch.bfloat16),
          "layers.0.attn.wq_a.scale": rnd((3, 4), torch.float32),
          "layers.0.attn.wo_a.weight": rnd((64, 128), torch.float8_e4m3fn),
          "layers.0.ffn.shared_experts.w1.weight": rnd((2304, 2560), torch.int8),
          "layers.0.ffn.shared_experts.w2.scale": rnd((5120, 72), E8M0),
          "layers.0.engram.embed.weight": rnd((512, 64), torch.bfloat16)}
    s1.update(experts("layers.0", range(0, 2)))
    s2 = experts("layers.0", range(2, 4))
    s2["layers.0.ffn.gate.weight"] = rnd((4, 5120), torch.bfloat16)
    s3 = {"mtp.0.attn.wq_a.weight": rnd((96, 128), torch.bfloat16),
          "mtp.0.ffn.shared_experts.w2.weight": rnd((5120, 1152), torch.int8)}
    s3.update(experts("mtp.0", range(N_DRAFT)))
    files = []
    for i, sd in enumerate((s1, s2, s3)):
        p = os.path.join(d, f"model-{i + 1:05d}-of-00003.safetensors")
        st.save_file(sd, p)
        files.append(p)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"text_config": {"n_routed_experts": N_TARGET}}, f)
    return files


# ---- the engine's narrowing: FusedMoE._load_w13 / _load_w2 when sglang is importable ----
try:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    ENGINE = "FusedMoE._load_w13/_load_w2"
except Exception as exc:  # noqa: BLE001
    FusedMoE = None
    ENGINE = f"mirror of FusedMoE narrowing (sglang not importable: {type(exc).__name__})"


def engine_copy(lw, proj, moe_tp_rank, moe_tp_size, shape):
    """What one rank's expert parameter holds after the FusedMoE loader copied ``lw`` into it."""
    shard_dim = fl._TP_SLICE_DIM[proj]
    n = shape[shard_dim] // moe_tp_size
    if proj in ("w1", "w3"):
        ed_shape = (2 * n,) + tuple(shape[1:])
    else:
        ed_shape = tuple(shape[:1]) + (n,)
    ed = torch.zeros(ed_shape, dtype=torch.uint8).view(lw.dtype)
    if FusedMoE is not None:
        fake = types.SimpleNamespace(
            moe_tp_rank=moe_tp_rank, moe_tp_size=moe_tp_size, moe_runner_config=types.SimpleNamespace(is_gated=True),
            quant_method=types.SimpleNamespace(load_up_proj_weight_first=True), use_padded_loading=False,
            use_presharded_weights=False, use_triton_kernels=False, quant_config=None)
        loader = FusedMoE._load_w2 if proj == "w2" else FusedMoE._load_w13
        kwargs = dict(expert_data=ed, shard_dim=shard_dim, shard_id=proj, loaded_weight=lw)
        if "tp_rank" in inspect.signature(loader).parameters:  # SGLang before #41814 passed the rank
            kwargs["tp_rank"] = moe_tp_rank
        loader(fake, **kwargs)
    else:
        ls = lw.shape[shard_dim] // moe_tp_size
        src = lw.narrow(shard_dim, ls * moe_tp_rank, ls)
        start = n if proj == "w1" else 0            # load_up_proj_weight_first: [w3; w1]
        ed.narrow(shard_dim, start, n if proj != "w2" else ed_shape[1]).copy_(src)
    return ed.view(torch.uint8).clone()


def raw_bytes(t):
    return t.contiguous().view(torch.uint8).reshape(-1)


def routed(name):
    if fl._expert_id(name) is None:
        return None
    proj, kind = name.rsplit(".", 2)[1:]
    return proj if proj in fl._TP_SLICE_DIM and kind in fl._TP_SLICE_KINDS else None


def run_rank(files, phase, ep, tp, mode):
    """One rank's load of every shard through the wrapped safe_open; checks bytes, returns stats."""
    if mode is None:
        os.environ.pop("DSV41_FAST_LOAD_TP_SLICE", None)
    else:
        os.environ["DSV41_FAST_LOAD_TP_SLICE"] = mode
    fl._ep_info = lambda: ep
    fl._moe_tp_info = lambda: tp
    fl._state["phase"] = phase
    seen_plans = []
    orig_init = fl._EagerShard.__init__

    def spy(self, inner, path, names):
        seen_plans.append((path, names))
        orig_init(self, inner, path, names)

    fl._EagerShard.__init__ = spy
    Slice = fl._tp_slice_cls()
    stats = {"sliced": 0, "whole_eager": 0, "copies": 0, "kept_per_shard": []}
    before = dict(fl._state)
    try:
        for path in files:
            with safetensors.safe_open(path, framework="pt", device="cpu") as f, \
                    RAW_OPEN(path, framework="pt", device="cpu") as r:
                got = {k: f.get_tensor(k) for k in f.keys()}   # as _load_file does
                kept = 0
                eager = set(seen_plans[-1][1]) if seen_plans and seen_plans[-1][0] == path else set()
                for k, t in got.items():
                    ref = r.get_tensor(k)
                    proj = routed(k)
                    moe_ep_owned = proj is not None and (
                        phase == "draft" or k.startswith("mtp.") is False and ep is not None and
                        fl._expert_id(k) // (N_TARGET // ep[1]) == ep[0])
                    if isinstance(t, Slice):
                        assert proj is not None and k in eager, k
                        stats["sliced"] += 1
                        kept += t._dsv41_resident_nbytes
                        assert tuple(t.shape) == tuple(ref.shape) and t.dtype == ref.dtype, k
                    elif k in eager:
                        stats["whole_eager"] += 1
                        kept += t.numel() * t.element_size()
                    if moe_ep_owned and tp is not None:
                        # the bytes that land in this rank's expert parameter
                        a = engine_copy(t, proj, tp[0], tp[1], list(ref.shape))
                        b = engine_copy(ref, proj, tp[0], tp[1], list(ref.shape))
                        assert torch.equal(a, b), (k, phase, ep, tp)
                        stats["copies"] += 1
                    else:
                        assert not isinstance(t, Slice), k
                        assert t.dtype == ref.dtype and t.shape == ref.shape, k
                        assert torch.equal(raw_bytes(t), raw_bytes(ref)), k
                stats["kept_per_shard"].append(kept)
    finally:
        fl._EagerShard.__init__ = orig_init
    stats["read"] = fl._state["bytes"] - before["bytes"]
    stats["resident"] = fl._state["resident"] - before["resident"]
    stats["plans"] = seen_plans
    return stats


def main():
    d = tempfile.mkdtemp(prefix="fl_tp_slice_")
    try:
        files = build(d)
        fl.install_weight_utils(types.SimpleNamespace(safetensors=None))
        assert safetensors.safe_open is not RAW_OPEN
        n = fl._n_routed_experts(files[0])
        assert n == N_TARGET, n

        # -- per-expert accounting on the real shapes --
        info = {k: {"dtype": "I8" if v[1] == torch.int8 else "F8_E8M0", "shape": list(v[0]),
                    "data_offsets": [0, v[0][0] * v[0][1]]} for k, v in EXPERT.items()}
        full = sum(i["data_offsets"][1] for i in info.values())
        ep1 = [fl.spec_bytes(i, fl.slice_spec("layers.0.ffn.experts.0." + k, i, (1, 4))) for k, i in info.items()]
        read1, kept1 = sum(r for r, _ in ep1), sum(h for _, h in ep1)
        assert full == 18_800_640 and read1 == 9_400_320 and kept1 == 4_700_160, (full, read1, kept1)
        print(f"per expert: whole {full / 1e6:.2f} MB; EP1 slice reads {read1 / 1e6:.2f} MB, keeps {kept1 / 1e6:.2f} MB")
        print(f"projected per rank (384 experts x 40 layers): EP2 whole owned {192 * 40 * full / 1e9:.1f} GB read; "
              f"EP1 sliced {384 * 40 * read1 / 1e9:.1f} GB read, {384 * 40 * kept1 / 1e9:.1f} GB kept")

        # -- EP2 default: exactly today's read list (a plain list = every tensor whole) --
        legacy_total = {}
        for r in range(4):
            ep, tp = (r // 2, 2), (r % 2, 2)
            for phase in ("target", "draft"):
                s = run_rank(files, phase, ep, tp, None)
                assert s["sliced"] == 0
                for path, names in s["plans"]:
                    assert isinstance(names, list), "EP2 must hand _EagerShard the legacy name list"
                    assert names == fl.needed_names(path, phase, ep if phase == "target" else None, n)
                want = sum(b - a for f in files
                           for a, b in fl.needed_ranges(f, phase, ep if phase == "target" else None, n))
                assert s["read"] == want == s["resident"], (s["read"], want)
                legacy_total[(r, phase)] = s
        print(f"EP2 default: legacy read lists on all 4 ranks, {ENGINE} copies bitwise equal")

        # -- EP1: every rank slices every routed expert (target and draft) --
        for r in range(4):
            ep, tp = (0, 1), (r, 4)
            for phase in ("target", "draft"):
                s = run_rank(files, phase, ep, tp, None)
                n_exp = (N_TARGET if phase == "target" else N_DRAFT) * len(EXPERT)
                assert s["sliced"] == n_exp and s["copies"] == n_exp, (phase, s["sliced"], s["copies"])
                want = sum(b - a for f in files for a, b in fl.plan_ranges(
                    f, fl.read_plan(f, phase, ep if phase == "target" else None, n, tp)))
                assert s["read"] == want, (s["read"], want)
                e2 = legacy_total[(r, phase)]
                # at most the EP2 bytes read, and a smaller resident window per shard
                assert s["read"] <= e2["read"] + 1, (phase, s["read"], e2["read"])
                assert max(s["kept_per_shard"]) <= max(e2["kept_per_shard"]), (s["kept_per_shard"], e2["kept_per_shard"])
                if phase == "target":
                    assert s["read"] - (e2["read"] - N_TARGET // 2 * full) == N_TARGET * read1
        print(f"EP1: 4 ranks x (target, draft) sliced, {ENGINE} copies bitwise equal to stock")

        # -- EP2 with DSV41_FAST_LOAD_TP_SLICE=1: owned experts sliced by moe_tp 2 --
        for r in range(4):
            s = run_rank(files, "target", (r // 2, 2), (r % 2, 2), "1")
            assert s["sliced"] == N_TARGET // 2 * len(EXPERT) and s["read"] < legacy_total[(r, "target")]["read"]
        print("EP2 + TP_SLICE=1: owned experts sliced, bitwise equal")

        # -- EP1 with TP_SLICE=0: experts left to mmap, never read whole eagerly --
        s = run_rank(files, "target", (0, 1), (2, 4), "0")
        assert s["sliced"] == 0
        for path, names in s["plans"]:
            assert all(fl._expert_id(k) is None for k in names), path
        s = run_rank(files, "target", (0, 1), None, None)   # moe_tp unknown: same guard
        for path, names in s["plans"]:
            assert all(fl._expert_id(k) is None for k in names), path
        print("EP1 + TP_SLICE=0 / unknown moe_tp: routed experts stay on the stock mmap path")

        # -- out-of-slice access raises --
        os.environ.pop("DSV41_FAST_LOAD_TP_SLICE", None)
        fl._ep_info, fl._moe_tp_info, fl._state["phase"] = (lambda: (0, 1)), (lambda: (1, 4)), "target"
        with safetensors.safe_open(files[0], framework="pt", device="cpu") as f, \
                RAW_OPEN(files[0], framework="pt", device="cpu") as r:
            w1 = f.get_tensor("layers.0.ffn.experts.1.w1.weight")
            w2 = f.get_tensor("layers.0.ffn.experts.1.w2.scale")
            ref1 = r.get_tensor("layers.0.ffn.experts.1.w1.weight")
            ref2 = r.get_tensor("layers.0.ffn.experts.1.w2.scale")
            rest = {k: f.get_tensor(k) for k in f.keys()}
        assert torch.equal(w1.narrow(0, 576, 576), ref1.narrow(0, 576, 576))
        assert torch.equal(torch.narrow(w1, 0, 600, 10), ref1[600:610])
        assert torch.equal(w1.narrow(-2, 576, 576), ref1[576:1152])
        assert torch.equal(raw_bytes(w2.narrow(1, 18, 18)), raw_bytes(ref2[:, 18:36]))
        assert w1.shape == ref1.shape and w1.size(0) == 2304 and w1.dim() == 2 and w1.device.type == "cpu"
        assert "TpSliceTensor" in repr(w1)
        bad = [lambda: w1.narrow(0, 0, 576), lambda: w1.narrow(0, 576, 577), lambda: w1.narrow(1, 0, 10),
               lambda: w1.narrow(0, 1100, 100), lambda: w2.narrow(0, 0, 10), lambda: w2.narrow(1, 0, 18),
               lambda: w1[0], lambda: w1[576:1152], lambda: w1.view(-1), lambda: w1.clone(),
               lambda: w1.contiguous(), lambda: w1.float(), lambda: w1.numpy(), lambda: w1.t(),
               lambda: w1.to("cpu"), lambda: torch.zeros(2304, 2560, dtype=torch.int8).copy_(w1),
               lambda: torch.equal(w1, ref1), lambda: w1.untyped_storage(), lambda: w1 + 1]
        for i, fn in enumerate(bad):
            try:
                fn()
            except RuntimeError as exc:
                assert "DSV41 fast load" in str(exc), (i, exc)
            else:
                raise AssertionError(f"out-of-slice access #{i} did not raise")
        del w1, w2, rest
        print(f"out-of-slice access: {len(bad)} cases raise")

        # -- pacing counts the resident slice, not the full shape (one buffer per tensor: a view
        #    into a slab charges its whole slab instead, see test_fast_load_slab.py) --
        os.environ["DSV41_FAST_LOAD_SLAB_MB"] = "0"
        with safetensors.safe_open(files[1], framework="pt", device="cpu") as f:
            parts = [f.get_tensor(k) for k in f.keys() if k.endswith(".w1.weight") or k.endswith(".w3.weight")]
        os.environ.pop("DSV41_FAST_LOAD_SLAB_MB", None)
        one = parts[0]._dsv41_resident_nbytes
        full_one = parts[0].numel()
        assert one * 4 == full_one
        os.environ["DSV41_FAST_LOAD_INFLIGHT_GB"] = repr(3 * one / 2**30)

        def orig(*, executor, futures, use_async, func, func_args=(), func_kwargs=None):
            futures.append(executor.submit(func, *func_args, **(func_kwargs or {})))

        mod = types.SimpleNamespace(maybe_executor_submit=orig)
        fl.install_deepseek_v4(mod)
        lock, cur, peak, conc, pconc = threading.Lock(), [0], [0], [0], [0]
        param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        def work(p, t):
            with lock:
                cur[0] += t._dsv41_resident_nbytes; conc[0] += 1
                peak[0], pconc[0] = max(peak[0], cur[0]), max(pconc[0], conc[0])
            time.sleep(0.003)
            with lock:
                cur[0] -= t._dsv41_resident_nbytes; conc[0] -= 1

        futures = []
        with concurrent.futures.ThreadPoolExecutor(24) as ex:
            for i in range(60):
                mod.maybe_executor_submit(executor=ex, futures=futures, use_async=True, func=work,
                                          func_args=(param, parts[i % len(parts)]))
            for fu in concurrent.futures.as_completed(futures):
                fu.result()
        assert peak[0] <= 3 * one and pconc[0] >= 2, (peak[0], pconc[0])
        print(f"pacing: peak {peak[0] / 1e6:.1f} MB <= budget {3 * one / 1e6:.1f} MB with {pconc[0]} slices in "
              f"flight (full-shape accounting would allow 1)")
        del parts

        # -- bounce buffers: one per reader thread, bounded, released --
        threads = int(os.environ.get("DSV41_FAST_LOAD_THREADS", "16"))
        assert 0 < len(fl._bounces) <= threads, len(fl._bounces)
        assert all(len(b) == fl._CHUNK for b in fl._bounces), [len(b) for b in fl._bounces]
        nb = len(fl._bounces)
        fl._release_all()
        assert not fl._bounces and fl._pool is None
        print(f"bounce buffers: {nb} x {fl._CHUNK >> 20} MB, released")
        print("fast_load TP slice OK")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
