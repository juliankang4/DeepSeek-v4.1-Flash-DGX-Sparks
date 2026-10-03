"""Prefill sequence parallel for DeepSeek-V4.1 at TP4 (DSV41_PREFILL_SP=comm | 1).

Today every TP rank runs every row of a prefill chunk through the row-local work (the
hyper-connection post / combine / norm / stats on the [M, 4, 5120] residual, the Engram `wkv`
GEMM and gate), and each sublayer ends with an all-reduce of [M, 5120]. Here, on the full-row
layers (0 .. late_layer_start-1 under decoder SWA bounded replay), the residual stays split by
rows: rank r owns rows [r*M/W, (r+1)*M/W).

  attention  : all-gather x (bf16) -> stock attention on all M rows, wo_b without its
               all-reduce -> reduce-scatter along rows
  MoE        : all-gather x (bf16) -> stock DeepseekV2MoE under mlp_reduce_scatter=True
               (skips the post-experts all-reduce) -> reduce-scatter along rows
  Engram     : full-row lookup (unchanged, so the prefetch still matches) -> reduce-scatter
               -> wkv GEMM + gate on the shard (wkv per (layer, M) bit-exact self-check,
               agreed over TP with MIN, full-M fallback)
  hc post / combine / norm / stats: stock code on the shard; every kernel choice that the
               engine keys on the row count is keyed on the FULL chunk row count
               (sp_logical_rows(), used by _hc_combine / hc_post / _hc_post_with_combine here,
               by hc_fused and by hc_prefill_fused)
  layer 21   : the tail rows of R and prev_pre are gathered (byte copy) and the late layers
               run stock
  no tail    : R and the last pre are gathered at exit

SGLang v0.5.21 prefill paths for 4096 <= M <= 65536 rows, and what happens to them here
  post+combine(+norm)  _hc_post_with_combine fuses hc_post with the next sublayer's collapse
               (and RMSNorm): after attention always, after the MoE when the next layer takes
               the cross-layer hand-off (combined= / normalized=). The gate keys on the chunk
               row count M here. On the shard the post/combine is the engine's own Triton
               _mhc_post_combine and the norm its _hc_norm_prefill (both one row per program,
               no row-count check), which is what the engine itself runs for these rows when
               the 16-byte alignment check fails; the fused CUDA kernel it prefers refuses
               fewer than 4096 rows. That the two are bit-identical is the engine's design,
               CHECKED here, not assumed: the first chunk of every size compares the shard result
               with the stock fused kernel on the gathered rows, agrees over TP (MIN), and on any
               difference serves that chunk exactly (gathered stock) and runs later chunks of that
               size stock (no SP).
  cross-layer collapse  the copied loop takes the engine's decision (Engram-free next layer,
               no bounded-replay tail, extend) on M, so the shard runs it whenever stock does.
               Under bounded replay (production) the engine never takes it during prefill.
  prefill stats stream  the hc stats of a sublayer go to a side stream and start just before
               the sublayer's all-reduce (MhcPostFusion overlap_only, which moves the attention
               all-reduce out of wo_b). Inside the region there is no such all-reduce to overlap
               (it is a reduce-scatter here), so _get_hc_stats_stream answers None there and the
               stats run on the current stream: the same kernels, so the same bits.

Modes
  DSV41_PREFILL_SP=comm  P0, transport only: nothing is sharded; the same three reduction sites
                         (wo_b, MoE output, Engram lookup) run reduce-scatter + all-gather
                         instead of the all-reduce. Same numerics as stage 1 fast by
                         construction (same reduce-scatters on the same tensors).
  DSV41_PREFILL_SP=1     stage 1, bf16 gathers.
  DSV41_PREFILL_SP_EXACT=1  (stage 1) every reduce-scatter is all-reduce + keep the quarter:
                         must reproduce production bit for bit.
  DSV41_PREFILL_SP_MIN_ROWS  default 2048; chunks below it or not divisible by the TP size run
                         the stock path in every mode (same predicate).
  DSV41_PREFILL_SP_WKV=full  never run wkv on the shard (gather the lookup, full-M GEMM).

Scope: extend without speculation only (plain prefill chunks), eager (never under graph
capture or torch.compile), TP only (no DP attention, no CP, no a2a MoE, PP 1). Decode,
verify, draft and CUDA graphs never enter this code. Every decision depends only on shape,
forward mode and static config, so it is identical on every rank.

Drift guard: sha256[:16] of the engine sources this relies on are checked at install; any
difference raises (refuses to boot). hc_prefill_fused, when enabled, must key its gates on
sp_logical_rows() (it declares SP_LOGICAL_ROWS = True); otherwise install raises.

Engine: SGLang v0.5.21 (+ RoCE and GB10 patches). Ported from the dsv4.1 f80c91a4b version.
"""
import hashlib
import inspect
import logging
import os
import threading
from contextlib import nullcontext

import torch

logger = logging.getLogger(__name__)

_RAW = os.environ.get("DSV41_PREFILL_SP", "0").strip().lower()
MODE = "comm" if _RAW == "comm" else ("shard" if _RAW in ("1", "on", "true", "shard") else "off")
EXACT = os.environ.get("DSV41_PREFILL_SP_EXACT", "0").strip().lower() in ("1", "on", "true")
MIN_ROWS = int(os.environ.get("DSV41_PREFILL_SP_MIN_ROWS", "2048"))
WKV_SHARD = os.environ.get("DSV41_PREFILL_SP_WKV", "shard").strip().lower() != "full"
# Stage 2 (DSV41_PREFILL_SP_FP8=1, with DSV41_PREFILL_SP=1): on full-row layers whose attention
# reads its input only through wqkv_a (no compressor / indexer: all but the kv/index sources 2, 8,
# 14, 20), each rank MXFP8-quantizes its shard (per row, per 32 columns; linear scale layout),
# all-gathers the fp8 bytes + ue8m0 scales, swizzles the scales to 128x4 and hands attention the
# pre-quantized input (x_quant) instead of the bf16 rows. Exact by construction if the quantizer is
# row-local; checked, not assumed: the first chunk of every size compares the gathered MXFP8 input
# with the stock quantization of the gathered bf16 rows AND wqkv_a's output on both, agrees over
# TP (MIN), and keeps the bf16 gather for that size on any difference.
FP8 = os.environ.get("DSV41_PREFILL_SP_FP8", "0").strip().lower() in ("1", "on", "true")
# Debug (never in production): DSV41_PREFILL_SP_DEBUG=compare,fp (either or both).
#   fp       per traced prefill chunk, rank 0 logs the chunk layout and a checksum of every
#            layer's input / attention in+out / MoE in+out / output residual and pre, over the full
#            rows. Works with DSV41_PREFILL_SP unset too (then only the tracing is installed), so an
#            off boot and an SP boot can be diffed chunk by chunk.
#   compare  (stage 1) next to every sharded op, the stock op runs on the same (gathered) inputs
#            and every rank logs each op whose bits differ: layer, op, differing elements, max abs.
#            wo_b and the MoE also run their stock all-reduce path on the same partials.
# DSV41_PREFILL_SP_DEBUG_CHUNKS  traced chunks (default 64, every extend forward counts)
# DSV41_PREFILL_SP_DEBUG_DUMP    directory: torch.save of the first mismatch per op and rank
DEBUG = {t.strip() for t in os.environ.get("DSV41_PREFILL_SP_DEBUG", "").lower().split(",") if t.strip()}
if "compare" in DEBUG:
    DEBUG.add("fp")
DEBUG_CHUNKS = int(os.environ.get("DSV41_PREFILL_SP_DEBUG_CHUNKS", "64"))
DEBUG_DUMP = os.environ.get("DSV41_PREFILL_SP_DEBUG_DUMP", "").strip()

# Engine sources (SGLang v0.5.21 + RoCE + GB10 patches), sha256[:16] of
# inspect.getsource(inspect.unwrap(obj)). The loop below is a copy of the first; the others are
# the call sites and kernel gates whose behaviour the exactness argument depends on.
# Values computed from the v0.5.21 source tree, NOT yet in the image: confirm with
# NEW_HASHES.md's script before shipping (a difference refuses to boot, it never runs wrong).
_EXPECTED = {
    "v4.DeepseekV4Model._forward_layers_hc_pre_from_prev": "64f2801ffd57d158",
    "v4.DeepseekV4DecoderLayer.forward_hc_pre_from_prev": "49f22482d5370465",
    "v4.DeepseekV4DecoderLayer._hc_combine": "02b26ea5f805ed54",
    "v4.DeepseekV4DecoderLayer._hc_mix_stats": "a94d95cd06bd9676",
    "v4.DeepseekV4DecoderLayer.hc_post": "5f5dc4cf67cf091b",
    "v4.DeepseekV4DecoderLayer._hc_post_with_combine": "0a23e4a2729b5adc",
    "v4.DeepseekV4DecoderLayer._get_hc_stats_stream": "9a92da0e2ce75ab9",
    "v4.DeepseekV4DecoderLayer._run_moe_ffn_dp_sync": "1100ee58ff5fc3ff",
    "v4.DeepseekV4Model.forward": "8fe30b4e1874a893",
    "v4.MQALayer.forward": "78d101dc4c4baec9",
    "v2.DeepseekV2MoE.forward": "79f4c814db857b0d",
    "v2.DeepseekV2MoE.forward_normal": "6771a662f03d71ab",
    "engram.Engram.forward": "19df65b564f45d15",
    "engram.EngramEmbedding.forward": "138d015825c95c0e",
    "engram.EngramEmbedding._lookup": "bd991c49e8e3e8dc",
    "engram.engram_gate": "34b0bfeb52ab1218",
    "hcn.hc_combine_norm": "f6dfc00796090fc3",
    "hcn._hc_combine_norm_prefill": "74f71ebf85b010ee",
    "mpc.mhc_post_combine": "a3f32180a2fbe7c8",
    "mpc._mhc_post_combine": "a634b84e16d3fbbb",
    "mpc.hc_norm_prefill": "33aacdba1d46530c",
    "mpc._hc_norm_prefill": "dea91349eed3279d",
    "mpcn.mhc_post_combine_norm_prefill": "6b353220faf832f0",
    "linear.RowParallelLinear.forward": "6e429216a343a251",
}
# Non-Python engine sources (sha256[:16] of the file text), relative to sglang/kernels/jit/csrc:
# the fused prefill post/combine/norm kernel the shard's Triton pair is checked against.
_EXPECTED_FILES = {
    "deepseek_v4/mhc_post_combine_norm_prefill.cuh": "a8805d1f7b8a2ebc",
}


# sglang.srt.layers.attention.deepseek_v4_backend: the tail-row semantics _tail_rows reproduces.
# Stage 2 additionally relies on these (checked at install when DSV41_PREFILL_SP_FP8=1): which
# inputs attention's prefill path reads, and when it takes a pre-quantized input.
# v0.5.21 changed only _forward_prepare, in its CP and NPU branches (excluded by the plan / CUDA);
# the CUDA non-CP path still reads x only through wqkv_a (x_linear) plus x.dtype / x.shape.
_EXPECTED_FP8 = {
    "v4.MQALayer._forward_prepare": "33d18c9cc577a883",
    "v4.MQALayer.accepts_mxfp8_swizzled_input": "5bb4fddfc5b329ca",
    "v4.MQALayer._compute_kv_to_cache": "f52f600c4986428c",
    "v4.MQALayer._compute_kv_bf16": "00c4ab90b403d1e9",
}
_EXPECTED_TAIL = {
    "LateLayerTail.rows": "c58392d3ef651408",
    "LateLayerTail.real_rows": "c78962c2e01e0e3d",
    "_tail_rows": "54a3cbff5e674e84",
}


class _Ctx(threading.local):
    def __init__(self):
        self.plan = None        # _Plan while the layer loop is inside the full-row region
        self.inner = 0          # > 0 inside a gathered attention / MoE call
        self.skip_wo_b = None   # the wo_b whose all-reduce the current attention call skips
        self.dbg = None         # _Dbg while a traced chunk runs


_ctx = _Ctx()
_M = {}                         # engine modules, filled by install
_STATIC = {"checked": False}
_WKV_OK = {}                    # (layer_id, M) -> bool (agreed over TP)
_PCN_OK = {}                    # M -> bool: shard post+combine+norm == stock fused (agreed over TP)
_SEEN_M = set()
_REQUIRE_CUDA = True            # CPU tests switch this off
_DBG = {"chunks": 0, "dumped": 0}


def _tp_group():
    """The TP GroupCoordinator. v0.5.21 deprecates get_tp_group() (deepseek_v4 no longer imports
    it); it is get_parallel().tp_group, the group every engine all-reduce here runs on (_ATTN_TP
    and _MOE_TP are this same object at attn_tp == moe_tp == tp)."""
    return _M["v4"].get_parallel().tp_group


def _no_mhc_fusion():
    """Clear the MhcPostFusion hand-off around a gathered attention / MoE call. In v0.5.21 a set
    overlap_only fusion makes MQALayer run its all-reduce outside wo_b (attn_tp_all_reduce), which
    would reduce what this adapter reduce-scatters. The region never sets one (no stats stream,
    and the fused ones are for <= 8 / 128..384-row decode and verify), so this is a guard: with it
    cleared the layer still materializes its stats from the fusion object after the call."""
    mpf = _M.get("mpf")
    return mpf.use_mhc_post_fusion(None) if mpf is not None else nullcontext()


def _agree_min(p, local):
    """Rank-agreed AND of a local boolean (MIN over the TP group)."""
    flag = torch.tensor([1 if local else 0], dtype=torch.int32, device=torch.cuda.current_device())
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=p.group.device_group)
    return bool(flag.item())


def _mega_moe(moe, x):
    try:
        from sglang.srt.layers.moe.mega_moe import should_use_mega_moe
    except ImportError:
        return False
    return should_use_mega_moe(moe, x)


class _Dbg:
    def __init__(self, idx, rows, plan, rank):
        self.idx, self.rows, self.plan, self.rank = idx, rows, plan, rank
        self.compare = "compare" in DEBUG and plan is not None and plan.sharded
        self.layer = -1
        self.fps = []           # (layer, key, fp)
        self.checked = 0
        self.bad = []           # (layer, op, ndiff, numel, maxabs)
        self.dumped = set()


def _fp(t):
    """Order-sensitive checksum of a tensor's bytes (GPU, chunked int64 sums)."""
    if t is None:
        return "none"
    t = t.contiguous().reshape(-1)
    nbytes = t.numel() * t.element_size()
    v = t.view(torch.int32) if nbytes % 4 == 0 else t.view(torch.uint8)
    s1 = torch.zeros((), dtype=torch.int64, device=v.device)
    s2 = torch.zeros((), dtype=torch.int64, device=v.device)
    step = 1 << 22
    for i in range(0, v.numel(), step):
        c = v[i:i + step].to(torch.int64)
        w = torch.arange(i, i + c.numel(), device=v.device, dtype=torch.int64) % 65521 + 1
        s1 += c.sum()
        s2 += (c * w).sum()
    return f"{(int(s1) * 1000003 + int(s2)) & ((1 << 48) - 1):012x}"


def _full_rows(t):
    """t over every row of the chunk: gathered when it is this rank's shard (collective)."""
    p = _ctx.plan
    if (t is not None and p is not None and p.sharded and isinstance(t, torch.Tensor)
            and t.dim() >= 1 and t.shape[0] == p.shard and p.shard != p.rows):
        return _gather_rows(p, t)
    return t


def _rec(key, t, full=False):
    d = _ctx.dbg
    if d is None:
        return
    d.fps.append((d.layer, key, _fp(t if full else _full_rows(t))))


class _Inner:
    def __enter__(self):
        _ctx.inner += 1

    def __exit__(self, *a):
        _ctx.inner -= 1


def _comparing():
    d = _ctx.dbg
    return d is not None and d.compare and _ctx.plan is not None and _ctx.plan.sharded


def _cmp(op, ours, ref, inputs=None):
    """Bits of our result vs the stock op's on the same inputs; log and dump differences."""
    d = _ctx.dbg
    if d is None:
        return
    outs = ours if isinstance(ours, (tuple, list)) else (ours,)
    refs = ref if isinstance(ref, (tuple, list)) else (ref,)
    d.checked += 1
    for k, (a, b) in enumerate(zip(outs, refs)):
        name = op if len(outs) == 1 else f"{op}[{k}]"
        if a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b):
            continue
        if a.shape != b.shape:
            nd, mx = -1, float("nan")
        else:
            diff = (a.float() - b.float()).abs()
            nd, mx = int((a != b).sum()), float(diff.max())
        d.bad.append((d.layer, name, nd, a.numel(), mx))
        if len(d.bad) <= 24:
            print(f"DSV41_PREFILL_SP compare MISMATCH chunk={d.idx} M={d.rows} rank={d.rank} "
                  f"layer={d.layer} op={name} shape={tuple(a.shape)}/{tuple(b.shape)} "
                  f"differing={nd}/{a.numel()} max_abs={mx:.4g}", flush=True)
        if DEBUG_DUMP and op not in d.dumped and _DBG["dumped"] < 16:
            d.dumped.add(op)
            _DBG["dumped"] += 1
            try:
                os.makedirs(DEBUG_DUMP, exist_ok=True)
                path = os.path.join(DEBUG_DUMP, f"psp_c{d.idx}_r{d.rank}_L{d.layer}_{op.replace(' ', '_')}.pt")
                blob = {"op": name, "layer": d.layer, "chunk": d.idx, "rows": d.rows, "rank": d.rank,
                        "ours": a.detach().cpu(), "ref": b.detach().cpu()}
                if inputs:
                    blob["inputs"] = {n: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                                      for n, v in inputs.items()}
                torch.save(blob, path)
                print(f"DSV41_PREFILL_SP compare dumped {path}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"DSV41_PREFILL_SP compare dump failed: {exc!r}", flush=True)


def _debug_begin(model, hidden_states, forward_batch, input_ids, p):
    if not DEBUG or _DBG["chunks"] >= DEBUG_CHUNKS:
        return None
    if torch.compiler.is_compiling() or (torch.cuda.is_available()
                                         and torch.cuda.is_current_stream_capturing()):
        return None
    if not forward_batch.forward_mode.is_extend_without_speculative():
        return None
    _DBG["chunks"] += 1
    rank = p.rank if p is not None else _tp_group().rank_in_group
    d = _Dbg(_DBG["chunks"], hidden_states.shape[0], p, rank)
    if rank == 0:
        lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        pref = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        mode = "off" if p is None else (MODE + ("-exact" if p.exact else "") if p.sharded else MODE)
        print(f"DSV41_PREFILL_SP fp chunk={d.idx} M={d.rows} mode={mode} reqs={len(lens) if lens is not None else '?'} "
              f"extend={list(lens)[:16] if lens is not None else '?'} prefix={list(pref)[:16] if pref is not None else '?'} "
              f"ids={_fp(input_ids)} R0={_fp(hidden_states)}", flush=True)
    return d


def _debug_end(d, out):
    try:
        hs, pre = out[0], out[1]
        fh, fp_ = _fp(hs), _fp(pre)
    except Exception as exc:  # noqa: BLE001
        fh = fp_ = f"err {exc!r}"
    if d.rank == 0:
        by_layer = {}
        for layer, key, f in d.fps:
            by_layer.setdefault(layer, []).append(f"{key}={f}")
        for layer, items in by_layer.items():
            print(f"DSV41_PREFILL_SP fp chunk={d.idx} L={layer} " + " ".join(items), flush=True)
        print(f"DSV41_PREFILL_SP fp chunk={d.idx} out R={fh} pre={fp_}", flush=True)
    if d.compare:
        first = f" first: layer {d.bad[0][0]} {d.bad[0][1]}" if d.bad else ""
        print(f"DSV41_PREFILL_SP compare chunk={d.idx} M={d.rows} rank={d.rank}: {d.checked} ops checked, "
              f"{len(d.bad)} mismatched{first}", flush=True)


def _make_layer_forward(orig):
    """Tracing only: record each layer's input / output residual and pre."""
    def forward_hc_pre_from_prev(self, *args, **kwargs):
        d = _ctx.dbg
        if d is None or _ctx.inner:
            return orig(self, *args, **kwargs)
        hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[1]
        d.layer = getattr(self, "layer_id", d.layer + 1)
        _rec("Rin", hs)
        out = orig(self, *args, **kwargs)
        _rec("Rout", out[0])
        _rec("pre", out[1])
        return out

    forward_hc_pre_from_prev.__wrapped__ = orig
    return forward_hc_pre_from_prev


class _Plan:
    __slots__ = ("rows", "world", "rank", "shard", "lo", "hi", "group", "sharded", "exact")

    def __init__(self, rows, world, rank, group, sharded, exact):
        self.rows, self.world, self.rank, self.group = rows, world, rank, group
        self.shard = rows // world
        self.lo, self.hi = rank * self.shard, (rank + 1) * self.shard
        self.sharded, self.exact = sharded, exact

    @property
    def in_rows(self):
        return self.shard if self.sharded else self.rows


# ------------------------------------------------------------------------------------------
# the logical-row helper (kernel gates key on the full chunk row count)
# ------------------------------------------------------------------------------------------
def sp_logical_rows(x):
    """Row count a kernel gate must use for `x`: the full chunk row count M when `x` is this
    rank's shard of a sequence-parallel chunk, else x.shape[0]. Cheap and safe to call when
    the adapter is off."""
    n = x.shape[0]
    p = _ctx.plan
    if p is None or _ctx.inner or not p.sharded:
        return n
    return p.rows if n == p.shard else n


def enabled():
    return MODE != "off"


# ------------------------------------------------------------------------------------------
# row partition (pure; unit-tested on CPU)
# ------------------------------------------------------------------------------------------
def eligible_rows(rows, world, min_rows=None):
    """Shape predicate shared by every mode: rows >= MIN_ROWS and divisible by the TP size."""
    min_rows = MIN_ROWS if min_rows is None else min_rows
    return world > 1 and rows >= min_rows and rows % world == 0


def shard_range(rows, world, rank):
    s = rows // world
    return rank * s, (rank + 1) * s


def tail_plan(rows, world, tail_len):
    """'full' when gathering all rows is no more traffic than gathering world * T rows."""
    return "full" if world * tail_len >= rows else "owners"


def owner_local(idx, shard):
    owner = torch.div(idx, shard, rounding_mode="floor")
    return owner, idx - owner * shard


# ------------------------------------------------------------------------------------------
# collectives (the group is a sglang GroupCoordinator; tests pass a fake one)
# ------------------------------------------------------------------------------------------
def _gather_rows(p, x):
    """[S, ...] shard on every rank -> [M, ...] (byte copy)."""
    x = x.contiguous()
    out = x.new_empty((p.rows, *x.shape[1:]))
    p.group.all_gather_into_tensor(out, x)
    return out


def _reduce_rows(p, x):
    """[M, ...] per-rank partial -> this rank's reduced [S, ...] rows."""
    x = x.contiguous()
    if p.exact:
        full = p.group.all_reduce(x)
        return full[p.lo:p.hi]
    out = x.new_empty((p.shard, *x.shape[1:]))
    p.group.reduce_scatter_tensor(out, x)
    return out


def _reduce_out(p, x):
    """Output of a reduction site: this rank's rows (stage 1) or every row (P0: RS + AG)."""
    if p.sharded:
        return _reduce_rows(p, x)
    x = x.contiguous()
    part = x.new_empty((p.shard, *x.shape[1:]))
    p.group.reduce_scatter_tensor(part, x)
    return _gather_rows(p, part)


def _tail_real_rows(p, tail, t):
    """tail.real_rows(concat of every rank's shard t), as a byte copy."""
    if tail.contiguous_start is not None:
        start, idx = int(tail.contiguous_start), None
        n_tail = p.rows - start
    else:
        start, idx = None, tail.token_indices
        n_tail = int(idx.shape[0])
    if tail_plan(p.rows, p.world, n_tail) == "full" or n_tail == 0:
        full = _gather_rows(p, t)
        return full[start:] if idx is None else full[idx]
    if idx is None:
        idx = torch.arange(start, p.rows, device=t.device)
    owner, local = owner_local(idx.to(torch.int64), p.shard)
    mine = t.contiguous()[local.clamp(0, p.shard - 1)].contiguous()
    buf = t.new_empty((p.world * n_tail, *t.shape[1:]))
    p.group.all_gather_into_tensor(buf, mine)
    buf = buf.view(p.world, n_tail, *t.shape[1:])
    return buf[owner, torch.arange(n_tail, device=t.device)]


def _tail_rows(p, tail, t):
    rows = _tail_real_rows(p, tail, t)
    if tail.pad_rows:
        rows = torch.cat([rows, rows.new_zeros((tail.pad_rows, *rows.shape[1:]))])
    return rows


# ------------------------------------------------------------------------------------------
# drift guard
# ------------------------------------------------------------------------------------------
def _src_hash(obj):
    obj = getattr(obj, "fn", obj)                       # triton JITFunction
    obj = inspect.unwrap(obj)
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()[:16]


def _resolve(name):
    mod, attr = name.split(".", 1)
    obj = _M[mod]
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def engine_hashes():
    return {k: _src_hash(_resolve(k)) for k in _EXPECTED}


def _file_hash(rel):
    path = os.path.join(_M["jit_csrc"], rel)
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        return f"unreadable ({exc.__class__.__name__})"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def check_engine():
    exp = dict(_EXPECTED)
    if FP8:
        exp.update(_EXPECTED_FP8)
    got = {k: _src_hash(_resolve(k)) for k in exp}
    bad = [f"{k} {got[k]} != {exp[k]}" for k in got if exp[k] != got[k]]
    for rel, want in _EXPECTED_FILES.items():
        have = _file_hash(rel)
        if have != want:
            bad.append(f"csrc/{rel} {have} != {want}")
    if bad:
        raise RuntimeError("DSV41_PREFILL_SP: engine source drifted, refusing to run: " + "; ".join(bad))


def _load_modules(model_mod):
    import sglang.kernels.jit as jit
    from sglang.kernels.ops.layernorm import hc_combine_norm as hcn
    from sglang.kernels.ops.layernorm import mhc
    from sglang.kernels.ops.layernorm import mhc_post_combine as mpc
    from sglang.kernels.ops.layernorm import mhc_post_combine_norm_prefill as mpcn
    from sglang.srt.layers import engram, linear
    from sglang.srt.layers.moe import mhc_post_fusion as mpf
    from sglang.srt.models import deepseek_v2 as v2

    _M.update(v4=model_mod, v2=v2, engram=engram, hcn=hcn, mhc=mhc, linear=linear, mpc=mpc,
              mpcn=mpcn, mpf=mpf, jit_csrc=os.path.join(os.path.dirname(jit.__file__), "csrc"))


def _check_hc_prefill_fused():
    if os.environ.get("DSV41_HC_PREFILL_FUSED", "0").strip() in ("0", "off", "false", ""):
        return
    import hc_prefill_fused as hpf
    if not getattr(hpf, "SP_LOGICAL_ROWS", False):
        raise RuntimeError(
            "DSV41_PREFILL_SP=1 with DSV41_HC_PREFILL_FUSED=1 needs hc_prefill_fused to key its "
            "row gates on prefill_sp.sp_logical_rows() (and declare SP_LOGICAL_ROWS = True); "
            "refusing to run. See diagnostics/dsv41-prefill-sp/IMPL.md.")


# ------------------------------------------------------------------------------------------
# plan (rank-invariant: shape, forward mode, static config)
# ------------------------------------------------------------------------------------------
def _check_tail_engine():
    from sglang.srt.layers.attention import deepseek_v4_backend as b
    bad = []
    for k, want in _EXPECTED_TAIL.items():
        obj = b
        for part in k.split("."):
            obj = getattr(obj, part)
        got = _src_hash(obj)
        if got != want:
            bad.append(f"deepseek_v4_backend.{k} {got} != {want}")
    if bad:
        raise RuntimeError("DSV41_PREFILL_SP: engine source drifted, refusing to run: " + "; ".join(bad))


def _static_check(model, world):
    v4 = _M["v4"]
    if not _STATIC.get("tail_checked"):
        _check_tail_engine()
        _STATIC["tail_checked"] = True
    par = v4.get_parallel()
    errs = []
    if model.pp_group.world_size != 1:
        errs.append("PP > 1")
    if par.attn_dp_size != 1:
        errs.append(f"attn_dp_size {par.attn_dp_size}")
    if par.tp_size != world or par.attn_tp_size != world:
        errs.append(f"tp {par.tp_size} / attn_tp {par.attn_tp_size} / group {world}")
    if not v4.get_moe_a2a_backend().is_none():
        errs.append("a2a MoE backend")
    if not model.hc_pre_from_prev_sublayer or model.hc_mult != 4:
        errs.append("hc layout")
    if v4.get_platform().is_sm100:
        errs.append("sm100 (_hc_mix_stats bf16x3/deepgemm gates are row-count keyed)")
    if MODE == "comm" and v4.envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get():
        errs.append("deterministic inference (reduce_scatter_tensor is all_reduce there)")
    if MIN_ROWS < 256:
        errs.append("DSV41_PREFILL_SP_MIN_ROWS below 256 would reach decode/verify sizes")
    last = model.late_layer_start if model.late_layer_start is not None else model.end_layer
    for i in range(model.start_layer, last):
        layer = model.layers[i]
        attn, mlp = layer.self_attn, layer.mlp
        if not isinstance(attn, v4.MQALayer) or not isinstance(mlp, _M["v2"].DeepseekV2MoE):
            errs.append(f"layer {i}: {type(attn).__name__}/{type(mlp).__name__}")
            continue
        rpl = getattr(_M.get("linear"), "RowParallelLinear", None)
        if rpl is not None and not isinstance(attn.wo_b, rpl):
            errs.append(f"layer {i}: wo_b is {type(attn.wo_b).__name__}")
        if not (attn.wo_b.reduce_results and attn.wo_b.tp_size == world and attn.attn_tp_size == world):
            errs.append(f"layer {i}: wo_b reduce/tp")
        if getattr(mlp, "_shared_expert_tp1", False) or getattr(mlp, "_enable_a2a_moe", False):
            errs.append(f"layer {i}: TP1 shared expert or a2a MoE")
        if mlp.tp_size != world:
            errs.append(f"layer {i}: moe tp {mlp.tp_size}")
        if getattr(layer, "dsa_enable_prefill_cp", False) or layer.use_fused_mhc_post_pre:
            errs.append(f"layer {i}: prefill CP / fused post-pre")
        if layer.engram is not None and layer.engram.embed.tp_size != world:
            errs.append(f"layer {i}: engram tp")
    if errs:
        raise RuntimeError("DSV41_PREFILL_SP: unsupported configuration, refusing to run: " + "; ".join(errs))
    if FP8 and MODE == "shard" and world > 1:
        fp8_layers = [i for i in range(model.start_layer, last)
                      if fp8_eligible_attn(model.layers[i].self_attn)]
        if _tp_group().rank_in_group == 0:
            print(f"DSV41_PREFILL_SP fp8 attention gather on layers {fp8_layers} "
                  f"({len(fp8_layers)} of {last - model.start_layer}; the rest gather bf16)", flush=True)
    _STATIC["checked"] = True


def _plan(model, hidden_states, forward_batch):
    if MODE == "off":
        return None
    v4 = _M["v4"]
    if torch.compiler.is_compiling() or (torch.cuda.is_available()
                                         and torch.cuda.is_current_stream_capturing()):
        return None
    if not forward_batch.forward_mode.is_extend_without_speculative():
        return None
    if v4.is_in_breakable_cuda_graph() or v4.is_cp_active(forward_batch):
        return None
    if v4.get_attn_tp_context().input_scattered:
        return None
    if not (hidden_states.dim() == 3 and hidden_states.dtype == torch.bfloat16
            and (hidden_states.is_cuda or not _REQUIRE_CUDA) and hidden_states.is_contiguous()):
        return None
    group = _tp_group()
    rows, world = hidden_states.shape[0], group.world_size
    if not eligible_rows(rows, world):
        return None
    if MODE == "shard" and _PCN_OK.get(rows) is False:
        return None             # the shard post+combine+norm differed from stock at this size
    if not _STATIC["checked"]:
        _static_check(model, world)
    sharded = MODE == "shard"
    p = _Plan(rows, world, group.rank_in_group, group, sharded, EXACT and sharded)
    if rows not in _SEEN_M:
        _SEEN_M.add(rows)
        if p.rank == 0 and len(_SEEN_M) <= 8:
            print(f"DSV41_PREFILL_SP {MODE}{' exact' if p.exact else ''}: chunk of {rows} rows -> "
                  f"{'shards of ' + str(p.shard) if sharded else 'reduce-scatter + all-gather'}",
                  flush=True)
    return p


# ------------------------------------------------------------------------------------------
# Engram on the shard
# ------------------------------------------------------------------------------------------
def _wkv_shard(p, engram, emb):
    """kv rows [lo, hi) from the shard's lookup rows, bit-identical to the full-M GEMM."""
    if not WKV_SHARD:
        return engram.wkv(_gather_rows(p, emb).flatten(-2))[0][p.lo:p.hi]
    key = (engram.layer_hash_index, p.rows)
    ok = _WKV_OK.get(key)
    if ok is None:
        full = engram.wkv(_gather_rows(p, emb).flatten(-2))[0]
        mine = full[p.lo:p.hi].clone()
        del full                # the [M, 25600] output is not held across the agreement
        part = engram.wkv(emb.flatten(-2))[0]
        local = bool(torch.equal(mine, part))
        ok = _WKV_OK[key] = _agree_min(p, local)
        if p.rank == 0 and (len(_WKV_OK) <= 4 or not ok):
            print(f"DSV41_PREFILL_SP wkv (engram {engram.layer_hash_index}, M={p.rows}): shard GEMM "
                  f"{'bit-exact on every rank -> ON' if ok else 'differs -> full-M GEMM'} "
                  f"(rank 0 local {local})", flush=True)
        return part if ok else mine
    if ok:
        return engram.wkv(emb.flatten(-2))[0]
    return engram.wkv(_gather_rows(p, emb).flatten(-2))[0][p.lo:p.hi]


def _engram_lookup_partial(engram, ids):
    embed = engram.embed
    if embed._shared:           # host-shared table: every rank already has every row
        return embed(ids), False
    return embed._owned_rows(ids), True


def _engram_gate(engram, x, kv):
    """The gate Engram.forward applies (v0.5.21 calls engram_gate directly; apply_gate is gone)."""
    return _M["engram"].engram_gate(x, kv, engram.q_weight, engram.k_weight, engram.eps,
                                    engram.clamp_value)


def _sp_engram(p, engram, x, ids, forward_batch=None):
    """Stock Engram.forward on a shard of x: lookup on all rows, reduce, wkv + gate on rows."""
    emb, partial = _engram_lookup_partial(engram, ids)
    if p.sharded:
        emb = _reduce_rows(p, emb) if partial else emb[p.lo:p.hi]
        kv = _wkv_shard(p, engram, emb)
        out = _engram_gate(engram, x, kv.contiguous())
        if _comparing():
            with _Inner():
                xg = _gather_rows(p, x)
                # slices are cloned at once: the [M, ...] references are not held across collectives
                ref_kv = engram.wkv(_gather_rows(p, emb).flatten(-2))[0][p.lo:p.hi].clone()
                ref = engram(xg, ids, forward_batch, cp_all_tokens=False)[p.lo:p.hi].clone()
                del xg
            _cmp("engram wkv", kv, ref_kv)
            _cmp("engram out", out, ref, {"x": x, "kv": kv})
        return out
    if partial:                 # P0: reduce-scatter + all-gather in place of the all-reduce
        emb = _reduce_out(p, emb)
    kv, _ = engram.wkv(emb.flatten(-2))
    return _engram_gate(engram, x, kv)


# ------------------------------------------------------------------------------------------
# the layer loop (copy of DeepseekV4Model._forward_layers_hc_pre_from_prev, hash-pinned)
# ------------------------------------------------------------------------------------------
def _sp_forward_layers(self, p, positions, hidden_states, forward_batch, input_ids,
                       input_ids_global, capture_dspark, dspark_aux_hidden_states):
    v4 = _M["v4"]
    assert self.pp_group.world_size == 1, "pre-mix hand-off across PP is not wired"
    # The plan excludes prefill CP (whole-prompt hashing, cp_all_tokens) and breakable CUDA
    # graphs (graph-side hash ids): the engine's plain hasher call is the one that applies.
    hash_ids = None
    if self.engram_hasher is not None:
        hash_ids = self.engram_hasher(input_ids, forward_batch)
    tail = None
    attn_backend = None
    if self.late_layer_start is not None and forward_batch.forward_mode.is_extend_without_speculative():
        self._check_late_layer_tail_readers(forward_batch)
        attn_backend = v4.get_attn_backend()
        tail = attn_backend.tail_forward_metadata.late_layer_tail
    in_region = True
    R = hidden_states[p.lo:p.hi] if p.sharded else hidden_states
    ids_rows = None
    saved_full = None
    prev_pre = None
    precomputed_attn = None
    combined_attn = None
    normalized_attn = None
    _ctx.plan = p
    try:
        for i in range(self.start_layer, self.end_layer):
            if tail is not None and i == self.late_layer_start:
                combined_attn = None
                normalized_attn = None
                saved_full = attn_backend.enter_late_layer_tail(forward_batch)
                _ctx.plan = None
                in_region = False
                if p.sharded:
                    R_sh, pre_sh = R, prev_pre
                    R, prev_pre = _tail_rows(p, tail, R), _tail_rows(p, tail, prev_pre)
                    if _ctx.dbg is not None and _ctx.dbg.compare:
                        _cmp("tail rows R", R, tail.rows(_gather_rows(p, R_sh)))
                        _cmp("tail rows pre", prev_pre, tail.rows(_gather_rows(p, pre_sh)))
                    del R_sh, pre_sh
                else:
                    R, prev_pre = tail.rows(R), tail.rows(prev_pre)
                input_ids, input_ids_global = tail.rows(input_ids), tail.rows(input_ids_global)
                positions = tail.positions
                if hash_ids is not None:
                    hash_ids = tail.rows(hash_ids)
            engram = self.layers[i].engram
            if engram is not None:
                precomputed_attn = None
                combined_attn = None
                normalized_attn = None
                before_engram = R
                ids = hash_ids[:, engram.layer_hash_index]
                if in_region:
                    R = _sp_engram(p, engram, R, ids, forward_batch)
                else:
                    R = engram(R, ids, forward_batch, cp_all_tokens=False)
                if self.config.model_type == "deepseek_v41" and self.config.vision_n_layers > 0:
                    if in_region and p.sharded:
                        if ids_rows is None:
                            ids_rows = input_ids[p.lo:p.hi]
                        vis = ids_rows
                    else:
                        vis = input_ids
                    R = torch.where((vis == self.config.image_token_id)[:, None, None],
                                    before_engram, R)
            if capture_dspark and i in self.dspark_layers_to_capture:
                aux = R
                if in_region and p.sharded:
                    aux = _tail_rows(p, tail, R) if tail is not None else _gather_rows(p, R)
                elif tail is not None and i < self.late_layer_start:
                    aux = tail.rows(aux)
                dspark_aux_hidden_states.append(aux.mean(dim=1))
            ctx = (
                nullcontext()
                if v4.check_cuda_graph_backend(v4.Phase.PREFILL, v4.Backend.TC_PIECEWISE)
                else v4.get_global_expert_distribution_recorder().with_current_layer(i)
            )
            # Cross-layer collapse (v0.5.21): the next layer takes this layer's FFN post fused with
            # its own input collapse (+ RMSNorm) when no Engram or row selection lies in between.
            # The engine keys it on the row count: here on the chunk's (M), on every rank alike.
            rows = p.rows if in_region else R.shape[0]
            next_norm = None
            next_input = []
            next_combined = (
                []
                if (
                    self.config.model_type == "deepseek_v41"
                    and (
                        128 <= rows <= 384
                        or (
                            4096 <= rows <= 65536
                            and forward_batch.forward_mode.is_extend_without_speculative()
                        )
                    )
                    and i + 1 < self.end_layer
                    and tail is None
                    and self.layers[i + 1].engram is None
                )
                else None
            )
            if next_combined is not None and rows >= 4096:
                next_norm = self.layers[i + 1].input_layernorm
            # The engine's other next_norm (<= 8 rows, decode / target verify) never applies to a
            # prefill chunk; next_input is only filled by those fused decode paths.
            with ctx:
                R, prev_pre = self.layers[i].forward_hc_pre_from_prev(
                    positions=positions,
                    hidden_states=R,
                    input_ids=input_ids,
                    forward_batch=forward_batch,
                    input_ids_global=input_ids_global,
                    prev_pre=prev_pre,
                    precomputed_attn=precomputed_attn,
                    next_norm=next_norm,
                    next_input=next_input,
                    combined_attn=combined_attn,
                    normalized_attn=normalized_attn,
                    next_combined=next_combined,
                )
            precomputed_attn = next_input[0] if next_input else None
            combined_attn, normalized_attn = (
                next_combined[0] if next_combined else (None, None)
            )
    finally:
        _ctx.plan = None
    if saved_full is not None:
        attn_backend.exit_late_layer_tail(saved_full, forward_batch)
        return R, prev_pre, tail
    if p.sharded:
        R, prev_pre = _gather_rows(p, R), _gather_rows(p, prev_pre)
    return R, prev_pre, None


# ------------------------------------------------------------------------------------------
# stage 2: MXFP8 attention-input gather
# ------------------------------------------------------------------------------------------
_FP8_M = {}                     # M -> (ok, scale shape, scale dtype) agreed over TP


def fp8_eligible_attn(attn):
    """Attention whose prefill path reads x only through wqkv_a (plus shape / dtype)."""
    v = getattr(attn, "_dsv41_sp_fp8", None)
    if v is None:
        try:
            v = bool(getattr(attn, "compressor", None) is None
                     and getattr(attn, "indexer", None) is None
                     and getattr(attn, "fuse_wqa_wkv", False)
                     and getattr(attn, "wqkv_a", None) is not None
                     and attn.accepts_mxfp8_swizzled_input())
        except Exception:  # noqa: BLE001
            v = False
        attn._dsv41_sp_fp8 = v
    return v


def _mxfp8_quantize(x, swizzled):
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize
    return flashinfer_mxfp8_quantize(x, is_sf_swizzled_layout=swizzled, alignment=32)


def swizzle_128x4(s_lin):
    """[M, G] linear ue8m0 scales -> the 128x4 tiled layout (rows padded to 128 with zeros):
    tile (m // 128, g // 4) in row-tile-major order, inside a tile [m % 32][(m % 128) // 32][g % 4]."""
    m, g = s_lin.shape
    assert g % 4 == 0
    mt = (m + 127) // 128
    if mt * 128 != m:
        s_lin = torch.cat([s_lin, s_lin.new_zeros((mt * 128 - m, g))])
    return s_lin.view(mt, 4, 32, g // 4, 4).permute(0, 3, 2, 1, 4).contiguous().view(-1)


def _mxfp8_input_cls():
    from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput
    return Mxfp8SwizzledInput


def mxfp8_gather(p, x):
    """This rank's bf16 shard [S, K] -> the whole chunk's (fp8 [M, K], linear scales [M, K/32])."""
    s_rows, k = x.shape
    q, sf = _mxfp8_quantize(x.contiguous(), False)
    sf = sf.view(torch.uint8).reshape(-1)[: s_rows * (k // 32)].view(s_rows, k // 32)
    qg = _gather_rows(p, q.view(torch.uint8)).view(q.dtype)
    sg = _gather_rows(p, sf)
    return qg, sg


def _fp8_input(p, attn, x):
    """Pre-quantized attention input for this chunk, or None (bf16 gather) with the bf16 rows
    when the first-chunk check had to gather them anyway."""
    state = _FP8_M.get(p.rows)
    if state is not None and not state[0]:
        return None, None
    qg, sg = mxfp8_gather(p, x)
    sw = swizzle_128x4(sg)
    if state is None:
        xb = _gather_rows(p, x)
        rq, rs = _mxfp8_quantize(xb, True)
        rs_u8 = rs.view(torch.uint8).reshape(-1)
        local = bool(rq.shape == qg.shape and torch.equal(rq.view(torch.uint8), qg.view(torch.uint8))
                     and rs_u8.numel() == sw.numel() and torch.equal(rs_u8, sw))
        if local:
            xq = _mxfp8_input_cls()(qg, sw.view(rs.dtype).view(rs.shape))
            with _Inner():
                local = bool(torch.equal(attn.wqkv_a(xb)[0], attn.wqkv_a(xq)[0]))
        ok = _agree_min(p, local)
        _FP8_M[p.rows] = state = (ok, tuple(rs.shape), rs.dtype)
        if p.rank == 0 and (len(_FP8_M) <= 8 or not ok):
            print(f"DSV41_PREFILL_SP fp8 attention input (M={p.rows}): MXFP8 bytes + scales and wqkv_a "
                  f"output {'equal the stock bf16 path on every rank -> ON' if ok else 'differ -> bf16 gather'}"
                  f" (rank 0 local {local})", flush=True)
        if not ok:
            return None, xb
    return _mxfp8_input_cls()(qg, sw.view(state[2]).view(state[1])), None


# ------------------------------------------------------------------------------------------
# attention / MoE: gather in, reduce-scatter out
# ------------------------------------------------------------------------------------------
def _hook_wo_b(attn):
    wo_b = attn.wo_b
    if getattr(wo_b, "_dsv41_sp_hooked", False):
        return
    orig = wo_b.forward

    def forward(input_, skip_all_reduce=False, *a, **kw):
        if _ctx.skip_wo_b is not wo_b:
            return orig(input_, skip_all_reduce, *a, **kw)
        out = orig(input_, True, *a, **kw)
        if _comparing():
            # stock: the same GEMM with the engine's own all-reduce, vs our all-reduce of the partial
            ref = orig(input_, False, *a, **kw)[0]
            ours = _ctx.plan.group.all_reduce(out[0].clone())
            _cmp("wo_b all-reduce", ours, ref)
        return out

    wo_b.forward = forward
    wo_b._dsv41_sp_hooked = True


def _make_attn_forward(orig):
    def forward(self, x, positions, forward_batch, x_quant=None):
        p = _ctx.plan
        if (p is None or _ctx.inner or x_quant is not None or not isinstance(x, torch.Tensor)
                or x.dim() != 2 or x.shape[0] != p.in_rows):
            if _ctx.dbg is None or _ctx.inner:
                return orig(self, x, positions, forward_batch, x_quant)
            _rec("ain", x, True)
            o = orig(self, x, positions, forward_batch, x_quant)
            _rec("aout", o, True)
            return o
        _hook_wo_b(self)
        xq = xb = None
        if p.sharded and FP8 and fp8_eligible_attn(self):
            xq, xb = _fp8_input(p, self, x)
        if xq is not None:
            # shape / dtype only: this layer's attention reads x solely through wqkv_a(x_quant).
            # NaN, so any other read would poison the output instead of passing silently.
            xf = x.new_full((1, 1), float("nan")).expand(p.rows, x.shape[1])
            if _ctx.dbg is not None:
                xb = _gather_rows(p, x)
                _rec("ain", xb, True)
                if _comparing():
                    rq, rs = _mxfp8_quantize(xb, True)
                    _cmp("attn x mxfp8 data", xq[0].view(torch.uint8), rq.view(torch.uint8))
                    _cmp("attn x mxfp8 scales", xq[1].view(torch.uint8).reshape(-1),
                         rs.view(torch.uint8).reshape(-1))
                    with _Inner():
                        _cmp("attn wqkv_a", self.wqkv_a(xq)[0], self.wqkv_a(xb)[0])
                    del rq, rs
                del xb
        else:
            xf = xb if xb is not None else (_gather_rows(p, x) if p.sharded else x)
            if _ctx.dbg is not None:
                _rec("ain", xf, True)
        _ctx.inner += 1
        _ctx.skip_wo_b = self.wo_b
        try:
            with _no_mhc_fusion():          # wo_b must own the (skipped) all-reduce
                o = orig(self, xf, positions, forward_batch, xq)
        finally:
            _ctx.inner -= 1
            _ctx.skip_wo_b = None
        if o.shape[0] != p.rows:
            raise RuntimeError(f"DSV41_PREFILL_SP: attention returned {tuple(o.shape)} for {p.rows} rows")
        out = _reduce_out(p, o)
        if _ctx.dbg is not None:
            _rec("aout", out)
        return out

    forward.__wrapped__ = orig
    return forward


def _make_moe_forward(orig):
    def forward(self, hidden_states, *args, **kwargs):
        p = _ctx.plan
        if (p is None or _ctx.inner or not isinstance(hidden_states, torch.Tensor)
                or hidden_states.dim() != 2 or hidden_states.shape[0] != p.in_rows
                or kwargs.get("skip_shared_experts", False)):
            if _ctx.dbg is None or _ctx.inner or not isinstance(hidden_states, torch.Tensor):
                return orig(self, hidden_states, *args, **kwargs)
            _rec("min", hidden_states, True)
            o = orig(self, hidden_states, *args, **kwargs)
            _rec("mout", o, True)
            return o
        v4 = _M["v4"]
        xf = _gather_rows(p, hidden_states) if p.sharded else hidden_states
        if _mega_moe(self, xf):
            raise RuntimeError("DSV41_PREFILL_SP: mega MoE path is not wired")
        if _ctx.dbg is not None:
            _rec("min", xf, True)
        x_ref = xf.clone() if _comparing() else None     # the MoE may write its input in place
        _ctx.inner += 1
        try:
            # mlp_reduce_scatter=True: forward_normal skips post_experts_all_reduce (and with it
            # the stats-stream start, which only precedes an all-reduce that runs)
            with v4.get_forward().scoped(mlp_reduce_scatter=True), _no_mhc_fusion():
                out = orig(self, xf, *args, **kwargs)
        finally:
            _ctx.inner -= 1
        if out.shape[0] != p.rows:
            raise RuntimeError(f"DSV41_PREFILL_SP: MoE returned {tuple(out.shape)} for {p.rows} rows")
        if x_ref is not None:
            # stock: the same MoE on the same rows with its own post-experts all-reduce
            ours = p.group.all_reduce(out.clone())
            with _Inner():
                ref = orig(self, x_ref, *args, **kwargs)
            _cmp("moe all-reduce", ours, ref)
            del x_ref, ref, ours
        res = _reduce_out(p, out)
        if _ctx.dbg is not None:
            _rec("mout", res)
        return res

    forward.__wrapped__ = orig
    return forward


# ------------------------------------------------------------------------------------------
# hc gates keyed on the logical row count (stage 1 only)
# ------------------------------------------------------------------------------------------
def _in_shard(x):
    """The plan when x is this rank's shard of a sharded chunk at the top level, else None."""
    p = _ctx.plan
    if p is None or _ctx.inner or not p.sharded or x.shape[0] != p.shard:
        return None
    return p


def _batch_invariant():
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    return is_batch_invariant_mode_enabled()


def _combine_norm_fused(layer, x, apply_pre, norm, m):
    """The engine's _hc_combine fused-norm gate (apply_pre path), evaluated at row count m."""
    v4 = _M["v4"]
    return bool(
        x.is_cuda
        and v4.get_platform().is_blackwell
        and (0 < m <= 96 or (layer.config.model_type == "deepseek_v41" and 4096 <= m <= 65536))
        and layer.hc_mult == 4
        and x.flatten(1).shape[1] == 20480
        and x.dtype == norm.weight.dtype == torch.bfloat16
        and apply_pre.stride(1) == 1
        and not norm.cast_x_before_out_mul
        and norm.variance_size_override is None
        and not _batch_invariant()
    )


def _norm_prefill_gate(norm, m):
    """The engine's _hc_combine gate for a collapsed (combined=) input, at row count m."""
    return bool(
        4096 <= m <= 65536
        and norm.weight.dtype == torch.bfloat16
        and not norm.cast_x_before_out_mul
        and norm.variance_size_override is None
    )


def hc_combine_norm_rows(x_flat, pre, weight, eps, logical_m):
    """hc_combine_norm on a row slice with the kernel the full chunk (logical_m rows) uses."""
    hcn = _M["hcn"]
    m = x_flat.shape[0]
    assert 4096 <= logical_m <= 65536 and x_flat.shape == (m, 20480)
    assert pre.shape == (m, 4) and pre.stride(1) == 1
    assert weight.shape == (5120,) and weight.is_contiguous()
    assert x_flat.dtype == weight.dtype == torch.bfloat16 and x_flat.stride(1) == 1
    y = torch.empty((m, 5120), dtype=x_flat.dtype, device=x_flat.device)
    hcn._hc_combine_norm_prefill[(m,)](
        x_flat, pre, weight, y, x_flat.stride(0), pre.stride(0), eps, num_warps=4
    )
    return y


def norm_prefill_rows(combined, weight, eps, logical_m):
    """mhc_post_combine.hc_norm_prefill on a row slice (its wrapper refuses < 4096 rows; the
    kernel is one program per row)."""
    mpc = _M["mpc"]
    assert 4096 <= logical_m <= 65536 and combined.shape[1] == 5120
    assert weight.shape == (5120,)
    assert combined.dtype == weight.dtype == torch.bfloat16
    assert combined.is_contiguous() and weight.is_contiguous()
    out = torch.empty_like(combined)
    if combined.shape[0]:
        mpc._hc_norm_prefill[(combined.shape[0],)](combined, weight, out, eps, num_warps=4)
    return out


def post_combine_rows(x, residual, post, comb, pre, logical_m):
    """mhc_post_combine on a row slice with the launch the full chunk (logical_m rows) uses.
    Element-wise per (row, column): the block width only tiles the columns."""
    mpc = _M["mpc"]
    assert x.dtype == residual.dtype == torch.bfloat16
    assert post.dtype == comb.dtype == pre.dtype == torch.float32
    assert x.ndim == 2 and x.shape[1] == 5120
    assert residual.shape == (x.shape[0], 4, x.shape[1])
    assert post.shape == pre.shape == (x.shape[0], 4)
    assert comb.shape == (x.shape[0], 4, 4)
    assert all(t.is_contiguous() for t in (x, residual, post, comb, pre))
    updated = torch.empty_like(residual)
    combined = torch.empty_like(x)
    if x.shape[0]:
        block = 1024 if logical_m <= 192 or logical_m >= 4096 else 512
        mpc._mhc_post_combine[(x.shape[0], (x.shape[1] + block - 1) // block)](
            x, residual, post, comb, pre, updated, combined,
            H=x.shape[1], B=block, num_warps=4, enable_fp_fusion=False,
        )
    return updated, combined


def _make_hc_combine(orig):
    def _hc_combine(self, x, apply_pre, norm, stats_stream=None, quantized=None, normalized=None,
                    precomputed=None, combined=None):
        p = _in_shard(x)
        if p is None or precomputed is not None:
            return orig(self, x, apply_pre, norm, stats_stream, quantized, normalized, precomputed,
                        combined)
        if stats_stream is not None and torch.cuda.is_available():
            # never set inside the region (see _make_stats_stream); keep the engine's ordering
            # (a shard has >= 64 rows, so never the engine's "tiny" case)
            stats_stream.wait_stream(torch.cuda.current_stream())
        if normalized is not None:
            # the engine returns it as is; its assert keys on the chunk's row count
            assert quantized is None or 4096 <= p.rows <= 65536
            return normalized
        if combined is not None:
            y = _sp_norm_combined(self, orig, p, x, norm, combined)
        else:
            y = _sp_combine(self, orig, p, x, apply_pre, norm, quantized)
        if _comparing():
            with _Inner():
                xg = _gather_rows(p, x)
                pg = _gather_rows(p, apply_pre) if apply_pre is not None else None
                cg = _gather_rows(p, combined) if combined is not None else None
                ref = orig(self, xg, pg, norm, None, None, None, None, cg)[p.lo:p.hi].clone()
                del xg, pg, cg
            which = "attn" if norm is getattr(self, "input_layernorm", None) else "ffn"
            what = " (collapsed)" if combined is not None else ("" if apply_pre is not None else " (stream 0)")
            _cmp(f"hc_combine {which}{what}", y, ref, {"x": x, "pre": apply_pre})
        return y

    def _sp_norm_combined(self, orig, p, x, norm, combined):
        full = _norm_prefill_gate(norm, p.rows)
        if full == _norm_prefill_gate(norm, combined.shape[0]):
            return orig(self, x, None, norm, None, None, None, None, combined)
        if full:
            return norm_prefill_rows(combined, norm.weight, norm.variance_epsilon, p.rows)
        return norm(combined)

    def _sp_combine(self, orig, p, x, apply_pre, norm, quantized):
        if apply_pre is None:
            return orig(self, x, apply_pre, norm, None, quantized, None, None, None)
        m = p.rows
        full = _combine_norm_fused(self, x, apply_pre, norm, m)
        here = _combine_norm_fused(self, x, apply_pre, norm, x.shape[0])
        if full == here:
            return orig(self, x, apply_pre, norm, None, quantized, None, None, None)
        x_flat = x.flatten(1)
        if full:
            return hc_combine_norm_rows(x_flat, apply_pre, norm.weight, norm.variance_epsilon, m)
        return norm(_M["mhc"].hc_combine(x_flat, apply_pre, self.hc_mult, x.dtype))

    _hc_combine.__wrapped__ = orig
    return _hc_combine


def _post_split_h(layer, x, residual, post, comb, m):
    v4 = _M["v4"]
    return bool(
        v4._is_cuda
        and v4.get_platform().is_blackwell
        and layer.hc_pre_from_prev_sublayer
        and layer.hc_mult == 4
        and x.shape[1] == 5120
        and m <= 384
        and x.dtype == residual.dtype == torch.bfloat16
        and post.dtype == comb.dtype == torch.float32
        and all(t.is_contiguous() for t in (x, residual, post, comb))
    )


def _make_hc_post(orig):
    def hc_post(self, x, residual, post, comb):
        p = _in_shard(x)
        if p is None:
            return orig(self, x, residual, post, comb)
        out = _sp_post(self, x, residual, post, comb, p)
        if _comparing():
            with _Inner():
                ref = orig(self, _gather_rows(p, x), _gather_rows(p, residual), _gather_rows(p, post),
                           _gather_rows(p, comb))[p.lo:p.hi].clone()
            _cmp("hc_post", out, ref, {"x": x, "post": post, "comb": comb})
        return out

    def _sp_post(self, x, residual, post, comb, p):
        v4 = _M["v4"]
        full = _post_split_h(self, x, residual, post, comb, p.rows)
        if full == _post_split_h(self, x, residual, post, comb, x.shape[0]):
            return orig(self, x, residual, post, comb)
        if full:
            return v4.mhc_post_split_h(x, residual, post, comb)
        # the full chunk skips split-h: the rest of the engine's CUDA chain
        if v4.envs.SGLANG_OPT_USE_FLASHINFER_MHC.get():
            from flashinfer.mhc import mhc_post
            return mhc_post(x, residual, post, comb)
        if v4.envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get():
            if (self.hc_pre_from_prev_sublayer and v4.get_platform().is_sm90
                    and 1 <= p.rows <= 64):
                return v4.mhc_post_split_h(x, residual, post, comb)
            return _M["mhc"].mhc_post(x, residual, post, comb)
        raise RuntimeError("DSV41_PREFILL_SP: hc_post fallback chain not wired for this platform")

    hc_post.__wrapped__ = orig
    return hc_post


def _post_combine_gate(layer, x, residual, post, comb, pre, forward_batch, m):
    """The engine's _hc_post_with_combine outer gate (fused post + collapse), at row count m."""
    v4 = _M["v4"]
    mode = forward_batch.forward_mode
    return bool(
        layer.config.model_type == "deepseek_v41"
        and x.is_cuda
        and v4.get_platform().is_blackwell
        and (
            (128 <= m <= 384 and (mode.is_decode() or mode.is_target_verify()))
            or (
                4096 <= m <= 65536
                and mode.is_extend_without_speculative()
                and v4.envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get()
                and not v4.envs.SGLANG_OPT_USE_FLASHINFER_MHC.get()
            )
        )
        and x.shape[1] == 5120
        and layer.hc_mult == 4
        and x.dtype == residual.dtype == torch.bfloat16
        and post.dtype == comb.dtype == pre.dtype == torch.float32
        and all(t.is_contiguous() for t in (x, residual, post, comb, pre))
        and v4.get_parallel().attn_dp_size == 1
        and not v4.get_forward().sp_active
        and not layer.dsa_enable_prefill_cp
        and not _batch_invariant()
    )


def _post_norm_ok(norm):
    """The engine's choice of the fused post+collapse+RMSNorm, minus its 16-byte pointer check
    (see _make_hc_post_with_combine)."""
    return bool(
        norm is not None
        and not norm.cast_x_before_out_mul
        and norm.variance_size_override is None
        and norm.weight.dtype == torch.bfloat16
        and norm.weight.shape == (5120,)
        and norm.weight.is_contiguous()
    )


def _make_hc_post_with_combine(orig):
    """v0.5.21's hc_post fused with the next sublayer's input collapse (+ norm), chosen on the
    chunk's row count. With m >= 4096 and a norm the engine prefers the CUDA kernel
    mhc_post_combine_norm_prefill (refuses < 4096 rows) and falls back to the Triton pair
    mhc_post_combine + hc_norm_prefill when a pointer is not 16-byte aligned, i.e. it treats the
    two as interchangeable. The shard always runs the Triton pair (no pointer-dependent branch, so
    every rank takes the same path and the self-check collective below cannot diverge), and the
    first chunk of every size checks it against the CUDA kernel on the gathered rows."""

    def _hc_post_with_combine(self, x, residual, post, comb, pre, forward_batch, norm=None):
        p = _in_shard(x)
        if p is None:
            return orig(self, x, residual, post, comb, pre, forward_batch, norm)
        out = _sp(self, p, x, residual, post, comb, pre, forward_batch, norm)
        if _comparing():
            with _Inner():
                ref = orig(self, _gather_rows(p, x), _gather_rows(p, residual), _gather_rows(p, post),
                           _gather_rows(p, comb), _gather_rows(p, pre), forward_batch, norm)
                ref = tuple(None if t is None else t[p.lo:p.hi].clone() for t in ref)
            # the stock call may take the other (pointer-alignment) form: compare what both have
            names = ("updated", "combined", "normalized")
            for k in range(3):
                if out[k] is not None and ref[k] is not None:
                    _cmp(f"hc_post+combine {names[k]}", out[k], ref[k], {"x": x, "post": post})
            del ref
        return out

    def _sp(self, p, x, residual, post, comb, pre, forward_batch, norm):
        m = p.rows
        full = _post_combine_gate(self, x, residual, post, comb, pre, forward_batch, m)
        if not full:
            # the chunk takes plain hc_post (our row-count-aware wrapper)
            return self.hc_post(x, residual, post, comb), None, None
        if m >= 4096 and _post_norm_ok(norm):
            return _post_combine_norm(p, x, residual, post, comb, pre, norm)
        updated, combined = post_combine_rows(x, residual, post, comb, pre, m)
        return updated, combined, None

    _hc_post_with_combine.__wrapped__ = orig
    return _hc_post_with_combine


def _post_combine_norm(p, x, residual, post, comb, pre, norm):
    """(updated, None, normalized) for this rank's rows, bit-identical to the engine's fused
    prefill kernel on the whole chunk: checked once per chunk size, agreed over TP."""
    ok = _PCN_OK.get(p.rows)
    if ok is False:             # this chunk started before the check failed: serve it exactly
        u, n = _post_combine_norm_gathered(p, x, residual, post, comb, pre, norm)
        return u, None, n
    updated, combined = post_combine_rows(x, residual, post, comb, pre, p.rows)
    normalized = norm_prefill_rows(combined, norm.weight, norm.variance_epsilon, p.rows)
    del combined
    if ok is None:
        ref_u, ref_n = _post_combine_norm_gathered(p, x, residual, post, comb, pre, norm)
        local = bool(torch.equal(updated, ref_u) and torch.equal(normalized, ref_n))
        ok = _PCN_OK[p.rows] = _agree_min(p, local)
        if p.rank == 0 and (len(_PCN_OK) <= 8 or not ok):
            print(f"DSV41_PREFILL_SP post+combine+norm (M={p.rows}): shard Triton pair "
                  f"{'equals the stock fused kernel on every rank -> ON' if ok else 'differs -> this chunk exact via gather, later chunks of this size run stock'}"
                  f" (rank 0 local {local})", flush=True)
        if not ok:
            return ref_u, None, ref_n
        del ref_u, ref_n
    return updated, None, normalized


def _post_combine_norm_gathered(p, x, residual, post, comb, pre, norm):
    """The engine's fused kernel on the gathered chunk, this rank's rows (exact, expensive)."""
    with _Inner():
        u, n = _M["mpcn"].mhc_post_combine_norm_prefill(
            _gather_rows(p, x), _gather_rows(p, residual), _gather_rows(p, post),
            _gather_rows(p, comb), _gather_rows(p, pre), norm.weight, norm.variance_epsilon)
    out = u[p.lo:p.hi].clone(), n[p.lo:p.hi].clone()
    del u, n
    return out


def _make_stats_stream(orig):
    """No prefill stats stream inside the region (any mode). The engine starts the sublayer's
    stats just before its all-reduce, and for attention moves that all-reduce out of wo_b; here
    the reduction is a reduce-scatter (or RS + all-gather) issued by this adapter. Stats on the
    current stream run the same kernels on the same rows, so the bits are the stock bits."""
    def _get_hc_stats_stream(self, hidden_states, forward_batch):
        p = _ctx.plan
        if p is not None and not _ctx.inner and hidden_states.shape[0] == p.in_rows:
            return None
        return orig(self, hidden_states, forward_batch)

    _get_hc_stats_stream.__wrapped__ = orig
    return _get_hc_stats_stream


def _make_hc_mix_stats(orig):
    """Debug compare only: the stats on the shard vs the stock stats on the gathered rows."""
    def _hc_mix_stats(self, x, hc_fn, hc_scale, hc_base, stats_stream=None):
        out = orig(self, x, hc_fn, hc_scale, hc_base, stats_stream)
        p = _ctx.plan
        if (_comparing() and not _ctx.inner and stats_stream is None and x.shape[0] == p.shard
                and p.shard != p.rows):
            with _Inner():
                ref = orig(self, _gather_rows(p, x), hc_fn, hc_scale, hc_base, None)
            which = "attn" if hc_fn is getattr(self, "hc_attn_fn", None) else "ffn"
            _cmp(f"hc_mix_stats {which}", tuple(out), tuple(r[p.lo:p.hi].clone() for r in ref))
        return out

    _hc_mix_stats.__wrapped__ = orig
    return _hc_mix_stats


# ------------------------------------------------------------------------------------------
# install
# ------------------------------------------------------------------------------------------
def install(model_mod):
    """sglang.srt.models.deepseek_v4. Must run BEFORE hc_prefill_fused.install, so that its
    'stock' _hc_combine / hc_post / _hc_post_with_combine are these logical-row-aware ones."""
    if MODE == "off" and not DEBUG:
        return
    if getattr(model_mod, "_dsv41_prefill_sp", False):
        return
    _load_modules(model_mod)
    check_engine()
    if MODE == "shard":
        _check_hc_prefill_fused()
    Model, Layer = model_mod.DeepseekV4Model, model_mod.DeepseekV4DecoderLayer
    MoE = _M["v2"].DeepseekV2MoE
    stock_layers = Model._forward_layers_hc_pre_from_prev

    def _forward_layers_hc_pre_from_prev(self, positions, hidden_states, forward_batch, input_ids,
                                         input_ids_global, capture_dspark, dspark_aux_hidden_states):
        p = _plan(self, hidden_states, forward_batch)
        d = _debug_begin(self, hidden_states, forward_batch, input_ids, p) if DEBUG else None
        _ctx.dbg = d
        try:
            if p is None:
                out = stock_layers(self, positions, hidden_states, forward_batch, input_ids,
                                   input_ids_global, capture_dspark, dspark_aux_hidden_states)
            else:
                out = _sp_forward_layers(self, p, positions, hidden_states, forward_batch, input_ids,
                                         input_ids_global, capture_dspark, dspark_aux_hidden_states)
        finally:
            _ctx.dbg = None
        if d is not None:
            _debug_end(d, out)
        return out

    _forward_layers_hc_pre_from_prev.__wrapped__ = stock_layers
    Model._forward_layers_hc_pre_from_prev = _forward_layers_hc_pre_from_prev
    model_mod.MQALayer.forward = _make_attn_forward(model_mod.MQALayer.forward)
    MoE.forward = _make_moe_forward(MoE.forward)
    if MODE != "off":
        Layer._get_hc_stats_stream = _make_stats_stream(Layer._get_hc_stats_stream)
    if MODE == "shard":
        Layer._hc_combine = _make_hc_combine(Layer._hc_combine)
        Layer.hc_post = _make_hc_post(Layer.hc_post)
        Layer._hc_post_with_combine = _make_hc_post_with_combine(Layer._hc_post_with_combine)
        if "compare" in DEBUG:
            Layer._hc_mix_stats = _make_hc_mix_stats(Layer._hc_mix_stats)
    if DEBUG:
        Layer.forward_hc_pre_from_prev = _make_layer_forward(Layer.forward_hc_pre_from_prev)
        print(f"DSV41_PREFILL_SP DEBUG {sorted(DEBUG)} ARMED for the first {DEBUG_CHUNKS} extend "
              f"forwards{' (dumps in ' + DEBUG_DUMP + ')' if DEBUG_DUMP else ''}: NOT FOR PRODUCTION",
              flush=True)
    model_mod._dsv41_prefill_sp = True
    if MODE == "off":
        return
    what = ("reduce-scatter + all-gather at wo_b / MoE / Engram (comm only, nothing sharded)"
            if MODE == "comm" else
            f"rows sharded over TP on the full-row layers, bf16 gathers"
            f"{', EXACT (all-reduce + keep the quarter)' if EXACT else ''}"
            f"{', MXFP8 attention-input gathers (checked per chunk size)' if FP8 else ''}"
            f", wkv {'on the shard (self-checked)' if WKV_SHARD else 'full-M'}"
            f", post+combine+norm on the shard (self-checked per chunk size)")
    print(f"DSV41_PREFILL_SP={_RAW} ARMED: prefill chunks >= {MIN_ROWS} rows: {what}", flush=True)
    if EXACT and MODE == "comm":
        print("DSV41_PREFILL_SP: DSV41_PREFILL_SP_EXACT has no effect in comm mode", flush=True)
