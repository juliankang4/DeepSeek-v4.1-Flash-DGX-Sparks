"""CPU check for adapter/prefill_sp.py: the row partition, the tail-row gathers, and the whole
sharded layer loop against the engine's stock loop (SGLang v0.5.21 structure), with four TP ranks
emulated by threads over a fake process group (all-gather = concatenation, all-reduce = rank-order
bf16 sum, reduce-scatter = a different summation order, like NCCL's).

The model is a toy with the engine's structure: residual [M, 4, D], a row-mixing "attention"
whose wo_b all-reduces unless told to skip, a hash-routed "MoE" that honours mlp_reduce_scatter,
an Engram (owned-row lookup + all-reduce, wkv, engram_gate), bounded replay (late layers on the
tail rows), DSpark aux capture and the vision `where`. v0.5.21's prefill paths for
4096 <= M <= 65536 are modelled too:
  * the prefill stats stream: with it the layer sets an overlap-only MhcPostFusion and the
    attention runs its all-reduce outside wo_b (as MQALayer does), so a region that let it through
    would reduce twice;
  * _hc_post_with_combine: hc_post fused with the next sublayer's collapse + a "prefill" norm whose
    rounding differs from the plain norm; the fused kernel is a separate callable (the CUDA kernel)
    that the shard's pair (post_combine_rows + norm_prefill_rows, the Triton kernels) must match;
  * the cross-layer collapse (combined= / normalized=) when no tail and no Engram intervene.
The adapter's real wrappers for _hc_combine / hc_post / _hc_post_with_combine /
_get_hc_stats_stream are installed on the toy; their GPU-only gates are swapped for toy ones.
The toy's own apply_pre combine picks one of two numerically different reductions by row count
keyed on sp_logical_rows() (what the adapter's _hc_combine achieves on GPU): a gate keyed on the
shard size would change bits, and the test shows it.

Checks, for several chunk sizes (8192 / 4096 = the v0.5.21 fused band, 2052 = odd shards, 2048
with a full-row tail, 2050 = not divisible, below MIN_ROWS) and tail layouts (one request,
requests straddling shard boundaries, many short requests, no bounded replay = cross-layer path):
  * stage 1 EXACT == stock, bit for bit, on every rank;
  * stage 1 fast == P0 (comm), bit for bit (same reduce-scatters on the same tensors);
  * P0 differs from stock only through the summation order (and matches it with an exact RS);
  * the post+combine+norm self-check turns ON when the pair matches the fused kernel, and when it
    does not, the chunk stays exact (gathered stock) and later chunks of that size run stock;
  * without the stats-stream / fusion guards the attention reduces twice (the test is sensitive);
  * the wkv self-check falls back to the full-M GEMM when the shard GEMM is not bit-exact.

  PYTHONPATH=adapter python tests/test_prefill_sp.py
"""
import os
import threading
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

os.environ.setdefault("DSV41_PREFILL_SP", "1")
import torch  # noqa: E402

import prefill_sp as sp  # noqa: E402

W = 4
D = 64
BF = torch.bfloat16


# ------------------------------------------------------------------------------------------
# fake process group over threads
# ------------------------------------------------------------------------------------------
class ThreadGroup:
    def __init__(self, world, rs_order="ring"):
        self.world_size = world
        self.unique_name = "tp"
        self.device_group = None
        self._bar = threading.Barrier(world)
        self._slots = [None] * world
        self._tl = threading.local()
        self.rs_order = rs_order
        self.calls = {"ag": 0, "ar": 0, "rs": 0}

    @property
    def rank_in_group(self):
        return self._tl.rank

    def bind(self, rank):
        self._tl.rank = rank

    def _exchange(self, t):
        r = self.rank_in_group
        self._slots[r] = t.clone()
        self._bar.wait()
        vals = list(self._slots)
        self._bar.wait()
        return vals

    def all_gather_into_tensor(self, out, x):
        vals = self._exchange(x.contiguous())
        if self.rank_in_group == 0:
            self.calls["ag"] += 1
        out.view(-1).copy_(torch.cat([v.reshape(-1) for v in vals]))

    def all_reduce(self, x):
        vals = self._exchange(x)
        if self.rank_in_group == 0:
            self.calls["ar"] += 1
        acc = vals[0].clone()
        for v in vals[1:]:
            acc = acc + v                               # rounds to the dtype at every hop
        return acc

    def reduce_scatter_tensor(self, out, x):
        vals = self._exchange(x.contiguous())
        r, s = self.rank_in_group, out.shape[0]
        if r == 0:
            self.calls["rs"] += 1
        parts = [v[r * s:(r + 1) * s] for v in vals]
        order = list(range(self.world_size)) if self.rs_order == "same" else \
            [(r + 1 + k) % self.world_size for k in range(self.world_size)]
        acc = parts[order[0]].clone()
        for o in order[1:]:
            acc = acc + parts[o]
        out.copy_(acc)


def run_ranks(group, fn):
    out, errs = [None] * group.world_size, []

    def body(r):
        group.bind(r)
        try:
            out[r] = fn(r)
        except BaseException as exc:  # noqa: BLE001
            errs.append((r, exc))
            group._bar.abort()

    ts = [threading.Thread(target=body, args=(r,)) for r in range(group.world_size)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errs:
        raise errs[0][1]
    return out


# ------------------------------------------------------------------------------------------
# fake engine
# ------------------------------------------------------------------------------------------
_fwd = threading.local()


class _Forward:
    sp_active = False

    @property
    def mlp_reduce_scatter(self):
        return getattr(_fwd, "mrs", False)

    @contextmanager
    def scoped(self, mlp_reduce_scatter=False):
        prev = getattr(_fwd, "mrs", False)
        _fwd.mrs = mlp_reduce_scatter
        try:
            yield
        finally:
            _fwd.mrs = prev


FORWARD = _Forward()
GATE = {"logical": True}


class _Fusion:
    """sglang.srt.layers.moe.mhc_post_fusion: the scoped MhcPostFusion hand-off."""
    _tl = threading.local()

    @classmethod
    def current(cls):
        return getattr(cls._tl, "cur", None)

    @classmethod
    @contextmanager
    def use_mhc_post_fusion(cls, state):
        prev = getattr(cls._tl, "cur", None)
        cls._tl.cur = state
        try:
            yield
        finally:
            cls._tl.cur = prev


def hc_rows(x):
    return sp.sp_logical_rows(x) if GATE["logical"] else x.shape[0]


class Norm:
    """RMSNorm stand-in with the attributes the engine's gates read."""

    def __init__(self, g):
        self.weight = (1 + 0.2 * torch.randn(D, generator=g)).to(BF)
        self.variance_epsilon = 1e-6
        self.cast_x_before_out_mul = False
        self.variance_size_override = None

    def __call__(self, y):
        yf = y.float()
        return (yf * torch.rsqrt(yf.square().mean(-1, keepdim=True) + self.variance_epsilon)
                * self.weight.float()).to(BF)


# the v0.5.21 prefill kernels (row-local, as the engine's are)
def toy_post(x, R, post, comb):
    out = post.unsqueeze(-1) * x.float().unsqueeze(1) + (comb.unsqueeze(-1) * R.float().unsqueeze(2)).sum(1)
    return out.to(BF)


def toy_post_combine(x, R, post, comb, pre, logical_m=None):
    """mhc_post_combine: updated streams (bf16) and their sequential collapse (bf16)."""
    updated = toy_post(x, R, post, comb)
    acc = torch.zeros(x.shape, dtype=torch.float32)
    for c in range(4):
        acc = acc + pre[:, c:c + 1] * updated[:, c].float()
    return updated, acc.to(BF)


def toy_norm_prefill(combined, weight, eps, logical_m=None):
    """hc_norm_prefill: rounds differently from Norm.__call__ (weight folded into the scale)."""
    yf = combined.float()
    inv = torch.rsqrt((yf * yf).sum(-1, keepdim=True) / yf.shape[-1] + eps)
    return (yf * (weight.float() * inv)).to(BF)


def _fused_ok(x, R, post, comb, pre, weight, eps):
    u, c = toy_post_combine(x, R, post, comb, pre)
    return u, toy_norm_prefill(c, weight, eps)


def _fused_bad(x, R, post, comb, pre, weight, eps):
    u, n = _fused_ok(x, R, post, comb, pre, weight, eps)
    return u, n.view(torch.int16).add(1).view(BF)       # one ulp off everywhere


MPCN = SimpleNamespace(mhc_post_combine_norm_prefill=_fused_ok)   # the fused CUDA kernel


def _prefill_band(m, fb):
    return 4096 <= m <= 65536 and fb.forward_mode.is_extend_without_speculative()


class RowParallel(torch.nn.Module):
    def __init__(self, group, weights):
        super().__init__()
        self.group, self.w = group, weights
        self.reduce_results, self.tp_size = True, group.world_size

    def forward(self, input_, skip_all_reduce=False):
        out = (input_.float() @ self.w[self.group.rank_in_group]).to(BF)
        if not skip_all_reduce:
            out = self.group.all_reduce(out)
        return out, None


class FakeAttn(torch.nn.Module):
    def __init__(self, group, g):
        super().__init__()
        self.group = group
        self.scale = torch.randn(group.world_size, D, generator=g)
        self.wo_b = RowParallel(group, torch.randn(group.world_size, D, D, generator=g) / D ** 0.5)
        self.attn_tp_size = group.world_size
        self.seen_rows = []

    def forward(self, x, positions, forward_batch, x_quant=None):
        assert positions.shape[0] == x.shape[0], "attention needs the positions of every row"
        self.seen_rows.append(x.shape[0])
        # causal row mixing: attention needs every earlier row
        h = x.float().cumsum(0) / torch.arange(1, x.shape[0] + 1).unsqueeze(1)
        h = (h * self.scale[self.group.rank_in_group]).to(BF)
        mhc = _Fusion.current()
        o, _ = self.wo_b(h, skip_all_reduce=mhc is not None)
        if mhc is not None:             # v0.5.21 overlap_only: attn_tp_all_reduce outside wo_b
            o = self.group.all_reduce(o)
        return o

    def maybe_use_decode_attn_tp(self, fb):
        return nullcontext()

    def accepts_mxfp8_swizzled_input(self):
        return False


class FakeMoE(torch.nn.Module):
    def __init__(self, group, g):
        super().__init__()
        self.group = group
        self.table = torch.randn(16, group.world_size, D, generator=g)
        self.tp_size = group.world_size
        self._shared_expert_tp1 = False
        self._enable_a2a_moe = False

    def forward(self, hidden_states, forward_batch=None, gemm_output_zero_allocator=None,
                input_ids=None, input_ids_global=None, skip_shared_experts=False):
        assert input_ids.shape[0] == hidden_states.shape[0], "hash routing needs every row's id"
        w = self.table[input_ids % 16, self.group.rank_in_group]
        out = (hidden_states.float() * torch.tanh(w)).to(BF)
        if not FORWARD.mlp_reduce_scatter:      # stats start (if any) does not change the sum
            out = self.group.all_reduce(out)
        return out


class FakeEmbed:
    def __init__(self, group, g, vocab, cols, dim):
        self.group, self.tp_size, self._shared = group, group.world_size, False
        self.table = torch.randn(vocab, cols, dim, generator=g).to(BF)
        self.vocab = vocab

    def _owned_rows(self, ids):
        r = self.group.rank_in_group
        lo, hi = self.vocab * r // self.tp_size, self.vocab * (r + 1) // self.tp_size
        owned = (ids >= lo) & (ids < hi)                                 # [T, cols]
        vals = self.table[ids, torch.arange(ids.shape[1])]              # [T, cols, dim]
        return vals.masked_fill(~owned.unsqueeze(-1), 0)


def fake_engram_gate(x, kv, q_weight, k_weight, eps, clamp_value):
    """engram_gate stand-in: row-local, reads every argument the adapter passes."""
    k = (kv.float() * q_weight.float() * k_weight.float()).clamp_min(-1 / clamp_value)
    return (x.float() + torch.sigmoid(k + eps).unsqueeze(1)).to(x.dtype)


class FakeEngram:
    def __init__(self, group, g, hash_index, m_dependent=False):
        self.group, self.layer_hash_index = group, hash_index
        self.embed = FakeEmbed(group, g, 97, 3, 8)
        self.wkv_w = torch.randn(24, D, generator=g) / 5
        self.q_weight = 1 + 0.1 * torch.randn(D, generator=g)
        self.k_weight = 1 + 0.1 * torch.randn(D, generator=g)
        self.eps, self.clamp_value = 1e-6, 1e-6
        self.m_dependent = m_dependent
        self.wkv_calls = []

    def wkv(self, t):
        self.wkv_calls.append(t.shape[0])
        out = (t.float() @ self.wkv_w)
        if self.m_dependent and t.shape[0] >= 2048:   # a GEMM whose tactic changes with M
            out = out * (1 + 2 ** -7)
        return out.to(BF), None

    def __call__(self, x, ids, forward_batch=None, cp_all_tokens=False):
        emb = self.group.all_reduce(self.embed._owned_rows(ids))
        kv, _ = self.wkv(emb.flatten(-2))
        return fake_engram_gate(x, kv, self.q_weight, self.k_weight, self.eps, self.clamp_value)


class FakeLayer:
    """DeepseekV4DecoderLayer's v0.5.21 prefill structure, numerics replaced by toys."""

    def __init__(self, group, g, layer_id, engram=None):
        self.layer_id = layer_id
        self.self_attn = FakeAttn(group, g)
        self.mlp = FakeMoE(group, g)
        self.engram = engram
        self.fn = torch.randn(2, 4, D, generator=g)
        self.input_layernorm, self.post_attention_layernorm = Norm(g), Norm(g)
        self.dsa_enable_prefill_cp = False
        self.use_fused_mhc_post_pre = False
        self.hc_mult = 4
        self.stats_stream_seen = []

    def _get_hc_stats_stream(self, hidden_states, forward_batch):
        return "side-stream" if _prefill_band(hidden_states.shape[0], forward_batch) else None

    def _hc_combine(self, R, apply_pre, norm, stats_stream=None, quantized=None, normalized=None,
                    precomputed=None, combined=None):
        if precomputed is not None:
            return precomputed[0]
        if normalized is not None:
            return normalized
        if combined is not None:
            if 4096 <= combined.shape[0] <= 65536:
                return toy_norm_prefill(combined, norm.weight, norm.variance_epsilon)
            return norm(combined)
        if apply_pre is None:
            return norm(R[:, 0, :].contiguous())
        p = apply_pre.float()
        if hc_rows(R) >= 4096:          # "fused" reduction order
            y = ((R[:, 0].float() * p[:, 0:1] + R[:, 1].float() * p[:, 1:2])
                 + (R[:, 2].float() * p[:, 2:3] + R[:, 3].float() * p[:, 3:4]))
        else:                           # "unfused": sequential, rounded before the norm
            y = R[:, 0].float() * p[:, 0:1]
            for c in (1, 2, 3):
                y = y + R[:, c].float() * p[:, c:c + 1]
            y = y.to(BF).float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-6) * norm.weight.float()
        return y.to(BF)

    def _stats(self, R, which):
        mix = (R.float() * self.fn[which]).sum(-1)                         # [N, 4]
        pre = torch.sigmoid(mix)
        post = 2 * torch.sigmoid(mix.flip(-1))
        comb = torch.softmax(mix.unsqueeze(-1) * mix.unsqueeze(-2) / 8, dim=-1)
        return pre, post, comb

    def hc_post(self, x, R, post, comb):
        return toy_post(x, R, post, comb)

    def _hc_post_with_combine(self, x, residual, post, comb, pre, forward_batch, norm=None):
        if _prefill_band(x.shape[0], forward_batch):
            if norm is not None:
                updated, normalized = MPCN.mhc_post_combine_norm_prefill(
                    x, residual, post, comb, pre, norm.weight, norm.variance_epsilon)
                return updated, None, normalized
            updated, combined = toy_post_combine(x, residual, post, comb, pre)
            return updated, combined, None
        return self.hc_post(x, residual, post, comb), None, None

    def forward_hc_pre_from_prev(self, positions, hidden_states, input_ids, forward_batch,
                                 input_ids_global, prev_pre, precomputed_attn=None,
                                 next_norm=None, next_input=None, combined_attn=None,
                                 normalized_attn=None, next_combined=None):
        stats_stream = self._get_hc_stats_stream(hidden_states, forward_batch)
        self.stats_stream_seen.append(stats_stream)
        R = hidden_states
        x = self._hc_combine(R, apply_pre=prev_pre, norm=self.input_layernorm,
                             stats_stream=stats_stream, quantized=None, precomputed=precomputed_attn,
                             combined=combined_attn, normalized=normalized_attn)
        attn_mhc = object() if stats_stream is not None else None       # overlap_only fusion
        ctx = _Fusion.use_mhc_post_fusion(attn_mhc) if attn_mhc is not None else nullcontext()
        with ctx:
            x = self.self_attn(x=x, positions=positions, forward_batch=forward_batch, x_quant=None)
        attn_pre, post, comb = self._stats(R, 0)
        R, ffn_combined, ffn_normalized = self._hc_post_with_combine(
            x, R, post, comb, attn_pre, forward_batch, norm=self.post_attention_layernorm)
        x = self._hc_combine(R, apply_pre=attn_pre, norm=self.post_attention_layernorm,
                             stats_stream=stats_stream, normalized=ffn_normalized,
                             combined=ffn_combined)
        mhc = object() if stats_stream is not None else None
        ctx = _Fusion.use_mhc_post_fusion(mhc) if mhc is not None else nullcontext()
        with ctx, FORWARD.scoped(mlp_reduce_scatter=False):
            x = self.mlp(x, forward_batch, input_ids=input_ids, input_ids_global=input_ids_global,
                         skip_shared_experts=False)
        ffn_pre, post, comb = self._stats(R, 1)
        if next_combined is not None:
            R, combined, normalized = self._hc_post_with_combine(
                x, R, post, comb, ffn_pre, forward_batch, norm=next_norm)
            if combined is not None or normalized is not None:
                next_combined.append((combined, normalized))
        else:
            R = self.hc_post(x, R, post, comb)
        return R, ffn_pre


class Tail:
    def __init__(self, lens, window=128, pad_rows=0):
        idx = []
        pos = 0
        for n in lens:
            k = min(window, n)
            idx.extend(range(pos + n - k, pos + n))
            pos += n
        self.token_indices = torch.tensor(idx, dtype=torch.int64)
        self.contiguous_start = (pos - min(window, lens[0])) if len(lens) == 1 else None
        self.positions = self.token_indices.clone()
        self.pad_rows = pad_rows
        self.cp_metadata = None

    def real_rows(self, t):
        return t[self.contiguous_start:] if self.contiguous_start is not None else t[self.token_indices]

    def rows(self, t):
        r = self.real_rows(t)
        if self.pad_rows:
            r = torch.cat([r, r.new_zeros((self.pad_rows, *r.shape[1:]))])
        return r


class Backend:
    def __init__(self, tail):
        self.tail_forward_metadata = SimpleNamespace(late_layer_tail=tail)
        self.entered = 0

    def enter_late_layer_tail(self, fb):
        self.entered += 1
        return "saved"

    def exit_late_layer_tail(self, saved, fb):
        assert saved == "saved"


class FakeModel:
    def __init__(self, group, seed, n_layers=6, late=4, vision=False, m_dependent_wkv=False):
        g = torch.Generator().manual_seed(seed)
        self.pp_group = SimpleNamespace(world_size=1)
        self.hidden_size, self.hc_mult, self.hc_pre_from_prev_sublayer = D, 4, True
        self.start_layer, self.end_layer, self.late_layer_start = 0, n_layers, late
        self.config = SimpleNamespace(model_type="deepseek_v41", vision_n_layers=1 if vision else 0,
                                      image_token_id=7)
        eng = {1: FakeEngram(group, g, 0, m_dependent_wkv), 3: FakeEngram(group, g, 1, m_dependent_wkv)}
        self.layers = [FakeLayer(group, g, i, eng.get(i)) for i in range(n_layers)]
        self.hash_a = torch.randint(1, 97, (2, 3), generator=g)
        self.dspark_layers_to_capture = [2, 5]

    def engram_hasher(self, input_ids, fb):
        return (input_ids[:, None, None] * self.hash_a + torch.arange(3)) % 97

    def _check_late_layer_tail_readers(self, fb):
        pass


def fake_engine(group, backend):
    return SimpleNamespace(
        get_attn_backend=lambda: backend,
        check_cuda_graph_backend=lambda *a: False,
        Phase=SimpleNamespace(PREFILL="prefill"),
        Backend=SimpleNamespace(TC_PIECEWISE="pcg"),
        get_global_expert_distribution_recorder=lambda: SimpleNamespace(
            with_current_layer=lambda i: nullcontext()),
        get_forward=lambda: FORWARD,
        is_in_breakable_cuda_graph=lambda: False,
        is_cp_active=lambda fb: False,
        get_attn_tp_context=lambda: SimpleNamespace(input_scattered=False),
        # v0.5.21: no get_tp_group here; the TP group is get_parallel().tp_group
        get_parallel=lambda: SimpleNamespace(attn_dp_size=1, tp_size=group.world_size,
                                             attn_tp_size=group.world_size, tp_group=group),
        get_moe_a2a_backend=lambda: SimpleNamespace(is_none=lambda: True),
        get_platform=lambda: SimpleNamespace(is_sm100=False, is_blackwell=True, is_sm90=False),
        envs=SimpleNamespace(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=SimpleNamespace(get=lambda: False)),
        MQALayer=FakeAttn,
        _is_cuda=False,
    )


FakeAttn.forward = sp._make_attn_forward(FakeAttn.forward)
FakeMoE.forward = sp._make_moe_forward(FakeMoE.forward)
FakeLayer.forward_hc_pre_from_prev = sp._make_layer_forward(FakeLayer.forward_hc_pre_from_prev)
_STOCK_STREAM = FakeLayer._get_hc_stats_stream
FakeLayer._get_hc_stats_stream = sp._make_stats_stream(FakeLayer._get_hc_stats_stream)
FakeLayer._hc_combine = sp._make_hc_combine(FakeLayer._hc_combine)
FakeLayer.hc_post = sp._make_hc_post(FakeLayer.hc_post)
FakeLayer._hc_post_with_combine = sp._make_hc_post_with_combine(FakeLayer._hc_post_with_combine)
sp._REQUIRE_CUDA = False

# the wrappers' GPU-only gates and kernels, as toys (the real ones check is_cuda / Triton)
sp._post_combine_gate = lambda layer, x, residual, post, comb, pre, fb, m: _prefill_band(m, fb)
sp._post_norm_ok = lambda norm: norm is not None
sp.post_combine_rows = toy_post_combine
sp.norm_prefill_rows = toy_norm_prefill


def _agree_min(p, local):
    vals = p.group._exchange(torch.tensor([1 if local else 0]))
    return bool(min(int(v) for v in vals))


sp._agree_min = _agree_min


# ------------------------------------------------------------------------------------------
# reference: the engine's stock loop (v0.5.21 _forward_layers_hc_pre_from_prev), same fake model
# ------------------------------------------------------------------------------------------
def stock_forward_layers(self, positions, hidden_states, forward_batch, input_ids, input_ids_global,
                         capture_dspark, aux_out):
    hash_ids = self.engram_hasher(input_ids, forward_batch)
    tail = None
    if self.late_layer_start is not None:
        backend = sp._M["v4"].get_attn_backend()
        tail = backend.tail_forward_metadata.late_layer_tail
    prev_pre = None
    saved = None
    precomputed_attn = combined_attn = normalized_attn = None
    for i in range(self.start_layer, self.end_layer):
        if tail is not None and i == self.late_layer_start:
            combined_attn = normalized_attn = None
            saved = backend.enter_late_layer_tail(forward_batch)
            hidden_states, prev_pre, input_ids, input_ids_global = (
                tail.rows(hidden_states), tail.rows(prev_pre), tail.rows(input_ids),
                tail.rows(input_ids_global))
            positions = tail.positions
            hash_ids = tail.rows(hash_ids)
        engram = self.layers[i].engram
        if engram is not None:
            precomputed_attn = combined_attn = normalized_attn = None
            before = hidden_states
            hidden_states = engram(hidden_states, hash_ids[:, engram.layer_hash_index], forward_batch)
            if self.config.vision_n_layers > 0:
                hidden_states = torch.where((input_ids == self.config.image_token_id)[:, None, None],
                                            before, hidden_states)
        if capture_dspark and i in self.dspark_layers_to_capture:
            aux = hidden_states
            if tail is not None and i < self.late_layer_start:
                aux = tail.rows(aux)
            aux_out.append(aux.mean(dim=1))
        rows = hidden_states.shape[0]
        next_norm, next_input = None, []
        next_combined = (
            [] if ((128 <= rows <= 384 or _prefill_band(rows, forward_batch))
                   and i + 1 < self.end_layer and tail is None
                   and self.layers[i + 1].engram is None) else None)
        if next_combined is not None and rows >= 4096:
            next_norm = self.layers[i + 1].input_layernorm
        hidden_states, prev_pre = self.layers[i].forward_hc_pre_from_prev(
            positions=positions, hidden_states=hidden_states, input_ids=input_ids,
            forward_batch=forward_batch, input_ids_global=input_ids_global, prev_pre=prev_pre,
            precomputed_attn=precomputed_attn, next_norm=next_norm, next_input=next_input,
            combined_attn=combined_attn, normalized_attn=normalized_attn,
            next_combined=next_combined)
        precomputed_attn = next_input[0] if next_input else None
        combined_attn, normalized_attn = next_combined[0] if next_combined else (None, None)
    if saved is not None:
        backend.exit_late_layer_tail(saved, forward_batch)
        return hidden_states, prev_pre, tail
    return hidden_states, prev_pre, None


FB = SimpleNamespace(forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True,
                                                  is_decode=lambda: False,
                                                  is_target_verify=lambda: False))


def run(mode, rows, lens, *, exact=False, late=4, vision=False, rs_order="ring", m_dep=False, seed=1,
        debug=False, keep_pcn=False):
    """Every rank's (hidden, pre, aux list) for one forward in `mode` (stock | comm | shard)."""
    group = ThreadGroup(W, rs_order)
    model = FakeModel(group, seed, late=late, vision=vision, m_dependent_wkv=m_dep)
    tail = Tail(lens) if late is not None else None
    backend = Backend(tail)
    sp._M.clear()
    sp._M.update(v4=fake_engine(group, backend), v2=SimpleNamespace(DeepseekV2MoE=FakeMoE),
                 engram=SimpleNamespace(engram_gate=fake_engram_gate), mpcn=MPCN, mpf=_Fusion)
    sp._STATIC["checked"] = False
    sp._STATIC["tail_checked"] = True         # the engine's tail helpers are hash-checked on GPU
    sp._WKV_OK.clear()
    if not keep_pcn:
        sp._PCN_OK.clear()
    g = torch.Generator().manual_seed(seed + 100)
    R0 = torch.randn(rows, 4, D, generator=g).to(BF)
    ids = torch.randint(0, 40, (rows,), generator=g)
    pos = torch.arange(rows)
    old = sp.MODE, sp.EXACT

    def body(r):
        aux = []
        if mode == "stock":
            out = stock_forward_layers(model, pos, R0.clone(), FB, ids, ids, True, aux)
            return out[0], out[1], aux, None
        p = sp._plan(model, R0, FB)
        if p is None:
            out = stock_forward_layers(model, pos, R0.clone(), FB, ids, ids, True, aux)
            return out[0], out[1], aux, None
        d = sp._Dbg(1, rows, p, r) if debug else None
        sp._ctx.dbg = d
        try:
            out = sp._sp_forward_layers(model, p, pos, R0.clone(), FB, ids, ids, True, aux)
        finally:
            sp._ctx.dbg = None
        if d is not None:
            sp._debug_end(d, out)
            return out[0], out[1], aux, (p, d)
        return out[0], out[1], aux, p

    sp.MODE, sp.EXACT = ("off" if mode == "stock" else mode), exact
    try:
        res = run_ranks(group, body)
    finally:
        sp.MODE, sp.EXACT = old
    return res, model, group


def same(a, b):
    ha, pa, xa, _ = a
    hb, pb, xb, _ = b
    return (torch.equal(ha, hb) and torch.equal(pa, pb) and len(xa) == len(xb)
            and all(torch.equal(u, v) for u, v in zip(xa, xb)))


def main():
    torch.set_num_threads(1)
    # --- partition -------------------------------------------------------------------------
    for rows in (4096, 2048, 2052, 1500, 128, 65536):
        if not sp.eligible_rows(rows, W, 128):
            continue
        cover = []
        for r in range(W):
            lo, hi = sp.shard_range(rows, W, r)
            cover.extend(range(lo, hi))
        assert cover == list(range(rows)), rows
    assert not sp.eligible_rows(2050, W, 1024) and not sp.eligible_rows(2044, W, 2048)
    assert sp.eligible_rows(2048, W, 2048) and not sp.eligible_rows(4096, 1, 2048)
    idx = torch.tensor([0, 511, 512, 1023, 1024, 4095])
    o, l = sp.owner_local(idx, 1024)
    assert o.tolist() == [0, 0, 0, 0, 1, 3] and l.tolist() == [0, 511, 512, 1023, 0, 1023]
    assert sp.tail_plan(4096, 4, 128) == "owners" and sp.tail_plan(2048, 4, 512) == "full"

    # --- tail-row gathers against tail.rows(full) ---------------------------------------------
    for rows, lens, pad in ((4096, [4096], 0), (4096, [1000, 50, 2046, 1000], 0),
                            (2052, [513, 1, 1025, 513], 0), (2048, [64] * 32, 0),
                            (4096, [4000, 96], 3), (8192, [2047, 2049, 4096], 0)):
        group = ThreadGroup(W)
        tail = Tail(lens, pad_rows=pad)
        full = torch.randn(rows, 4, D).to(BF)
        pre = torch.rand(rows, 4)

        def body(r, rows=rows, tail=tail, full=full, pre=pre, group=group):
            p = sp._Plan(rows, W, r, group, True, False)
            return sp._tail_rows(p, tail, full[p.lo:p.hi]), sp._tail_rows(p, tail, pre[p.lo:p.hi])

        for got_h, got_p in run_ranks(group, body):
            assert torch.equal(got_h, tail.rows(full)) and torch.equal(got_p, tail.rows(pre)), (rows, lens)

    # --- whole loop ----------------------------------------------------------------------------
    MIN = sp.MIN_ROWS
    sp.MIN_ROWS = 1024
    try:
        cases = [
            ("one request", 4096, [4096], dict()),
            ("straddling requests", 4096, [1000, 50, 2046, 1000], dict()),
            ("8192, cross-layer", 8192, [8192], dict(late=None)),
            ("16384, shard in band", 16384, [16384], dict()),
            ("odd shards + vision", 2052, [513, 1, 1025, 513], dict(vision=True)),
            ("tail == all rows", 2048, [64] * 32, dict()),
            ("no bounded replay", 4096, [4096], dict(late=None)),
            ("not divisible (stock)", 2050, [2050], dict()),
            ("below MIN_ROWS (stock)", 1020, [1020], dict()),
        ]
        for name, rows, lens, kw in cases:
            stock, m_st, _ = run("stock", rows, lens, **kw)
            exact, m_ex, g_ex = run("shard", rows, lens, exact=True, **kw)
            pcn = dict(sp._PCN_OK)
            fast, m_fast, g_fast = run("shard", rows, lens, **kw)
            comm, _, g_comm = run("comm", rows, lens, **kw)
            comm_same, _, _ = run("comm", rows, lens, rs_order="same", **kw)
            planned = exact[0][3] is not None
            for r in range(W):
                assert same(exact[r], stock[r]), (name, "exact != stock", r)
                assert same(fast[r], comm[r]), (name, "fast != P0", r)
                assert same(comm_same[r], stock[r]), (name, "P0 with a same-order RS != stock", r)
            band = 4096 <= rows <= 65536
            if band:                # stock took the stats stream on every full-row layer
                assert "side-stream" in m_st.layers[0].stats_stream_seen, name
            if planned:
                assert not same(fast[0], stock[0]), (name, "the RS order should show")
                attn = m_fast.layers[0].self_attn
                assert set(attn.seen_rows) == {rows}, (name, attn.seen_rows)
                # never a stats stream inside the region (any mode)
                assert set(m_fast.layers[0].stats_stream_seen) == {None}, name
                late = kw.get("late", 4)
                tail_ars = 0 if late is None else 2 * (len(m_fast.layers) - late)   # stock late layers
                assert g_fast.calls["ar"] == tail_ars == g_comm.calls["ar"], (name, g_fast.calls)
                eng_shard = m_fast.layers[1].engram.wkv_calls
                assert rows // W in eng_shard, (name, eng_shard)
                assert pcn == ({rows: True} if band else {}), (name, pcn)
            else:
                assert same(fast[0], stock[0]) and same(comm[0], stock[0]), name
            print(f"  {name:24s} rows={rows}: exact==stock, fast==P0"
                  f"{'' if planned else ' (stock path)'}; collectives fast {g_fast.calls}")

        # gates keyed on the shard size instead of the chunk: bits change (the test is sensitive)
        GATE["logical"] = False
        try:
            wrong, _, _ = run("shard", 4096, [4096], exact=True)
        finally:
            GATE["logical"] = True
        stock, _, _ = run("stock", 4096, [4096])
        assert not same(wrong[0], stock[0]), "a shard-keyed gate should change the result"

        # the post+combine+norm decision keyed on the shard (no wrapper): bits change
        wrapped = FakeLayer._hc_post_with_combine
        FakeLayer._hc_post_with_combine = wrapped.__wrapped__
        try:
            wrong, _, _ = run("shard", 4096, [4096], exact=True)
        finally:
            FakeLayer._hc_post_with_combine = wrapped
        assert not same(wrong[0], stock[0]), "a shard-keyed post+combine gate should change the result"

        # without the stats-stream and fusion guards the attention would reduce twice. The hazard
        # needs >= 4096 rows inside the region: P0 at M = 4096, stage 1 at M = 16384 (shard 4096).
        def guarded(which):
            """{mode: equal to stock on every rank} with only `which` guards in place."""
            guard, no_mhc = FakeLayer._get_hc_stats_stream, sp._no_mhc_fusion
            if "stream" not in which:
                FakeLayer._get_hc_stats_stream = _STOCK_STREAM
            if "fusion" not in which:
                sp._no_mhc_fusion = nullcontext
            try:
                res = {}
                for mode, rows, kw in (("comm", 4096, dict(rs_order="same")),
                                       ("shard", 16384, dict(exact=True))):
                    ref, _, _ = run("stock", rows, [rows])
                    got, _, _ = run(mode, rows, [rows], **kw)
                    res[mode] = all(same(got[r], ref[r]) for r in range(W))
                return res
            finally:
                FakeLayer._get_hc_stats_stream, sp._no_mhc_fusion = guard, no_mhc

        assert guarded(()) == {"comm": False, "shard": False}, "a double all-reduce should show"
        for which in (("stream",), ("fusion",), ("stream", "fusion")):
            assert guarded(which) == {"comm": True, "shard": True}, which

        # fused kernel != the shard pair: this chunk exact via gathers, later ones stock
        MPCN.mhc_post_combine_norm_prefill = _fused_bad
        try:
            stock_bad, _, _ = run("stock", 4096, [1000, 3096])
            exact, _, _ = run("shard", 4096, [1000, 3096], exact=True)
            assert all(same(exact[r], stock_bad[r]) for r in range(W)), "post+combine fallback not exact"
            assert exact[0][3] is not None and sp._PCN_OK == {4096: False}, sp._PCN_OK
            again, _, _ = run("shard", 4096, [1000, 3096], exact=True, keep_pcn=True)
            assert again[0][3] is None, "a failed size must run stock afterwards"
            assert all(same(again[r], stock_bad[r]) for r in range(W))
        finally:
            MPCN.mhc_post_combine_norm_prefill = _fused_ok
            sp._PCN_OK.clear()

        # wkv whose result depends on M: the self-check must fall back and stay exact
        stock, _, _ = run("stock", 4096, [1000, 3096], m_dep=True)
        exact, m_ex, _ = run("shard", 4096, [1000, 3096], exact=True, m_dep=True)
        assert all(same(exact[r], stock[r]) for r in range(W)), "wkv fallback not exact"
        assert sp._WKV_OK and not any(sp._WKV_OK.values()), sp._WKV_OK
        stock, _, _ = run("stock", 4096, [1000, 3096])
        exact, _, _ = run("shard", 4096, [1000, 3096], exact=True)
        assert all(same(exact[r], stock[r]) for r in range(W))
        assert sp._WKV_OK and all(sp._WKV_OK.values()), sp._WKV_OK

        # debug compare + fingerprints: on (exact and fast), results unchanged, nothing flagged
        old_dbg = set(sp.DEBUG)
        sp.DEBUG.clear()
        sp.DEBUG.update({"compare", "fp"})
        try:
            for rows, lens, kw in ((4096, [1000, 50, 2046, 1000], dict()),
                                   (2052, [513, 1, 1025, 513], dict()),
                                   (4096, [4096], dict(late=None))):
                stock, _, _ = run("stock", rows, lens, **kw)
                for exact_ in (True, False):
                    dbg, _, _ = run("shard", rows, lens, exact=exact_, debug=True, **kw)
                    ref, _, _ = run("shard", rows, lens, exact=exact_, **kw)
                    for r in range(W):
                        d = dbg[r][3][1]
                        assert same(dbg[r][:3] + (None,), ref[r][:3] + (None,)), "debug changed the result"
                        assert d.checked >= 10 and not d.bad, (rows, d.checked, d.bad)
                        assert {k for _, k, _ in d.fps} == {"Rin", "ain", "aout", "min", "mout", "Rout", "pre"}
                        assert {layer for layer, _, _ in d.fps} == set(range(6)), d.fps[:3]
                    if exact_:
                        assert all(same(dbg[r][:3] + (None,), stock[r]) for r in range(W))
        finally:
            sp.DEBUG.clear()
            sp.DEBUG.update(old_dbg)
    finally:
        sp.MIN_ROWS = MIN
    print("test_prefill_sp: ok")


if __name__ == "__main__":
    main()
