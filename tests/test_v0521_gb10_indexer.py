"""CPU logic checks for the patched v41_indexer (no GPU, no DeepGEMM, no Triton).

Kernel entry points are replaced by torch references; what is checked is the
python around them:
  1. row_split: the per-rank partitions, packed and gathered, give exactly the
     unsplit selections and block ids (publish, consume, full top-k), including
     through DenseBlocksBackend / FullTopKIndexer with a fake TP group.
  2. paged_masks (its torch fallback, same contract as the Triton kernels): the
     source / consumer selections equal the dense_blocks scheme's on the same
     logits, i.e. the two candidate schemes pick the same positions.
Run: python tests/test_cpu_logic.py <patched python/ dir> <v0.5.21 candidate_blocks.py>
"""

import ast
import importlib.util
import sys
import types
import typing

import torch
import torch.nn.functional as F

ROOT = sys.argv[1]
PKG = "sglang.srt.layers.attention.dsv4.v41_indexer"
CB = sys.argv[2]


def mod(name, **attrs):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


for name in [
    "sglang",
    "sglang.kernels",
    "sglang.kernels.ops",
    "sglang.kernels.ops.attention",
    "sglang.srt",
    "sglang.srt.layers",
    "sglang.srt.layers.attention",
    "sglang.srt.layers.attention.dsv4",
    "sglang.srt.mem_cache",
]:
    mod(name)


# ---- torch references for the kernels ----
def ragged_topk(scores, seq_lens, *, out_offsets, out_indices, row_starts=None):
    k = out_indices.shape[1]
    out_indices.fill_(-1)
    for i in range(scores.shape[0]):
        n = int(seq_lens[i])
        if n <= 0:
            continue
        top = scores[i, :n].topk(min(k, n)).indices
        out_indices[i, : top.numel()] = (top + int(out_offsets[i])).to(out_indices.dtype)


def paged_topk(scores, seq_lens, page_tables, out_page_indices, page_size, out_raw=None, *_):
    k = out_page_indices.shape[1]
    lens = seq_lens.reshape(-1)
    out_page_indices.fill_(-1)
    if out_raw is not None:
        out_raw.fill_(-1)
    for i in range(scores.shape[0]):
        n = int(lens[i])
        top = scores[i, :n].topk(min(k, n))
        idx = top.indices[top.values > -torch.inf]
        slots = page_tables[i, idx // page_size] * page_size + idx % page_size
        out_page_indices[i, : idx.numel()] = slots.to(out_page_indices.dtype)
        if out_raw is not None:
            out_raw[i, : idx.numel()] = idx.to(out_raw.dtype)


def topk_from_metadata(logits, metadata, page_indices, raw_indices=None, *, rows=None, topk_metadata=None):
    sl = slice(None) if rows is None else rows
    paged_topk(
        logits,
        metadata.compressed_seq_lens[sl],
        metadata.page_table[sl],
        page_indices[sl],
        metadata.compressed_page_size,
        raw_indices[sl] if raw_indices is not None else None,
    )


KV = {}


def flat_tiles(*, q, kv, weights, starts, lengths, context_lengths, budget_bytes, width_align=4):
    rows = q[0].shape[0]
    width = -(-max(context_lengths, default=0) // width_align) * width_align
    if rows == 0 or width == 0:
        return
    k = kv[0]
    per_tile = KV["rows_per_tile"]
    for off in range(0, rows, per_tile):
        tile = slice(off, min(off + per_tile, rows))
        out = torch.randn(tile.stop - tile.start, width, dtype=torch.float64) * 1e3  # garbage
        for i in range(tile.start, tile.stop):
            s, n = int(starts[i]), int(lengths[i])
            out[i - tile.start, :n] = torch.relu(q[0][i] @ k[s : s + n].T) .mul(weights[i][:, None]).sum(0)
        yield tile, out


# select_candidate_block_ids / topk_among_blocks: the real torch code of v0.5.21
src = open(CB).read()
tree = ast.parse(src)
fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("select_candidate_block_ids", "topk_among_blocks")]
ns = {"torch": torch, "F": F, "Union": typing.Union, "Optional": typing.Optional}  # Python 3.12 evaluates annotations at def time
exec(compile(ast.Module(body=fns, type_ignores=[]), CB, "exec"), ns)

mod("sglang.kernels.ops.attention.dsv4", topk_transform_ragged_v2=ragged_topk)
mod("sglang.kernels.ops.attention.dsv4.candidate_blocks", select_candidate_block_ids=ns["select_candidate_block_ids"], topk_among_blocks=ns["topk_among_blocks"])
mod("sglang.kernels.ops.attention.dsv4.fp4_indexer", fp4_index_logits_decode=None)
mod("sglang.kernels.ops.attention.dsv4.index_logits", flat_index_logits_tiles=flat_tiles, deep_gemm_fp4_paged_mqa_logits=None)
mod("sglang.kernels.ops.attention.dsv4.topk", topk_transform_paged_torch=None)
mod("sglang.srt.layers.attention.dsv4.indexer", topk_transform_paged_from_metadata=topk_from_metadata)

base = ROOT + "/sglang/srt/layers/attention/dsv4/v41_indexer/"
pkg = mod(PKG)
pkg.__path__ = [base]


def load(short):
    spec = importlib.util.spec_from_file_location(f"{PKG}.{short}", base + short + ".py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


types_m = load("types")
scoring = load("scoring")
row_split = load("row_split")
dense_blocks = load("dense_blocks")
full_topk = load("full_topk")
paged_masks = load("paged_masks")

# CPU tensors: drop only the is_cuda / capture part of the gate
_eligible = row_split.RowSplit.eligible
row_split.RowSplit.eligible = lambda self, data: (
    data.num_rows >= self.min_rows
    and max(data.lens_per_request, default=0) >= self.min_context
)
GATHERS = [0]

torch.manual_seed(0)
H, D, TOPK, TOPK_BLOCKS, BS = 4, 16, 24, 6, 8


def make_data(rows_per_request, lens_per_request):
    starts, s = [], 0
    for lc in lens_per_request:
        starts.append(s)
        s += lc
    total = s
    rows = sum(rows_per_request)
    k = torch.randn(total + 64, D, dtype=torch.float64)
    KV["k"] = k
    request_starts = torch.repeat_interleave(torch.tensor(starts, dtype=torch.int32), torch.tensor(rows_per_request))
    # each request's rows see a growing causal prefix ending at its length
    lens = []
    for n, lc in zip(rows_per_request, lens_per_request):
        lens += [max(0, lc - (n - 1 - r)) for r in range(n)]
    data = scoring.DeepGEMMPrefillData(
        k_slots=torch.arange(total, dtype=torch.int64),
        request_starts=request_starts,
        lens_per_request=list(lens_per_request),
        rows_per_request=list(rows_per_request),
        compress_lens=torch.tensor(lens, dtype=torch.int32),
        q_fp4=torch.randn(rows, H, D, dtype=torch.float64),
        q_sf=torch.zeros(rows, H, dtype=torch.int32),
        weights=torch.rand(rows, H, dtype=torch.float64),
    )
    return data, (k, None)


class FakeGroup:
    """Rank 0 of `world`; the other ranks' packs come from `others(packed)`."""

    def __init__(self, world, others):
        self.world_size, self.rank_in_group, self.others = world, 0, others

    def all_gather(self, packed, dim=0):
        GATHERS[0] += 1
        return torch.cat([packed] + self.others(packed), dim=0)


def per_rank_packs(data, world, fn):
    """Every rank's padded pack, computed as that rank would."""
    packs = []
    for r in range(world):
        g = types.SimpleNamespace(world_size=world, rank_in_group=r)
        sp = row_split.RowSplit(g, 1, 1)
        start, end, extent = sp.partition(data.num_rows)
        local = fn(sp.local(data, start, end), start, end)
        pad = local.new_full((extent, local.shape[1]), -1)
        pad[: local.shape[0]] = local
        packs.append(pad)
    return packs


def check_split(rows_per_request, lens_per_request, world, rows_per_tile):
    KV["rows_per_tile"] = rows_per_tile
    data, kv = make_data(rows_per_request, lens_per_request)
    sel, blocks = dense_blocks._publish_prefill_blocks(data=data, kv=kv, topk=TOPK, topk_blocks=TOPK_BLOCKS, block_size=BS)

    def pub(local, s, e):
        a, b = dense_blocks._publish_prefill_blocks(data=local, kv=kv, topk=TOPK, topk_blocks=TOPK_BLOCKS, block_size=BS)
        return torch.cat([a, b], 1)

    packed = torch.cat(per_rank_packs(data, world, pub))[: data.num_rows]
    assert torch.equal(packed[:, :TOPK], sel), "publish picks differ"
    assert torch.equal(packed[:, TOPK:], blocks), "publish block ids differ"

    con = dense_blocks._consume_prefill_blocks(data=data, kv=kv, topk=TOPK, blocks=blocks, block_size=BS)
    packed = torch.cat(per_rank_packs(data, world, lambda local, s, e: dense_blocks._consume_prefill_blocks(data=local, kv=kv, topk=TOPK, blocks=blocks[s:e], block_size=BS)))[: data.num_rows]
    assert torch.equal(packed, con), "consume picks differ"

    full = scoring.dense_prefill_topk(data, kv, topk=TOPK)
    packed = torch.cat(per_rank_packs(data, world, lambda local, s, e: scoring.dense_prefill_topk(local, kv, topk=TOPK)))[: data.num_rows]
    assert torch.equal(packed, full), "full top-k picks differ"
    return data, kv, sel, blocks, con, full


def check_backends(rows_per_request, lens_per_request, world, rows_per_tile):
    data, kv, sel, blocks, con, full = check_split(rows_per_request, lens_per_request, world, rows_per_tile)
    rows = data.num_rows

    def selection():
        return types_m.Selection(page_indices=torch.full((rows, TOPK), -7, dtype=torch.int32), raw_indices=torch.full((rows, TOPK), -7, dtype=torch.int32))

    pool = types.SimpleNamespace(get_low_ratio_index_k_fp4=lambda layer_id, slots: kv)
    inputs = types.SimpleNamespace(allow_row_split=True, layer_id=0, indexer=types.SimpleNamespace(index_topk=TOPK))
    dense_blocks.get_deep_gemm_prefill_data = lambda inputs, req_to_token: data
    full_topk.get_deep_gemm_prefill_data = lambda inputs, req_to_token: data

    def others_for(fn):
        return lambda packed: per_rank_packs(data, world, fn)[1:]

    def run(backend_fn, split_fn, use_split):
        out = selection()
        split = None
        if use_split:
            split = row_split.RowSplit(FakeGroup(world, others_for(split_fn)), 1, 1)
        return out, split

    # publish
    ref_out = selection()
    nb = dense_blocks.DenseBlocksBackend(token_to_kv_pool=pool, req_to_token=None, candidate_topk_blocks=TOPK_BLOCKS, candidate_block_size=BS, use_deep_gemm_prefill=True)
    ref_pub = nb.publish_prefill(inputs, ref_out)
    sb = dense_blocks.DenseBlocksBackend(token_to_kv_pool=pool, req_to_token=None, candidate_topk_blocks=TOPK_BLOCKS, candidate_block_size=BS, use_deep_gemm_prefill=True,
                                         row_split=row_split.RowSplit(FakeGroup(world, others_for(lambda local, s, e: torch.cat(dense_blocks._publish_prefill_blocks(data=local, kv=kv, topk=TOPK, topk_blocks=TOPK_BLOCKS, block_size=BS), 1))), 1, 1))
    out = selection()
    pub = sb.publish_prefill(inputs, out)
    assert torch.equal(out.page_indices, ref_out.page_indices) and torch.equal(out.raw_indices, ref_out.raw_indices), "backend publish selection differs"
    assert torch.equal(pub.blocks, ref_pub.blocks), "backend published blocks differ"
    # consume
    ref_out = selection()
    nb.consume_prefill(inputs, ref_pub, ref_out)
    sb.row_split = row_split.RowSplit(FakeGroup(world, others_for(lambda local, s, e: dense_blocks._consume_prefill_blocks(data=local, kv=kv, topk=TOPK, blocks=ref_pub.blocks[s:e], block_size=BS))), 1, 1)
    out = selection()
    sb.consume_prefill(inputs, pub, out)
    assert torch.equal(out.page_indices, ref_out.page_indices) and torch.equal(out.raw_indices, ref_out.raw_indices), "backend consume selection differs"
    # gate off -> unsplit path
    off = types.SimpleNamespace(allow_row_split=False, layer_id=0, indexer=inputs.indexer)
    out = selection()
    sb.consume_prefill(off, pub, out)
    assert torch.equal(out.page_indices, ref_out.page_indices)
    # full top-k
    ref_out = selection()
    ft = full_topk.FullTopKIndexer(token_to_kv_pool=pool, req_to_token=None, use_deep_gemm_prefill=True, use_deep_gemm_decode=True)
    ft.topk_prefill(inputs, ref_out)
    ft.row_split = row_split.RowSplit(FakeGroup(world, others_for(lambda local, s, e: scoring.dense_prefill_topk(local, kv, topk=TOPK))), 1, 1)
    out = selection()
    ft.topk_prefill(inputs, out)
    assert torch.equal(out.page_indices, ref_out.page_indices) and torch.equal(out.raw_indices, ref_out.raw_indices), "backend full top-k differs"


# ragged requests, an empty request, partitions cutting through requests, tiles
# not aligned to requests or partitions, one rank with no rows
check_backends([300, 0, 211, 489], [700, 50, 211, 1200], 4, 96)
check_backends([1000], [3000], 4, 4096)
check_backends([5, 60], [40, 60], 4, 7)  # 65 rows < one 128-row tile: ranks 1..3 empty
check_backends([700, 333], [900, 2000], 8, 129)
assert GATHERS[0] == 4 * 3, GATHERS  # publish, consume, full top-k per case
print("row_split: OK")


# ---- paged_masks fallback vs the dense_blocks scheme on the same logits ----
def check_paged_masks(rows, width, page_size, ratio_topk):
    lens = torch.randint(1, width + 1, (rows,), dtype=torch.int32)
    lens[0] = width
    pages = width // page_size
    page_table = torch.stack([torch.randperm(10 * pages)[:pages] for _ in range(rows)]).to(torch.int32)
    logits_src = torch.randn(rows, width, dtype=torch.float32)
    logits_con = torch.randn(rows, width, dtype=torch.float32)
    for lg in (logits_src, logits_con):
        for i in range(rows):
            lg[i, int(lens[i]):] = float("nan")  # garbage past the length
    md = types.SimpleNamespace(
        compressed_seq_lens=lens, page_table=page_table, compressed_page_size=page_size,
        max_compressed_seq_len=width, deep_gemm_metadata=torch.empty(0), use_topk_v2=True,
        topk_metadata=None, topk_metadata_chunks=None,
    )
    feed = iter([logits_src.clone(), logits_con.clone()])
    paged_masks.get_deep_gemm_decode_data = lambda inputs, pool: scoring.DeepGEMMDecodeData(torch.empty(rows, 1), torch.empty(rows, 1), torch.empty(rows, 1), None)
    paged_masks.deep_gemm_fp4_paged_mqa_logits = lambda *a: next(feed)
    be = paged_masks.PagedMasksBackend(token_to_kv_pool=None, candidate_topk_blocks=TOPK_BLOCKS, candidate_block_size=BS)
    inputs = types.SimpleNamespace(paged_metadata=md)
    out_s = types_m.Selection(page_indices=torch.empty(rows, ratio_topk, dtype=torch.int32), raw_indices=torch.empty(rows, ratio_topk, dtype=torch.int32))
    pub = be.publish_decode(inputs, out_s)
    out_c = types_m.Selection(page_indices=torch.empty(rows, ratio_topk, dtype=torch.int32), raw_indices=torch.empty(rows, ratio_topk, dtype=torch.int32))
    be.consume_decode(inputs, pub, out_c)

    # dense_blocks scheme on the same logits (decode: -inf past lens, as decode_scores)
    col = torch.arange(width)
    s_src = logits_src.masked_fill(col[None] >= lens[:, None].long(), -torch.inf)
    s_con = logits_con.masked_fill(col[None] >= lens[:, None].long(), -torch.inf)
    ids = ns["select_candidate_block_ids"](s_src, lens[:, None].long(), TOPK_BLOCKS, BS)
    k = min(ratio_topk, width)
    src_pick = s_src.topk(k, dim=-1, sorted=False).indices
    con_pick = ns["topk_among_blocks"](s_con, lens.long(), ids, k, BS)

    def as_set(row_picks, row_lens):
        return {int(x) for x in row_picks if 0 <= int(x) < row_lens}

    for i in range(rows):
        n = int(lens[i])
        assert as_set(out_s.raw_indices[i], n) == as_set(src_pick[i], n), f"source row {i}"
        assert as_set(out_c.raw_indices[i], n) == as_set(con_pick[i], n), f"consumer row {i}"
        # mask == block ids
        m = pub.mask[i, :n].nonzero().flatten() // BS
        assert set(m.tolist()) == {int(b) for b in ids[i] if b >= 0}, f"mask row {i}"
        # slots consistent with raw positions
        for j in range(ratio_topk):
            r = int(out_c.raw_indices[i, j])
            p = int(out_c.page_indices[i, j])
            assert (r < 0 and p == -1) or p == int(page_table[i, r // page_size]) * page_size + r % page_size


check_paged_masks(rows=6, width=512, page_size=64, ratio_topk=24)
check_paged_masks(rows=16, width=1024, page_size=64, ratio_topk=40)
print("paged_masks fallback == dense_blocks selections: OK")
