"""GPU check for the one kernel claim prefill_sp's v0.5.21 port relies on (engine image, 1 GPU):

  on rows [r*M/4, (r+1)*M/4), the shard pair prefill_sp runs (the engine's Triton _mhc_post_combine
  + _hc_norm_prefill, launched as for the full chunk) gives the bits of the engine's fused CUDA
  mhc_post_combine_norm_prefill on the whole chunk; and the engine's own Triton fallback (its
  unaligned-pointer path) gives them on the whole chunk too.

  docker run --rm --gpus all --network none -v $PWD:/ds41 -w /ds41 -e PYTHONPATH=/ds41/adapter \
      --entrypoint python3 <image> tests/check_post_combine_gpu.py

The adapter also checks this at run time on the first chunk of every size (and falls back exactly),
so a failure here costs speed, not correctness; it says SP would run stock at that size.
"""
import os

os.environ.setdefault("DSV41_PREFILL_SP", "1")
import torch  # noqa: E402

import prefill_sp as sp  # noqa: E402
from sglang.kernels.ops.layernorm import mhc_post_combine as mpc  # noqa: E402
from sglang.kernels.ops.layernorm import mhc_post_combine_norm_prefill as mpcn  # noqa: E402

sp._M.update(mpc=mpc, mpcn=mpcn)
DEV, BF, H, W, EPS = "cuda", torch.bfloat16, 5120, 4, 1e-6


def inputs(m, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(m, H, device=DEV, generator=g, dtype=BF) * 0.3
    r = torch.randn(m, 4, H, device=DEV, generator=g, dtype=BF) * 0.5
    r[:, :, :64] *= 40                                   # outlier columns, as the residual has
    post = 2 * torch.sigmoid(torch.randn(m, 4, device=DEV, generator=g))
    comb = torch.softmax(torch.randn(m, 4, 4, device=DEV, generator=g), dim=-1)
    pre = torch.sigmoid(torch.randn(m, 4, device=DEV, generator=g)) + 1e-2
    w = (1 + 0.1 * torch.randn(H, device=DEV, generator=g)).to(BF)
    return x, r, post.contiguous(), comb.contiguous(), pre.contiguous(), w


def main():
    fails = []
    for m in (4096, 8192, 16384, 32768):
        x, r, post, comb, pre, w = inputs(m, m)
        fu, fn = mpcn.mhc_post_combine_norm_prefill(x, r, post, comb, pre, w, EPS)
        tu, tc = mpc.mhc_post_combine(x, r, post, comb, pre)
        tn = mpc.hc_norm_prefill(tc, w, EPS)
        engine_pair = torch.equal(tu, fu) and torch.equal(tn, fn)
        shards = []
        for k in range(W):
            lo, hi = sp.shard_range(m, W, k)
            u, c = sp.post_combine_rows(x[lo:hi].contiguous(), r[lo:hi].contiguous(),
                                        post[lo:hi].contiguous(), comb[lo:hi].contiguous(),
                                        pre[lo:hi].contiguous(), m)
            n = sp.norm_prefill_rows(c, w, EPS, m)
            shards.append(torch.equal(u, fu[lo:hi]) and torch.equal(n, fn[lo:hi]))
        nd = int((tn != fn).sum())
        print(f"  M={m} (shard {m // W}): engine Triton pair == fused CUDA on the chunk: {engine_pair}"
              f" ({nd} normalized elements differ); shard pair == fused slice per rank: {shards}",
              flush=True)
        if not (engine_pair and all(shards)):
            fails.append(m)
        del x, r, fu, fn, tu, tc, tn
        torch.cuda.empty_cache()
    assert not fails, f"shard pair != fused kernel at M={fails}: prefill_sp would run stock there"
    print("check_post_combine_gpu: ok")


if __name__ == "__main__":
    main()
