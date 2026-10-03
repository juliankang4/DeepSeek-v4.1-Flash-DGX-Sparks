"""Print sha256[:16] of inspect.getsource(inspect.unwrap(obj)) for every engine source prefill_sp
pins, next to the value the ported adapter expects. Run inside the serving image, no server:

  docker run --rm --gpus all --network none --entrypoint python3 <image> - < print_engine_hashes.py

(DSV41_SOURCE unset, so sitecustomize installs no adapter; the pins are of the stock sources.)
Exit status 1 if anything differs.
"""
import hashlib
import importlib
import inspect
import os
import sys

MODS = {
    "v4": "sglang.srt.models.deepseek_v4",
    "v2": "sglang.srt.models.deepseek_v2",
    "engram": "sglang.srt.layers.engram",
    "hcn": "sglang.kernels.ops.layernorm.hc_combine_norm",
    "mpc": "sglang.kernels.ops.layernorm.mhc_post_combine",
    "mpcn": "sglang.kernels.ops.layernorm.mhc_post_combine_norm_prefill",
    "linear": "sglang.srt.layers.linear",
    "tail": "sglang.srt.layers.attention.deepseek_v4_backend",
}
EXPECTED = {
    # _EXPECTED: changed since f80c91a4b
    "v4.DeepseekV4Model._forward_layers_hc_pre_from_prev": "64f2801ffd57d158",
    "v4.DeepseekV4DecoderLayer.forward_hc_pre_from_prev": "49f22482d5370465",
    "v4.DeepseekV4DecoderLayer._hc_combine": "02b26ea5f805ed54",
    "v4.DeepseekV4DecoderLayer._hc_mix_stats": "a94d95cd06bd9676",
    "v4.DeepseekV4DecoderLayer.hc_post": "5f5dc4cf67cf091b",
    "v4.DeepseekV4DecoderLayer._run_moe_ffn_dp_sync": "1100ee58ff5fc3ff",
    "v4.DeepseekV4Model.forward": "8fe30b4e1874a893",
    "v4.MQALayer.forward": "78d101dc4c4baec9",
    "v2.DeepseekV2MoE.forward": "79f4c814db857b0d",
    "v2.DeepseekV2MoE.forward_normal": "6771a662f03d71ab",
    "engram.Engram.forward": "19df65b564f45d15",
    "engram.EngramEmbedding.forward": "138d015825c95c0e",
    "engram.EngramEmbedding._lookup": "bd991c49e8e3e8dc",
    "hcn.hc_combine_norm": "f6dfc00796090fc3",
    "linear.RowParallelLinear.forward": "6e429216a343a251",
    # _EXPECTED: new keys
    "v4.DeepseekV4DecoderLayer._hc_post_with_combine": "0a23e4a2729b5adc",
    "v4.DeepseekV4DecoderLayer._get_hc_stats_stream": "9a92da0e2ce75ab9",
    "engram.engram_gate": "34b0bfeb52ab1218",
    "mpc.mhc_post_combine": "a3f32180a2fbe7c8",
    "mpc._mhc_post_combine": "a634b84e16d3fbbb",
    "mpc.hc_norm_prefill": "33aacdba1d46530c",
    "mpc._hc_norm_prefill": "dea91349eed3279d",
    "mpcn.mhc_post_combine_norm_prefill": "6b353220faf832f0",
    # _EXPECTED: unchanged since f80c91a4b
    "hcn._hc_combine_norm_prefill": "74f71ebf85b010ee",
    # _EXPECTED_FP8 (checked when DSV41_PREFILL_SP_FP8=1)
    "v4.MQALayer._forward_prepare": "33d18c9cc577a883",
    "v4.MQALayer.accepts_mxfp8_swizzled_input": "5bb4fddfc5b329ca",
    "v4.MQALayer._compute_kv_to_cache": "f52f600c4986428c",
    "v4.MQALayer._compute_kv_bf16": "00c4ab90b403d1e9",
    # _EXPECTED_TAIL (unchanged)
    "tail.LateLayerTail.rows": "c58392d3ef651408",
    "tail.LateLayerTail.real_rows": "c78962c2e01e0e3d",
    "tail._tail_rows": "54a3cbff5e674e84",
}
FILES = {"deepseek_v4/mhc_post_combine_norm_prefill.cuh": "a8805d1f7b8a2ebc"}   # _EXPECTED_FILES


def src_hash(obj):
    obj = inspect.unwrap(getattr(obj, "fn", obj))            # triton JITFunction -> .fn
    return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()[:16]


def main():
    bad = 0
    for key, want in EXPECTED.items():
        mod, attr = key.split(".", 1)
        obj = importlib.import_module(MODS[mod])
        for part in attr.split("."):
            obj = getattr(obj, part)
        got = src_hash(obj)
        bad += got != want
        print(f"{'ok  ' if got == want else 'DIFF'} {key:58s} {got} (expected {want})")
    import sglang.kernels.jit as jit
    for rel, want in FILES.items():
        with open(os.path.join(os.path.dirname(jit.__file__), "csrc", rel), encoding="utf-8") as f:
            got = hashlib.sha256(f.read().encode()).hexdigest()[:16]
        bad += got != want
        print(f"{'ok  ' if got == want else 'DIFF'} csrc/{rel:53s} {got} (expected {want})")
    print(f"{bad} differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
