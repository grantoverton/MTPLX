"""T4 fused KDA gated output norm (Glm5NextRMSNormGated) for glm5_next.

``glm5_kda_o_norm`` replaces the ~11-op hand-rolled gated RMSNorm epilogue
(astype f32, x*x, mean, +eps, rsqrt, mul, w astype+mul, gate astype,
sigmoid, mul, astype T) with one Metal dispatch at verify widths.
Covers: (a) kernel-level bit-exact parity vs the eager module across
S=1..8 and weight dtypes, (b) layer-level output equality at S=4,
(c) MTPLX_GLM5_KDA_ONORM_FUSED=0 is bit-identical to the eager chain,
(d) S=16 declines and stays eager, (e) the eval'd primitive census
actually shrinks.
"""

import collections
import io
import re

import mlx.core as mx
import pytest

import mtplx.vendor.glm5_omlx.glm5_next.language as glm5_lang
from mtplx.vendor.glm5_omlx.glm5_next import kda_o_norm
from mtplx.vendor.glm5_omlx.glm5_next.config import TextConfig
from mtplx.vendor.glm5_omlx.glm5_next.language import (
    Glm5NextLinearAttention,
    Glm5NextRMSNormGated,
)
from mlx_vlm.models.cache import ArraysCache


def _config():
    # Same trick as the T2 test: hidden_size shrunk so projections stay
    # cheap; the kernel only sees the fixed 64x128 o_norm geometry.
    return TextConfig(
        model_type="glm5_next",
        vocab_size=512,
        hidden_size=512,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        kv_lora_rank=64,
        q_lora_rank=64,
        qk_rope_head_dim=0,
        v_head_dim=32,
        qk_nope_head_dim=32,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=2.5,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        index_topk=8,
        index_head_dim=32,
        index_n_heads=2,
        layer_types=["linear_attention"],
        mlp_layer_types=["dense"],
        linear_attn_config={
            "num_heads": 64,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
        },
    )


@pytest.fixture()
def attn():
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal kernel needs the GPU")
    mx.random.seed(11)
    layer = Glm5NextLinearAttention(_config())
    layer.eval()
    from mlx.utils import tree_flatten

    # Production dtype contract: everything bf16 except the keep-fp32
    # recurrent params (A_log, dt_bias) from glm5_next_cast_predicate.
    weights = [
        (
            k,
            v.astype(mx.bfloat16)
            if mx.issubdtype(v.dtype, mx.floating)
            and not ("A_log" in k or "dt_bias" in k)
            else v,
        )
        for k, v in tree_flatten(layer.parameters())
    ]
    layer.load_weights(weights)
    mx.eval(layer.parameters())
    return layer


def _forward(attn, x):
    return attn(x, mask=None, cache=ArraysCache(size=2))


def _count_calls(monkeypatch):
    calls = {"n": 0}
    orig = kda_o_norm.glm5_kda_o_norm

    def counting(*a, **k):
        r = orig(*a, **k)
        if r is not None:
            calls["n"] += 1
        return r

    monkeypatch.setattr(glm5_lang, "glm5_kda_o_norm", counting)
    return calls


def test_fused_matches_eager_s4(attn, monkeypatch):
    mx.random.seed(5)
    x = (mx.random.normal((1, 4, 512)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)

    monkeypatch.setenv("MTPLX_GLM5_KDA_ONORM_FUSED", "0")
    eager = _forward(attn, x)
    mx.eval(eager)

    calls = _count_calls(monkeypatch)
    monkeypatch.setenv("MTPLX_GLM5_KDA_ONORM_FUSED", "1")
    fused = _forward(attn, x)
    mx.eval(fused)

    assert calls["n"] == 1, "fused dispatch did not run — vacuous parity"
    assert (fused == eager).all().item(), "fused layer output diverged"


def test_env_zero_is_bit_identical_eager(attn, monkeypatch):
    mx.random.seed(7)
    x = (mx.random.normal((1, 4, 512)) * 0.5).astype(mx.bfloat16)

    monkeypatch.setattr(glm5_lang, "glm5_kda_o_norm", lambda *a, **k: None)
    monkeypatch.setenv("MTPLX_GLM5_KDA_ONORM_FUSED", "1")
    reference = _forward(attn, x)
    mx.eval(reference)

    monkeypatch.undo()
    monkeypatch.setenv("MTPLX_GLM5_KDA_ONORM_FUSED", "0")
    off = _forward(attn, x)
    mx.eval(off)
    assert (reference == off).all().item(), "env=0 output diverged from eager"


def test_shape_gate_fallback_s16(attn, monkeypatch):
    calls = _count_calls(monkeypatch)
    monkeypatch.setenv("MTPLX_GLM5_KDA_ONORM_FUSED", "1")

    mx.random.seed(8)
    x16 = (mx.random.normal((1, 16, 512)) * 0.5).astype(mx.bfloat16)
    mx.eval(x16)
    out16 = _forward(attn, x16)
    mx.eval(out16)
    assert calls["n"] == 0, "S=16 must not reach the fused dispatch"


@pytest.mark.parametrize("S", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_kernel_parity_sweep(S, dtype):
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal kernel needs the GPU")
    mx.random.seed(30 + S)
    for w_dtype in (dtype, mx.float32):
        mod = Glm5NextRMSNormGated(128, eps=1e-5)
        mod.weight = (mx.random.normal((128,)) * 0.4 + 1.0).astype(w_dtype)
        x = (mx.random.normal((1, S, 64, 128)) * 0.8).astype(dtype)
        g = (mx.random.normal((1, S, 64, 128)) * 3.0).astype(dtype)
        mx.eval(x, g, mod.weight)

        fused = kda_o_norm.glm5_kda_o_norm(x, g, mod.weight, mod.eps)
        assert fused is not None, f"S={S} dtype={dtype} w={w_dtype} gated out"
        eager = mod(x, g)
        mx.eval(fused, eager)
        # Bit-exact: the kernel replicates the eager f32 op-for-op
        # (float4-staged squares + pairwise simd_sum + precise::rsqrt +
        # abs-form precise::exp sigmoid).
        assert (fused == eager).all().item(), (
            f"S={S} dtype={dtype} w={w_dtype}: "
            f"{int((fused != eager).sum())} mismatched elements"
        )


def test_wrong_geometry_declines():
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal kernel needs the GPU")
    mod = Glm5NextRMSNormGated(64, eps=1e-5)  # head_dim 64: not the GLM shape
    x = mx.zeros((1, 4, 64, 64), dtype=mx.bfloat16)
    g = mx.zeros((1, 4, 64, 64), dtype=mx.bfloat16)
    assert kda_o_norm.glm5_kda_o_norm(x, g, mod.weight, mod.eps) is None


_VIEW_PRIMS = {
    "Broadcast",
    "Reshape",
    "Transpose",
    "Squeeze",
    "ExpandDims",
    "StopGradient",
    "Depends",
}


def _graph_kernels(outputs):
    buf = io.StringIO()
    mx.export_to_dot(buf, *outputs)
    counts = collections.Counter(re.findall(r'label ="([^"]+)"', buf.getvalue()))
    nodes = sum(counts.values())
    kernels = nodes - sum(counts[p] for p in _VIEW_PRIMS)
    return nodes, kernels


def test_fused_launch_census(attn, monkeypatch):
    mx.random.seed(10)
    x = (mx.random.normal((1, 4, 512)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)

    monkeypatch.setenv("MTPLX_GLM5_KDA_ONORM_FUSED", "0")
    out_e = _forward(attn, x)
    ne, ke = _graph_kernels([out_e])

    monkeypatch.setenv("MTPLX_GLM5_KDA_ONORM_FUSED", "1")
    out_f = _forward(attn, x)
    nf, kf = _graph_kernels([out_f])

    print(
        f"\nKDA o_norm census: eager nodes={ne} kernels={ke} | "
        f"fused nodes={nf} kernels={kf} | delta={ke - kf}"
    )
    assert ke - kf >= 8, f"fused path saved only {ke - kf} kernels"
