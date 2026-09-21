"""T1 fused HyperConnection pre-branch parity for glm5_next (GPU: Metal).

The ``glm5_hc_normalized_norm`` single dispatch must reproduce the eager
chain (f32 upcast -> fp32 rms_norm -> tokenwise mix -> sinkhorn collapse ->
branch RMSNorm) through the decoder-layer boundary at verify widths. Covers:
(a) fused == eager numerics at S=1 (gate-off, bit-identical) and S=4
    (fused, bf16-rounding tolerance) for a KDA+MoE layer,
(b) MTPLX_GLM5_HC_FUSED=0 returns bit-identical eager output,
(c) the shape gate falls back at S=16 and under a mask,
(d) the eval'd primitive census actually shrinks.
An anti-vacuous counter asserts the fused dispatch ran on the fused arm.
"""

import collections
import io
import re

import mlx.core as mx
import pytest

import mtplx.vendor.glm5_omlx.glm5_next.language as glm5_lang
from mtplx.vendor.glm5_omlx.glm5_next import hc_fused
from mtplx.vendor.glm5_omlx.glm5_next.config import TextConfig
from mtplx.vendor.glm5_omlx.glm5_next.language import Glm5NextDecoderLayer
from mlx_vlm.models.cache import ArraysCache


def _tiny_config():
    # hidden_size/hc_mult are fixed by the kernel's 1024-thread vec4 layout;
    # every other width is shrunk so the layer is cheap to build.
    return TextConfig(
        model_type="glm5_next",
        vocab_size=512,
        hidden_size=4096,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=2.5,
        kv_lora_rank=64,
        q_lora_rank=64,
        qk_rope_head_dim=0,
        v_head_dim=32,
        qk_nope_head_dim=32,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        index_topk=8,
        index_head_dim=32,
        index_n_heads=2,
        layer_types=["linear_attention"],
        mlp_layer_types=["sparse"],
        linear_attn_config={
            "num_heads": 4,
            "head_dim": 64,
            "short_conv_kernel_size": 4,
        },
    )


@pytest.fixture()
def layer():
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal kernel needs the GPU")
    mx.random.seed(11)
    lay = Glm5NextDecoderLayer(_tiny_config(), 0)
    lay.eval()  # serve-mode: the gates refuse training modules
    # Mirror the production dtype contract: every float tensor bf16 except
    # the keep-fp32 set from glm5_next_cast_predicate / sanitize.
    from mlx.utils import tree_flatten

    keep_f32 = (
        "attn_hc.",
        "ffn_hc.",
        "A_log",
        "dt_bias",
        "mlp.gate.weight",
        "e_score_correction_bias",
    )
    weights = [
        (
            k,
            v.astype(mx.bfloat16)
            if mx.issubdtype(v.dtype, mx.floating)
            and not any(s in k for s in keep_f32)
            else v,
        )
        for k, v in tree_flatten(lay.parameters())
    ]
    lay.load_weights(weights)
    for hc in (lay.attn_hc, lay.ffn_hc):
        hc.fn = mx.random.normal(hc.fn.shape) * 0.02
        hc.base = mx.random.normal(hc.base.shape) * 0.05
    mx.eval(lay.parameters())
    return lay


def _forward(lay, x, mask=None):
    cache = ArraysCache(size=2)
    return lay(x, mask=mask, cache=cache)


def _max_abs(a, b):
    return mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()


def test_fused_matches_eager_s4(layer, monkeypatch):
    mx.random.seed(5)
    x = (mx.random.normal((1, 4, 4, 4096)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)

    monkeypatch.setenv("MTPLX_GLM5_HC_FUSED", "0")
    eager = _forward(layer, x)
    mx.eval(eager)

    calls = {"n": 0}
    orig = hc_fused.glm5_hc_normalized_norm

    def counting(*a, **k):
        r = orig(*a, **k)
        if r is not None:
            calls["n"] += 1
        return r

    monkeypatch.setattr(glm5_lang, "glm5_hc_normalized_norm", counting)
    monkeypatch.setenv("MTPLX_GLM5_HC_FUSED", "1")
    fused = _forward(layer, x)
    mx.eval(fused)
    assert calls["n"] == 2, "fused dispatch did not run — vacuous parity"

    scale = mx.abs(eager.astype(mx.float32)).max().item()
    diff = _max_abs(fused, eager)
    assert diff <= 0.008 * max(scale, 1.0) + 1e-3, (
        f"fused vs eager max abs diff {diff} at output scale {scale}"
    )


def test_fused_s1_stays_eager(layer, monkeypatch):
    mx.random.seed(6)
    x = (mx.random.normal((1, 1, 4, 4096)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)

    calls = {"n": 0}
    orig = hc_fused.glm5_hc_normalized_norm

    def counting(*a, **k):
        r = orig(*a, **k)
        if r is not None:
            calls["n"] += 1
        return r

    monkeypatch.setattr(glm5_lang, "glm5_hc_normalized_norm", counting)
    monkeypatch.setenv("MTPLX_GLM5_HC_FUSED", "1")
    out = _forward(layer, x)
    mx.eval(out)
    # The gate is 1 < S <= 8: decode rows take the stock chain (which is also
    # what the mx.compile'd _ffn_block was validated on).
    assert calls["n"] == 0


def test_env_zero_is_bit_identical_eager(layer, monkeypatch):
    mx.random.seed(7)
    x = (mx.random.normal((1, 4, 4, 4096)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)

    # Reference: fused helper forced to decline -- the stock eager chain.
    monkeypatch.setattr(
        glm5_lang, "glm5_hc_normalized_norm", lambda *a, **k: None
    )
    monkeypatch.setenv("MTPLX_GLM5_HC_FUSED", "1")
    reference = _forward(layer, x)
    mx.eval(reference)

    monkeypatch.undo()
    monkeypatch.setenv("MTPLX_GLM5_HC_FUSED", "0")
    off = _forward(layer, x)
    mx.eval(off)
    assert (reference == off).all().item(), "env=0 output diverged from eager"


def test_shape_gate_fallback_s16_and_mask(layer, monkeypatch):
    calls = {"n": 0}
    orig = hc_fused.glm5_hc_normalized_norm

    def counting(*a, **k):
        r = orig(*a, **k)
        if r is not None:
            calls["n"] += 1
        return r

    monkeypatch.setattr(glm5_lang, "glm5_hc_normalized_norm", counting)
    monkeypatch.setenv("MTPLX_GLM5_HC_FUSED", "1")

    mx.random.seed(8)
    x16 = (mx.random.normal((1, 16, 4, 4096)) * 0.5).astype(mx.bfloat16)
    mx.eval(x16)
    out16 = _forward(layer, x16)
    mx.eval(out16)
    assert calls["n"] == 0, "S=16 must not reach the fused dispatch"

    mx.random.seed(9)
    x4 = (mx.random.normal((1, 4, 4, 4096)) * 0.5).astype(mx.bfloat16)
    mask = mx.ones((1, 4), dtype=mx.bool_)
    out_masked = _forward(layer, x4, mask=mask)
    mx.eval(out_masked)
    assert calls["n"] == 0, "masked calls must not reach the fused dispatch"


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
    """Primitive count via mx.export_to_dot, minus stride-only view prims --
    the same bracket the op-census scripts use."""
    buf = io.StringIO()
    mx.export_to_dot(buf, *outputs)
    counts = collections.Counter(
        re.findall(r'label ="([^"]+)"', buf.getvalue())
    )
    nodes = sum(counts.values())
    kernels = nodes - sum(counts[p] for p in _VIEW_PRIMS)
    return nodes, kernels


def test_fused_launch_census(layer, monkeypatch):
    mx.random.seed(10)
    x = (mx.random.normal((1, 4, 4, 4096)) * 0.5).astype(mx.bfloat16)
    mx.eval(x, layer.parameters())

    monkeypatch.setenv("MTPLX_GLM5_HC_FUSED", "0")
    cache_e = ArraysCache(size=2)
    out_e = layer(x, mask=None, cache=cache_e)
    ne, ke = _graph_kernels([out_e, cache_e[0], cache_e[1]])

    monkeypatch.setenv("MTPLX_GLM5_HC_FUSED", "1")
    cache_f = ArraysCache(size=2)
    out_f = layer(x, mask=None, cache=cache_f)
    nf, kf = _graph_kernels([out_f, cache_f[0], cache_f[1]])

    # Two HC calls per layer x ~12 dispatches saved each; keep the bound
    # conservative against backend lowering noise.
    assert kf <= ke - 15, f"kernels eager={ke} fused={kf}"
    mx.eval(out_e, cache_e[0], cache_e[1], out_f, cache_f[0], cache_f[1])
