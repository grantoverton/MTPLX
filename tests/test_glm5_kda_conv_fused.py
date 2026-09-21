"""T2 fused KDA conv tape + q/k l2norm for glm5_next (GPU: Metal).

``glm5_kda_conv_norm`` replaces concat conv_state -> state roll -> conv1d
-> silu -> split -> f32 l2norm (q scaled) with one Metal dispatch at verify
widths. Covers: (a) fused == eager at S=4 through the layer boundary
(bit-identical on the captured q/k/v stash and on the layer output),
(b) MTPLX_GLM5_KDA_CONV_FUSED=0 is bit-identical to the eager chain,
(c) S=16 declines and stays eager, (d) kernel-level parity sweep S=2..6,
(e) the eval'd primitive census actually shrinks.
"""

import collections
import io
import re

import mlx.core as mx
import mlx.nn as nn
import pytest

import mtplx.vendor.glm5_omlx.glm5_next.language as glm5_lang
from mtplx.vendor.glm5_omlx.glm5_next import kda_conv_norm
from mtplx.vendor.glm5_omlx.glm5_next.config import TextConfig
from mtplx.vendor.glm5_omlx.glm5_next.language import Glm5NextLinearAttention
from mlx_vlm.models.cache import ArraysCache

_C = 24576
_QK = 8192


def _config():
    # The kernel is hardwired to the GLM-5.3 conv geometry (q/k/v 8192 =
    # 64 heads x 128, conv width 24576); hidden_size is shrunk so the
    # projections stay cheap to build.
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
    orig = kda_conv_norm.glm5_kda_conv_norm

    def counting(*a, **k):
        r = orig(*a, **k)
        if r is not None:
            calls["n"] += 1
        return r

    monkeypatch.setattr(glm5_lang, "glm5_kda_conv_norm", counting)
    return calls


def test_fused_matches_eager_s4(attn, monkeypatch):
    mx.random.seed(5)
    x = (mx.random.normal((1, 4, 512)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)

    monkeypatch.setenv("MTPLX_GLM5_KDA_CONV_FUSED", "0")
    cache_e = ArraysCache(size=2)
    with glm5_lang.verify_capture_scope():
        eager = attn(x, mask=None, cache=cache_e)
    mx.eval(eager)
    stash_e = cache_e._mtplx_verify_rows

    calls = _count_calls(monkeypatch)
    monkeypatch.setenv("MTPLX_GLM5_KDA_CONV_FUSED", "1")
    cache_f = ArraysCache(size=2)
    with glm5_lang.verify_capture_scope():
        fused = attn(x, mask=None, cache=cache_f)
    mx.eval(fused)
    stash_f = cache_f._mtplx_verify_rows

    assert calls["n"] == 1, "fused dispatch did not run — vacuous parity"
    # stash = (mixed, q, k, v, a, b_o): q/k/v came straight out of the kernel
    for name, fe, ee in zip(
        ("mixed", "q", "k", "v", "a", "b_o"), stash_f, stash_e
    ):
        assert (fe == ee).all().item(), f"stash {name} diverged"
    assert (fused == eager).all().item(), "fused layer output diverged"
    assert (cache_f[0] == cache_e[0]).all().item(), "conv state roll diverged"


def test_env_zero_is_bit_identical_eager(attn, monkeypatch):
    mx.random.seed(7)
    x = (mx.random.normal((1, 4, 512)) * 0.5).astype(mx.bfloat16)

    monkeypatch.setattr(glm5_lang, "glm5_kda_conv_norm", lambda *a, **k: None)
    monkeypatch.setenv("MTPLX_GLM5_KDA_CONV_FUSED", "1")
    reference = _forward(attn, x)
    mx.eval(reference)

    monkeypatch.undo()
    monkeypatch.setenv("MTPLX_GLM5_KDA_CONV_FUSED", "0")
    off = _forward(attn, x)
    mx.eval(off)
    assert (reference == off).all().item(), "env=0 output diverged from eager"


def test_shape_gate_fallback_s16(attn, monkeypatch):
    calls = _count_calls(monkeypatch)
    monkeypatch.setenv("MTPLX_GLM5_KDA_CONV_FUSED", "1")

    mx.random.seed(8)
    x16 = (mx.random.normal((1, 16, 512)) * 0.5).astype(mx.bfloat16)
    mx.eval(x16)
    out16 = _forward(attn, x16)
    mx.eval(out16)
    assert calls["n"] == 0, "S=16 must not reach the fused dispatch"


def _eager_chain(mixed, conv_state, conv_w):
    conv = nn.Conv1d(
        in_channels=_C,
        out_channels=_C,
        kernel_size=4,
        groups=_C,
        padding=0,
        bias=False,
    )
    conv.weight = conv_w
    conv_input = mx.concatenate([conv_state, mixed], axis=1)
    conv_out = nn.silu(conv(conv_input))
    q, k, v = mx.split(conv_out, [_QK, 2 * _QK], axis=-1)
    q = q.reshape(1, -1, 64, 128)
    k = k.reshape(1, -1, 64, 128)
    v = v.reshape(1, -1, 64, 128)
    eps = 1e-6
    q = (
        q.astype(mx.float32)
        * mx.rsqrt((q.astype(mx.float32) ** 2).sum(-1, keepdims=True) + eps)
        * (128**-0.5)
    ).astype(mixed.dtype)
    k = (
        k.astype(mx.float32)
        * mx.rsqrt((k.astype(mx.float32) ** 2).sum(-1, keepdims=True) + eps)
    ).astype(mixed.dtype)
    return q, k, v, conv_input[:, -3:, :]


@pytest.mark.parametrize("S", [2, 3, 4, 5, 6])
def test_kernel_parity_sweep(attn, S):
    mx.random.seed(20 + S)
    mixed = (mx.random.normal((1, S, _C)) * 0.5).astype(mx.bfloat16)
    state = (mx.random.normal((1, 3, _C)) * 0.5).astype(mx.bfloat16)
    w = (mx.random.normal((_C, 4, 1)) * 0.05).astype(mx.bfloat16)
    mx.eval(mixed, state, w)

    fused = kda_conv_norm.glm5_kda_conv_norm(mixed, state, w)
    assert fused is not None
    fq, fk, fv, fns = fused
    eq, ek, ev, ens = _eager_chain(mixed, state, w)
    mx.eval(fq, fk, fv, fns, eq, ek, ev, ens)

    # Bit-identical modulo a rare 1-ulp l2norm-sum-order element; allow a
    # small mismatch budget of 1 bf16 ulp at |x|~1.
    for name, f, e in (("q", fq, eq), ("k", fk, ek), ("v", fv, ev)):
        d = (f.reshape(e.shape).astype(mx.float32) - e.astype(mx.float32)).abs()
        mx.eval(d)
        mism = int((d > 0).sum().item())
        maxd = float(d.max().item())
        assert mism <= 2 and maxd <= 4e-3, f"S={S} {name}: {mism} mism, max {maxd}"
    assert (fns == ens).all().item()


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

    monkeypatch.setenv("MTPLX_GLM5_KDA_CONV_FUSED", "0")
    out_e = _forward(attn, x)
    ne, ke = _graph_kernels([out_e])

    monkeypatch.setenv("MTPLX_GLM5_KDA_CONV_FUSED", "1")
    out_f = _forward(attn, x)
    nf, kf = _graph_kernels([out_f])

    print(f"\nKDA conv census: eager nodes={ne} kernels={ke} | "
          f"fused nodes={nf} kernels={kf} | delta={ke - kf}")
    assert kf < ke, "fused path did not reduce kernel count"
