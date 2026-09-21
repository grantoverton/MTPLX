"""T5 fused MoE router post-matmul chain for glm5_next (Metal).

``glm5_moe_router_topk`` replaces the ~7-dispatch tail of
``group_expert_select`` at n_group==1 (sigmoid, +bias, neg, argpartition,
slice, take_along, sum, div, scale) with one Metal dispatch per gate call.
Covers: (a) kernel-level bit-exact parity (indices AND scores) across
S=1..8 incl. forced exact score ties, (b) module-level equality through
Glm5NextMoEGate, (c) MTPLX_GLM5_MOE_ROUTER_FUSED=0 is bit-identical,
(d) fallback gates (S=16, n_group>1), (e) census shrinks.
"""

import collections
import io
import re
import types

import mlx.core as mx
import pytest

import mtplx.vendor.glm5_omlx.glm5_next.language as glm5_lang
from mtplx.vendor.glm5_omlx.glm5_next import moe_router
from mtplx.vendor.glm5_omlx.glm5_next.language import Glm5NextMoEGate

_E = 288
_K = 8
_SCALE = 2.5


def _eager_chain(logits, bias, top_k, scale):
    """Uncompiled replica of group_expert_select at n_group==1 (verified
    bit-identical to the @mx.compile'd original)."""
    scores = mx.sigmoid(logits.astype(mx.float32))
    orig = scores
    corr = scores + bias
    inds = mx.argpartition(-corr, kth=top_k - 1, axis=-1)[..., :top_k]
    picked = mx.take_along_axis(orig, inds, axis=-1)
    if top_k > 1:
        picked = picked / picked.sum(axis=-1, keepdims=True)
    return inds, picked * scale


def _gate(**over):
    cfg = types.SimpleNamespace(
        num_experts_per_tok=_K,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=_SCALE,
        n_routed_experts=_E,
        hidden_size=512,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    g = Glm5NextMoEGate(cfg)
    mx.random.seed(3)
    g.weight = (mx.random.normal((_E, cfg.hidden_size)) * 0.02).astype(
        mx.float32
    )
    g.e_score_correction_bias = (
        mx.random.normal((_E,)) * 0.1
    ).astype(mx.float32)
    mx.eval(g.parameters())
    return g


def _count_calls(monkeypatch):
    calls = {"n": 0}
    orig = moe_router.glm5_moe_router_topk

    def counting(*a, **k):
        r = orig(*a, **k)
        if r is not None:
            calls["n"] += 1
        return r

    monkeypatch.setattr(glm5_lang, "glm5_moe_router_topk", counting)
    return calls


@pytest.mark.parametrize("S", [1, 2, 3, 4, 5, 6, 7, 8])
def test_kernel_parity_sweep(S):
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal kernel needs the GPU")
    mx.random.seed(40 + S)
    # Wide logit spreads exercise the selection boundary.
    logits = (
        mx.random.normal((1, S, _E)) * (1.0 + 0.4 * S)
    ).astype(mx.float32)
    bias = (mx.random.normal((_E,)) * 0.25).astype(mx.float32)
    mx.eval(logits, bias)

    fused = moe_router.glm5_moe_router_topk(
        logits, bias, _K, 1, True, _SCALE
    )
    assert fused is not None
    fi, fs = fused
    ei, es = _eager_chain(logits, bias, _K, _SCALE)
    mx.eval(fi, fs, ei, es)
    assert fi.dtype == ei.dtype == mx.uint32
    assert (fi == ei).all().item(), f"S={S}: indices diverged"
    assert (fs == es).all().item(), f"S={S}: scores diverged"


def test_exact_score_ties():
    """Identical corrected scores must resolve to ascending expert index
    (stable merge sort on -scores)."""
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal kernel needs the GPU")
    row = [-2.0] * _E
    # 12-way exact tie at the top: indices 5..16 all get logit 3.0.
    for i in range(5, 17):
        row[i] = 3.0
    logits = mx.array([row, row]).reshape(1, 2, _E).astype(mx.float32)
    bias = mx.zeros((_E,), mx.float32)
    mx.eval(logits, bias)

    fused = moe_router.glm5_moe_router_topk(logits, bias, _K, 1, True, _SCALE)
    ei, es = _eager_chain(logits, bias, _K, _SCALE)
    fi, fs = fused
    mx.eval(fi, fs, ei, es)
    assert (fi == ei).all().item()
    assert fi[0, 0].tolist() == list(range(5, 13)), fi[0, 0].tolist()
    assert (fs == es).all().item()


def test_module_matches_eager(monkeypatch):
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal kernel needs the GPU")
    gate = _gate()
    mx.random.seed(9)
    x = (mx.random.normal((1, 4, 512)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)

    monkeypatch.setenv("MTPLX_GLM5_MOE_ROUTER_FUSED", "0")
    ei, es = gate(x)
    mx.eval(ei, es)

    calls = _count_calls(monkeypatch)
    monkeypatch.setenv("MTPLX_GLM5_MOE_ROUTER_FUSED", "1")
    fi, fs = gate(x)
    mx.eval(fi, fs)

    assert calls["n"] == 1, "fused dispatch did not run — vacuous parity"
    assert (fi == ei).all().item(), "fused indices diverged"
    assert (fs == es).all().item(), "fused scores diverged"


def test_env_zero_is_bit_identical_eager(monkeypatch):
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal kernel needs the GPU")
    gate = _gate()
    mx.random.seed(7)
    x = (mx.random.normal((1, 4, 512)) * 0.5).astype(mx.bfloat16)

    monkeypatch.setattr(
        glm5_lang, "glm5_moe_router_topk", lambda *a, **k: None
    )
    monkeypatch.setenv("MTPLX_GLM5_MOE_ROUTER_FUSED", "1")
    ri, rs = gate(x)
    mx.eval(ri, rs)

    monkeypatch.undo()
    monkeypatch.setenv("MTPLX_GLM5_MOE_ROUTER_FUSED", "0")
    oi, os_ = gate(x)
    mx.eval(oi, os_)
    assert (ri == oi).all().item() and (rs == os_).all().item()


def test_fallback_gates(monkeypatch):
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal kernel needs the GPU")
    calls = _count_calls(monkeypatch)
    monkeypatch.setenv("MTPLX_GLM5_MOE_ROUTER_FUSED", "1")

    # S=16 exceeds the verify-width gate.
    gate = _gate()
    x16 = (mx.random.normal((1, 16, 512)) * 0.5).astype(mx.bfloat16)
    mx.eval(x16)
    i16, s16 = gate(x16)
    mx.eval(i16, s16)
    assert calls["n"] == 0, "S=16 must not reach the fused dispatch"

    # n_group>1 has a live group-mask block — stays eager.
    gate2 = _gate(n_group=2, topk_group=1)
    x4 = (mx.random.normal((1, 4, 512)) * 0.5).astype(mx.bfloat16)
    i4, s4 = gate2(x4)
    mx.eval(i4, s4)
    assert calls["n"] == 0, "n_group=2 must not reach the fused dispatch"


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


def test_fused_launch_census(monkeypatch):
    if mx.default_device().type != mx.DeviceType.gpu:
        pytest.skip("Metal kernel needs the GPU")
    gate = _gate()
    mx.random.seed(10)
    x = (mx.random.normal((1, 4, 512)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)

    monkeypatch.setenv("MTPLX_GLM5_MOE_ROUTER_FUSED", "0")
    ie, se = gate(x)
    ne, ke = _graph_kernels([ie, se])

    monkeypatch.setenv("MTPLX_GLM5_MOE_ROUTER_FUSED", "1")
    if_, sf = gate(x)
    nf, kf = _graph_kernels([if_, sf])

    print(
        f"\nMoE router census: eager nodes={ne} kernels={ke} | "
        f"fused nodes={nf} kernels={kf} | delta={ke - kf}"
    )
    # Eager tail = Sigmoid + CompiledAddNeg + ArgPartition + Slice +
    # GatherAxis + Sum + CompiledDivMul; fused = one CustomKernel.
    # Node-count delta is 5 (slice/view bookkeeping makes it conservative
    # -- argpartition itself is a multi-kernel sort internally).
    assert ke - kf >= 5, f"fused path saved only {ke - kf} kernels"
