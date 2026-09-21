# Copyright (c) 2026 Apple Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fused KDA gated output norm (Glm5NextRMSNormGated) for verify blocks (T4).

``Glm5NextRMSNormGated.__call__`` is the model's only hand-rolled norm --
per KDA layer it dispatches ~11 eager ops on (1, S, 64, 128):

    astype f32 -> mul(x*x) -> mean -> add eps -> rsqrt -> mul
    -> w.astype f32 -> mul -> gate.astype -> sigmoid -> mul -> astype T

This module replaces all of them with one ``mx.fast.metal_kernel``
dispatch (one simdgroup per 128-dim head row).

Bit-exactness vs the eager chain, verified on synthetic bf16/f16 inputs
(0 mismatched outputs, plus layer-level stash + output equality):

  * the (x*x) products must round to f32 before the sum -- they are kept
    in a ``float4`` so each add consumes a vector *extract*, which the
    compiler cannot contract back into an fma (the ``part += v*v`` form
    DOES contract and diverges ~1 ulp);
  * the per-lane partial is 4 *consecutive* elements summed sequentially
    (thread_reduce's N_READS=4 block layout in row_reduce_simple), then
    ``simd_sum`` -- which on Apple silicon is an adjacent-pairwise tree,
    matching the library reduce bitwise;
  * ``mean`` = sum/128 is an exact power-of-two scale;
  * ``mx.rsqrt`` f32 == ``metal::precise::rsqrt`` (plain ``metal::rsqrt``
    is the fast variant and fails by ~1 ulp);
  * ``mx.sigmoid`` f32 == MLX ``Sigmoid{}`` abs-form with
    ``metal::precise::exp`` (plain ``metal::exp`` is the fast variant and
    fails by ~1 ulp -- same lesson as T2's bf16 silu, transposed to f32);
  * the trailing ``w * (x*rs) * sigmoid`` f32 multiply chain keeps the
    eager rounding order (mul-only chains do not contract).

Gate: B=1, 1 <= S <= 8, exact (num_heads=64, head_dim=128) geometry,
x/gate bf16 or f16, weight bf16/f16/f32, GPU device, and a cached
pipeline probe. Anything else returns None and the caller runs the
module's eager path.
"""

import logging
import os
from functools import lru_cache

import mlx.core as mx

logger = logging.getLogger(__name__)

_NUM_HEADS = 64
_HEAD_DIM = 128
_MAX_S = 8
_TG_ROWS = 8  # one simdgroup per head row, 8 simdgroups per threadgroup

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""


def _src(eps: float) -> str:
    # eps is baked into the source (metal_kernel template args are
    # int/bool/Dtype only); repr round-trips the same f32 the eager
    # ``var + self.eps`` add consumes.
    return f"""
    constexpr int HD = {_HEAD_DIM};
    constexpr float EPS = {float(eps)!r}f;

    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint row = threadgroup_position_in_grid.y * {_TG_ROWS} + sg;
    if (row >= (uint)ROWS) return;

    const int base = row * HD + lane * 4;

    float4 xv;
    for (int i = 0; i < 4; ++i) xv[i] = (float)x[base + i];
    // f32 products must round before summing: sq is a vector fmul, and
    // extract-adds cannot be contracted into fmas.
    const float4 sq = xv * xv;
    const float part = ((sq[0] + sq[1]) + sq[2]) + sq[3];
    const float ssum = simd_sum(part);
    const float rs = metal::precise::rsqrt(ssum / 128.0f + EPS);

    float4 gv;
    for (int i = 0; i < 4; ++i) gv[i] = (float)g[base + i];
    for (int i = 0; i < 4; ++i) {{
        const float nrm = xv[i] * rs;
        const float sc = (float)w[lane * 4 + i] * nrm;
        const float ay =
            1.0f / (1.0f + metal::precise::exp(metal::abs(gv[i])));
        const float s = (gv[i] < 0.0f) ? ay : 1.0f - ay;
        out[base + i] = (T)(sc * s);
    }}
"""


@lru_cache(maxsize=8)
def _kernel(dtype_t, dtype_w, eps):
    return mx.fast.metal_kernel(
        name="mtplx_glm5_kda_onorm_"
        f"{str(dtype_t).split('.')[-1]}_{str(dtype_w).split('.')[-1]}_"
        f"{abs(hash(float(eps))) % 100000}",
        input_names=["x", "g", "w"],
        output_names=["out"],
        header=_HEADER,
        source=_src(eps),
    )


def _dispatch(x, g, w, eps, rows):
    ntg = (rows + _TG_ROWS - 1) // _TG_ROWS
    (out,) = _kernel(x.dtype, w.dtype, float(eps))(
        inputs=[x.reshape(-1), g.reshape(-1), w.reshape(-1)],
        template=[("T", x.dtype), ("WT", w.dtype), ("ROWS", int(rows))],
        grid=(32 * _TG_ROWS, ntg, 1),
        threadgroup=(32 * _TG_ROWS, 1, 1),
        output_shapes=[(rows * _HEAD_DIM,)],
        output_dtypes=[x.dtype],
    )
    return out.reshape(x.shape)


@lru_cache(maxsize=1)
def _device_supports_kda_onorm() -> bool:
    try:
        x = mx.zeros((1, 2, _NUM_HEADS, _HEAD_DIM), dtype=mx.bfloat16)
        g = mx.zeros((1, 2, _NUM_HEADS, _HEAD_DIM), dtype=mx.bfloat16)
        w = mx.zeros((_HEAD_DIM,), dtype=mx.bfloat16)
        mx.eval(_dispatch(x, g, w, 1e-5, 2 * _NUM_HEADS))
        return True
    except Exception as exc:
        logger.warning(
            "[mtplx] fused GLM5 KDA o_norm disabled: kernel dispatch failed; "
            "using the eager chain (%s)",
            exc,
        )
        return False


def _kda_onorm_fused_enabled() -> bool:
    """Opt-out env read: unset resolves ON, falsy disables."""
    return str(
        os.environ.get("MTPLX_GLM5_KDA_ONORM_FUSED", "1")
    ).strip().lower() not in ("0", "false", "no", "off")


def glm5_kda_o_norm(x, gate, weight, eps):
    """Fused Glm5NextRMSNormGated for one KDA layer, or None to fall back.

    x:      (B, S, 64, 128) recurrent output (bf16/f16)
    gate:   (B, S, 64, 128) g_b_proj output (same dtype)
    weight: (128,) norm weight
    Returns the gated-normed tensor in x's dtype/shape.
    """
    if not _kda_onorm_fused_enabled():
        return None
    if (
        x.ndim != 4
        or x.shape[0] != 1
        or not (1 <= int(x.shape[1]) <= _MAX_S)
        or x.shape[2:] != (_NUM_HEADS, _HEAD_DIM)
    ):
        return None
    if x.dtype not in (mx.bfloat16, mx.float16):
        return None
    if gate.shape != x.shape or gate.dtype != x.dtype:
        return None
    if weight.shape != (_HEAD_DIM,) or weight.dtype not in (
        mx.bfloat16,
        mx.float16,
        mx.float32,
    ):
        return None
    if mx.default_device() != mx.Device(mx.gpu):
        return None
    if not _device_supports_kda_onorm():
        return None
    rows = int(x.shape[1]) * _NUM_HEADS
    return _dispatch(mx.contiguous(x), mx.contiguous(gate), weight, eps, rows)
