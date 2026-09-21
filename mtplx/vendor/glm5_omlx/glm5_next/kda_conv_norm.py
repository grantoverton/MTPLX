# Copyright (c) 2026 Apple Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fused KDA conv tape + q/k l2norm for glm5_next verify blocks (T2).

One ``mx.fast.metal_kernel`` dispatch replaces the eager chain between the
fused in-proj matmul and the forget-gate matmul in
``Glm5NextLinearAttention.__call__``:

    concat conv_state -> slice/contiguous roll -> conv1d -> silu(2 ops)
    -> split -> l2norm(q) (~8 ops) -> l2norm(k) (~7 ops)

roughly 20+ eager dispatches, once per KDA layer per forward. Retemplated
from ``mtplx/kernels/gdn_conv_norm.py::_SRC_ROWS`` (qwen4_exp C=10240,
QK=2048, v 6144) to GLM-5.3 geometry: conv width 24576 = q 8192 | k 8192 |
v 8192, 64 heads x 128, state (3, 24576).

Unlike the qwen4 source, the conv->silu chain replicates the eager dtype
contract exactly: conv1d rounds its f32 accumulation to T, and each silu op
(sigmoid, multiply) rounds to T again before the l2norm reads the value
through an f32 upcast. The only remaining non-exact link vs eager is the
per-head sum order (4-simdgroup tree vs the library reduce kernel), which
lands inside one f32 sum and rounds back out to at most ~1 ulp on T.

Gate: B=1, 1 < S <= 8, bf16/f16, the exact GLM conv geometry above, GPU
with a 1024-thread-capable pipeline (probed once, issue-#400 pattern);
everything else returns None and the caller runs the stock chain.
"""

import logging
import os
from functools import lru_cache

import mlx.core as mx

logger = logging.getLogger(__name__)

_C = 24576
_QK = 8192
_STATE_ROWS = 3
_MAX_S = 8

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

_SRC = """
    constexpr int C = 24576;                  // conv channels = 3 * qkv
    constexpr int QK = 8192;                  // q width == k width == v width
    constexpr float INV_SCALE = 0.08838834764831845f;   // 128^-0.5

    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint c = threadgroup_position_in_grid.x * 1024 + tid;
    if (c >= (uint)C) return;

    threadgroup float tg_vals[1024];
    threadgroup float tg_partial[32];

    const float w0 = (float)cw[c * 4 + 0];
    const float w1 = (float)cw[c * 4 + 1];
    const float w2 = (float)cw[c * 4 + 2];
    const float w3 = (float)cw[c * 4 + 3];

    // stream(t): t<3 -> conv state row t, else xnew row t-3
    #define STREAM(t) ((t) < 3 ? (float)state[(t) * C + c] : (float)xnew[((t) - 3) * C + c])

    const bool is_v = (c >= (uint)(2 * QK));

    for (int s = 0; s < S; ++s) {
        const float acc = w0 * STREAM(s) + w1 * STREAM(s + 1)
                        + w2 * STREAM(s + 2) + w3 * STREAM(s + 3);
        // Eager dtype contract, verified bit-identical on 2^20 points:
        // conv1d rounds its f32 accumulation to T; nn.silu is x*sigmoid(x)
        // where MLX's Sigmoid{} is the abs-form evaluated in T arithmetic
        // (metal::exp on T -- the fast exp for bf16), then a T multiply.
        const T conv_t = (T)acc;
        const T y = (T)(1) / ((T)(1) + metal::exp(metal::abs(conv_t)));
        const T sig = (conv_t < (T)(0)) ? y : (T)((T)(1) - y);
        const T sv_t = conv_t * sig;
        const float sv = (float)sv_t;
        if (is_v) {
            v_out[s * (C - 2 * QK) + (c - 2 * QK)] = (T)sv;
            continue;
        }
        // q/k: per-head l2norm. This TG holds 8 aligned heads of 128
        // channels; 4 consecutive simdgroups own one head.
        tg_vals[tid] = sv;
        float part = simd_sum(sv * sv);
        if (lane == 0) tg_partial[sg] = part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint head_sg0 = (sg / 4) * 4;
        const float ssum = tg_partial[head_sg0] + tg_partial[head_sg0 + 1]
                         + tg_partial[head_sg0 + 2] + tg_partial[head_sg0 + 3];
        const float inv = metal::rsqrt(ssum + 1e-6f);
        const float normed = tg_vals[tid] * inv;
        if (c < (uint)QK) {
            q_out[s * QK + c] = (T)(normed * INV_SCALE);
        } else {
            k_out[s * QK + (c - QK)] = (T)normed;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // rolled conv state: last 3 rows of the stream
    state_out[0 * C + c] = (T)STREAM(S + 0);
    state_out[1 * C + c] = (T)STREAM(S + 1);
    state_out[2 * C + c] = (T)STREAM(S + 2);
    #undef STREAM
"""


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="mtplx_glm5_kda_conv_norm_rows",
        input_names=["xnew", "state", "cw"],
        output_names=["q_out", "k_out", "v_out", "state_out"],
        header=_HEADER,
        source=_SRC,
    )


def _dispatch(mixed: mx.array, conv_state: mx.array, conv_w: mx.array):
    s_rows = int(mixed.shape[1])
    cw = conv_w.reshape(_C, 4)
    q, k, v, ns = _kernel()(
        inputs=[
            mixed.reshape(-1),
            conv_state.reshape(-1),
            cw.reshape(-1),
        ],
        template=[("T", mixed.dtype), ("S", s_rows)],
        grid=(_C, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[
            (s_rows, _QK),
            (s_rows, _QK),
            (s_rows, _QK),
            (_STATE_ROWS, _C),
        ],
        output_dtypes=[mixed.dtype] * 4,
    )
    return q, k, v, ns


@lru_cache(maxsize=1)
def _device_supports_kda_conv_norm() -> bool:
    try:
        rows = mx.zeros((1, 2, _C), dtype=mx.bfloat16)
        state = mx.zeros((1, _STATE_ROWS, _C), dtype=mx.bfloat16)
        cw = mx.zeros((_C, 4, 1), dtype=mx.bfloat16)
        mx.eval(*_dispatch(rows, state, cw))
        return True
    except Exception as exc:
        logger.warning(
            "[mtplx] fused GLM5 KDA conv+norm disabled: this GPU cannot "
            "dispatch its 1024-thread pipeline; using the eager chain (%s)",
            exc,
        )
        return False


def _kda_conv_fused_enabled() -> bool:
    """Opt-out env read: unset resolves ON, falsy disables."""
    return str(
        os.environ.get("MTPLX_GLM5_KDA_CONV_FUSED", "1")
    ).strip().lower() not in ("0", "false", "no", "off")


def glm5_kda_conv_norm(mixed, conv_state, conv_w):
    """Fused conv tape for one KDA layer at verify width, or None to fall
    back to the eager chain.

    mixed:      (B, S, 24576) post in-proj stream (already mask-zeroed)
    conv_state: (B, 3, 24576) rolling depthwise-conv state
    conv_w:     (24576, 4, 1) depthwise kernel
    Returns (q, k, v, new_state): q (S, 8192) l2normed+128^-0.5 scaled,
    k (S, 8192) l2normed, v (S, 8192) silu'd, new_state (B, 3, 24576).
    """
    if not _kda_conv_fused_enabled():
        return None
    if (
        mixed.ndim != 3
        or mixed.shape[0] != 1
        or not (1 < int(mixed.shape[1]) <= _MAX_S)
        or mixed.shape[2] != _C
    ):
        return None
    if conv_state.shape != (1, _STATE_ROWS, _C):
        return None
    if conv_w.ndim != 3 or conv_w.shape[0] != _C or conv_w.shape[1] != 4:
        return None
    if mixed.dtype not in (mx.bfloat16, mx.float16):
        return None
    if conv_state.dtype != mixed.dtype or conv_w.dtype != mixed.dtype:
        return None
    if mx.default_device() != mx.Device(mx.gpu):
        return None
    if not _device_supports_kda_conv_norm():
        return None
    q, k, v, ns = _dispatch(mixed, conv_state, conv_w)
    return q, k, v, ns.reshape(1, _STATE_ROWS, _C)
