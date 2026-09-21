# Copyright (c) 2026 Apple Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fused MoE router post-matmul chain for glm5_next verify blocks (T5).

``Glm5NextMoEGate.__call__`` runs ``x.astype(f32) @ weight.T`` and then
``group_expert_select`` -- with ``n_group == 1`` the whole group-mask
block is dead code and the remaining chain is ~7 dispatches on
(B, S, 288) f32:

    sigmoid -> +e_score_correction_bias -> neg -> argpartition(-s, kth)
    -> slice[:k] -> take_along_axis(orig sigmoid) -> sum -> div
    -> * routed_scaling_factor

This module folds that tail into ONE ``mx.fast.metal_kernel`` dispatch:
one threadgroup per token row, E threads, each computing its own
sigmoid/corrected score into threadgroup memory, then an O(E) in-thread
rank scan, then the shared top-K denominator and scale. The matmul stays
eager: its steel-GEMM f32 accumulation order cannot be reproduced
in-kernel, and identical *logits* are what make the selection indices
bit-identical downstream.

Bit-exactness vs the eager chain (0 index mismatches, 0 score
mismatches, incl. forced exact ties):

  * ``mx.argpartition`` on GPU is a full stable merge sort
    (``ArgPartition::eval_gpu`` -> ``gpu_merge_sort``; 288 elements is a
    single-block sort whose strict ``b < a`` merge keeps the left side
    on ties). So ``inds`` = the K largest corrected scores in
    *descending* order, ties broken by *ascending* expert index. The
    kernel reproduces that exactly with
    ``rank(e) = #{j: corr_j > corr_e} + #{j < e: corr_j == corr_e}``
    (NaN handled to sort last, matching LessThan's NaN rule -- never
    hit with finite weights).
  * ``mx.sigmoid`` f32 == abs-form with ``metal::precise::exp`` (plain
    ``metal::exp`` is the fast variant -- ~1 ulp off, see T2/T4 notes).
  * ``denominator`` sums the K gathered *original* (pre-bias) sigmoid
    scores in rank order -- the same left-to-right order the
    single-thread row_reduce_small path uses for an 8-element row
    (verified bitwise).
  * ``scores / denom`` then ``* scale`` keep the two eager roundings.

Gate: n_group == 1, B=1, 1 <= S <= 8, f32 logits, f32 bias,
E <= 1024, 1 <= K <= E, GPU device, cached dispatch probe. Anything
else returns None and the caller runs ``group_expert_select``.
"""

import logging
import os
from functools import lru_cache

import mlx.core as mx

logger = logging.getLogger(__name__)

_MAX_S = 8
_MAX_E = 1024  # one thread per expert, single threadgroup

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""


def _src(scale: float) -> str:
    # scale is baked into the source (metal_kernel template args are
    # int/bool/Dtype only); repr round-trips the f32 the eager
    # ``* routed_scaling_factor`` multiply consumes.
    return f"""
    constexpr float SCALE = {float(scale)!r}f;

    threadgroup float tg_sig[E];
    threadgroup float tg_corr[E];
    threadgroup float tg_sel[K];
    threadgroup float tg_den[1];

    const uint e = thread_position_in_threadgroup.x;
    const uint row = threadgroup_position_in_grid.y;
    if (e >= (uint)E) return;

    const float lg = logits[row * E + e];
    // MLX Sigmoid{{}} abs-form, f32 precise exp (== mx.sigmoid bitwise).
    const float ay = 1.0f / (1.0f + metal::precise::exp(metal::abs(lg)));
    const float s = (lg < 0.0f) ? ay : 1.0f - ay;
    tg_sig[e] = s;
    const float c = s + bias[e];
    tg_corr[e] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stable descending sort position: count strictly-greater scores
    // plus equal scores at lower indices (replicates gpu_merge_sort's
    // stable merge on -scores). NaNs order last, stable among themselves.
    uint rank = 0;
    const bool cn = metal::isnan(c);
    for (int j = 0; j < E; ++j) {{
        const float o = tg_corr[j];
        bool before;
        if (cn) {{
            before = metal::isnan(o) ? (j < (int)e) : true;
        }} else if (metal::isnan(o)) {{
            before = false;
        }} else {{
            before = (o > c) || (o == c && j < (int)e);
        }}
        rank += before ? 1u : 0u;
    }}
    if (rank < (uint)K) {{
        inds[row * K + rank] = e;
        tg_sel[rank] = s;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (e == 0) {{
        float d = 0.0f;
        for (int r = 0; r < K; ++r) d += tg_sel[r];
        tg_den[0] = d;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (e < (uint)K) {{
        float v = tg_sel[e];
        if (NORM) v = v / tg_den[0];
        scores[row * K + e] = v * SCALE;
    }}
"""


@lru_cache(maxsize=8)
def _kernel(num_experts, top_k, norm_topk_prob, scale):
    return mx.fast.metal_kernel(
        name="mtplx_glm5_moe_router_"
        f"{num_experts}_{top_k}_{int(bool(norm_topk_prob))}_"
        f"{abs(hash(float(scale))) % 100000}",
        input_names=["logits", "bias"],
        output_names=["inds", "scores"],
        header=_HEADER,
        source=_src(scale),
    )


def _dispatch(logits, bias, top_k, norm_topk_prob, scale):
    rows = logits.size // logits.shape[-1]
    inds, scores = _kernel(
        logits.shape[-1], int(top_k), bool(norm_topk_prob), float(scale)
    )(
        inputs=[logits.reshape(-1), bias.reshape(-1)],
        template=[
            ("E", int(logits.shape[-1])),
            ("K", int(top_k)),
            ("NORM", bool(norm_topk_prob and top_k > 1)),
        ],
        grid=(logits.shape[-1], rows, 1),
        threadgroup=(logits.shape[-1], 1, 1),
        output_shapes=[(rows, int(top_k)), (rows, int(top_k))],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return inds.reshape(*logits.shape[:-1], int(top_k)), scores.reshape(
        *logits.shape[:-1], int(top_k)
    )


@lru_cache(maxsize=1)
def _device_supports_moe_router() -> bool:
    try:
        logits = mx.zeros((1, 2, 288), dtype=mx.float32)
        bias = mx.zeros((288,), dtype=mx.float32)
        mx.eval(*_dispatch(logits, bias, 8, True, 2.5))
        return True
    except Exception as exc:
        logger.warning(
            "[mtplx] fused GLM5 MoE router disabled: kernel dispatch failed; "
            "using the eager chain (%s)",
            exc,
        )
        return False


def _moe_router_fused_enabled() -> bool:
    """Opt-out env read: unset resolves ON, falsy disables."""
    return str(
        os.environ.get("MTPLX_GLM5_MOE_ROUTER_FUSED", "1")
    ).strip().lower() not in ("0", "false", "no", "off")


def glm5_moe_router_topk(
    logits, e_score_correction_bias, top_k, n_group, norm_topk_prob,
    routed_scaling_factor,
):
    """Fused post-matmul ``group_expert_select`` for one MoE gate, or None
    to fall back to the eager chain.

    logits:                 (B, S, E) f32 gate-matmul output
    e_score_correction_bias:(E,) f32
    Returns (inds uint32, scores f32), both (B, S, top_k) -- same as
    ``group_expert_select`` with n_group == 1.
    """
    if not _moe_router_fused_enabled():
        return None
    if n_group != 1 or top_k is None or not (1 <= int(top_k)):
        return None
    if (
        logits.ndim != 3
        or logits.shape[0] != 1
        or not (1 <= int(logits.shape[1]) <= _MAX_S)
        or logits.shape[-1] > _MAX_E
        or int(top_k) > logits.shape[-1]
    ):
        return None
    if logits.dtype != mx.float32 or e_score_correction_bias.dtype != mx.float32:
        return None
    if e_score_correction_bias.shape != (logits.shape[-1],):
        return None
    if mx.default_device() != mx.Device(mx.gpu):
        return None
    if not _device_supports_moe_router():
        return None
    return _dispatch(
        mx.contiguous(logits),
        e_score_correction_bias,
        top_k,
        norm_topk_prob,
        routed_scaling_factor,
    )
