# Copyright (c) 2026 Apple Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fused HyperConnection collapse+norm for glm5_next verify blocks (T1).

One ``mx.fast.metal_kernel`` dispatch replaces the eager pre-branch chain
``astype f32 -> fast.rms_norm -> tokenwise mix (4 slices x contiguous+matmul +
concat) -> hc_sinkhorn_collapse -> branch RMSNorm`` (~12 dispatches per HC
call, 2 calls per decoder layer). This is the ``exact_hc_normalized_norm``
kernel from ``mlx_vlm.models.fast_ops`` re-dispatched with the mix-weight
template ``W`` relaxed to ``float``: glm5 keeps ``.attn_hc.``/``.ffn_hc.``
tensors in FP32 (``glm5_next_cast_predicate``), which the upstream gate
rejects. The kernel source is already ``W``-generic -- scalar ``source_w``
loads accumulate into float registers -- so the float instantiation changes
nothing beyond f32 accumulation order.

The kernel itself hardcodes the GLM-5.3 HC geometry (HC=4, D=4096: one vec4
per thread in a 1024-thread group), so the gate here is deliberately the same
exact-shape gate upstream uses: B=1, 1 < S <= DECODE_BLOCK_SIZE, bf16/f16
activations, fn (24, 16384). Anything else returns None and the caller falls
back to the eager chain.
"""

import logging
import os
from functools import lru_cache

import mlx.core as mx

logger = logging.getLogger(__name__)

try:
    from mlx_vlm.models.fast_ops import _hc_normalized_norm_kernel
except Exception:  # pragma: no cover - mlx_vlm without the kernel factory
    _hc_normalized_norm_kernel = None

_DECODE_BLOCK_SIZE = 8


def _hc_fused_enabled() -> bool:
    """Opt-out env read: unset resolves ON, falsy disables."""
    return str(os.environ.get("MTPLX_GLM5_HC_FUSED", "1")).strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _dispatch(connection, norm, x: mx.array):
    batch, length, hc_mult, width = x.shape
    kernel = _hc_normalized_norm_kernel(
        x.dtype,
        connection.fn.dtype,
        hc_mult,
        width,
        connection.sinkhorn_iters,
        connection.hc_eps,
        norm.eps,
    )
    return kernel(
        inputs=[
            mx.contiguous(x),
            connection.fn,
            connection.scale,
            connection.base,
            norm.weight,
        ],
        template=[
            ("T", x.dtype),
            ("W", connection.fn.dtype),
            ("HC", int(hc_mult)),
            ("D", int(width)),
            ("ITERS", int(connection.sinkhorn_iters)),
            ("HC_EPS_INT", round(connection.hc_eps / 1e-9)),
            ("NORM_EPS_INT", round(norm.eps / 1e-9)),
        ],
        grid=(batch * length * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[
            (batch, length, width),
            (batch, length, hc_mult),
            (batch, length, hc_mult, hc_mult),
        ],
        output_dtypes=[x.dtype, mx.float32, mx.float32],
    )


@lru_cache(maxsize=1)
def device_supports_hc_normalized_norm() -> bool:
    """One-shot dispatch probe: 1024-thread pipelines can exceed the register
    cap on some GPUs (same failure class as issue #400), so the real kernel is
    dispatched once on dummy inputs and the verdict cached."""
    try:
        import mlx.nn as nn

        from mlx_vlm.models.deepseek_v4.hyper_connection import HyperConnection

        cfg = type(
            "cfg",
            (),
            {
                "hc_mult": 4,
                "hc_sinkhorn_iters": 20,
                "hc_eps": 1e-6,
                "rms_norm_eps": 1e-5,
                "hidden_size": 4096,
            },
        )()
        hc = HyperConnection(cfg)
        norm = nn.RMSNorm(4096, eps=1e-5)
        norm.weight = norm.weight.astype(mx.bfloat16)
        x = mx.zeros((1, 2, 4, 4096), dtype=mx.bfloat16)
        mx.eval(*_dispatch(hc, norm, x))
        return True
    except Exception as exc:
        print(
            "[mtplx] fused HC normalized-norm disabled: this GPU cannot "
            f"dispatch its 1024-thread pipeline; using the eager chain ({exc})",
            flush=True,
        )
        return False


def glm5_hc_normalized_norm(connection, norm, x: mx.array):
    """Return ``(normalized, post, comb)`` for the fused HC pre-branch chain,
    or None when the dispatch does not apply (caller falls back to eager).

    Same contract as ``exact_hc_normalized_norm`` plus the f32 mix-weight
    instantiation: ``normalized`` is the collapse + branch RMSNorm output the
    attention/MLP consumes; ``post``/``comb`` feed ``hc_expand`` downstream.
    """
    if not _hc_fused_enabled() or _hc_normalized_norm_kernel is None:
        return None
    weight = connection.fn
    if (
        not mx.metal.is_available()
        or mx.default_device() != mx.gpu
        or x.ndim != 4
        or x.shape[0] != 1
        or x.shape[1] <= 1
        or x.shape[1] > _DECODE_BLOCK_SIZE
        or x.dtype not in (mx.bfloat16, mx.float16)
        or x.shape[2:] != (4, 4096)
        or weight.ndim != 2
        or weight.shape != (24, 4 * 4096)
        or weight.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or connection.scale.dtype != mx.float32
        or connection.base.dtype != mx.float32
        or norm.weight.dtype != x.dtype
        or connection.norm_eps != norm.eps
        or not device_supports_hc_normalized_norm()
    ):
        return None
    return _dispatch(connection, norm, x)
