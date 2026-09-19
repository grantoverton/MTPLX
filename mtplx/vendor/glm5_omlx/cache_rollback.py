"""Vendored subset of omlx.patches.mlx_lm_mtp.cache_rollback.

PoolingCache consults these flags to decide whether to stash an undo-log row
for a verify block. Phase-2 arms them around MTP-managed forwards: the GLM
``mtp_forward`` seam wraps its draft-layer call in :func:`pool_undo_arm`, so
draft steps (L == 1) and verify/history windows (L <= 8) both record the
state ``trim`` needs to roll back rejected positions.
"""

from contextlib import contextmanager
from contextvars import ContextVar

_undo_armed: ContextVar[bool] = ContextVar("glm5_pool_undo_armed", default=False)
_decode_consistent_armed: ContextVar[bool] = ContextVar(
    "glm5_pool_undo_decode_consistent", default=False
)


def _is_undo_armed() -> bool:
    return _undo_armed.get()


def _is_decode_consistent_armed() -> bool:
    return _decode_consistent_armed.get()


@contextmanager
def pool_undo_arm():
    """Arm PoolingCache undo logging for the duration of one MTP forward.

    Both flags are raised together: ``_undo_armed`` covers single-token
    draft steps, ``_decode_consistent_armed`` lets consecutive multi-token
    windows chain their undo stashes (batched verify / history replay).
    """

    undo_token = _undo_armed.set(True)
    chain_token = _decode_consistent_armed.set(True)
    try:
        yield
    finally:
        _decode_consistent_armed.reset(chain_token)
        _undo_armed.reset(undo_token)


def apply() -> bool:
    return False
