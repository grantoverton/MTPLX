"""Vendored subset of omlx.patches.mlx_lm_mtp.cache_rollback.

PoolingCache consults these flags to decide whether to stash an undo-log row
for a verify block. Phase-1 keeps them permanently disarmed; Phase-2 ports
the full arming/rollback machinery.
"""


def _is_undo_armed() -> bool:
    return False


def _is_decode_consistent_armed() -> bool:
    return False


def apply() -> bool:
    return False
