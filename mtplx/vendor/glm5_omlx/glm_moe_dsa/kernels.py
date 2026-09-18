# SPDX-License-Identifier: Apache-2.0
"""Fast-kernel dispatch for the vendored GLM MoE DSA stack (MTPLX port).

Uses the locally built ``_ext`` extension (py3.13 build of oMLX's
``custom_kernels/glm_moe_dsa``) via the vendored ``fast.py`` module, which
itself falls back to ``mx.fast`` when the native ext is absent.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from . import fast as _native_fast  # module, or None-equivalent shim


class _FastDispatch:
    """Expose the oMLX ``fast`` module API plus the ``has``/``missing``
    helpers the vendored patch code calls on ``kernels.fast``."""

    def __getattr__(self, name: str) -> Any:
        return getattr(_native_fast, name)

    def __dir__(self) -> list[str]:
        return dir(_native_fast)

    def has_symbol(self, name: str) -> bool:
        return _native_fast.has_symbol(name)

    def has(self, name: str) -> bool:
        return self.has_symbol(name) or hasattr(mx.fast, name)

    def missing(self, required: tuple[str, ...]) -> list[str]:
        return [name for name in required if not self.has(name)]

    def native_available(self) -> bool:
        return _native_fast.is_native_available()

    def native_import_error(self):
        return getattr(_native_fast, "_IMPORT_ERROR", None)


fast = _FastDispatch()

__all__ = ["fast"]
