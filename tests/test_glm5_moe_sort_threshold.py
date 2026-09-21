"""MTPLX_GLM5_MOE_SORT_MIN_ROUTES override on deepseek_v4 _sort_threshold.

The env override (e.g. =32) must lower the sort threshold for every
bit-width -- including the 4-bit path that otherwise returns a hard-coded
64 -- so the native gather-QMM block kernels + fused weighted-sum epilogue
engage at verify route counts (S=4 -> indices.size=32). Unset must preserve
the existing behavior exactly.
"""

import mlx.core as mx
import pytest

from mtplx.vendor.glm5_omlx.deepseek_v4 import switch_layers


def _qsl(bits: int = 4, group_size: int = 64):
    return switch_layers.QuantizedSwitchLinear(
        input_dims=group_size,
        output_dims=16,
        num_experts=4,
        bias=False,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )


def test_unset_preserves_current_behavior(monkeypatch):
    monkeypatch.delenv("MTPLX_GLM5_MOE_SORT_MIN_ROUTES", raising=False)
    mx.eval(_qsl(bits=4).parameters())
    assert switch_layers._sort_threshold(_qsl(bits=4)) == 64
    # The 2/3-bit g64 affine path keeps its own env-backed threshold.
    assert (
        switch_layers._sort_threshold(_qsl(bits=3, group_size=64))
        == switch_layers._SORT_MIN_ROUTES
    )


def test_override_applies_to_every_bit_width(monkeypatch):
    monkeypatch.setenv("MTPLX_GLM5_MOE_SORT_MIN_ROUTES", "32")
    assert switch_layers._sort_threshold(_qsl(bits=4)) == 32
    assert switch_layers._sort_threshold(_qsl(bits=3, group_size=64)) == 32
    assert switch_layers._sort_threshold(_qsl(bits=8, group_size=64)) == 32


def test_override_values(monkeypatch):
    monkeypatch.setenv("MTPLX_GLM5_MOE_SORT_MIN_ROUTES", "16")
    assert switch_layers._sort_threshold(_qsl(bits=4)) == 16


def test_invalid_value_falls_back(monkeypatch):
    monkeypatch.setenv("MTPLX_GLM5_MOE_SORT_MIN_ROUTES", "bogus")
    assert switch_layers._sort_threshold(_qsl(bits=4)) == 64
