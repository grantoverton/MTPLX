"""Regression tests for the sampled draft chain's device-side distribution.

The sampled chain (``_sampled_chain_eligible`` in ``generate_mtpk``) replaces
the per-step host softmax + ``.item()`` sync with a lazy-token chain: each
level draws from ``_device_draft_q_arrays`` on-device and reports that exact
q as the proposal distribution for the accept/residual law.  These tests pin
that the device helper produces the same distribution the serial host lane
would report — the chain is only exact if the reported q is the distribution
actually drawn from.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlx.core as mx

from mtplx.fast_sampling import (
    SamplerConfig,
    sparse_distribution_from_mlx_logits,
)
from mtplx.generation import _device_draft_q_arrays


def _reference_dict(logits: mx.array, config: SamplerConfig) -> dict[int, float]:
    dist = sparse_distribution_from_mlx_logits(logits, config)
    assert dist is not None
    return {int(t): float(p) for t, p in zip(dist.token_ids, dist.probs)}


def _chain_dict(row: mx.array, *, temperature: float, top_k: int, top_p: float) -> dict[int, float]:
    top_idx, q = _device_draft_q_arrays(
        row, temperature=temperature, top_k=top_k, top_p=top_p
    )
    mx.eval(top_idx, q)
    ids = np.asarray(top_idx)
    probs = np.asarray(q)
    return {int(t): float(p) for t, p in zip(ids, probs) if float(p) > 0.0}


@pytest.mark.parametrize("top_p", [1.0, 0.9])
@pytest.mark.parametrize("top_k", [20, 1024])
def test_device_q_matches_serial_distribution(top_k: int, top_p: float):
    rng = np.random.default_rng(0)
    logits = mx.array(rng.normal(size=32000).astype(np.float32))
    config = SamplerConfig(temperature=1.0, top_k=top_k, top_p=top_p)
    ref = _reference_dict(logits, config)
    got = _chain_dict(
        logits, temperature=1.0, top_k=min(top_k, 32000), top_p=top_p
    )
    # Same support (modulo zero-prob padding in the chain output).
    assert set(got) == set(ref)
    for tok, p in ref.items():
        assert got[tok] == pytest.approx(p, rel=2e-3, abs=1e-6)


def test_device_q_temperature_scaling():
    logits = mx.array(np.linspace(-3, 3, 512).astype(np.float32))
    hot = _chain_dict(logits, temperature=2.0, top_k=64, top_p=1.0)
    cold = _chain_dict(logits, temperature=0.5, top_k=64, top_p=1.0)
    top_tok = int(mx.argmax(logits))
    # Lower temperature concentrates mass on the argmax.
    assert cold[top_tok] > hot[top_tok]


def test_device_q_probs_normalize():
    rng = np.random.default_rng(1)
    logits = mx.array(rng.normal(size=8192).astype(np.float32))
    _, q = _device_draft_q_arrays(logits, temperature=1.0, top_k=256, top_p=0.95)
    mx.eval(q)
    assert float(mx.sum(q)) == pytest.approx(1.0, abs=1e-5)
