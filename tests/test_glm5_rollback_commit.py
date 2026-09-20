"""Regression tests for the glm5_next verify-rollback / capture-commit fix.

Synthetic — no model load. Covers the corruption path found 2026-09-19:
context-copy block verifies (25-33 tokens) overflowed the PoolingCache
undo stash (L <= 8), so a boundary trim refused, CacheList.is_trimmable()
AND-ed the refusal, and rollback_after_verify skipped the whole KV+pool
pair — leaving foreign verify tokens in trunk KV.

Pinning:

* PoolingCache stashes undo coverage for verify-sized updates (L <= 64).
* can_trim(n) reports n-token rollback feasibility before mutation.
* GLMOffsetCacheList.can_trim AND-s the pair.
* trim_verified_window_to_prefix preflights before mutating any child.
* rollback_after_verify falls back to per-child trims so the KV half can
  never keep rejected tokens.
* commit_verified_window replays the KDA gated-delta recurrence over the
  kept prefix and refuses atomically.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mtplx.vendor.glm5_omlx.cache_rollback import pool_undo_arm
from mtplx.vendor.glm5_omlx.deepseek_v4.cache_extras import PoolingCache
from mtplx.cache_state import (
    CacheSnapshot,
    rollback_after_verify,
    trim_verified_window_to_prefix,
)

mx.random.seed(7)

RATIO = 4
D1, D2 = 8, 4


def _rows(n, d=D1):
    return mx.random.normal((1, n, d)).astype(mx.float32)


def _pool_update(pool, n):
    """Feed n tokens through accumulate_windows (verify/draft-sized)."""
    pool.accumulate_windows(_rows(n), mx.random.normal((1, n, D2)), 0)


def _pool_fingerprint(pool):
    return (
        pool.remainder,
        0 if pool.pooled is None else pool.pooled.shape[1],
        None
        if pool.pooled is None
        else float(mx.sum(pool.pooled).item()),
    )


class TestPoolingCacheUndo:
    def test_wide_verify_update_trims_back(self):
        """A 25-token cc-block verify update must be fully undoable."""
        pool = PoolingCache(RATIO)
        with pool_undo_arm():
            _pool_update(pool, 24)  # fills windows -> remainder 0 boundary
            assert pool.remainder == 0
            pre = _pool_fingerprint(pool)
            _pool_update(pool, 25)
            assert pool.can_trim(25)
            assert pool.trim(25) == 25
        assert _pool_fingerprint(pool) == pre

    def test_oversize_update_clears_coverage_and_refuses(self):
        """Updates beyond the stash bound clear undo; boundary trim refuses."""
        pool = PoolingCache(RATIO)
        with pool_undo_arm():
            _pool_update(pool, 24)
            _pool_update(pool, 100)  # >64: stash dropped
            # remainder = 100 % 4 == 0 -> sits exactly on a window edge
            assert pool.remainder == 0
            assert not pool.can_trim(20)
            assert pool.trim(20) == 0

    def test_can_trim_remainder_and_undo_semantics(self):
        pool = PoolingCache(RATIO)
        with pool_undo_arm():
            _pool_update(pool, 10)  # remainder 2, 2 windows compressed
            assert pool.remainder == 2
            assert pool.can_trim(2)      # pure remainder trim
            assert pool.can_trim(10)     # remainder + undo window coverage
            assert not pool.can_trim(11)  # beyond this update's coverage

    def test_undo_chain_stays_bounded(self):
        """Chained multi-token updates cap the undo log at 64 rows."""
        pool = PoolingCache(RATIO)
        with pool_undo_arm():
            _pool_update(pool, 32)
            _pool_update(pool, 32)  # chains to 64
            assert pool._undo is not None
            _pool_update(pool, 4)   # 68 > 64 -> fresh stash, not a chain
            assert pool._undo is not None
            assert pool._undo[4].shape[1] == 4


class _StubPair:
    """Minimal GLMOffsetCacheList-shaped pair for preflight tests."""

    def __init__(self, kv_ok, pool_ok):
        def _child(ok):
            c = SimpleNamespace(is_trimmable=lambda: ok, trimmed=[])

            def _trim(n):
                if not ok:
                    return 0
                c.trimmed.append(n)
                c.offset -= n
                return n

            c.can_trim = lambda n: ok
            c.trim = _trim
            return c

        self.caches = [_child(kv_ok), _child(pool_ok)]
        self.caches[0].offset = 100
        self.caches[1].offset = 100

    def is_trimmable(self):
        return self.caches[0].is_trimmable() and self.caches[1].is_trimmable()

    def can_trim(self, n):
        return self.caches[0].can_trim(n) and self.caches[1].can_trim(n)

    def trim(self, n):
        for c in self.caches:
            c.trim(n)
        return n

    @property
    def offset(self):
        return self.caches[0].offset


class TestPairPreflight:
    def test_trim_prefix_preflights_before_mutating(self):
        """A pool-side refusal must leave the KV half untouched."""
        pair = _StubPair(kv_ok=True, pool_ok=False)
        snap = CacheSnapshot(states=(None,), meta_states=(None,))
        ok = trim_verified_window_to_prefix(
            [pair], snap, verified_tokens=25, keep_tokens=2
        )
        assert not ok
        assert pair.caches[0].trimmed == []
        assert pair.offset == 100

    def test_trim_prefix_commits_when_both_cover(self):
        pair = _StubPair(kv_ok=True, pool_ok=True)
        snap = CacheSnapshot(states=(None,), meta_states=(None,))
        ok = trim_verified_window_to_prefix(
            [pair], snap, verified_tokens=25, keep_tokens=2
        )
        assert ok
        assert pair.offset == 77

    def test_rollback_falls_back_to_per_child_trim(self):
        """Pair-level refusal still trims the KV child — no stranded tokens."""
        pair = _StubPair(kv_ok=True, pool_ok=False)
        snap = CacheSnapshot(states=(None,), meta_states=(None,))
        rollback_after_verify([pair], snap, verified_tokens=20)
        assert pair.caches[0].trimmed == [20]
        assert pair.caches[1].trimmed == []


class TestCommitVerifiedWindow:
    """commit_verified_window on Glm5NextModel, driven over synthetic layers.

    The method only reads ``self.layers``/``self.config``-free surfaces, so a
    SimpleNamespace stands in for the model.
    """

    def _kda_layer(self, heads=2, head_dim=4, conv_k=4):
        fg = SimpleNamespace(
            A_log=mx.random.normal((heads,)),
            dt_bias=mx.random.normal((heads, head_dim)),
            safe_gate_lower_bound=-5.0,
        )
        attn = SimpleNamespace(
            conv_kernel_size=conv_k,
            num_heads=heads,
            head_dim=head_dim,
            forget_gate=fg,
        )
        return SimpleNamespace(is_linear=True, self_attn=attn)

    def _sparse_layer(self):
        return SimpleNamespace(is_linear=False)

    def _commit(self, model, cache, states, keep, verified):
        from mtplx.vendor.glm5_omlx.glm5_next.language import (
            Glm5NextModel,
        )

        return Glm5NextModel.commit_verified_window(
            model, cache, states, keep_tokens=keep, verified_tokens=verified
        )

    def _kda_entry(self, verified=5, heads=2, head_dim=4, conv_k=4):
        """ArraysCache-like KDA entry with stashed verify rows."""
        from mlx_lm.models.cache import ArraysCache

        entry = ArraysCache(size=2)
        mixed = _rows(verified, 3 * heads * head_dim)
        q = _rows(verified, heads * head_dim).reshape(
            1, verified, heads, head_dim
        )
        k = _rows(verified, heads * head_dim).reshape(
            1, verified, heads, head_dim
        )
        v = _rows(verified, heads * head_dim).reshape(
            1, verified, heads, head_dim
        )
        a = _rows(verified, heads * head_dim).reshape(
            1, verified, heads, head_dim
        )
        b = mx.random.normal((1, verified, heads))
        entry._mtplx_verify_rows = (mixed, q, k, v, a, b)
        conv_tail = _rows(conv_k - 1, 3 * heads * head_dim)
        gdn = mx.random.normal((1, heads, head_dim, head_dim))
        entry[0] = conv_tail
        entry[1] = gdn
        return entry, conv_tail, gdn

    def test_kda_replay_matches_reference(self):
        """Replayed state equals a fresh recurrence over the kept prefix."""
        from mtplx.vendor.glm5_omlx.glm5_next.gated_delta import (
            gated_delta_update,
        )

        layer = self._kda_layer()
        model = SimpleNamespace(layers=[layer])
        entry, conv_tail, gdn = self._kda_entry(verified=5)
        cache = [entry]
        states = [[conv_tail, gdn]]
        mixed, q, k, v, a, b = entry._mtplx_verify_rows
        keep = 2

        _, ref_state = gated_delta_update(
            q[:, :keep],
            k[:, :keep],
            v[:, :keep],
            a[:, :keep],
            b[:, :keep],
            layer.self_attn.forget_gate.A_log.reshape(2, 1),
            layer.self_attn.forget_gate.dt_bias.reshape(2, 4),
            state=gdn,
            lower_bound=layer.self_attn.forget_gate.safe_gate_lower_bound,
        )
        ref_conv = mx.contiguous(
            mx.concatenate([conv_tail, mixed[:, :keep]], axis=1)[:, -3:, :]
        )

        assert self._commit(model, cache, states, keep, 5)
        assert mx.allclose(entry[1], ref_state, atol=1e-5)
        assert mx.allclose(entry[0], ref_conv, atol=1e-6)
        assert entry._mtplx_verify_rows is None

    def test_refusal_when_rows_missing(self):
        layer = self._kda_layer()
        model = SimpleNamespace(layers=[layer])
        from mlx_lm.models.cache import ArraysCache

        entry = ArraysCache(size=2)
        entry[0] = _rows(3, 24)
        entry[1] = _rows(2, 16)
        # no _mtplx_verify_rows stashed
        ok = self._commit(
            model, [entry], [[entry[0], entry[1]]], 2, 5,
        )
        assert not ok

    def test_full_accept_clears_rows_without_replay(self):
        layer = self._kda_layer()
        model = SimpleNamespace(layers=[layer])
        entry, conv_tail, gdn = self._kda_entry(verified=4)
        cache = [entry]
        states = [[conv_tail, gdn]]
        assert self._commit(model, cache, states, keep=4, verified=4)
        # state untouched (verify forward already left post-window state)
        assert entry[1] is gdn
        assert entry._mtplx_verify_rows is None

    def test_sparse_entry_trims_through_can_trim(self):
        layer = self._sparse_layer()
        model = SimpleNamespace(layers=[layer])
        pair = _StubPair(kv_ok=True, pool_ok=True)
        cache = [pair]
        states = [None]
        assert self._commit(model, cache, states, keep=2, verified=5)
        assert pair.caches[0].trimmed == [3]

    def test_sparse_refusal_blocks_commit_atomically(self):
        kda = self._kda_layer()
        sparse = self._sparse_layer()
        model = SimpleNamespace(layers=[kda, sparse])
        kda_entry, conv_tail, gdn = self._kda_entry(verified=5)
        pair = _StubPair(kv_ok=True, pool_ok=False)
        cache = [kda_entry, pair]
        states = [[conv_tail, gdn], None]
        ok = self._commit(model, cache, states, keep=2, verified=5)
        assert not ok
        # the KDA replay must NOT have run — preflight before mutation
        assert kda_entry._mtplx_verify_rows is not None
        assert pair.caches[0].trimmed == []
