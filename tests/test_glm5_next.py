"""glm5_next (GLM-5.3-Flash) port surface: registry, descriptor, MTP weight
selection and the forge-sidecar regression — all synthetic, no model load."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mtplx.backends.descriptors import (
    GLM5_NEXT_DESCRIPTOR,
    context_window_policy_for_model,
    descriptor_for_architecture_id,
    descriptor_for_backend_id,
    kv_quant_policy_for_model,
    reasoning_policy_for_model,
)
from mtplx.backends.registry import (
    architecture_support_for,
)
from mtplx.glm_mtp_patch import (
    _candidate_weight_files,
    _rewrite_glm_mtp_weights,
    is_glm_mtp_config,
)


def _glm5_config(**overrides):
    config = {
        "model_type": "glm5_next",
        "text_config": {"model_type": "glm5_next_text", "num_hidden_layers": 45},
        "num_nextn_predict_layers": 1,
    }
    config.update(overrides)
    return config


def test_glm5_next_architecture_registered():
    arch = architecture_support_for("glm5-next-mtp")
    assert arch is not None
    assert arch.backend == "glm5_next"
    # model_type aliases resolve to the same support row.
    by_alias = architecture_support_for("glm5_next")
    assert by_alias is arch
    assert descriptor_for_backend_id("glm5_next") is GLM5_NEXT_DESCRIPTOR


def test_glm5_next_descriptor_resolution():
    descriptor = descriptor_for_architecture_id("glm5-next-mtp")
    assert descriptor is GLM5_NEXT_DESCRIPTOR
    assert descriptor.backend_id == "glm5_next"
    # GLM-5.3 generation_config: temp 1.0, top_p 0.95, no top_k.
    assert descriptor.sampler_defaults.temperature == 1.0
    assert descriptor.sampler_defaults.top_k == 0
    codec = descriptor.reasoning_codec
    assert codec.parser == "qwen3"
    assert codec.effort_levels == ("low", "high", "xhigh")


def test_glm5_next_family_policies_use_glm5_descriptor():
    # Family "glm" resolves for both glm_moe_dsa and glm5_next artifacts;
    # family-level policies must not fall back to the GLM-4 descriptor when
    # the resolved descriptor is glm5_next.
    assert (
        kv_quant_policy_for_model(descriptor=GLM5_NEXT_DESCRIPTOR)
        is GLM5_NEXT_DESCRIPTOR.kv_quant_policy
    )
    assert (
        reasoning_policy_for_model(descriptor=GLM5_NEXT_DESCRIPTOR)
        is GLM5_NEXT_DESCRIPTOR.reasoning_codec
    )
    ctx = context_window_policy_for_model(descriptor=GLM5_NEXT_DESCRIPTOR)
    assert ctx.maximum == GLM5_NEXT_DESCRIPTOR.context_window_policy.maximum


def test_is_glm_mtp_config_glm5():
    assert is_glm_mtp_config(_glm5_config())


def test_sidecar_keeps_lm_head_shard(tmp_path):
    """Regression for the forge zero-acceptance bug: a configured
    mtp.safetensors early-returned the weight-file list, so the shared
    lm_head that feeds shared_head_head never loaded and the draft head
    proposed uninitialized tokens (0% acceptance)."""
    (tmp_path / "mtp.safetensors").write_bytes(b"sidecar")
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"one")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"two")
    index = {
        "weight_map": {
            "language_model.lm_head.weight": "model-00001-of-00002.safetensors",
            "language_model.model.layers.0.x": "model-00002-of-00002.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    config = _glm5_config(mlx_lm_extra_tensors={"mtp_file": "mtp.safetensors"})
    files = _candidate_weight_files(tmp_path, config)
    names = [f.name for f in files]
    assert "mtp.safetensors" in names
    assert "model-00001-of-00002.safetensors" in names
    assert "model-00002-of-00002.safetensors" not in names


def test_rewrite_embedded_and_lm_head_lands_shared_head():
    sentinel = object()
    raw = {
        "language_model.mtp.0.enorm.weight": object(),
        "language_model.mtp.0.hnorm.weight": object(),
        "language_model.mtp.0.eh_proj.weight": object(),
        "language_model.mtp.0.norm.weight": object(),
        "language_model.mtp.0.block.mlp.gate.weight": object(),
        "language_model.mtp.0.block.self_attn.embed_q.weight": object(),
        "language_model.lm_head.weight": sentinel,
        "language_model.lm_head.scales": object(),
    }
    args = SimpleNamespace(n_routed_experts=0)
    mapped = _rewrite_glm_mtp_weights(
        raw, args=args, start_layer=45, num_mtp_layers=1, rewrite_mla_kv_b=False
    )
    assert mapped["layers.0.enorm.weight"] is raw["language_model.mtp.0.enorm.weight"]
    assert "layers.0.mtp_block.mlp.gate.weight" in mapped
    assert "layers.0.shared_head_norm.weight" in mapped
    assert mapped["layers.0.shared_head_head.weight"] is sentinel
    assert "layers.0.shared_head_head.scales" in mapped


def test_rewrite_sidecar_dialect_also_maps():
    raw = {
        "mtp.0.enorm.weight": object(),
        "mtp.0.hnorm.weight": object(),
        "mtp.0.eh_proj.weight": object(),
        "mtp.0.norm.weight": object(),
        "mtp.0.block.self_attn.q_a_proj.weight": object(),
        "mtp.0.block.self_attn.embed_q.weight": object(),
        "lm_head.weight": object(),
    }
    args = SimpleNamespace(n_routed_experts=0)
    mapped = _rewrite_glm_mtp_weights(
        raw, args=args, start_layer=45, num_mtp_layers=1, rewrite_mla_kv_b=False
    )
    assert "layers.0.enorm.weight" in mapped
    assert "layers.0.mtp_block.self_attn.q_a_proj.weight" in mapped
    assert "layers.0.shared_head_norm.weight" in mapped
    assert "layers.0.shared_head_head.weight" in mapped


def test_rewrite_bf16_appended_layer_dialect_maps():
    """zai-org GLM-5.3-Flash-BF16 keeps the head as an appended decoder
    layer model.language_model.layers.45.* (text_config declares
    num_nextn_predict_layers=1) with no shared_head.head — the output
    projection is filled from the top-level lm_head.weight."""
    sentinel = object()
    raw = {
        "model.language_model.layers.45.enorm.weight": object(),
        "model.language_model.layers.45.hnorm.weight": object(),
        "model.language_model.layers.45.eh_proj.weight": object(),
        "model.language_model.layers.45.shared_head.norm.weight": object(),
        "model.language_model.layers.45.input_layernorm.weight": object(),
        "model.language_model.layers.45.self_attn.kv_a_proj_with_mqa.weight": object(),
        "model.language_model.layers.45.self_attn.embed_q.weight": object(),
        "model.language_model.layers.45.mlp.gate.weight": object(),
        "lm_head.weight": sentinel,
    }
    args = SimpleNamespace(n_routed_experts=0)
    mapped = _rewrite_glm_mtp_weights(
        raw, args=args, start_layer=45, num_mtp_layers=1, rewrite_mla_kv_b=False
    )
    assert "layers.0.enorm.weight" in mapped
    assert "layers.0.mtp_block.self_attn.kv_a_proj_with_mqa.weight" in mapped
    assert "layers.0.mtp_block.mlp.gate.weight" in mapped
    assert "layers.0.shared_head_norm.weight" in mapped
    assert mapped["layers.0.shared_head_head.weight"] is sentinel


def test_mlx_lm_convert_module_routes_glm5(tmp_path):
    from mtplx.commands.forge import _mlx_lm_convert_module

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm5_next",
                "text_config": {"model_type": "glm5_next_text"},
            }
        )
    )
    assert _mlx_lm_convert_module(tmp_path) == "mtplx.commands.forge_glm5_convert"
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_next"}))
    assert _mlx_lm_convert_module(tmp_path) == "mlx_lm"


def test_vision_prefix_recognizes_vision_model():
    from mtplx.vision.qwen3_vl_tower import resolve_vision_prefix

    assert (
        resolve_vision_prefix({"vision_model.blocks.0.attn.weight": "shard"})
        == "vision_model."
    )


# --- Draft-cache rollback surface (Fable blockers 1+2) ---------------------


def test_pool_undo_arm_toggles_flags():
    from mtplx.vendor.glm5_omlx import cache_rollback

    assert not cache_rollback._is_undo_armed()
    assert not cache_rollback._is_decode_consistent_armed()
    with cache_rollback.pool_undo_arm():
        assert cache_rollback._is_undo_armed()
        assert cache_rollback._is_decode_consistent_armed()
        with cache_rollback.pool_undo_arm():  # nested arms restore correctly
            assert cache_rollback._is_undo_armed()
        assert cache_rollback._is_undo_armed()
    assert not cache_rollback._is_undo_armed()
    assert not cache_rollback._is_decode_consistent_armed()


def test_draft_cache_offset_and_trim_delegate():
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm.models.cache")
    pytest.importorskip("mlx_vlm")
    from mtplx.models.glm5_next import mtp_impl

    impl = mtp_impl(_glm5_config())
    cache = impl["cache_factory"]()
    kv, pool = cache.caches
    assert cache.offset == kv.offset == 0
    kv.offset = 11
    pool.remainder = 3  # pool counts the same token stream
    assert cache.offset == 11
    trimmed = cache.trim(2)
    assert kv.offset == 9
    assert pool.remainder == 1
    assert trimmed == 2  # CacheList returns the last child's result


def test_mtp_impl_reads_config_index_kpool():
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm")
    impl = __import__("mtplx.models.glm5_next", fromlist=["mtp_impl"]).mtp_impl(
        _glm5_config(text_config={"index_kpool": 7})
    )
    cache = impl["cache_factory"]()
    assert cache.caches[1].ratio == 7


def test_model_classes_memoized():
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm")
    from mtplx.models import glm5_next

    first = glm5_next.model_classes()
    assert glm5_next.model_classes() is first


def test_pooling_cache_meta_state_guard_and_from_state():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm.models.cache")
    from mtplx.vendor.glm5_omlx.deepseek_v4.cache_extras import PoolingCache

    pool = PoolingCache(4)
    # A mismatched/placeholder meta must not poison the constructed ratio
    # (the ratio=None crash on the near-prefix restore path).
    pool.meta_state = None
    assert pool.ratio == 4
    pool.meta_state = ("256", "40", "default")  # a KVCache-shaped meta
    assert pool.ratio == 4
    pool.meta_state = 4
    assert pool.ratio == 4

    # from_state must build through __init__ (upstream _BaseCache uses
    # __new__ + setters, leaving ratio/undo attrs unset).
    rebuilt = PoolingCache.from_state(pool.state, 4)
    assert rebuilt.ratio == 4
    with pytest.raises((TypeError, ValueError)):
        PoolingCache.from_state(pool.state, None)


def test_pooling_cache_undo_rollback_round_trip():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm.models.cache")
    from mtplx.vendor.glm5_omlx import cache_rollback
    from mtplx.vendor.glm5_omlx.deepseek_v4.cache_extras import PoolingCache

    kv = mx.zeros((1, 4, 8), dtype=mx.float16)
    gate = mx.zeros((1, 4, 4), dtype=mx.float16)
    pooled_row = mx.zeros((1, 1, 8), dtype=mx.float16)

    # Single-update rollback works armed or not — the one-update undo log is
    # always stashed for decode-sized updates.
    pool = PoolingCache(4)
    with cache_rollback.pool_undo_arm():
        pool.accumulate_windows(kv, gate, 0)  # completes pool window 0
        pool.update_and_fetch(pooled_row)
    assert pool.size() == 1
    assert pool.trim(1) == 1
    assert pool.remainder == 3
    assert pool.size() == 0

    # Arming extends the undo log across consecutive updates (rejection
    # spanning a whole verify window + draft steps). Without it the second
    # update overwrites the log and a trim that deep is a clean refusal.
    armed = PoolingCache(4)
    with cache_rollback.pool_undo_arm():
        armed.accumulate_windows(kv, gate, 0)
        armed.update_and_fetch(pooled_row)
        armed.accumulate_windows(kv, gate, 4)
        armed.update_and_fetch(pooled_row)
    assert armed.size() == 2
    assert armed.trim(5) == 5
    assert armed.remainder == 3
    assert armed.size() == 0

    unarmed = PoolingCache(4)
    unarmed.accumulate_windows(kv, gate, 0)
    unarmed.update_and_fetch(pooled_row)
    unarmed.accumulate_windows(kv, gate, 4)
    unarmed.update_and_fetch(pooled_row)
    assert unarmed.trim(5) == 0  # last-update log only reaches 4 rows
    assert unarmed.size() == 2


def test_mtp_weights_present_on_disk_bf16_dialect(tmp_path):
    """The BF16 appended-layer head must count as on-disk MTP weights."""
    from mtplx.artifacts import mtp_weights_present_on_disk

    index = {
        "weight_map": {
            "model.language_model.layers.45.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
            "model.language_model.layers.45.enorm.weight": "model-00001-of-00001.safetensors",
            "lm_head.weight": "model-00001-of-00001.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"shard")
    assert mtp_weights_present_on_disk(tmp_path, _glm5_config())


def test_sidecar_no_index_falls_back_to_trunk_shard(tmp_path):
    """A sidecar beside an unindexed single-file trunk must still load the
    trunk shard so shared_head_head gets its lm_head."""
    (tmp_path / "mtp.safetensors").write_bytes(b"sidecar")
    (tmp_path / "model.safetensors").write_bytes(b"trunk")
    config = _glm5_config(mlx_lm_extra_tensors={"mtp_file": "mtp.safetensors"})
    names = [f.name for f in _candidate_weight_files(tmp_path, config)]
    assert names == ["model.safetensors", "mtp.safetensors"] or set(names) == {
        "model.safetensors",
        "mtp.safetensors",
    }


def test_sidecar_lm_head_quant_leaves_in_other_shard(tmp_path):
    """Quantized lm_head splits weight/scales across shards; all must load."""
    (tmp_path / "mtp.safetensors").write_bytes(b"sidecar")
    index = {
        "weight_map": {
            "language_model.lm_head.weight": "model-00001-of-00003.safetensors",
            "language_model.lm_head.scales": "model-00002-of-00003.safetensors",
            "language_model.lm_head.biases": "model-00002-of-00003.safetensors",
            "language_model.model.layers.0.x": "model-00003-of-00003.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    config = _glm5_config(mlx_lm_extra_tensors={"mtp_file": "mtp.safetensors"})
    names = {f.name for f in _candidate_weight_files(tmp_path, config)}
    assert names == {
        "mtp.safetensors",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
    }


def test_rewrite_completeness_requires_shared_head():
    """The payload gate must reject a mapping that silently missed lm_head —
    that exact miss produced the zero-acceptance forge defect."""
    raw = {
        "language_model.mtp.0.enorm.weight": object(),
        "language_model.mtp.0.hnorm.weight": object(),
        "language_model.mtp.0.eh_proj.weight": object(),
        "language_model.mtp.0.block.mlp.gate.weight": object(),
        "language_model.mtp.0.block.self_attn.embed_q.weight": object(),
        # no lm_head.* / embed_tokens.* anywhere
    }
    args = SimpleNamespace(n_routed_experts=0)
    assert (
        _rewrite_glm_mtp_weights(
            raw, args=args, start_layer=45, num_mtp_layers=1, rewrite_mla_kv_b=False
        )
        == {}
    )


def test_rewrite_tied_embedding_fills_shared_head():
    sentinel = object()
    raw = {
        "language_model.mtp.0.enorm.weight": object(),
        "language_model.mtp.0.hnorm.weight": object(),
        "language_model.mtp.0.eh_proj.weight": object(),
        "language_model.mtp.0.block.mlp.gate.weight": object(),
        "language_model.mtp.0.block.self_attn.embed_q.weight": object(),
        "language_model.model.embed_tokens.weight": sentinel,
    }
    args = SimpleNamespace(n_routed_experts=0)
    mapped = _rewrite_glm_mtp_weights(
        raw, args=args, start_layer=45, num_mtp_layers=1, rewrite_mla_kv_b=False
    )
    assert mapped["layers.0.shared_head_head.weight"] is sentinel


def test_trunk_pair_offset_gates_session_bank_trims():
    """The wrapped trunk pair must let ``_trim_cache_ref_to_prefix`` see real
    offsets: mid-window restores trim; an untrimmable pool declines instead
    of silently serving an over-long cache (the "TheThe" corruption)."""
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm.models.cache")
    pytest.importorskip("mlx_vlm")
    from mlx_lm.models.cache import CacheList, KVCache
    from mtplx.models.glm5_next import mtp_impl
    from mtplx.session_bank import _trim_cache_ref_to_prefix
    from mtplx.vendor.glm5_omlx.deepseek_v4.cache_extras import PoolingCache

    impl = mtp_impl(_glm5_config())
    pair_cls = impl["cache_pair_cls"]

    # Bare CacheList (what the vendored make_cache returned): offset stays
    # invisible so the helper computes delta=0 and "succeeds" without
    # trimming — the silent-corruption path.
    bare_kv, bare_pool = KVCache(), PoolingCache(4)
    bare_kv.offset = 6
    bare = CacheList(bare_kv, bare_pool)
    assert _trim_cache_ref_to_prefix([bare], 6) is True
    assert bare_kv.offset == 6  # untrimmed — this was the bug

    # Wrapped pair, mid-window: trims the real delta on both halves.
    kv, pool = KVCache(), PoolingCache(4)
    kv.offset = 6
    pool.remainder = 3
    wrapped = pair_cls(kv, pool)
    assert _trim_cache_ref_to_prefix([wrapped], 6) is True
    assert kv.offset == 5
    assert pool.remainder == 2

    # Wrapped pair at a pool window boundary with no undo log: the pool
    # cannot give the token back, so the restore must decline cleanly —
    # and the atomic pair trim must leave the KV half untouched (a
    # refused trim that still shortened KV would desync the pair).
    kv2, pool2 = KVCache(), PoolingCache(4)
    kv2.offset = 6
    wrapped2 = pair_cls(kv2, pool2)
    assert _trim_cache_ref_to_prefix([wrapped2], 6) is False
    assert kv2.offset == 6


def test_trunk_pair_wrap_fires_on_vendored_cache_classes():
    """Regression for the dead-wrap defect: the vendored ``make_cache``
    builds ``mlx_vlm.models.cache.CacheList`` pairs, not the ``mlx_lm``
    class — an isinstance-based wrap never fired. ``_needs_offset_wrap``
    must duck-type both flavors."""
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm")
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.cache import CacheList as VLMCacheList
    from mlx_vlm.models.cache import KVCache as VLMKVCache
    from mtplx.glm_mtp_patch import _needs_offset_wrap
    from mtplx.models.glm5_next import mtp_impl
    from mtplx.vendor.glm5_omlx.deepseek_v4.cache_extras import PoolingCache

    impl = mtp_impl(_glm5_config())
    pair_cls = impl["cache_pair_cls"]

    # The vendored trunk pair shape: mlx_vlm CacheList of [KVCache, Pool].
    vlm_pair = VLMCacheList(VLMKVCache(), PoolingCache(4))
    assert _needs_offset_wrap(vlm_pair) is True
    wrapped = pair_cls(*vlm_pair.caches)
    assert wrapped.offset == 0
    wrapped.caches[0].offset = 7
    assert wrapped.offset == 7

    # Linear-layer arrays are not pairs and must pass through untouched.
    assert _needs_offset_wrap(ArraysCache(size=2)) is False

    # Idempotent: an already-wrapped pair exposes offset and is skipped.
    assert _needs_offset_wrap(wrapped) is False


def test_kv_b_split_for_bf16_head_lands_embed_q():
    """BF16 exports carry the fused ``kv_b_proj``; the MTP block's sparse
    attention needs the split ``embed_q``/``unembed_out`` — the gate must
    see them after rewrite_mla_kv_b, not an init-time projection."""
    mx = pytest.importorskip("mlx.core")
    import mlx.core  # noqa: F401

    heads, qk_nope, v_head, kv_lora = 2, 4, 4, 8
    fused = mx.random.normal((heads * (qk_nope + v_head), kv_lora))
    raw = {
        "model.language_model.layers.45.enorm.weight": object(),
        "model.language_model.layers.45.hnorm.weight": object(),
        "model.language_model.layers.45.eh_proj.weight": object(),
        "model.language_model.layers.45.self_attn.kv_a_proj_with_mqa.weight": object(),
        "model.language_model.layers.45.self_attn.kv_b_proj.weight": fused,
        "model.language_model.layers.45.mlp.gate.weight": object(),
        "lm_head.weight": object(),
    }
    args = SimpleNamespace(
        n_routed_experts=0,
        num_attention_heads=heads,
        qk_nope_head_dim=qk_nope,
        v_head_dim=v_head,
        kv_lora_rank=kv_lora,
    )
    mapped = _rewrite_glm_mtp_weights(
        raw, args=args, start_layer=45, num_mtp_layers=1, rewrite_mla_kv_b=True
    )
    embed_q = mapped["layers.0.mtp_block.self_attn.embed_q.weight"]
    unembed_out = mapped["layers.0.mtp_block.self_attn.unembed_out.weight"]
    assert embed_q.shape == (heads, kv_lora, qk_nope)
    assert unembed_out.shape == (heads, v_head, kv_lora)
    assert "layers.0.mtp_block.self_attn.kv_b_proj.weight" not in mapped
    assert "layers.0.shared_head_head.weight" in mapped
