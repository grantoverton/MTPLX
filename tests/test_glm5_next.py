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
    }
    args = SimpleNamespace(n_routed_experts=0)
    mapped = _rewrite_glm_mtp_weights(
        raw, args=args, start_layer=45, num_mtp_layers=1, rewrite_mla_kv_b=False
    )
    assert "layers.0.enorm.weight" in mapped
    assert "layers.0.mtp_block.self_attn.q_a_proj.weight" in mapped
    assert "layers.0.shared_head_norm.weight" in mapped


def test_vision_prefix_recognizes_vision_model():
    from mtplx.vision.qwen3_vl_tower import resolve_vision_prefix

    assert (
        resolve_vision_prefix({"vision_model.blocks.0.attn.weight": "shard"})
        == "vision_model."
    )
