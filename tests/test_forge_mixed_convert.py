"""Unit coverage for the Forge module_overrides mixed-precision lane."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtplx.commands.forge import ForgeError, _mixed_convert_command
from mtplx.commands.forge_mixed_convert import build_predicate

SPEED_RECIPE = {
    "body_bits": 4,
    "body_group_size": 32,
    "body_mode": "affine",
    "mtp_policy": "keep_bf16",
    "module_overrides": [
        {"suffix": "embed_tokens", "bits": 8, "group_size": 64},
        {"suffix": "lm_head", "bits": 8, "group_size": 64},
        {"suffix": "linear_attn.out_proj", "bits": 8, "group_size": 64},
        {"suffix": "mlp.gate_proj", "layers": [56, 63], "bits": 8, "group_size": 64},
    ],
}


def test_predicate_suffix_override() -> None:
    predicate = build_predicate(SPEED_RECIPE)
    result = predicate("language_model.model.layers.12.linear_attn.out_proj", None)
    assert result == {"bits": 8, "group_size": 64, "mode": "affine"}


def test_predicate_layer_pinning() -> None:
    predicate = build_predicate(SPEED_RECIPE)
    assert predicate("language_model.model.layers.63.mlp.gate_proj", None) == {
        "bits": 8,
        "group_size": 64,
        "mode": "affine",
    }
    # Layer outside the pinned set falls back to the recipe body params.
    assert predicate("language_model.model.layers.12.mlp.gate_proj", None) is True


def test_predicate_unmatched_module_uses_body() -> None:
    predicate = build_predicate(SPEED_RECIPE)
    assert predicate("language_model.model.layers.5.self_attn.q_proj", None) is True


def test_predicate_quantize_false_preserves_source_precision() -> None:
    recipe = {
        "body_mode": "affine",
        "module_overrides": [
            {
                "suffix": "linear_attn.in_proj_a",
                "quantize": False,
            }
        ],
    }
    predicate = build_predicate(recipe)
    assert (
        predicate("language_model.model.layers.5.linear_attn.in_proj_a", None)
        is False
    )
    assert predicate("language_model.model.layers.5.mlp.gate_proj", None) is True


def test_predicate_prefix_agnostic() -> None:
    predicate = build_predicate(SPEED_RECIPE)
    assert predicate("model.layers.3.linear_attn.out_proj", None) == {
        "bits": 8,
        "group_size": 64,
        "mode": "affine",
    }
    assert predicate("lm_head", None) == {"bits": 8, "group_size": 64, "mode": "affine"}


def test_predicate_first_match_wins() -> None:
    recipe = {
        "body_mode": "affine",
        "module_overrides": [
            {"suffix": "mlp.down_proj", "bits": 6, "group_size": 64},
            {"suffix": "down_proj", "bits": 8, "group_size": 32},
        ],
    }
    predicate = build_predicate(recipe)
    assert predicate("model.layers.1.mlp.down_proj", None) == {
        "bits": 6,
        "group_size": 64,
        "mode": "affine",
    }


def test_predicate_rejects_empty_suffix() -> None:
    with pytest.raises(SystemExit):
        build_predicate({"module_overrides": [{"bits": 8}]})


def test_mixed_command_shape() -> None:
    command = _mixed_convert_command(
        Path("/src"),
        Path("/dst"),
        recipe=SPEED_RECIPE,
        source_format="bf16_native",
    )
    assert "-m" in command and "mtplx.commands.forge_mixed_convert" in command
    recipe_json = command[command.index("--recipe-json") + 1]
    assert json.loads(recipe_json) == SPEED_RECIPE
    assert "--dtype" not in command  # bf16 host default passes no dtype


def test_mixed_command_refuses_packed_sources() -> None:
    with pytest.raises(ForgeError):
        _mixed_convert_command(
            Path("/src"),
            Path("/dst"),
            recipe=SPEED_RECIPE,
            source_format="compressed_tensors_awq",
        )


def test_mixed_command_refuses_unquantized_body() -> None:
    with pytest.raises(ForgeError):
        _mixed_convert_command(
            Path("/src"),
            Path("/dst"),
            recipe={"body_bits": 0, "module_overrides": [{"suffix": "lm_head"}]},
            source_format="bf16_native",
        )


def test_glm5_source_registers_vendored_arch(tmp_path, monkeypatch) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text(
        json.dumps({"model_type": "glm5_next"}), encoding="utf-8"
    )
    import importlib
    import sys

    convert_mod = importlib.import_module("mlx_lm.convert")
    calls = {}
    monkeypatch.setattr(
        convert_mod,
        "convert",
        lambda **kwargs: calls.setdefault("called", True),
    )
    from mtplx.commands.forge_mixed_convert import main

    main(
        [
            "--source",
            str(src),
            "--destination",
            str(tmp_path / "dst"),
            "--recipe-json",
            json.dumps({"body_bits": 4}),
        ]
    )
    assert calls["called"]
    assert "mlx_lm.models.glm5_next" in sys.modules


def test_non_glm5_source_skips_vendored_registration(tmp_path, monkeypatch) -> None:
    import importlib
    import sys

    convert_mod = importlib.import_module("mlx_lm.convert")
    sys.modules.pop("mlx_lm.models.glm5_next", None)
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text(
        json.dumps({"model_type": "qwen3_next"}), encoding="utf-8"
    )
    monkeypatch.setattr(convert_mod, "convert", lambda **kwargs: None)
    from mtplx.commands.forge_mixed_convert import main

    main(
        [
            "--source",
            str(src),
            "--destination",
            str(tmp_path / "dst"),
            "--recipe-json",
            json.dumps({"body_bits": 4}),
        ]
    )
    assert "mlx_lm.models.glm5_next" not in sys.modules
