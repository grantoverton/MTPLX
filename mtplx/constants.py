"""Project constants for MTPLX gates and defaults."""

from __future__ import annotations

from pathlib import Path

PROJECT_NAME = "MTPLX"
PRIMARY_MODEL_REPO = "trevon/Qwen3.6-27B-mtp"
OFFICIAL_MODEL_REPO = "Qwen/Qwen3.6-27B"
DFLASH_MODEL_REPO = "z-lab/Qwen3.6-27B-DFlash"

PRIMARY_MODEL_DIR = Path("models/Qwen3.6-27B-mtp")
DEFAULT_RUNTIME_MODEL_DIR = Path("models/Qwen3.6-27B-MTPLX-Optimized-Speed")
LEGACY_SPEED_BASELINE_MODEL_DIR = Path("models/Qwen3.6-27B-MLXCommunity-4bit-mtp-graft")

DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20

EXPECTED_MTP_TENSOR_COUNT = 15
EXPECTED_GLM5_MTP_KEYS = (
    "mtp.0.block.input_layernorm.weight",
    "mtp.0.block.mlp.gate.e_score_correction_bias",
    "mtp.0.block.mlp.gate.weight",
    "mtp.0.block.mlp.shared_experts.down_proj.biases",
    "mtp.0.block.mlp.shared_experts.down_proj.scales",
    "mtp.0.block.mlp.shared_experts.down_proj.weight",
    "mtp.0.block.mlp.shared_experts.gate_proj.biases",
    "mtp.0.block.mlp.shared_experts.gate_proj.scales",
    "mtp.0.block.mlp.shared_experts.gate_proj.weight",
    "mtp.0.block.mlp.shared_experts.up_proj.biases",
    "mtp.0.block.mlp.shared_experts.up_proj.scales",
    "mtp.0.block.mlp.shared_experts.up_proj.weight",
    "mtp.0.block.mlp.switch_mlp.down_proj.biases",
    "mtp.0.block.mlp.switch_mlp.down_proj.scales",
    "mtp.0.block.mlp.switch_mlp.down_proj.weight",
    "mtp.0.block.mlp.switch_mlp.gate_proj.biases",
    "mtp.0.block.mlp.switch_mlp.gate_proj.scales",
    "mtp.0.block.mlp.switch_mlp.gate_proj.weight",
    "mtp.0.block.mlp.switch_mlp.up_proj.biases",
    "mtp.0.block.mlp.switch_mlp.up_proj.scales",
    "mtp.0.block.mlp.switch_mlp.up_proj.weight",
    "mtp.0.block.post_attention_layernorm.weight",
    "mtp.0.block.self_attn.embed_q.biases",
    "mtp.0.block.self_attn.embed_q.scales",
    "mtp.0.block.self_attn.embed_q.weight",
    "mtp.0.block.self_attn.indexer.index_kpool_compress_ape",
    "mtp.0.block.self_attn.indexer.index_kpool_compress_gate",
    "mtp.0.block.self_attn.indexer.k_norm.bias",
    "mtp.0.block.self_attn.indexer.k_norm.weight",
    "mtp.0.block.self_attn.indexer.weights_proj.biases",
    "mtp.0.block.self_attn.indexer.weights_proj.scales",
    "mtp.0.block.self_attn.indexer.weights_proj.weight",
    "mtp.0.block.self_attn.indexer.wk.biases",
    "mtp.0.block.self_attn.indexer.wk.scales",
    "mtp.0.block.self_attn.indexer.wk.weight",
    "mtp.0.block.self_attn.indexer.wq_b.biases",
    "mtp.0.block.self_attn.indexer.wq_b.scales",
    "mtp.0.block.self_attn.indexer.wq_b.weight",
    "mtp.0.block.self_attn.kv_a_layernorm.weight",
    "mtp.0.block.self_attn.kv_a_proj_with_mqa.biases",
    "mtp.0.block.self_attn.kv_a_proj_with_mqa.scales",
    "mtp.0.block.self_attn.kv_a_proj_with_mqa.weight",
    "mtp.0.block.self_attn.o_proj.biases",
    "mtp.0.block.self_attn.o_proj.scales",
    "mtp.0.block.self_attn.o_proj.weight",
    "mtp.0.block.self_attn.q_a_layernorm.weight",
    "mtp.0.block.self_attn.q_a_proj.biases",
    "mtp.0.block.self_attn.q_a_proj.scales",
    "mtp.0.block.self_attn.q_a_proj.weight",
    "mtp.0.block.self_attn.q_b_proj.biases",
    "mtp.0.block.self_attn.q_b_proj.scales",
    "mtp.0.block.self_attn.q_b_proj.weight",
    "mtp.0.block.self_attn.unembed_out.biases",
    "mtp.0.block.self_attn.unembed_out.scales",
    "mtp.0.block.self_attn.unembed_out.weight",
    "mtp.0.eh_proj.weight",
    "mtp.0.enorm.weight",
    "mtp.0.hnorm.weight",
    "mtp.0.norm.weight",
)


EXPECTED_MTP_KEYS = (
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
)

EXPECTED_QWEN_MOE_MTP_TENSOR_COUNT = 19
EXPECTED_QWEN_MOE_MTP_KEYS = (
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.experts.down_proj",
    "mtp.layers.0.mlp.experts.gate_up_proj",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
)

EXPECTED_QWEN_MOE_SWITCH_MLP_MTP_KEYS = (
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight",
    "mtp.layers.0.mlp.switch_mlp.down_proj.weight",
    "mtp.layers.0.mlp.switch_mlp.gate_proj.weight",
    "mtp.layers.0.mlp.switch_mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
)
EXPECTED_QWEN_MOE_SWITCH_MLP_MTP_TENSOR_COUNT = len(
    EXPECTED_QWEN_MOE_SWITCH_MLP_MTP_KEYS
)

MTP_QUANTIZED_LINEAR_WEIGHT_KEYS = (
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
)

MTP_ALL_QUANTIZED_LINEAR_WEIGHT_KEYS = (
    "mtp.fc.weight",
    *MTP_QUANTIZED_LINEAR_WEIGHT_KEYS,
)

EXPECTED_PREQUANTIZED_MTP_KEYS = tuple(
    sorted(
        EXPECTED_MTP_KEYS
        + tuple(key.rsplit(".", 1)[0] + ".scales" for key in MTP_QUANTIZED_LINEAR_WEIGHT_KEYS)
        + tuple(key.rsplit(".", 1)[0] + ".biases" for key in MTP_QUANTIZED_LINEAR_WEIGHT_KEYS)
    )
)
EXPECTED_PREQUANTIZED_MTP_TENSOR_COUNT = len(EXPECTED_PREQUANTIZED_MTP_KEYS)

EXPECTED_ALL_PREQUANTIZED_MTP_KEYS = tuple(
    sorted(
        EXPECTED_MTP_KEYS
        + tuple(
            key.rsplit(".", 1)[0] + ".scales"
            for key in MTP_ALL_QUANTIZED_LINEAR_WEIGHT_KEYS
        )
        + tuple(
            key.rsplit(".", 1)[0] + ".biases"
            for key in MTP_ALL_QUANTIZED_LINEAR_WEIGHT_KEYS
        )
    )
)
EXPECTED_ALL_PREQUANTIZED_MTP_TENSOR_COUNT = len(EXPECTED_ALL_PREQUANTIZED_MTP_KEYS)

QWEN_MOE_MTP_QUANTIZED_LINEAR_WEIGHT_KEYS = (
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
)

QWEN_MOE_SWITCH_MLP_MTP_QUANTIZED_LINEAR_WEIGHT_KEYS = (
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight",
    "mtp.layers.0.mlp.switch_mlp.down_proj.weight",
    "mtp.layers.0.mlp.switch_mlp.gate_proj.weight",
    "mtp.layers.0.mlp.switch_mlp.up_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
)

EXPECTED_QWEN_MOE_SWITCH_MLP_PREQUANTIZED_MTP_KEYS = tuple(
    sorted(
        EXPECTED_QWEN_MOE_SWITCH_MLP_MTP_KEYS
        + tuple(
            key.rsplit(".", 1)[0] + ".scales"
            for key in QWEN_MOE_SWITCH_MLP_MTP_QUANTIZED_LINEAR_WEIGHT_KEYS
        )
        + tuple(
            key.rsplit(".", 1)[0] + ".biases"
            for key in QWEN_MOE_SWITCH_MLP_MTP_QUANTIZED_LINEAR_WEIGHT_KEYS
        )
    )
)
EXPECTED_QWEN_MOE_SWITCH_MLP_PREQUANTIZED_MTP_TENSOR_COUNT = len(
    EXPECTED_QWEN_MOE_SWITCH_MLP_PREQUANTIZED_MTP_KEYS
)

EXPECTED_QWEN_MOE_PREQUANTIZED_MTP_KEYS = tuple(
    sorted(
        EXPECTED_QWEN_MOE_MTP_KEYS
        + tuple(
            key.rsplit(".", 1)[0] + ".scales"
            for key in QWEN_MOE_MTP_QUANTIZED_LINEAR_WEIGHT_KEYS
        )
        + tuple(
            key.rsplit(".", 1)[0] + ".biases"
            for key in QWEN_MOE_MTP_QUANTIZED_LINEAR_WEIGHT_KEYS
        )
    )
)
EXPECTED_QWEN_MOE_PREQUANTIZED_MTP_TENSOR_COUNT = len(
    EXPECTED_QWEN_MOE_PREQUANTIZED_MTP_KEYS
)

_MTP_LAYER_KEY_MARKER = "mtp.layers.0."


def expand_mtp_layer_keys(keys: tuple[str, ...] | set[str], n_layers: int) -> set[str]:
    """Expand a depth-1 MTP key template across ``n_layers`` draft layers.

    Every expected-key set in this module describes the canonical
    single-layer (``mtp.layers.0.*``) head. Upstream checkpoints may declare
    ``mtp_num_hidden_layers > 1`` (the config key is N-generic in the vLLM
    reference contract); their weight layout replicates the per-layer
    template at each index. Identity for ``n_layers <= 1``.
    """
    n = max(int(n_layers), 1)
    expanded: set[str] = set()
    for key in keys:
        if _MTP_LAYER_KEY_MARKER in key:
            for index in range(n):
                expanded.add(key.replace(_MTP_LAYER_KEY_MARKER, f"mtp.layers.{index}.", 1))
        else:
            expanded.add(key)
    return expanded


MULTIMODAL_SIDECARS = (
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)
