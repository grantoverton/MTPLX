"""GLM-5.3 (``glm5_next``) in-tree model classes for MTPLX.

No ``glm5_next`` implementation exists in the pinned mlx-lm (0.31.3). The
architecture lives in mlx-vlm upstream (>=0.6.17) but *that* implementation
expects a different checkpoint dialect — the oQ artifacts this fleet ships
(oMLX-vendored convention: ``forget_gate.f_a_proj``, ``conv1d``, unfused
``q_proj``/``k_proj``/``v_proj``, ``vision_model.*`` tower) were converted
against oMLX's vendored glm5_next, which lives in
``omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next/``.

So the runtime here is the vendored copy of exactly that tree, at
``mtplx/vendor/glm5_omlx/`` — model package plus its two ``omlx.patches``
dependency packages (``glm_moe_dsa`` sparse-MLA/DSA helpers and
``deepseek_v4`` SwitchGLU/PoolingCache). Native oMLX custom kernels are
optional: every ``omlx.custom_kernels.*`` import falls back to ``mx.fast``
or a pure-MLX path when absent.

Two load-time adaptations happen in ``Model.sanitize``:

* ``language_model.mtp.*`` — the nextn MTP block's 59 tensors have no module
  to land on in the AR path. They are stashed on ``model._mtp_weight_stash``
  (keyed minus the ``language_model.`` prefix) for the Phase-2 attach.
* Quantization dict keys are already module-tree paths (the vendored tree
  keeps the unfused names), so ``config["quantization"]`` resolves directly.

``PoolingCache`` is injected into ``mlx_lm.models.cache`` before the model is
constructed — the vendored ``make_cache`` imports it from there, matching the
oMLX ``deepseek_v4`` patch behavior.
"""

from __future__ import annotations

import sys
from typing import Any, Dict


def _install_pooling_cache() -> None:
    """Make ``mlx_lm.models.cache.PoolingCache`` resolve to the vendored one."""
    import mlx_lm.models.cache as lm_cache

    if getattr(lm_cache, "PoolingCache", None) is None:
        from mtplx.vendor.glm5_omlx.deepseek_v4.cache_extras import PoolingCache

        lm_cache.PoolingCache = PoolingCache


def model_classes() -> tuple[type, type]:
    """Return (Model, ModelArgs) for the vendored glm5_next tree."""
    _install_pooling_cache()

    from mtplx.vendor.glm5_omlx.glm5_next.config import (
        ModelConfig,
        TextConfig,
        VisionConfig,
    )
    from mtplx.vendor.glm5_omlx.glm5_next.glm5_next import Model as _VLMModel

    class Model(_VLMModel):
        def sanitize(self, weights: Dict[str, Any]) -> Dict[str, Any]:
            stash: Dict[str, Any] = {}
            remapped: Dict[str, Any] = {}
            for key, value in weights.items():
                if key.startswith("language_model.mtp."):
                    stash[key[len("language_model.") :]] = value
                    continue
                remapped[key] = value
            self._mtp_weight_stash = stash
            return super().sanitize(remapped)

    class ModelArgs(ModelConfig):
        @classmethod
        def from_dict(cls, params):
            params = dict(params)
            tc = params.get("text_config")
            if isinstance(tc, dict):
                params["text_config"] = TextConfig.from_dict(tc)
            vc = params.get("vision_config")
            if isinstance(vc, dict):
                params["vision_config"] = VisionConfig.from_dict(vc)
            return super().from_dict(params)

    return Model, ModelArgs


# ``mtplx.runtime._model_classes_for_config`` calls the registered loader,
# which lands here.


def mtp_impl():
    """``glm_mtp_patch._glm_impl`` entry for the vendored glm5_next tree.

    The GLM-5.3 MTP head is a single plain decoder block (no HyperConnection):
    ``enorm``/``hnorm``/``eh_proj`` fusion feeding ``block`` (sparse/indexer
    attention + MoE), then ``norm`` + the shared lm_head. The checkpoint stores
    it as ``language_model.mtp.0.*``.
    """
    _install_pooling_cache()

    import mlx.nn as nn
    from mlx_lm.models.cache import CacheList, KVCache, PoolingCache

    from mtplx.vendor.glm5_omlx.glm5_next.language import (
        Glm5NextMLP,
        Glm5NextMoE,
        Glm5NextSparseAttention,
    )
    from mtplx.vendor.glm5_omlx.glm5_next.config import TextConfig

    class Glm5NextMTPDecoderLayer(nn.Module):
        """``Glm5NextDecoderLayer`` minus HyperConnection.

        The MTP block carries no ``hc_attn_*``/``hc_ffn_*`` tensors; it is a
        plain pre-norm layer with additive residuals on the (B, S, D) hidden.
        Attention is always the sparse/indexer variant.
        """

        def __init__(self, config, layer_idx=None):
            super().__init__()
            self.self_attn = Glm5NextSparseAttention(config)
            n_routed = getattr(config, "n_routed_experts", None)
            self.mlp = Glm5NextMoE(config) if n_routed else Glm5NextMLP(config)
            self.input_layernorm = nn.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_attention_layernorm = nn.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

        def __call__(self, x, mask=None, cache=None):
            x = x + self.self_attn(self.input_layernorm(x), mask, cache)
            return x + self.mlp(self.post_attention_layernorm(x))

    def cache_factory(config):
        kpool = int(getattr(config, "index_kpool", 4) or 4)

        def factory():
            return CacheList(KVCache(), PoolingCache(kpool))

        return factory

    return {
        "args_cls": TextConfig,
        "layer_cls": Glm5NextMTPDecoderLayer,
        "cache_factory": cache_factory,
        "return_array_mask": True,
        "rewrite_mla_kv_b": False,
        # glm5 checkpoints are VLM-wrapped: patch language_model, not model.
        # The remaining accessors then resolve on the patched language_model
        # itself (default .model / .lm_head / .model.embed_tokens shape).
        "mtp_target": lambda model: model.language_model,
        # DSA attention wants the [KVCache, PoolingCache] pair; the mask only
        # needs the KV half for its offset.
        "mask_cache": lambda cache: cache[0]
        if isinstance(cache, CacheList)
        else cache,
        # Weight dialect: language_model.mtp.{i}.{enorm,hnorm,eh_proj,norm,block.*}
        "embedded_mtp_prefix": "language_model.mtp.",
    }
