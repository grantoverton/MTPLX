"""GLM-5.3 (``glm5_next``) in-tree model classes for MTPLX.

No ``glm5_next`` implementation exists in the pinned mlx-lm (0.31.3). The
architecture lives in mlx-vlm upstream (>=0.7,<0.8) but *that* implementation
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

One load-time adaptation happens in ``Model.sanitize``:

* ``language_model.mtp.*`` — the nextn MTP block's tensors have no module to
  land on in the AR path and are dropped from the trunk load; the Phase-2
  attach (``glm_mtp_patch.inject_glm_mtp_support``) reloads them from disk
  through ``_candidate_weight_files``.
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


_MODEL_CLASSES_CACHE: tuple[type, type] | None = None


def model_classes() -> tuple[type, type]:
    """Return (Model, ModelArgs) for the vendored glm5_next tree."""
    global _MODEL_CLASSES_CACHE
    if _MODEL_CLASSES_CACHE is not None:
        return _MODEL_CLASSES_CACHE
    _install_pooling_cache()

    from mtplx.vendor.glm5_omlx.glm5_next.config import (
        ModelConfig,
        TextConfig,
        VisionConfig,
    )
    from mtplx.vendor.glm5_omlx.glm5_next.glm5_next import Model as _VLMModel

    class Model(_VLMModel):
        def verify_capture_scope(self):
            return self.language_model.model.verify_capture_scope()

        def commit_verified_window(
            self, cache, snapshot_states, *, keep_tokens, verified_tokens
        ):
            return self.language_model.model.commit_verified_window(
                cache,
                snapshot_states,
                keep_tokens=keep_tokens,
                verified_tokens=verified_tokens,
            )

        def sanitize(self, weights: Dict[str, Any]) -> Dict[str, Any]:
            remapped = {
                key: value
                for key, value in weights.items()
                if not key.startswith("language_model.mtp.")
            }
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

    _MODEL_CLASSES_CACHE = (Model, ModelArgs)
    return _MODEL_CLASSES_CACHE


# ``mtplx.runtime._model_classes_for_config`` calls the registered loader,
# which lands here.


def mtp_impl(config: Dict[str, Any] | None = None):
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

    class GLMOffsetCacheList(CacheList):
        """CacheList exposing the engine's offset/rollback surface.

        ``generation._mtp_cache_offset`` / ``_rollback_mtp_cache`` read
        ``mtp_cache[0].offset`` and call ``trim(n)``, and the session bank's
        ``_trim_cache_ref_to_*`` helpers gate restores on ``entry.offset``.
        A plain CacheList has ``trim`` (delegated to both halves) but no
        ``offset``, so recorded base offsets stayed 0, rejected draft
        positions were never rolled back, and restored prefix caches kept
        tokens the engine then replayed on top of committed KV. ``offset``
        reports the KV half's token count (PoolingCache's ``offset`` is a
        compressed-row count, not tokens). ``trim`` deliberately keeps
        base ``CacheList`` semantics (every child trims ``n``, last result
        returned) — measured against the alternative: an atomic
        pool-first trim leaves ``KV.offset`` un-decremented whenever the
        pool refuses (its undo log stops at a >8-token update), and
        ``KVCache.meta_state`` is empty so ``rollback_after_verify``'s
        ``restore_cache`` cannot heal the stranded KV span — every later
        forward then reads the rejected verify tokens' positions and
        draft acceptance collapses at depth. Under base ``trim`` the KV
        half still decrements while the pool half's meta is restored, and
        restore callers (``_trim_cache_to_offset``,
        ``_trim_cache_ref_to_*``) already decline on a short count, so a
        pool refusal rejects the restore candidate instead of serving a
        desynced pair. Used for both the MTP draft cache and — via
        ``glm_mtp_patch.make_cache`` — the vendored trunk pair.
        """

        @property
        def offset(self):
            return self.caches[0].offset

        def is_trimmable(self):
            # Pinned AND-semantics: the pair reports trimmable only when both
            # halves can trim, independent of upstream CacheList convention.
            return self.caches[0].is_trimmable() and self.caches[1].is_trimmable()

        def can_trim(self, n: int) -> bool:
            # n-token form: a pair can roll a verify window back only when
            # BOTH halves cover it — the pool's undo log is the binding
            # constraint at window boundaries.
            for child in self.caches:
                child_can_trim = getattr(child, "can_trim", None)
                if callable(child_can_trim):
                    if not child_can_trim(n):
                        return False
                elif not child.is_trimmable():
                    return False
            return True

        def reserve_indexer_capacity(self, *args, **kwargs):
            # Decline marker honored by qsa_mtp_outer_device_core_supported:
            # like the QSA cache it was named for, the pooling half keeps
            # Python-side state (_pool_len, remainder, the undo log) that an
            # mx.compile state replay freezes at trace time — silently
            # desyncing pool and KV. Opting out keeps --draft-core device /
            # device-d2 honest: the lane declines instead of corrupting
            # committed history.
            return None

    tcfg = (config or {}).get("text_config", config or {})
    kpool = int(tcfg.get("index_kpool", 4) or 4)

    def cache_factory():
        return GLMOffsetCacheList(KVCache(), PoolingCache(kpool))

    return {
        "args_cls": TextConfig,
        "layer_cls": Glm5NextMTPDecoderLayer,
        "cache_factory": cache_factory,
        # Re-wraps the vendored trunk ``make_cache`` pairs (bare CacheList)
        # so session-bank restores can see and trim real token offsets.
        "cache_pair_cls": GLMOffsetCacheList,
        "return_array_mask": True,
        # BF16 exports keep the fused kv_b_proj; the MTP block's sparse
        # attention wants the split embed_q/unembed_out names. No-op on oQ
        # artifacts, which already ship the split leaves.
        "rewrite_mla_kv_b": True,
        # glm5 checkpoints are VLM-wrapped: patch language_model, not model.
        # The remaining accessors then resolve on the patched language_model
        # itself (default .model / .lm_head / .model.embed_tokens shape).
        "mtp_target": lambda model: model.language_model,
        # DSA attention wants the [KVCache, PoolingCache] pair; the mask only
        # needs the KV half for its offset.
        "mask_cache": lambda cache: cache[0]
        if isinstance(cache, CacheList)
        else cache,
    }
