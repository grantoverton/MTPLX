"""Runtime MTP injection for GLM-4 MoE-family MLX models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .artifacts import expected_mtp_file, text_config

logger = logging.getLogger(__name__)

GLM_MTP_MODEL_TYPES = {
    "glm4_moe",
    "glm4_moe_lite",
    "glm5_next",
    "glm5_next_text",
}


def _num_mtp_layers(config: dict[str, Any]) -> int:
    tcfg = text_config(config)
    return int(
        tcfg.get("num_nextn_predict_layers")
        or tcfg.get("mtp_num_hidden_layers")
        or config.get("num_nextn_predict_layers")
        or 0
    )


def _model_type(config: dict[str, Any]) -> str:
    tcfg = text_config(config)
    return str(tcfg.get("model_type") or config.get("model_type") or "").lower()


def is_glm_mtp_config(config: dict[str, Any]) -> bool:
    return _model_type(config) in GLM_MTP_MODEL_TYPES and _num_mtp_layers(config) > 0


def _glm_impl(config: dict[str, Any]) -> dict[str, Any]:
    model_type = _model_type(config)
    if model_type in {"glm5_next", "glm5_next_text"}:
        from mlx_lm.models.cache import CacheList, KVCache, PoolingCache

        from mtplx.models.glm5_next import mtp_impl

        impl = mtp_impl()
        tcfg = text_config(config)
        kpool = int(tcfg.get("index_kpool", 4) or 4)
        impl["cache_factory"] = lambda: CacheList(KVCache(), PoolingCache(kpool))
        return impl
    if model_type == "glm4_moe_lite":
        from mlx_lm.models import glm4_moe_lite as impl
        from mlx_lm.models.cache import KVCache

        return {
            "module": impl,
            "args_cls": impl.ModelArgs,
            "layer_cls": impl.Glm4MoeLiteDecoderLayer,
            "cache_factory": KVCache,
            "return_array_mask": True,
            "rewrite_mla_kv_b": True,
        }

    from mlx_lm.models import glm4_moe as impl
    from mlx_lm.models.cache import KVCache

    return {
        "module": impl,
        "args_cls": impl.ModelArgs,
        "layer_cls": impl.DecoderLayer,
        "cache_factory": KVCache,
        "return_array_mask": False,
        "rewrite_mla_kv_b": False,
    }


def _load_weight_file(path: Path) -> dict[str, Any]:
    import mlx.core as mx

    if path.suffix == ".json":
        return {}
    return dict(mx.load(str(path)))


def _candidate_weight_files(model_path: Path, config: dict[str, Any]) -> list[Path]:
    mtp_file = expected_mtp_file(model_path, config)
    if mtp_file.exists():
        files = [mtp_file]
        # A configured sidecar carries the head block only; the shared
        # lm_head that feeds shared_head_head still lives in the trunk
        # shards, so the draft output projection needs those files too.
        index_path = model_path / "model.safetensors.index.json"
        if index_path.exists():
            try:
                weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
            except Exception:
                weight_map = {}
            for rel in sorted({
                rel for key, rel in weight_map.items()
                if str(key).endswith("lm_head.weight")
            }):
                shard = model_path / rel
                if shard not in files:
                    files.append(shard)
        return files

    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
        except Exception:
            weight_map = {}
        start = int(text_config(config).get("num_hidden_layers") or config.get("num_hidden_layers") or 0)
        count = _num_mtp_layers(config)
        wanted_prefixes = tuple(f"model.layers.{start + i}." for i in range(count))
        selected = {
            model_path / rel
            for key, rel in weight_map.items()
            if str(key).startswith(wanted_prefixes)
        }
        if not selected:
            # Embedded dialect: ``language_model.mtp.*`` / ``mtp.*`` tensors live
            # inside the trunk shards (glm5_next, DeepSeek-V3 style sidecars).
            selected = {
                model_path / rel
                for key, rel in weight_map.items()
                if ".mtp." in str(key) or str(key).startswith("mtp.")
            }
            # The shared lm_head feeds the MTP block's shared_head_head.
            selected.update(
                model_path / rel
                for key, rel in weight_map.items()
                if str(key).endswith("lm_head.weight")
            )
        if selected:
            return sorted(selected)

    return sorted(model_path.glob("model*.safetensors"))


def _rewrite_kv_b_projection(weights: dict[str, Any], prefix: str, args: Any) -> None:
    import mlx.core as mx

    weight_key = f"{prefix}.self_attn.kv_b_proj.weight"
    if weight_key not in weights:
        return

    quantized = f"{prefix}.self_attn.kv_b_proj.scales" in weights
    v = weights.pop(weight_key)
    head_dim = int(args.qk_nope_head_dim) + int(args.v_head_dim)

    bits = None
    group_size = None
    if quantized:
        dims = int(args.kv_lora_rank)
        scales = weights.pop(f"{prefix}.self_attn.kv_b_proj.scales")
        biases = weights.pop(f"{prefix}.self_attn.kv_b_proj.biases")
        bits = (int(v.shape[-1]) * 32) // dims
        group_size = dims // int(scales.shape[-1])
        v = mx.dequantize(v, scales, biases, bits=bits, group_size=group_size)

    num_heads = int(args.num_attention_heads)
    v = v.reshape(num_heads, head_dim, -1)
    wk = mx.contiguous(v[:, : int(args.qk_nope_head_dim), :].swapaxes(-1, -2))
    wv = mx.contiguous(v[:, int(args.qk_nope_head_dim) :, :])

    if quantized:
        wk, wk_scales, wk_biases = mx.quantize(wk, bits=bits, group_size=group_size)
        wv, wv_scales, wv_biases = mx.quantize(wv, bits=bits, group_size=group_size)
        weights[f"{prefix}.self_attn.embed_q.scales"] = wk_scales
        weights[f"{prefix}.self_attn.embed_q.biases"] = wk_biases
        weights[f"{prefix}.self_attn.unembed_out.scales"] = wv_scales
        weights[f"{prefix}.self_attn.unembed_out.biases"] = wv_biases

    weights[f"{prefix}.self_attn.embed_q.weight"] = wk
    weights[f"{prefix}.self_attn.unembed_out.weight"] = wv


def _stack_moe_experts(weights: dict[str, Any], prefix: str, args: Any) -> None:
    import mlx.core as mx

    n_routed = int(getattr(args, "n_routed_experts", 0) or 0)
    if n_routed <= 0:
        return
    for module in ("gate_proj", "down_proj", "up_proj"):
        for leaf in ("weight", "scales", "biases"):
            first = f"{prefix}.mlp.experts.0.{module}.{leaf}"
            if first not in weights:
                continue
            values = [
                weights.pop(f"{prefix}.mlp.experts.{idx}.{module}.{leaf}")
                for idx in range(n_routed)
            ]
            weights[f"{prefix}.mlp.switch_mlp.{module}.{leaf}"] = mx.stack(values)


def _rewrite_glm_mtp_weights(
    raw: dict[str, Any],
    *,
    args: Any,
    start_layer: int,
    num_mtp_layers: int,
    rewrite_mla_kv_b: bool,
) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    shared_lm_head: dict[str, Any] = {}
    for key, value in raw.items():
        key = key.removeprefix("language_model.")
        if "rotary_emb.inv_freq" in key:
            continue
        if key in {"lm_head.weight", "lm_head.scales", "lm_head.biases"}:
            shared_lm_head[key.removeprefix("lm_head.")] = value
            continue
        if key.startswith("mtp."):
            # Embedded dialect: mtp.{i}.{enorm,hnorm,eh_proj,norm,block.*} ->
            # layers.{i}.{enorm,hnorm,eh_proj} / mtp_block / shared_head_norm.
            rest = key.removeprefix("mtp.")
            head, _, suffix = rest.partition(".")
            if head.isdigit() and suffix:
                local_prefix = f"layers.{head}"
                if suffix.startswith("block."):
                    mapped[f"{local_prefix}.mtp_block.{suffix.removeprefix('block.')}"] = value
                elif suffix.startswith(("enorm.", "hnorm.", "eh_proj.")):
                    mapped[f"{local_prefix}.{suffix}"] = value
                elif suffix.startswith("norm."):
                    mapped[f"{local_prefix}.shared_head_norm.{suffix.removeprefix('norm.')}"] = value
                else:
                    mapped[f"{local_prefix}.{suffix}"] = value
            else:
                mapped[rest] = value
            continue
        if key.startswith("layers."):
            mapped[key] = value
            continue
        for local_idx in range(num_mtp_layers):
            spec_idx = start_layer + local_idx
            prefix = f"model.layers.{spec_idx}."
            if not key.startswith(prefix):
                continue
            suffix = key.removeprefix(prefix)
            local_prefix = f"layers.{local_idx}"
            if suffix.startswith("shared_head.norm."):
                mapped[f"{local_prefix}.shared_head_norm.{suffix.removeprefix('shared_head.norm.')}"] = value
            elif suffix.startswith("shared_head.head."):
                mapped[f"{local_prefix}.shared_head_head.{suffix.removeprefix('shared_head.head.')}"] = value
            elif suffix.startswith(("enorm.", "hnorm.", "eh_proj.")):
                mapped[f"{local_prefix}.{suffix}"] = value
            elif suffix.startswith("embed_tokens."):
                # GLM MTP shares the target embedding; the runtime reuses it.
                pass
            else:
                mapped[f"{local_prefix}.mtp_block.{suffix}"] = value
            break

    for local_idx in range(num_mtp_layers):
        block_prefix = f"layers.{local_idx}.mtp_block"
        if rewrite_mla_kv_b:
            _rewrite_kv_b_projection(mapped, block_prefix, args)
        _stack_moe_experts(mapped, block_prefix, args)
        if f"layers.{local_idx}.shared_head_head.weight" not in mapped:
            for leaf, value in shared_lm_head.items():
                mapped[f"layers.{local_idx}.shared_head_head.{leaf}"] = value

    if not _has_complete_glm_mtp_payload(mapped, num_mtp_layers=num_mtp_layers):
        return {}

    return mapped


def _has_complete_glm_mtp_payload(
    weights: dict[str, Any],
    *,
    num_mtp_layers: int,
) -> bool:
    """Return true only when every declared GLM MTP layer has real layer weights."""

    for local_idx in range(num_mtp_layers):
        prefix = f"layers.{local_idx}."
        required = (
            f"{prefix}enorm.weight",
            f"{prefix}hnorm.weight",
            f"{prefix}eh_proj.weight",
        )
        if not all(key in weights for key in required):
            return False
        if not any(key.startswith(f"{prefix}mtp_block.") for key in weights):
            return False
    return True


def _quantize_for_loaded_weights(mtp: Any, config: dict[str, Any], weights: dict[str, Any]) -> None:
    import mlx.nn as nn

    quantization = config.get("quantization") or config.get("quantization_config") or {}
    if not quantization:
        return
    if "group_size" not in quantization or "bits" not in quantization:
        return

    tcfg = text_config(config)
    per_module = (
        tcfg.get("quantization")
        or config.get("quantization")
        or config.get("quantization_config")
        or {}
    )

    def dict_key_for(path: str) -> str | None:
        """Module path -> quantization-dict key across the known dialects."""
        # layers.{i}.mtp_block.X -> language_model.mtp.{i}.block.X (glm5 embedded)
        # layers.{i}.{enorm,hnorm,eh_proj,shared_head_norm} -> language_model.mtp.{i}.X
        parts = path.split(".")
        if parts[:1] == ["layers"] and parts[1].isdigit():
            idx = parts[1]
            tail = ".".join(parts[2:])
            if tail.startswith("mtp_block."):
                return f"language_model.mtp.{idx}.block.{tail.removeprefix('mtp_block.')}"
            if tail.startswith("shared_head_head"):
                return "language_model.lm_head"
            return f"language_model.mtp.{idx}.{tail}"
        return path

    def spec_for(path: str):
        scales_key = f"{path}.scales"
        if scales_key not in weights:
            return None
        spec = per_module.get(dict_key_for(path)) or {}
        bits = int(spec.get("bits", quantization["bits"]))
        group_size = spec.get("group_size")
        if group_size is None:
            # Derive gs from tensor geometry: in = packed_cols * 32 / bits.
            packed = weights.get(f"{path}.weight")
            scales = weights[scales_key]
            if packed is None or not scales.shape[-1]:
                group_size = int(quantization["group_size"])
            else:
                in_features = int(packed.shape[-1]) * 32 // bits
                derived = in_features // int(scales.shape[-1])
                group_size = (
                    derived
                    if in_features % int(scales.shape[-1]) == 0
                    else int(quantization["group_size"])
                )
        return {
            "group_size": int(group_size),
            "bits": bits,
            "mode": spec.get("mode", quantization.get("mode", "affine")),
        }

    def class_predicate(path: str, module: Any):
        if not hasattr(module, "to_quantized"):
            return False
        return spec_for(path) or False

    nn.quantize(
        mtp,
        group_size=int(quantization["group_size"]),
        bits=int(quantization["bits"]),
        mode=quantization.get("mode", "affine"),
        class_predicate=class_predicate,
    )


def _make_glm_mtp_module(config: dict[str, Any], args: Any):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.base import create_attention_mask

    impl = _glm_impl(config)
    layer_cls = impl["layer_cls"]
    return_array_mask = bool(impl["return_array_mask"])
    start_layer = int(getattr(args, "num_hidden_layers"))
    num_mtp_layers = _num_mtp_layers(config)

    class _GLMMTPLayer(nn.Module):
        def __init__(self, layer_idx: int):
            super().__init__()
            self.enorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.hnorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.eh_proj = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
            self.mtp_block = layer_cls(args, layer_idx=layer_idx)
            self.shared_head_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.shared_head_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

        def __call__(self, input_ids, previous_hidden_states, *, embed_tokens, cache=None):
            inputs_embeds = embed_tokens(input_ids)
            mixed = self.eh_proj(
                mx.concatenate(
                    [self.enorm(inputs_embeds), self.hnorm(previous_hidden_states)],
                    axis=-1,
                )
            )
            mask_cache = cache
            selector = impl.get("mask_cache")
            if selector is not None:
                mask_cache = selector(cache)
            mask = create_attention_mask(mixed, mask_cache, return_array=return_array_mask)
            hidden = self.mtp_block(mixed, mask=mask, cache=cache)
            logits = self.shared_head_head(self.shared_head_norm(hidden))
            return logits, hidden

    class _GLMMTP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [_GLMMTPLayer(start_layer + idx) for idx in range(num_mtp_layers)]
            self.start_layer = start_layer
            self.num_mtp_layers = num_mtp_layers

    return _GLMMTP()


def inject_glm_mtp_support(
    model: Any,
    model_path: Path | str,
    config: dict[str, Any],
    contract: Any | None = None,
) -> bool:
    """Attach GLM-4 MoE-family native MTP support to a loaded mlx-lm model."""
    import mlx.core as mx

    if not is_glm_mtp_config(config):
        return False

    model_path = Path(model_path)
    tcfg = text_config(config)
    impl = _glm_impl(config)
    target_model = impl.get("mtp_target", lambda m: m)(model)
    trunk_getter = impl.get("trunk", lambda m: m.model)
    lm_head_getter = impl.get("lm_head", lambda m: m.lm_head)
    embed_getter = impl.get("embed_tokens", lambda m: m.model.embed_tokens)
    args = getattr(target_model, "args", None) or getattr(model, "args", None)
    if args is None:
        args = impl["args_cls"].from_dict(tcfg)

    mtp = _make_glm_mtp_module(config, args)
    raw_weights: dict[str, Any] = {}
    for file in _candidate_weight_files(model_path, config):
        raw_weights.update(_load_weight_file(file))

    mapped = _rewrite_glm_mtp_weights(
        raw_weights,
        args=args,
        start_layer=int(getattr(args, "num_hidden_layers")),
        num_mtp_layers=_num_mtp_layers(config),
        rewrite_mla_kv_b=bool(impl["rewrite_mla_kv_b"]),
    )
    if not mapped:
        logger.warning("[GLM MTP inject] No GLM MTP weights found in %s", model_path)
        return False

    _quantize_for_loaded_weights(mtp, config, mapped)
    mtp.load_weights(list(mapped.items()), strict=False)
    mx.eval(mtp.parameters())

    cache_factory = impl["cache_factory"]
    original_outer_class = target_model.__class__

    class _MTPLXGLMModel(original_outer_class):
        def __call__(
            self,
            inputs,
            cache=None,
            return_hidden: bool = False,
            input_embeddings=None,
            hidden_variant: str | None = None,
            **kwargs,
        ):
            if input_embeddings is not None:
                raise ValueError("GLM MTP backend does not support input_embeddings")
            hidden = trunk_getter(self)(inputs, cache)
            logits = lm_head_getter(self)(hidden)
            if not return_hidden:
                return logits
            return logits, hidden

        def mtp_forward(
            self,
            hidden_states,
            next_token_ids,
            cache=None,
            mtp_cache=None,
            concat_order=None,
            return_hidden: bool = False,
            mtp_hidden_variant: str = "post_norm",
            position_offset: int | None = None,
            mtp_depth: int | None = None,
        ):
            if concat_order not in {None, "embedding_hidden"}:
                raise ValueError("GLM MTP backend supports embedding_hidden concat order only")
            depth = 0 if mtp_depth is None else max(int(mtp_depth) - 1, 0)
            depth %= len(self.mtp.layers)
            layer_cache = None
            if mtp_cache is not None:
                layer_cache = mtp_cache[depth] if isinstance(mtp_cache, list) else mtp_cache
            logits, hidden = self.mtp.layers[depth](
                next_token_ids,
                hidden_states,
                embed_tokens=embed_getter(self),
                cache=layer_cache,
            )
            if not return_hidden:
                return logits
            return logits, hidden

        def mtp_update_cache(
            self,
            hidden_states,
            next_token_ids,
            mtp_cache=None,
            concat_order=None,
            position_offset: int | None = None,
            mtp_depth: int | None = None,
            mtp_hidden_variant: str | None = None,
            input_embeddings=None,
        ):
            _logits, hidden = self.mtp_forward(
                hidden_states,
                next_token_ids,
                mtp_cache=mtp_cache,
                concat_order=concat_order,
                return_hidden=True,
                mtp_depth=mtp_depth,
            )
            return hidden

        def make_mtp_cache(self):
            return [cache_factory() for _ in self.mtp.layers]

        def make_cache(self):
            make_cache = getattr(super(), "make_cache", None)
            if callable(make_cache):
                return make_cache()
            layers = getattr(getattr(self, "model", None), "layers", ())
            return [cache_factory() for _ in layers]

    target_model.mtp = mtp
    target_model.__class__ = _MTPLXGLMModel

    if target_model is not model:
        # VLM-wrapped trunk (glm5_next): the engine drives the outer model, so
        # it needs the MTP surface too. Delegate to the patched language_model
        # and forward MTP kwargs through __call__ (return_hidden et al.).
        outer_class = model.__class__

        class _MTPLXGLMOuterFacade(outer_class):
            def __call__(
                self,
                inputs=None,
                cache=None,
                inputs_embeds=None,
                mask=None,
                return_hidden: bool = False,
                **kwargs,
            ):
                return self.language_model(
                    inputs,
                    inputs_embeds=inputs_embeds,
                    cache=cache,
                    return_hidden=return_hidden,
                    **kwargs,
                )

            def mtp_forward(self, *a, **k):
                return self.language_model.mtp_forward(*a, **k)

            def mtp_update_cache(self, *a, **k):
                return self.language_model.mtp_update_cache(*a, **k)

            def make_mtp_cache(self):
                return self.language_model.make_mtp_cache()

        model.__class__ = _MTPLXGLMOuterFacade
    logger.info("[GLM MTP inject] Loaded %d tensors from %s", len(mapped), model_path)
    return True
