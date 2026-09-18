"""mlx-lm convert lane for glm5_next sources.

``mlx_lm convert`` resolves ``config["model_type"]`` to
``mlx_lm.models.<model_type>``; upstream mlx-lm has no ``glm5_next``, so a
GLM-5.3 BF16 source fails with "Model type glm5_next not supported". Forge
runs this driver instead: it pre-registers the vendored implementation
under that module name, then forwards the argv to ``mlx_lm.cli.main``
unchanged (``python -m mtplx.commands.forge_glm5_convert convert ...``).

The appended-layer MTP head (``model.language_model.layers.N.*``) is not
part of the vendored trunk module, so body conversion drops it the same
way ``sanitize`` drops foreign keys; Forge repacks it into ``mtp.safetensors``
from the source afterwards.
"""

from __future__ import annotations

import sys
import types


def _register_vendored_architectures() -> None:
    from mtplx.models.glm5_next import model_classes

    model, model_args = model_classes()
    for model_type in ("glm5_next", "glm5_next_text"):
        module = types.ModuleType(f"mlx_lm.models.{model_type}")
        module.Model = model
        module.ModelArgs = model_args
        sys.modules[module.__name__] = module


def main() -> None:
    _register_vendored_architectures()
    from mlx_lm.cli import main as mlx_lm_main

    mlx_lm_main()


if __name__ == "__main__":
    main()
