"""GLM-5.3 (glm5_next) native backend facade.

Autoregressive serving rides the in-tree ``mtplx.models.glm5_next`` wrapper
around mlx-vlm's implementation. The speculative path loads through the shared
``mtplx.runtime`` contract gate; MTP execution is wired through
``generation.py`` plus ``mtplx/glm_mtp_patch.py`` (the port of oMLX's
``glm5_next_vlm_runtime`` — MTP block attach, ``mtp_forward``,
``make_mtp_cache``, KDA replay-rollback, PoolingCache undo).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import DraftTokens, ModelState, MTPBackend, VerifyOutput
from mtplx.profiles import DEFAULT_PROFILE_NAME


class Glm5NextMTPBackend(MTPBackend):
    arch_id = "glm5-next-mtp"

    def load(self, model_path: Path) -> ModelState:
        from mtplx.mtp_patch import MTPContract
        from mtplx.runtime import load

        runtime = load(model_path, mtp=True, contract=MTPContract())
        return ModelState(
            model_path=Path(model_path),
            runtime=runtime,
            metadata={"arch_id": self.arch_id, "contract_gated": True},
        )

    def verify(self, state: ModelState, draft_tokens: DraftTokens, hidden: Any) -> VerifyOutput:
        raise NotImplementedError("Glm5NextMTPBackend.verify is wired through generation.py")

    def propose(self, state: ModelState, hidden: Any) -> DraftTokens:
        raise NotImplementedError("Glm5NextMTPBackend.propose is wired through generation.py")

    def recommended_profile(self) -> str:
        return DEFAULT_PROFILE_NAME

    def health(self) -> dict[str, Any]:
        return {
            "arch_id": self.arch_id,
            "runtime_path": "mtplx.runtime + mtplx.glm_mtp_patch + mtplx.generation",
            "support_level": "experimental-native-contract-gated",
            "contract_required": True,
            "requires": "mlx-vlm>=0.7,<0.8",
            "supported_model_types": ["glm5_next", "glm5_next_text"],
            "references": [
                "REFERENCES:mlx-vlm/mlx_vlm/models/glm5_next/language.py",
                "REFERENCES:oMLX/omlx/patches/mlx_vlm_mtp/glm5_next_vlm_runtime.py",
            ],
        }
