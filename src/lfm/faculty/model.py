"""LanguageFaculty — central compositor for the LFM pipeline.

The ``LanguageFaculty`` wires the generator (pretrained multilingual VAE
decoder) into a single coherent forward pass.  Agent embeddings are
projected into the VAE latent space and decoded into linguistically
structured output.
"""

from __future__ import annotations

import importlib
from collections import OrderedDict
from typing import Any

import torch
from torch import Tensor, nn

from lfm._registry import create
from lfm._types import AgentState, Mask, TokenEmbeddings, TokenIds
from lfm.core.module import LFMModule
from lfm.faculty.config import FacultyConfig

# Concrete module files that must be imported so their @register decorators
# fire before we call ``create()``.
_CONCRETE_MODULES: list[str] = [
    "lfm.generator.multilingual_vae",
    "lfm.data.loaders.leipzig",
]

_registry_loaded = False


def _ensure_registry() -> None:
    """Import concrete module files to populate the global registry."""
    global _registry_loaded  # noqa: PLW0603
    if _registry_loaded:
        return
    for mod in _CONCRETE_MODULES:
        importlib.import_module(mod)
    _registry_loaded = True


class LanguageFaculty(nn.Module):
    """Compositor that orchestrates the LFM generator pipeline.

    The pipeline projects agent embeddings through a pretrained VAE
    generator that produces linguistically structured output::

        agent_state → generator (project → z → frozen decoder → IPA tokens)

    The generator is optional (set ``config.generator = None`` to disable).
    When disabled, the agent state passes through as a single-step embedding.

    Args:
        config: A ``FacultyConfig`` with generator configuration.
    """

    def __init__(self, config: FacultyConfig) -> None:
        super().__init__()
        self.config = config

        _ensure_registry()

        # -- Generator (pretrained VAE decoder) --
        self.generator: LFMModule | None = None
        if config.generator is not None:
            self.generator = create(
                "generator", config.generator.name, config.generator
            )

        # -- Projection layers for dimension mismatches --
        self.projections = nn.ModuleDict(self._build_projections())

    # ------------------------------------------------------------------
    # Projection builder
    # ------------------------------------------------------------------

    def _build_projections(self) -> OrderedDict[str, nn.Linear]:
        """Create ``nn.Linear`` layers to bridge dimension mismatches."""
        projections: OrderedDict[str, nn.Linear] = OrderedDict()
        dim = self.config.dim

        # Pre-tokenized external input → faculty dim.
        if self.config.pretokenized_dim is not None:
            if self.config.pretokenized_dim != dim:
                projections["pretokenized_to_faculty"] = nn.Linear(
                    self.config.pretokenized_dim, dim
                )

        # Generator decoder_hidden_dim → faculty dim.
        if self.generator is not None:
            g_cfg = self.config.generator
            assert g_cfg is not None
            if g_cfg.decoder_hidden_dim != dim:
                projections["generator_to_faculty"] = nn.Linear(
                    g_cfg.decoder_hidden_dim, dim
                )

        return projections

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        agent_state: AgentState | None = None,
        mask: Mask | None = None,
        *,
        tokens: TokenIds | None = None,
        embeddings: TokenEmbeddings | None = None,
    ) -> dict[str, Any]:
        """Run the LFM pipeline.

        Accepts either a raw ``agent_state`` or pre-tokenized
        ``tokens`` + ``embeddings``.

        Args:
            agent_state: Continuous agent state ``(batch, dim)``.
            mask: Optional boolean padding mask ``(batch, seq_len)``.
            tokens: Pre-tokenized token indices (keyword-only).
            embeddings: Pre-tokenized dense embeddings (keyword-only).

        Returns:
            Namespaced output dictionary with generator outputs and
            ``"extra_losses"``.
        """
        outputs: dict[str, Any] = {}

        has_pretokenized = tokens is not None or embeddings is not None
        has_agent_state = agent_state is not None

        if has_pretokenized and has_agent_state:
            raise ValueError(
                "Provide agent_state OR (tokens, embeddings), not both."
            )
        if not has_pretokenized and not has_agent_state:
            raise ValueError(
                "Must provide agent_state or both tokens and embeddings."
            )

        if has_pretokenized:
            if tokens is None or embeddings is None:
                raise ValueError(
                    "Pre-tokenized path requires both tokens and embeddings."
                )
            outputs["pretokenized.tokens"] = tokens
            outputs["pretokenized.embeddings"] = embeddings

            if "pretokenized_to_faculty" in self.projections:
                embeddings = self.projections["pretokenized_to_faculty"](
                    embeddings
                )
            elif embeddings.size(-1) != self.config.dim:
                raise ValueError(
                    f"Pre-tokenized embeddings dim {embeddings.size(-1)} "
                    f"!= faculty dim {self.config.dim} and no "
                    f"pretokenized_dim configured."
                )

            if mask is None:
                mask = torch.ones(
                    tokens.size(0),
                    tokens.size(1),
                    dtype=torch.bool,
                    device=tokens.device,
                )
        else:
            assert agent_state is not None
            embeddings = agent_state.unsqueeze(1)  # (batch, 1, dim)
            if mask is None:
                mask = torch.ones(
                    agent_state.size(0),
                    1,
                    dtype=torch.bool,
                    device=agent_state.device,
                )

        # ---- Generator ----
        if self.generator is not None and embeddings is not None:
            assert mask is not None
            g_out = self.generator(embeddings, mask)
            self._merge(outputs, self.generator.output_prefix, g_out)

            embeddings = g_out["embeddings"]
            if (
                "generator_to_faculty" in self.projections
                and embeddings.size(-1)
                == self.config.generator.decoder_hidden_dim
            ):
                embeddings = self.projections["generator_to_faculty"](
                    embeddings
                )

            tokens = g_out["tokens"]
            mask = g_out["mask"]

        # ---- Extra losses ----
        outputs["extra_losses"] = self.extra_losses()

        return outputs

    # ------------------------------------------------------------------
    # Aliases / helpers
    # ------------------------------------------------------------------

    def encode(
        self,
        agent_state: AgentState | None = None,
        mask: Mask | None = None,
        *,
        tokens: TokenIds | None = None,
        embeddings: TokenEmbeddings | None = None,
    ) -> dict[str, Any]:
        """Alias for ``forward``."""
        return self.forward(
            agent_state, mask=mask, tokens=tokens, embeddings=embeddings
        )

    def extra_losses(self) -> dict[str, Tensor]:
        """Aggregate ``extra_losses`` from the generator."""
        losses: dict[str, Tensor] = {}
        if self.generator is not None:
            for k, v in self.generator.extra_losses().items():
                losses[f"{self.generator.output_prefix}.{k}"] = v
        return losses

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _merge(
        target: dict[str, Tensor],
        prefix: str,
        source: dict[str, Tensor],
    ) -> None:
        """Namespace-merge *source* into *target*."""
        for k, v in source.items():
            target[f"{prefix}.{k}"] = v
