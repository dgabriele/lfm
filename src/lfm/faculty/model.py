"""LanguageFaculty — central compositor for the LFM pipeline.

The ``LanguageFaculty`` wires together all LFM sub-modules (quantizer,
phonology, morphology, syntax, sentence, channel) into a single coherent
forward pass.  Each sub-module is optional and instantiated from its config
via the global registry.  Dimension-mismatch projections are inserted
automatically where adjacent stages disagree on embedding size.
"""

from __future__ import annotations

import importlib
from collections import OrderedDict
from typing import Any

import torch
from torch import Tensor, nn

from lfm._registry import create
from lfm._types import AgentState, Mask
from lfm.core.module import LFMModule
from lfm.faculty.config import FacultyConfig

# Concrete module files that must be imported so their @register decorators
# fire before we call ``create()``.  We use importlib to avoid top-level
# coupling to every concrete implementation.
_CONCRETE_MODULES: list[str] = [
    "lfm.quantization.vqvae",
    "lfm.quantization.fsq",
    "lfm.quantization.lfq",
    "lfm.phonology.surface",
    "lfm.morphology.segmenter",
    "lfm.morphology.composer",
    "lfm.morphology.tree",
    "lfm.syntax.agreement",
    "lfm.syntax.attention",
    "lfm.syntax.ordering",
    "lfm.sentence.typing",
    "lfm.sentence.boundary",
    "lfm.channel.gumbel",
    "lfm.channel.straight_through",
    "lfm.channel.noisy",
    "lfm.losses.structural",
    "lfm.losses.compositionality",
    "lfm.losses.information",
    "lfm.losses.diversity",
    "lfm.losses.morphological",
]

_registry_loaded = False


def _ensure_registry() -> None:
    """Import all concrete module files to populate the global registry.

    This is idempotent — repeated calls are no-ops after the first
    successful load.
    """
    global _registry_loaded  # noqa: PLW0603
    if _registry_loaded:
        return
    for mod in _CONCRETE_MODULES:
        importlib.import_module(mod)
    _registry_loaded = True


class LanguageFaculty(nn.Module):
    """Compositor that orchestrates the full LFM pipeline.

    The pipeline runs in fixed order::

        quantizer -> phonology -> morphology -> syntax -> sentence -> channel

    Each stage is skipped when its config is ``None``.  Output tensors from
    every active stage are namespaced with the stage's ``output_prefix``
    (e.g. ``"quantization.tokens"``) and merged into one flat dictionary.

    Args:
        config: A ``FacultyConfig`` aggregating all sub-module configs.
    """

    # Ordered list of (attribute_name, registry_category, config_attr) tuples
    # that define the pipeline stages and their execution order.
    _STAGES: list[tuple[str, str]] = [
        ("quantizer", "quantizer"),
        ("phonology", "phonology"),
        ("morphology", "morphology"),
        ("syntax", "syntax"),
        ("sentence", "sentence"),
        ("channel", "channel"),
    ]

    def __init__(self, config: FacultyConfig) -> None:
        super().__init__()
        self.config = config

        _ensure_registry()

        # -- Instantiate sub-modules from configs via registry ---------------
        self.quantizer: LFMModule | None = None
        self.phonology: LFMModule | None = None
        self.morphology: LFMModule | None = None
        self.syntax: LFMModule | None = None
        self.sentence: LFMModule | None = None
        self.channel: LFMModule | None = None

        stage_configs: dict[str, Any] = {
            "quantizer": config.quantizer,
            "phonology": config.phonology,
            "morphology": config.morphology,
            "syntax": config.syntax,
            "sentence": config.sentence,
            "channel": config.channel,
        }

        for attr, category in self._STAGES:
            stage_cfg = stage_configs[attr]
            if stage_cfg is not None:
                module = create(category, stage_cfg.name, stage_cfg)
                setattr(self, attr, module)

        # -- Build projection layers for dimension mismatches ----------------
        self.projections = nn.ModuleDict(self._build_projections())

    # ------------------------------------------------------------------
    # Projection builder
    # ------------------------------------------------------------------

    def _build_projections(self) -> OrderedDict[str, nn.Linear]:
        """Create ``nn.Linear`` layers to bridge dimension mismatches.

        Returns:
            An ordered dict of named projection layers.  Keys follow the
            pattern ``"{source}_to_{target}"`` (e.g.
            ``"quantizer_to_faculty"``).
        """
        projections: OrderedDict[str, nn.Linear] = OrderedDict()
        dim = self.config.dim

        # Quantizer output dim is codebook_dim; project to faculty dim if
        # they differ.
        if self.quantizer is not None:
            q_cfg = self.config.quantizer
            assert q_cfg is not None
            if q_cfg.codebook_dim != dim:
                projections["quantizer_to_faculty"] = nn.Linear(q_cfg.codebook_dim, dim)

        # Syntax latent dim -> faculty dim (for constituent representations).
        if self.syntax is not None:
            s_cfg = self.config.syntax
            assert s_cfg is not None
            if s_cfg.latent_dim != dim:
                projections["syntax_latent_to_faculty"] = nn.Linear(s_cfg.latent_dim, dim)

        # Channel needs logits of shape (batch, seq_len, vocab_size).
        # Project from faculty dim to vocab_size.
        if self.channel is not None:
            c_cfg = self.config.channel
            assert c_cfg is not None
            vocab = self._resolve_vocab_size(c_cfg)
            if vocab is not None:
                projections["faculty_to_channel"] = nn.Linear(dim, vocab)

        return projections

    def _resolve_vocab_size(self, channel_cfg: Any) -> int | None:
        """Determine the channel vocabulary size.

        If the channel config specifies ``vocab_size`` explicitly, use that.
        Otherwise fall back to the quantizer's codebook size if available.

        Returns:
            The resolved vocabulary size, or ``None`` if it cannot be
            determined.
        """
        if channel_cfg.vocab_size is not None:
            return channel_cfg.vocab_size
        if self.quantizer is not None and self.config.quantizer is not None:
            return self.config.quantizer.codebook_size
        return None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        agent_state: AgentState,
        mask: Mask | None = None,
    ) -> dict[str, Any]:
        """Run the full LFM pipeline.

        Args:
            agent_state: Continuous agent state tensor of shape
                ``(batch, dim)``.
            mask: Optional boolean padding mask of shape
                ``(batch, seq_len)`` where ``True`` marks valid positions.
                If ``None``, a mask of all ``True`` is created after the
                quantizer produces a sequence length.

        Returns:
            A flat dictionary whose keys are namespaced as
            ``"{output_prefix}.{key}"`` for each active stage, plus an
            ``"extra_losses"`` key mapping loss names to scalar tensors.
        """
        outputs: dict[str, Any] = {}
        batch = agent_state.size(0)

        # Flowing representations that get refined at each stage.
        tokens: Tensor | None = None
        embeddings: Tensor | None = None

        # ---- Quantizer ---------------------------------------------------
        if self.quantizer is not None:
            q_out = self.quantizer(agent_state)
            self._merge(outputs, self.quantizer.output_prefix, q_out)

            tokens = q_out["tokens"]
            embeddings = q_out["embeddings"]

            # Project codebook_dim -> faculty dim if necessary.
            if "quantizer_to_faculty" in self.projections:
                embeddings = self.projections["quantizer_to_faculty"](embeddings)

            # Build a default mask if none was provided.
            if mask is None:
                seq_len = tokens.size(1)
                mask = torch.ones(batch, seq_len, dtype=torch.bool, device=agent_state.device)
        else:
            # Without a quantizer the agent_state itself is treated as a
            # single-step embedding: (batch, dim) -> (batch, 1, dim).
            embeddings = agent_state.unsqueeze(1)
            if mask is None:
                mask = torch.ones(batch, 1, dtype=torch.bool, device=agent_state.device)

        # ---- Phonology ---------------------------------------------------
        if self.phonology is not None and embeddings is not None:
            if tokens is None:
                # Without a quantizer there are no discrete tokens; create
                # dummy zero indices so the phonology forward signature is
                # satisfied.
                tokens = torch.zeros(
                    embeddings.shape[:2],
                    dtype=torch.long,
                    device=embeddings.device,
                )

            p_out = self.phonology(tokens, embeddings)
            self._merge(outputs, self.phonology.output_prefix, p_out)

            # Phonology may refine embeddings.
            if "embeddings" in p_out:
                embeddings = p_out["embeddings"]

        # ---- Morphology --------------------------------------------------
        if self.morphology is not None and embeddings is not None:
            if tokens is None:
                tokens = torch.zeros(
                    embeddings.shape[:2],
                    dtype=torch.long,
                    device=embeddings.device,
                )

            m_out = self.morphology(tokens, embeddings)
            self._merge(outputs, self.morphology.output_prefix, m_out)

            # Morphology produces recomposed embeddings.
            if "composed" in m_out:
                embeddings = m_out["composed"]

        # ---- Syntax ------------------------------------------------------
        assert mask is not None  # guaranteed by quantizer or fallback above
        if self.syntax is not None and embeddings is not None:
            gram_feats = outputs.get("morphology.grammatical_features")
            sx_out = self.syntax(embeddings, mask, grammatical_features=gram_feats)
            self._merge(outputs, self.syntax.output_prefix, sx_out)

        # ---- Sentence ----------------------------------------------------
        if self.sentence is not None and embeddings is not None:
            sn_out = self.sentence(embeddings, mask)
            self._merge(outputs, self.sentence.output_prefix, sn_out)

        # ---- Channel -----------------------------------------------------
        if self.channel is not None and embeddings is not None:
            if "faculty_to_channel" in self.projections:
                logits = self.projections["faculty_to_channel"](embeddings)
            else:
                # If no projection was created, assume embeddings dim already
                # matches vocab_size.
                logits = embeddings

            ch_out = self.channel(logits, mask)
            self._merge(outputs, self.channel.output_prefix, ch_out)

        # ---- Extra losses ------------------------------------------------
        outputs["extra_losses"] = self.extra_losses()

        return outputs

    # ------------------------------------------------------------------
    # Aliases / helpers
    # ------------------------------------------------------------------

    def encode(
        self,
        agent_state: AgentState,
        mask: Mask | None = None,
    ) -> dict[str, Any]:
        """Alias for ``forward`` — encode an agent state through the faculty.

        Args:
            agent_state: Continuous agent state tensor of shape
                ``(batch, dim)``.
            mask: Optional boolean padding mask.

        Returns:
            The same namespaced output dictionary as ``forward``.
        """
        return self.forward(agent_state, mask=mask)

    def extra_losses(self) -> dict[str, Tensor]:
        """Aggregate ``extra_losses`` from all active sub-modules.

        Returns:
            A dictionary mapping ``"{prefix}.{loss_name}"`` to scalar
            loss tensors.
        """
        losses: dict[str, Tensor] = {}
        for attr, _category in self._STAGES:
            module: LFMModule | None = getattr(self, attr, None)
            if module is not None:
                for k, v in module.extra_losses().items():
                    losses[f"{module.output_prefix}.{k}"] = v
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
        """Namespace-merge *source* into *target*.

        Each key ``k`` in *source* becomes ``"{prefix}.{k}"`` in *target*.
        """
        for k, v in source.items():
            target[f"{prefix}.{k}"] = v
