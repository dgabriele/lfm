"""Procedural scene generation for LFM games.

Provides :class:`SceneGenerator`, a stateless (no learnable parameters) utility
that produces batches of relational scenes as GPU tensors using pure PyTorch
operations.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lfm.games.config import SceneConfig


class SceneGenerator:
    """Generates batches of relational scenes as GPU tensors.

    Each scene consists of a fixed number of objects, each described by a
    vector of categorical attributes, plus a set of directed binary relations
    between every ordered pair of distinct objects.

    This class holds no learnable parameters and is not a :class:`torch.nn.Module`.
    All generation is performed via batched PyTorch operations on the target
    device for maximum throughput.

    Parameters:
        config: Scene configuration controlling the number of objects,
            attributes, relations, and sampling parameters.
        device: Torch device (or device string) on which to allocate tensors.
    """

    def __init__(self, config: SceneConfig, device: torch.device | str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)

    def generate(self, batch_size: int) -> dict[str, Tensor]:
        """Generate a batch of random scenes.

        Returns:
            A dictionary with the following entries:

            ``"object_attrs"``
                Integer tensor of shape ``(batch_size, n_obj, n_attr)`` where
                *n_obj* is the number of objects and *n_attr* is the number of
                attributes.  Entry ``[b, i, d]`` is sampled uniformly from
                ``[0, cardinalities[d])``.

            ``"relations"``
                Float tensor of shape ``(batch_size, n_obj, n_obj, n_rel)``
                where *n_rel* is the number of relation types.  Each entry is
                Bernoulli-sampled with probability
                ``config.relation_density``.  The diagonal ``[b, i, i, :]`` is
                zeroed (no self-relations).  If
                ``config.symmetric_relations`` is ``True``, the lower triangle
                mirrors the upper triangle.
        """
        cfg = self.config
        n_obj = cfg.num_objects
        n_attr = cfg.num_attributes
        n_rel = cfg.num_relation_types

        # -- Object attributes -------------------------------------------------
        # Sample each attribute column independently according to its cardinality.
        attr_columns: list[Tensor] = [
            torch.randint(
                low=0,
                high=int(cfg.attribute_cardinalities[d]),
                size=(batch_size, n_obj),
                device=self.device,
            )
            for d in range(n_attr)
        ]
        # Stack along the last dimension: (batch, n_obj, n_attr)
        object_attrs = torch.stack(attr_columns, dim=-1)  # int64

        # -- Relations ----------------------------------------------------------
        probs = torch.full(
            (batch_size, n_obj, n_obj, n_rel),
            cfg.relation_density,
            device=self.device,
        )
        relations = torch.bernoulli(probs)  # float32

        # Zero out self-relations along the diagonal (i == j).
        diag_idx = torch.arange(n_obj, device=self.device)
        relations[:, diag_idx, diag_idx, :] = 0.0

        # Enforce symmetry: keep only the strict upper triangle and mirror it.
        if cfg.symmetric_relations:
            upper_mask = torch.triu(
                torch.ones(n_obj, n_obj, device=self.device, dtype=torch.bool),
                diagonal=1,
            )
            # Extract upper-triangle values and mirror to lower triangle.
            upper_vals = relations * upper_mask.unsqueeze(0).unsqueeze(-1)
            relations = upper_vals + upper_vals.transpose(1, 2)

        return {
            "object_attrs": object_attrs,
            "relations": relations,
        }

    @property
    def object_feature_dim(self) -> int:
        """Total one-hot feature dimension for a full scene encoding.

        This equals the sum of all attribute cardinalities multiplied by the
        number of objects, plus the flattened relation tensor size
        (``n_obj * n_obj * n_rel``).
        """
        cfg = self.config
        n_obj = cfg.num_objects
        n_rel = cfg.num_relation_types
        attr_dim = sum(cfg.attribute_cardinalities) * n_obj
        rel_dim = n_obj * n_obj * n_rel
        return attr_dim + rel_dim

    def generate_distractors(
        self,
        batch_size: int,
        num_distractors: int,
    ) -> dict[str, Tensor]:
        """Generate distractor scenes for the referential game.

        Each batch element gets ``num_distractors`` independently sampled
        scenes.

        Returns:
            A dictionary with the following entries:

            ``"object_attrs"``
                Integer tensor of shape
                ``(batch_size, num_distractors, n_obj, n_attr)``.

            ``"relations"``
                Float tensor of shape
                ``(batch_size, num_distractors, n_obj, n_obj, n_rel)``.
        """
        cfg = self.config
        n_obj = cfg.num_objects
        n_attr = cfg.num_attributes
        n_rel = cfg.num_relation_types

        # -- Object attributes -------------------------------------------------
        attr_columns: list[Tensor] = [
            torch.randint(
                low=0,
                high=int(cfg.attribute_cardinalities[d]),
                size=(batch_size, num_distractors, n_obj),
                device=self.device,
            )
            for d in range(n_attr)
        ]
        object_attrs = torch.stack(attr_columns, dim=-1)  # (batch, K, n_obj, n_attr)

        # -- Relations ----------------------------------------------------------
        probs = torch.full(
            (batch_size, num_distractors, n_obj, n_obj, n_rel),
            cfg.relation_density,
            device=self.device,
        )
        relations = torch.bernoulli(probs)

        # Zero self-relations along the object-pair diagonal.
        diag_idx = torch.arange(n_obj, device=self.device)
        relations[:, :, diag_idx, diag_idx, :] = 0.0

        # Enforce symmetry: keep only the strict upper triangle and mirror it.
        if cfg.symmetric_relations:
            upper_mask = torch.triu(
                torch.ones(n_obj, n_obj, device=self.device, dtype=torch.bool),
                diagonal=1,
            )
            mask_5d = upper_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            upper_vals = relations * mask_5d
            relations = upper_vals + upper_vals.transpose(2, 3)

        return {
            "object_attrs": object_attrs,
            "relations": relations,
        }
