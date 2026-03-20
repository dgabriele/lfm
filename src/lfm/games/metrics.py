"""Game-specific evaluation metrics for LFM emergent communication.

Provides metrics for evaluating performance and communication quality in
reconstruction and referential games:

- ``AttributeAccuracy``: per-attribute reconstruction accuracy.
- ``RelationAccuracy``: binary relation prediction accuracy.
- ``ReferentialAccuracy``: top-1 accuracy of receiver target identification.
- ``MessageEntropy``: entropy of message token distributions.
- ``MessageUniqueness``: fraction of unique messages in a batch.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lfm.metrics.base import Metric


class AttributeAccuracy(Metric):
    """Per-attribute reconstruction accuracy (fraction correct).

    Reads per-attribute logits from the output dictionary, takes argmax
    predictions, and compares against ground-truth attribute indices.
    Returns the mean accuracy across all attribute dimensions and objects.

    Expected output keys:
        - ``"game.attr_logits.{d}"`` for each attribute dimension ``d``:
          ``(batch, N, C_d)`` float logits.
        - ``"game.object_attrs"``: ``(batch, N, D)`` int64 ground-truth
          attribute indices (may also be in a targets sub-dict merged
          into outputs).
    """

    def __init__(self) -> None:
        super().__init__("attribute_accuracy")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Compute mean attribute accuracy across all dimensions.

        Args:
            outputs: Combined output dictionary containing predicted
                logits and ground-truth attributes.

        Returns:
            Scalar accuracy in [0, 1].
        """
        if "game.object_attrs" not in outputs:
            return 0.0

        gt_attrs = outputs["game.object_attrs"]  # (batch, N, D)
        num_attrs = gt_attrs.shape[-1]

        correct_total = 0
        count_total = 0

        for d in range(num_attrs):
            key = f"game.attr_logits.{d}"
            if key not in outputs:
                break

            logits = outputs[key]  # (batch, N, C_d)
            preds = logits.argmax(dim=-1)  # (batch, N)
            targets = gt_attrs[:, :, d]  # (batch, N)

            correct_total += (preds == targets).sum().item()
            count_total += targets.numel()

        if count_total == 0:
            return 0.0

        return correct_total / count_total


class RelationAccuracy(Metric):
    """Relation prediction accuracy (threshold at 0.5).

    Applies a sigmoid to relation logits, thresholds at 0.5, and computes
    accuracy against ground-truth binary relation labels.

    Expected output keys:
        - ``"game.relation_logits"``: ``(batch, N, N, R)`` float logits.
        - ``"game.relations"``: ``(batch, N, N, R)`` float32 binary labels.
    """

    def __init__(self) -> None:
        super().__init__("relation_accuracy")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Compute binary relation prediction accuracy.

        Args:
            outputs: Combined output dictionary containing predicted
                relation logits and ground-truth relation labels.

        Returns:
            Scalar accuracy in [0, 1].
        """
        if "game.relation_logits" not in outputs or "game.relations" not in outputs:
            return 0.0

        logits = outputs["game.relation_logits"]  # (batch, N, N, R)
        targets = outputs["game.relations"]  # (batch, N, N, R)

        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct = (preds == targets).float().mean().item()

        return correct


class ReferentialAccuracy(Metric):
    """Top-1 accuracy of receiver choosing the correct target.

    Compares the receiver's argmax choice against the ground-truth
    target index.

    Expected output keys:
        - ``"game.receiver_logits"``: ``(batch, K)`` float logits.
        - ``"game.target_idx"``: ``(batch,)`` int64 target index.
    """

    def __init__(self) -> None:
        super().__init__("referential_accuracy")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Compute top-1 referential accuracy.

        Args:
            outputs: Combined output dictionary containing receiver
                logits and target indices.

        Returns:
            Scalar accuracy in [0, 1].
        """
        if "game.receiver_logits" not in outputs or "game.target_idx" not in outputs:
            return 0.0

        logits = outputs["game.receiver_logits"]  # (batch, K)
        target_idx = outputs["game.target_idx"]  # (batch,)

        preds = logits.argmax(dim=-1)  # (batch,)
        correct = (preds == target_idx).float().mean().item()

        return correct


class MessageEntropy(Metric):
    """Entropy of message token distribution.

    Computes the average per-position entropy of the token probability
    distribution in the transmitted message.  Higher entropy indicates
    more uniform token usage; lower entropy indicates the language is
    concentrating on fewer symbols.

    Searches for channel logits or quantization tokens in the output
    dictionary to compute entropy.
    """

    def __init__(self) -> None:
        super().__init__("message_entropy")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Compute mean per-position token entropy.

        Args:
            outputs: Pipeline output dictionary containing either
                ``"channel.logits"`` or ``"quantization.tokens"``.

        Returns:
            Scalar entropy value in nats.  Returns 0.0 if no suitable
            tensor is found.
        """
        # Prefer channel logits (pre-softmax) if available
        if "channel.logits" in outputs:
            logits = outputs["channel.logits"]  # (batch, seq_len, vocab)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            # Per-position entropy: -sum(p * log(p))
            entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq_len)
            return entropy.mean().item()

        # Fall back to token distribution from quantization
        if "quantization.tokens" in outputs:
            tokens = outputs["quantization.tokens"]  # (batch, seq_len)
            # Compute empirical distribution over token values per position
            if tokens.ndim < 2:
                return 0.0
            batch, seq_len = tokens.shape
            # Compute entropy from the empirical distribution across the batch
            # for each position
            max_token = tokens.max().item() + 1
            entropy_sum = 0.0
            for pos in range(seq_len):
                counts = torch.bincount(tokens[:, pos].long(), minlength=max_token).float()
                probs = counts / counts.sum()
                probs = probs[probs > 0]
                entropy_sum += -(probs * probs.log()).sum().item()
            return entropy_sum / seq_len

        return 0.0


class MessageUniqueness(Metric):
    """Fraction of unique messages in a batch.

    Treats each message as a sequence of discrete tokens and counts how
    many distinct messages appear in the batch.  A uniqueness of 1.0 means
    every message is different; lower values indicate the language is
    producing many identical messages (low expressivity).

    Searches for ``"quantization.tokens"`` or ``"channel.message"`` in
    the output dictionary.
    """

    def __init__(self) -> None:
        super().__init__("message_uniqueness")

    def compute(self, outputs: dict[str, Tensor]) -> float:
        """Compute the fraction of unique messages in a batch.

        Args:
            outputs: Pipeline output dictionary containing discrete
                token sequences.

        Returns:
            Scalar uniqueness fraction in [0, 1].  Returns 0.0 if no
            suitable tensor is found.
        """
        tokens: Tensor | None = None

        if "quantization.tokens" in outputs:
            tokens = outputs["quantization.tokens"]
        elif "channel.message" in outputs:
            msg = outputs["channel.message"]
            # If continuous, discretize by argmax along last dim
            if msg.ndim == 3:
                tokens = msg.argmax(dim=-1)
            elif msg.ndim == 2:
                tokens = msg

        if tokens is None or tokens.ndim < 2:
            return 0.0

        batch = tokens.shape[0]
        if batch == 0:
            return 0.0

        # Count unique rows
        unique_messages = torch.unique(tokens, dim=0)
        return unique_messages.shape[0] / batch
