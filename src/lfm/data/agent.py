"""Agent dataset implementation.

Provides a dataset class for wrapping agent-generated representations,
enabling integrated training where the LFM pipeline learns directly
from agent internal states.
"""

from __future__ import annotations

from torch import Tensor
from torch.utils.data import Dataset

from lfm.data.config import DataConfig


class AgentDataset(Dataset[dict[str, Tensor]]):
    """Dataset wrapping agent-generated representations for integrated training.

    Stores pre-computed agent state vectors alongside any associated
    metadata, enabling the LFM pipeline to train on representations
    produced by an upstream agent system.

    Args:
        config: Data configuration specifying batch size and preprocessing
            parameters.
    """

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.max_seq_len = config.max_seq_len

        # Placeholder for agent states — populated externally
        self._states: list[dict[str, Tensor]] = []

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError("AgentDataset.__len__() not yet implemented")

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return a single agent state sample by index.

        Args:
            index: Sample index.

        Returns:
            Dictionary with agent state tensors.
        """
        raise NotImplementedError("AgentDataset.__getitem__() not yet implemented")
