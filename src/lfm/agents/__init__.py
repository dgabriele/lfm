"""Agent communication games for the LFM framework.

Provides embedding-based agent games that train communication through
the frozen linguistic bottleneck.  Each game type lives in
``agents.games`` with its own config and game class.  Shared components
(message encoder, receiver, trainer) live at the ``agents`` level.
"""

from __future__ import annotations

from lfm.agents.components import MessageEncoder, Receiver
from lfm.agents.games.contrastive import ContrastiveGame, ContrastiveGameConfig
from lfm.agents.games.referential import ReferentialGame, ReferentialGameConfig
from lfm.agents.trainer import AgentTrainer

__all__ = [
    "AgentTrainer",
    "ContrastiveGame",
    "ContrastiveGameConfig",
    "MessageEncoder",
    "Receiver",
    "ReferentialGame",
    "ReferentialGameConfig",
]
