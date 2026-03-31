"""Agent communication games for the LFM framework.

Provides embedding-based agent games that train communication through
the frozen linguistic bottleneck.  Each game type lives in
``agents.games`` with its own config and game class.  Shared components
(message encoder, receiver, trainer) live at the ``agents`` level.
"""

from __future__ import annotations

from lfm.agents.components import MessageEncoder, Receiver
from lfm.agents.games.expression import (
    ExpressionGame,
    ExpressionGameConfig,
    ZSequenceGenerator,
)
from lfm.agents.games.referential import ReferentialGame, ReferentialGameConfig
from lfm.agents.trainer import AgentTrainer

__all__ = [
    "AgentTrainer",
    "ExpressionGame",
    "ExpressionGameConfig",
    "MessageEncoder",
    "Receiver",
    "ReferentialGame",
    "ReferentialGameConfig",
    "ZSequenceGenerator",
]
