"""Channel subsystem for the LFM framework.

Provides the abstract ``Channel`` base class and ``ChannelConfig``
for encoding and decoding messages through a communication channel.
"""

from __future__ import annotations

from lfm.channel.base import Channel
from lfm.channel.config import ChannelConfig

__all__ = ["Channel", "ChannelConfig"]
