"""Pluggable text corpus sources for Qwen-latent target building."""

from lfm.qwen_targets.corpora.base import CorpusSource, CorpusText
from lfm.qwen_targets.corpora.hf_streaming import HFStreamingCorpusSource
from lfm.qwen_targets.corpora.jsonl import JSONLCorpusSource, PlainTextCorpusSource
from lfm.qwen_targets.corpora.mixed import MixedCorpusLoader

__all__ = [
    "CorpusSource",
    "CorpusText",
    "HFStreamingCorpusSource",
    "JSONLCorpusSource",
    "PlainTextCorpusSource",
    "MixedCorpusLoader",
]
