"""Vocabulary for p2g VAE — IPA chars (input) and English chars (output).

Both vocabs are built once from the paired dataset.  Reserved ids:
    0 = <pad>
    1 = <bos>   (not used for non-AR decode; kept for optional use)
    2 = <eos>   (emitted at position L to signal end; training uses length loss)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
_SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]


@dataclass
class CharVocab:
    """Bidirectional char ↔ id map with reserved special ids at the top."""

    chars: list[str] = field(default_factory=list)
    char_to_id: dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(cls, texts: list[str]) -> "CharVocab":
        seen: set[str] = set()
        for t in texts:
            seen.update(t)
        ordered = _SPECIAL_TOKENS + sorted(seen)
        return cls(
            chars=ordered,
            char_to_id={c: i for i, c in enumerate(ordered)},
        )

    @property
    def size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[c] for c in text if c in self.char_to_id]

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        for i in ids:
            if i == EOS_ID:
                break
            if i < len(_SPECIAL_TOKENS):
                continue
            out.append(self.chars[i])
        return "".join(out)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"chars": self.chars}, ensure_ascii=False))

    @classmethod
    def from_json(cls, path: str | Path) -> "CharVocab":
        data = json.loads(Path(path).read_text())
        chars = data["chars"]
        return cls(chars=chars, char_to_id={c: i for i, c in enumerate(chars)})
