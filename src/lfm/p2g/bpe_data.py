"""Dataset + collate for BPE-output p2g.

Reads (ipa: str, token_ids: vlen-int32) pairs from HDF5 and pads to fixed
tensors.  The IPA side reuses CharVocab; the output side already has
integer ids assigned at dataset-build time.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

from .vocab import PAD_ID, CharVocab


class P2GBPEDataset(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        ipa_vocab: CharVocab,
        max_ipa_len: int,
        max_bpe_len: int,
    ) -> None:
        with h5py.File(h5_path, "r") as h:
            g = h["pairs"]
            self.ipa = [s.decode() if isinstance(s, bytes) else s for s in g["ipa"][:]]
            self.tok_seqs = [list(map(int, t)) for t in g["token_ids"][:]]
            self.spelling = [
                s.decode() if isinstance(s, bytes) else s for s in g["spelling"][:]
            ]
        self.ipa_vocab = ipa_vocab
        self.max_ipa_len = max_ipa_len
        self.max_bpe_len = max_bpe_len

    def __len__(self) -> int:
        return len(self.ipa)

    def __getitem__(self, idx: int) -> dict:
        ipa_ids = self.ipa_vocab.encode(self.ipa[idx])[: self.max_ipa_len]
        bpe_ids = self.tok_seqs[idx][: self.max_bpe_len]
        return {
            "ipa_ids": ipa_ids,
            "bpe_ids": bpe_ids,
            "bpe_len": len(bpe_ids),
            "ipa_text": self.ipa[idx],
            "spelling_text": self.spelling[idx],
        }


def collate_bpe(
    batch: list[dict], max_ipa_len: int, max_bpe_len: int,
) -> dict:
    B = len(batch)
    ipa = torch.full((B, max_ipa_len), PAD_ID, dtype=torch.long)
    bpe = torch.full((B, max_bpe_len), PAD_ID, dtype=torch.long)
    lens = torch.zeros(B, dtype=torch.long)
    for i, row in enumerate(batch):
        ipa_ids = row["ipa_ids"]
        bpe_ids = row["bpe_ids"]
        ipa[i, : len(ipa_ids)] = torch.tensor(ipa_ids, dtype=torch.long)
        bpe[i, : len(bpe_ids)] = torch.tensor(bpe_ids, dtype=torch.long)
        lens[i] = row["bpe_len"]
    return {
        "ipa_ids": ipa,
        "bpe_ids": bpe,
        "bpe_lens": lens,
        "ipa_text": [r["ipa_text"] for r in batch],
        "spelling_text": [r["spelling_text"] for r in batch],
    }


def build_ipa_vocab_from_bpe_h5(train_h5: str | Path) -> CharVocab:
    """Char vocab over IPA strings only (BPE side already has int ids)."""
    with h5py.File(train_h5, "r") as h:
        ipa_texts = [
            s.decode() if isinstance(s, bytes) else s
            for s in h["pairs/ipa"][:]
        ]
    return CharVocab.build(ipa_texts)
