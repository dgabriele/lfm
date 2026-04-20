"""Data loading for DepTreeVAE training."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import sentencepiece as spm
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from lfm.generator.dep_tree_vae.config import DEP_REL_TO_ID, DepTreeVAEConfig
from lfm.generator.dep_tree_vae.skeleton import SKEL_BOS, SKEL_EOS

logger = logging.getLogger(__name__)

PAD = 0
BOS = 1
EOS = 2


class DepTreeDataset(Dataset):
    """Dataset of dependency-annotated IPA sentences.

    Each sample is a parsed sentence from the JSONL file with:
        - Interleaved token sequence: [role_tok] ipa_tok [role_tok] ipa_tok ...
        - Role id sequence (for skeleton decoder supervision)
        - Content token ids (for bag-of-words supervision)

    Role tokens are mapped to vocab ids starting after the IPA SPM vocab
    so a single embedding table can handle both.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        sp: spm.SentencePieceProcessor,
        max_seq_len: int,
        max_samples: int | None = None,
    ) -> None:
        self.sp = sp
        self.max_seq_len = max_seq_len

        # The pretrained decoder uses: SPM native IDs (0-7999),
        # BOS = spm_size (8000), EOS = spm_size + 1 (8001).
        # No offset — SPM tokens keep their native IDs.
        spm_size = sp.get_piece_size()
        self._eos_id = spm_size + 1  # decoder EOS, not SPM's EOS
        self._bos_id = spm_size      # decoder BOS
        self._spm_offset = 0         # no offset — use native SPM IDs
        self._role_offset = spm_size + 2  # role tokens start after BOS/EOS

        self.samples: list[dict] = []
        filtered = 0

        with open(jsonl_path) as f:
            for line in f:
                if max_samples and len(self.samples) >= max_samples:
                    break
                rec = json.loads(line)
                sample = self._process(rec)
                if sample is not None:
                    self.samples.append(sample)
                else:
                    filtered += 1

        logger.info(
            "Loaded %d samples from %s (filtered %d)",
            len(self.samples), jsonl_path, filtered,
        )

    def _process(self, rec: dict) -> dict | None:
        """Convert a JSONL record to tensors."""
        ipa_words = rec["ipa"].split()
        dep_labels = rec["dep_labels"]
        dep_heads = rec["dep_heads"]

        if len(ipa_words) != len(dep_labels):
            return None

        # Tokenize each IPA word with SPM
        word_token_ids: list[list[int]] = []
        for word in ipa_words:
            ids = self.sp.encode(word, out_type=int)
            word_token_ids.append(ids)

        # Build interleaved sequence: [role] tok1 tok2 ... [role] tok1 ...
        # Each role token precedes the SPM tokens for that word.
        interleaved: list[int] = []
        role_sequence: list[int] = []

        for word_idx, (label, word_ids) in enumerate(
            zip(dep_labels, word_token_ids)
        ):
            role_id = DEP_REL_TO_ID.get(label, DEP_REL_TO_ID.get("dep", 0))
            role_tok = self._role_offset + role_id
            interleaved.append(role_tok)
            for tid in word_ids:
                interleaved.append(tid + self._spm_offset)
            role_sequence.append(role_id)

        # Add EOS (decoder convention: spm_size + 1)
        interleaved.append(self._eos_id)

        if len(interleaved) > self.max_seq_len:
            return None

        # Skeleton: BOS + role_sequence + EOS
        skeleton = [SKEL_BOS] + role_sequence + [SKEL_EOS]

        return {
            "interleaved": interleaved,
            "skeleton": skeleton,
            "n_words": len(ipa_words),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]

    @property
    def interleaved_vocab_size(self) -> int:
        """Total vocab: SPM tokens + specials + role tokens."""
        from lfm.generator.dep_tree_vae.config import NUM_DEP_RELATIONS
        return self.sp.get_piece_size() + 3 + NUM_DEP_RELATIONS


def collate_dep_tree(batch: list[dict]) -> dict[str, Tensor]:
    """Collate variable-length dep tree samples into padded tensors."""
    max_inter = max(len(s["interleaved"]) for s in batch)
    max_skel = max(len(s["skeleton"]) for s in batch)

    b = len(batch)
    tokens = torch.zeros(b, max_inter, dtype=torch.long)
    lengths = torch.zeros(b, dtype=torch.long)
    role_ids = torch.zeros(b, max_skel, dtype=torch.long)
    role_lengths = torch.zeros(b, dtype=torch.long)

    for i, s in enumerate(batch):
        inter = s["interleaved"]
        skel = s["skeleton"]
        tokens[i, : len(inter)] = torch.tensor(inter, dtype=torch.long)
        lengths[i] = len(inter)
        role_ids[i, : len(skel)] = torch.tensor(skel, dtype=torch.long)
        role_lengths[i] = len(skel)

    return {
        "tokens": tokens,
        "lengths": lengths,
        "role_ids": role_ids,
        "role_lengths": role_lengths,
    }


def build_dataloaders(
    cfg: DepTreeVAEConfig,
) -> tuple[DataLoader, DataLoader, spm.SentencePieceProcessor, int]:
    """Build train/val DataLoaders from config.

    Returns:
        train_loader, val_loader, sp (sentencepiece), vocab_size
    """
    sp = spm.SentencePieceProcessor()
    sp.load(cfg.spm_model_path)

    jsonl_path = Path(cfg.dataset_path) / "sentences.jsonl"
    dataset = DepTreeDataset(
        jsonl_path, sp, cfg.max_seq_len, max_samples=cfg.max_samples,
    )
    vocab_size = dataset.interleaved_vocab_size

    val_size = max(1, int(len(dataset) * cfg.val_fraction))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=generator,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_dep_tree,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_dep_tree,
    )

    logger.info(
        "Data: %d train, %d val, vocab=%d (spm=%d + 3 specials + %d roles)",
        train_size, val_size, vocab_size,
        sp.get_piece_size(),
        len(DEP_REL_TO_ID),
    )
    return train_loader, val_loader, sp, vocab_size
