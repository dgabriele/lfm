"""Deterministic IPA→spelling seq2seq baseline (no z-bottleneck).

Encoder-decoder transformer with cross-attention.  Same data, same
char vocabs, same training loop as P2GVAE, so the comparison isolates
the effect of the VAE bottleneck.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .vocab import BOS_ID, EOS_ID, PAD_ID


@dataclass
class P2GSeq2SeqConfig:
    input_vocab_size: int
    output_vocab_size: int
    d_model: int = 256
    encoder_layers: int = 3
    decoder_layers: int = 3
    nhead: int = 4
    max_ipa_len: int = 24
    max_spelling_len: int = 24
    dropout: float = 0.1
    label_smoothing: float = 0.0


class P2GSeq2Seq(nn.Module):
    def __init__(self, cfg: P2GSeq2SeqConfig):
        super().__init__()
        self.cfg = cfg
        self.src_emb = nn.Embedding(
            cfg.input_vocab_size, cfg.d_model, padding_idx=PAD_ID,
        )
        self.tgt_emb = nn.Embedding(
            cfg.output_vocab_size, cfg.d_model, padding_idx=PAD_ID,
        )
        self.src_pos = nn.Embedding(cfg.max_ipa_len, cfg.d_model)
        self.tgt_pos = nn.Embedding(cfg.max_spelling_len + 2, cfg.d_model)
        self.transformer = nn.Transformer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_encoder_layers=cfg.encoder_layers,
            num_decoder_layers=cfg.decoder_layers,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.to_logits = nn.Linear(cfg.d_model, cfg.output_vocab_size)
        self.max_ipa_len = cfg.max_ipa_len
        self.max_spell = cfg.max_spelling_len

    def _encode(self, ipa_ids: Tensor) -> tuple[Tensor, Tensor]:
        T = ipa_ids.size(1)
        pos = torch.arange(T, device=ipa_ids.device).clamp_max(self.max_ipa_len - 1)
        h = self.src_emb(ipa_ids) + self.src_pos(pos)[None]
        pad_mask = ipa_ids == PAD_ID
        memory = self.transformer.encoder(h, src_key_padding_mask=pad_mask)
        return memory, pad_mask

    def _decode_logits(
        self,
        memory: Tensor,
        memory_pad_mask: Tensor,
        decoder_input_ids: Tensor,
    ) -> Tensor:
        T = decoder_input_ids.size(1)
        pos = torch.arange(T, device=decoder_input_ids.device)
        tgt = self.tgt_emb(decoder_input_ids) + self.tgt_pos(pos)[None]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=tgt.device,
        )
        tgt_pad_mask = decoder_input_ids == PAD_ID
        h = self.transformer.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
            tgt_is_causal=True,
        )
        return self.to_logits(h)

    def forward(
        self,
        ipa_ids: Tensor,
        spelling_ids: Tensor,
        spelling_lens: Tensor,
        step: int,  # unused, signature-match P2GVAE
        loss_weight: Tensor | None = None,
    ) -> dict[str, Tensor]:
        memory, src_pad = self._encode(ipa_ids)

        B, L = spelling_ids.shape
        bos = torch.full((B, 1), BOS_ID, dtype=torch.long, device=ipa_ids.device)
        dec_in = torch.cat([bos, spelling_ids], dim=1)            # (B, 1+L)
        # Targets: shift left, place EOS at length index, PAD elsewhere.
        eos_scatter = torch.full_like(spelling_ids, PAD_ID)
        idx = spelling_lens.clamp_max(L - 1).unsqueeze(1)
        eos_scatter.scatter_(1, idx, EOS_ID)
        tgt = torch.where(spelling_ids == PAD_ID, eos_scatter, spelling_ids)

        logits = self._decode_logits(memory, src_pad, dec_in)     # (B, 1+L, V)
        logits = logits[:, :L, :]                                 # align with tgt
        V = logits.size(-1)
        if loss_weight is not None:
            per_token = F.cross_entropy(
                logits.reshape(-1, V), tgt.reshape(-1),
                ignore_index=PAD_ID, reduction="none",
                label_smoothing=self.cfg.label_smoothing,
            ).reshape(B, L)
            mask = (tgt != PAD_ID).float()
            per_sample = (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            recon = (per_sample * loss_weight.to(per_sample.device)).mean()
        else:
            recon = F.cross_entropy(
                logits.reshape(-1, V), tgt.reshape(-1),
                ignore_index=PAD_ID, reduction="mean",
                label_smoothing=self.cfg.label_smoothing,
            )
        with torch.no_grad():
            pred = logits.argmax(-1)
            mask = tgt != PAD_ID
            char_acc = ((pred == tgt) & mask).float().sum() / mask.float().sum().clamp_min(1)
            per_sample_correct = ((pred == tgt) | ~mask).all(dim=-1)
            word_acc = per_sample_correct.float().mean()

        zero = torch.zeros((), device=recon.device)
        return {
            "loss": recon,
            "recon": recon.detach(),
            "word_acc": word_acc.detach(),
            "length_loss": zero,
            "kl": zero,
            "kl_weight": zero,
            "char_acc": char_acc,
            "len_acc": zero,
            "mu": torch.zeros(B, 1, device=recon.device),
            "logvar": torch.zeros(B, 1, device=recon.device),
        }

    @torch.no_grad()
    def generate(self, ipa_ids: Tensor, sample: bool = False) -> list[list[int]]:
        memory, src_pad = self._encode(ipa_ids)
        B = ipa_ids.size(0)
        device = ipa_ids.device
        cur = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        outs: list[list[int]] = [[] for _ in range(B)]
        for _ in range(self.max_spell):
            logits = self._decode_logits(memory, src_pad, cur)[:, -1, :]
            nxt = logits.argmax(-1)
            for i in range(B):
                if finished[i]:
                    continue
                tok = int(nxt[i].item())
                if tok == EOS_ID:
                    finished[i] = True
                else:
                    outs[i].append(tok)
            if finished.all():
                break
            cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
        return outs

    @torch.no_grad()
    def decode_from_z(self, z: Tensor) -> list[list[int]]:
        # No latent space — return empty; here only for interface compat.
        raise NotImplementedError("seq2seq baseline has no z-bottleneck")
