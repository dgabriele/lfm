"""Referential game V2 — minimal single-expression game.

Hidden state CE loss only. No KL, no diversity, no length reg.
z-gen → decode → Phase 2 → encode → CE score.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lfm.agents.components import MessageEncoder, Receiver, ZDiversityLoss
from lfm.agents.config import CurriculumConfig, MessageEncoderConfig
from lfm.agents.decode import rerun_decoder_multiphrase_with_grad
from lfm.agents.diffusion import DiffusionZGenerator
from lfm.config.base import LFMBaseConfig
from lfm.faculty.config import FacultyConfig
from lfm.faculty.model import LanguageFaculty
from lfm.generator.config import GeneratorConfig


class ReferentialV2Config(LFMBaseConfig):
    embedding_dim: int = 384
    decoder_path: str = "data/vae_decoder.pt"
    spm_path: str = "data/spm.model"
    num_memory_tokens: int = 8
    max_output_len: int = 109

    z_hidden_dim: int = 512
    max_phrases: int = 3
    diffusion_steps: int = 4
    diffusion_layers: int = 4
    diffusion_heads: int = 8

    encoder: MessageEncoderConfig = MessageEncoderConfig()

    num_distractors: int = 15
    embedding_store_dir: str = "data/embeddings"

    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    steps: int = 2000
    gru_lr: float = 1e-4
    receiver_lr: float = 3e-4
    max_grad_norm: float = 1.0
    curriculum: CurriculumConfig = CurriculumConfig()

    checkpoint_every: int = 100
    log_every: int = 20
    output_dir: str = "data/refv2_game"
    device: str = "cuda"
    seed: int = 42

    def build_faculty_config(self) -> FacultyConfig:
        return FacultyConfig(
            dim=self.embedding_dim,
            generator=GeneratorConfig(
                pretrained_decoder_path=self.decoder_path,
                spm_model_path=self.spm_path,
                freeze_decoder=True,
                max_output_len=self.max_output_len,
                num_statements=1,
                num_memory_tokens=self.num_memory_tokens,
            ),
        )


class ReferentialV2(nn.Module):

    def __init__(self, config: ReferentialV2Config, faculty: LanguageFaculty):
        super().__init__()
        self.config = config
        self.faculty = faculty
        gen = faculty.generator
        gen.eval()
        device = next(gen.parameters()).device
        with torch.no_grad():
            faculty(torch.randn(1, config.embedding_dim, device=device))

        self.z_gen = DiffusionZGenerator(
            input_dim=config.embedding_dim,
            latent_dim=gen._latent_dim,
            d_model=config.z_hidden_dim,
            max_phrases=config.max_phrases,
            num_steps=config.diffusion_steps,
            num_layers=config.diffusion_layers,
            num_heads=config.diffusion_heads,
            variable_phrases=False,
            z_mean=gen._z_mean if gen._z_stats_initialized else None,
            z_std=gen._z_std if gen._z_stats_initialized else None,
        )
        self._max_tokens_per_phrase = 48
        self.encoder = MessageEncoder(
            gen.config.decoder_hidden_dim, config.embedding_dim,
            num_heads=config.encoder.num_heads,
            num_layers=config.encoder.num_layers,
        )
        self.receiver = Receiver(config.embedding_dim)
        if gen._z_stats_initialized:
            self.z_diversity = ZDiversityLoss(gen._z_mean, gen._z_std)
        else:
            self.z_diversity = None

    @property
    def gen(self):
        return self.faculty.generator

    def checkpoint_state(self):
        return {
            "z_gen": self.z_gen.state_dict(),
            "encoder": self.encoder.state_dict(),
            "receiver": self.receiver.state_dict(),
        }

    def load_checkpoint_state(self, ckpt):
        self.z_gen.load_state_dict(ckpt["z_gen"])
        self.encoder.load_state_dict(ckpt["encoder"])
        self.receiver.load_state_dict(ckpt["receiver"])

    def trainable_param_groups(self):
        c = self.config
        return [
            {"params": list(self.z_gen.parameters()), "lr": c.gru_lr},
            {"params": list(self.encoder.parameters()), "lr": c.receiver_lr},
            {"params": list(self.receiver.parameters()), "lr": c.receiver_lr},
        ]

    @torch.no_grad()
    def _decode(self, z_seq, z_weights):
        """Phase 1: KV-cached multi-phrase decode — copied from ExpressionGame."""
        from lfm.generator.layers import PhraseDecoder, multiscale_causal_mask, precompute_rope_freqs

        gen = self.gen
        batch, K, _ = z_seq.shape
        device = z_seq.device
        max_tok = self._max_tokens_per_phrase * K

        weighted_z = z_weights.unsqueeze(-1) * z_seq
        n_mem = gen._num_memory_tokens
        hdim = gen.config.decoder_hidden_dim
        memories = gen.latent_to_decoder(
            weighted_z.reshape(batch * K, -1),
        ).reshape(batch, K, n_mem, hdim)

        decoder = gen.decoder
        is_ld = isinstance(decoder, PhraseDecoder)

        if gen._full_causal_mask is None or gen._full_causal_mask.size(1) < max_tok + 1:
            gen._full_causal_mask = multiscale_causal_mask(
                max_tok + 1, gen.config.decoder_num_heads,
                gen.config.attention_head_windows,
                gen.config.attention_global_every, device,
            )

        tokens = torch.zeros(batch, max_tok, dtype=torch.long, device=device)
        mask = torch.zeros(batch, max_tok, dtype=torch.bool, device=device)
        bounds = torch.zeros(batch, K, dtype=torch.long, device=device)

        cur_phr = torch.zeros(batch, dtype=torch.long, device=device)
        tok_in_phr = torch.zeros(batch, dtype=torch.long, device=device)
        pos = torch.zeros(batch, dtype=torch.long, device=device)
        active = z_weights > 0.01
        done = ~active[:, 0]
        idx = torch.arange(batch, device=device)

        rope = gen._rope_freqs
        if rope is not None and rope.size(0) < max_tok + 1:
            rope = precompute_rope_freqs(
                hdim // gen.config.decoder_num_heads, max_tok + 1, device=device,
            )

        if is_ld:
            kv = decoder.make_kv_cache(batch, max_tok + 1, device, torch.float16)

        emb = gen.token_embedding(
            torch.full((batch, 1), gen.bos_id, dtype=torch.long, device=device),
        )

        def mem():
            i = cur_phr.clamp(max=K - 1)
            m = memories[idx, i]
            a = active[idx, i] & ~done
            return m * a.unsqueeze(-1).unsqueeze(-1).float()

        if is_ld:
            row = gen._full_causal_mask[:, 0:1, 0:1]
            out = decoder.forward_cached(emb, mem(), kv, rope_freqs=rope, tgt_mask_row=row)
            kv.advance()
        else:
            out = decoder(emb, mem())

        for t in range(max_tok):
            logits = gen.output_head(out[:, -1])
            nxt = logits.argmax(dim=-1)

            live = ~done
            tokens[idx, pos] = nxt * live.long()
            mask[idx, pos] = live
            pos += live.long()
            tok_in_phr += live.long()

            eos = (nxt == gen.eos_id) & (tok_in_phr >= 1)
            hit_max = tok_in_phr >= self._max_tokens_per_phrase
            switch = (eos | hit_max) & live

            cur_phr += switch.long()
            tok_in_phr *= ~switch

            sv = switch & (cur_phr < K)
            if sv.any():
                bounds[idx[sv], cur_phr[sv]] = pos[sv]

            clamped = cur_phr.clamp(max=K - 1)
            done = done | (cur_phr >= K) | (switch & ~active[idx, clamped])

            if done.all():
                break

            new_emb = gen.token_embedding(nxt.unsqueeze(1))
            if is_ld:
                sl = kv.seq_len + 1
                row = gen._full_causal_mask[:, kv.seq_len:kv.seq_len + 1, :sl]
                out = decoder.forward_cached(new_emb, mem(), kv, rope_freqs=rope, tgt_mask_row=row)
                kv.advance()
            else:
                out = decoder(new_emb, mem())

        return tokens, mask, bounds

    def forward(self, anchor, distractors, *, step=0, candidate_indices=None):
        batch = anchor.size(0)
        device = anchor.device

        z_seq, z_weights, num_phrases = self.z_gen(anchor)
        tokens, gen_mask, bounds = self._decode(z_seq, z_weights)
        hidden = rerun_decoder_multiphrase_with_grad(
            self.gen, z_seq, z_weights, tokens, gen_mask, bounds,
        )
        mask = gen_mask[:, :hidden.size(1)]
        message = self.encoder(hidden, mask)

        num_cand = distractors.size(1) + 1
        candidates = torch.cat([anchor.unsqueeze(1), distractors], dim=1)
        perm = torch.stack([torch.randperm(num_cand, device=device) for _ in range(batch)])
        candidates = torch.gather(candidates, 1, perm.unsqueeze(-1).expand_as(candidates))
        target_idx = (perm == 0).long().argmax(dim=1)

        logits = self.receiver(message, candidates)
        loss = F.cross_entropy(logits, target_idx)

        # Z diversity: push z-vectors apart
        if self.z_diversity is not None:
            div_loss, _ = self.z_diversity(z_seq, z_weights)
            loss = loss + 0.5 * div_loss

        with torch.no_grad():
            acc = (logits.argmax(1) == target_idx).float().mean()
            total_tok = mask.float().sum(dim=1).mean()

        return {
            "loss": loss,
            "accuracy": acc,
            "msg_lengths": total_tok.detach(),
            "num_phrases": num_phrases.mean().detach(),
            "hs_weight": torch.tensor(1.0),
            "_tokens": tokens.detach(),
            "_gen_mask": gen_mask.detach(),
        }
