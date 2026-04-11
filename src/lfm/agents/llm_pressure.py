"""LLM-pressure loss for the dialogue game.

Instead of training an agent to produce expressions in an arbitrary
phonotactic language and then trying to teach a downstream LLM what
that language means (the UNMT route), this module applies pressure
*during* agent game training so that the agent's output already
falls somewhere inside a pretrained LLM's latent distribution.  If
the agent's sequences are natively recognizable to the LLM, the
interpretation problem reduces to prompting that LLM in whatever
language we like — no fine-tuning or cross-lingual bridge needed.

The target distribution is the full aggregate distribution of the
frozen LLM (Qwen 2.5 0.5B by default).  Since Qwen was pretrained on
multilingual data, "inside its latent distribution" covers English,
Chinese, romanization, and various other forms simultaneously — we
do not specifically pressure the agent toward English.

Gradient path
-------------

The VAE decoder's SPM vocabulary is disjoint from Qwen's BPE
vocabulary.  To keep the whole pipeline differentiable, we learn a
projection matrix ``P ∈ R^{V_spm × d_qwen}`` that maps each SPM
token to a continuous embedding in Qwen's input-embedding space::

    agent output head
       → logits (V_spm)
       → Gumbel-softmax (hard=True, straight-through one-hot)
       → P   (learned bridge)
       → soft embeddings in Qwen input space
       → frozen Qwen forward (inputs_embeds)
       → Qwen next-token logits (V_qwen)
       → autoregressive NLL against target
       → scalar loss

The targets are also differentiable: for each position we compute a
"nearest Qwen token" from the soft embedding via ``lm_head`` (which
on Qwen 2.5 is tied to the input embedding matrix), then apply a
straight-through estimator so the forward uses a hard one-hot while
the backward pass sees the soft softmax gradient.  Both the input
and target sides of the autoregressive loss flow gradients back into
the projection matrix and, through the agent's output head, into the
VAE decoder hidden states that generated the logits.

Cost
----

A Qwen 2.5 0.5B forward pass at ``(B, T) = (batch × num_turns,
tokens_per_turn)`` adds ~1 GB weights plus ~500 MB of activations
in bf16.  This is significant overhead on an 8 GB consumer GPU —
expect to reduce the dialogue game's batch size or Phase 2 chunk
size to make room.  On a larger cloud GPU it is essentially free.
"""

from __future__ import annotations

import logging
from pathlib import Path

import sentencepiece as spm_lib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from lfm.translator.romanize import romanize_iso

logger = logging.getLogger(__name__)


class LLMPressureScorer(nn.Module):
    """Frozen LLM + learned projection, applied as a loss on agent logits.

    Args:
        spm_model: Loaded sentencepiece processor for the agent's
            vocabulary.  Used at construction to seed the projection
            matrix with a "romanized SPM piece → average Qwen
            embedding" lookup.
        spm_vocab_size: Total size of the agent's SPM vocabulary
            (matches ``gen._full_vocab`` in the dialogue game).
        llm_model_name: HuggingFace name of the frozen LLM.  Qwen 2.5
            0.5B is the default.
        dtype: Dtype for the frozen LLM.  bfloat16 is the default
            because Qwen 2.5 trains in bf16 and it halves VRAM.
        device: Device to place the LLM on.  Defaults to cuda if
            available.
    """

    def __init__(
        self,
        spm_model: spm_lib.SentencePieceProcessor,
        spm_vocab_size: int,
        llm_model_name: str = "Qwen/Qwen2.5-0.5B",
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        self._device = device

        logger.info("Loading frozen LLM %s (%s) for pressure loss", llm_model_name, dtype)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, torch_dtype=dtype,
        ).to(device)
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad_(False)

        # Gradient checkpointing: recomputes activations during the
        # backward pass instead of storing them, trading ~30% compute
        # for ~60% activation VRAM.  The frozen Qwen otherwise holds
        # all hidden states across 24 layers × B × T × 896 floats.
        # Requires use_cache=False (already set in forward()).
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )

        self._llm_dtype = dtype
        self._llm_vocab_size = self.llm.config.vocab_size
        d_qwen = self.llm.config.hidden_size

        logger.info(
            "Building SPM→LLM projection (V_spm=%d, d_llm=%d)",
            spm_vocab_size, d_qwen,
        )
        qwen_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        qwen_input_embeds = self.llm.get_input_embeddings().weight  # (V_llm, d_llm)

        projection = torch.zeros(spm_vocab_size, d_qwen, dtype=torch.float32)
        sp_max = spm_model.GetPieceSize()
        missing = 0
        for spm_id in range(spm_vocab_size):
            # IDs beyond the sentencepiece model's range are reserved
            # generator-side specials (EOS, BOS, ...).  They get a
            # small-Gaussian init so backprop has a starting point.
            if spm_id >= sp_max:
                projection[spm_id].normal_(mean=0.0, std=1.0 / (d_qwen ** 0.5))
                continue
            piece = spm_model.IdToPiece(spm_id)
            text = piece.replace("▁", " ").strip()
            if not text:
                missing += 1
                continue
            try:
                romanized = romanize_iso(text).strip()
            except Exception:
                romanized = text
            if not romanized:
                missing += 1
                continue
            qwen_ids = qwen_tokenizer.encode(romanized, add_special_tokens=False)
            if not qwen_ids:
                missing += 1
                continue
            embeds = qwen_input_embeds[qwen_ids].float().cpu()
            projection[spm_id] = embeds.mean(dim=0)
        logger.info(
            "Projection init: %d / %d SPM pieces matched Qwen tokens "
            "(%d generator-specials randomly initialized)",
            sp_max - missing, sp_max,
            max(0, spm_vocab_size - sp_max),
        )

        # Parameter so the matrix is updated by the trainer's optimizer.
        # Kept in fp32 for optimizer stability; cast to Qwen dtype at use.
        self.projection = nn.Parameter(projection.to(device))

    @property
    def llm_vocab_size(self) -> int:
        return self._llm_vocab_size

    def _soft_embeddings(
        self, agent_logits: torch.Tensor, tau: float,
    ) -> torch.Tensor:
        """Map agent SPM logits to soft Qwen-space embeddings.

        Uses Gumbel-softmax with ``hard=True`` so the forward pass
        sees a true one-hot while backward gets the soft-gradient
        straight-through signal.
        """
        one_hot_st = F.gumbel_softmax(agent_logits, tau=tau, hard=True, dim=-1)
        soft_emb = one_hot_st @ self.projection.to(agent_logits.dtype)
        return soft_emb

    def _target_st(self, soft_emb: torch.Tensor) -> torch.Tensor:
        """Non-detached straight-through target distribution over Qwen vocab.

        For each position, the target is the Qwen token id that has
        the highest dot-product with that position's projected
        embedding — i.e., the "nearest" Qwen token in the tied
        embedding space.  The straight-through estimator returns a
        one-hot in the forward pass but lets soft softmax gradients
        flow in the backward pass, so the agent receives gradient at
        every position (including the final token of the sequence,
        which the pure-detached formulation misses).
        """
        # lm_head output = soft_emb @ embed_tokens.weight.T  (tied weights on Qwen 2.5).
        sim_logits = self.llm.lm_head(soft_emb)  # (B, T, V_llm)
        soft_probs = F.softmax(sim_logits.float(), dim=-1)
        hard_ids = sim_logits.argmax(dim=-1)
        hard_one_hot = F.one_hot(hard_ids, num_classes=sim_logits.size(-1)).float()
        # Straight-through: forward uses hard_one_hot, backward sees soft_probs gradient.
        return hard_one_hot - soft_probs.detach() + soft_probs

    def _loss_chunk(
        self,
        agent_logits_chunk: torch.Tensor,
        mask_chunk: torch.Tensor,
        tau: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute LLM pressure loss for one sub-batch.

        Returns ``(loss_sum, token_count)`` — both summed, not
        averaged — so caller can combine chunks with a single
        division at the end.
        """
        soft_emb = self._soft_embeddings(
            agent_logits_chunk.to(self._llm_dtype), tau=tau,
        )
        attention_mask = mask_chunk.to(self._device, dtype=torch.long)

        qwen_out = self.llm(
            inputs_embeds=soft_emb,
            attention_mask=attention_mask,
            use_cache=False,
        )
        pred_logits = qwen_out.logits  # (B, T, V_llm)

        target_st = self._target_st(soft_emb)  # (B, T, V_llm)

        shift_pred_log_probs = F.log_softmax(
            pred_logits[:, :-1].float(), dim=-1,
        )  # (B, T-1, V_llm)
        shift_target = target_st[:, 1:]
        shift_mask = attention_mask[:, 1:].float()

        per_position_loss = -(shift_target * shift_pred_log_probs).sum(dim=-1)
        loss_sum = (per_position_loss * shift_mask).sum()
        token_count = shift_mask.sum()
        return loss_sum, token_count

    def forward(
        self,
        agent_logits: torch.Tensor,
        mask: torch.Tensor,
        tau: float = 1.0,
        chunk_size: int = 8,
    ) -> torch.Tensor:
        """Compute the LLM-pressure loss for one batch of agent logits.

        Processes the batch in chunks of ``chunk_size`` rows so the
        frozen Qwen forward pass fits in the available VRAM even at
        large effective batch sizes.  Chunking is along the batch
        dimension; the token axis is always passed whole.

        Args:
            agent_logits: ``(B, T, V_spm)`` raw logits from the VAE
                decoder's output head.  Must require grad.
            mask: ``(B, T)`` attention mask with ``1`` for real tokens
                and ``0`` for padding.
            tau: Gumbel-softmax temperature.
            chunk_size: Number of rows per Qwen forward pass.

        Returns:
            Scalar cross-entropy loss (lower = more Qwen-plausible).
        """
        batch = agent_logits.size(0)
        if chunk_size <= 0 or chunk_size >= batch:
            loss_sum, token_count = self._loss_chunk(agent_logits, mask, tau)
            return loss_sum / token_count.clamp(min=1.0)

        # Wrap each chunk in torch.utils.checkpoint so activations
        # drop after each forward and are recomputed during backward.
        # Peak memory is bounded to one chunk's worth of activations,
        # independent of the outer batch size.
        def _chunk_fn(logits_chunk, mask_chunk):
            return self._loss_chunk(logits_chunk, mask_chunk, tau)

        total_loss = torch.zeros((), device=agent_logits.device, dtype=torch.float32)
        total_tokens = torch.zeros((), device=agent_logits.device, dtype=torch.float32)
        for start in range(0, batch, chunk_size):
            end = min(start + chunk_size, batch)
            loss_sum, token_count = checkpoint(
                _chunk_fn,
                agent_logits[start:end],
                mask[start:end],
                use_reentrant=False,
            )
            total_loss = total_loss + loss_sum
            total_tokens = total_tokens + token_count
        return total_loss / total_tokens.clamp(min=1.0)
