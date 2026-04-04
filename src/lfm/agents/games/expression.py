"""Expression game with GRU z-sequence generation and PonderNet halting.

Replaces the REINFORCE tree-topology expression system with a fully
differentiable GRU that autoregressively produces z vectors.  The
frozen decoder decodes each z as a phrase (until EOS), with KV cache
persisting across boundaries for coarticulation.

Halting uses a geometric prior KL (PonderNet, Banino et al. 2021):
the learned halt distribution is regularized toward a geometric
distribution p(k) = lambda * (1-lambda)^{k-1}, naturally producing
Zipf-like expression lengths.

Two-phase forward (same as referential):
  1. GRU emits z sequence, decode phrases with no_grad (KV-cached).
  2. Re-run GRU + decoder in parallel with gradients through
     cross-attention to position-wise memory.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lfm.agents.components import (
    IPAEncoder,
    MessageEncoder,
    Receiver,
    ZDistributionLoss,
    ZDiversityLoss,
)
from lfm.agents.diffusion import DiffusionZGenerator, length_distribution_loss
from lfm.agents.refinement import RefinementDenoiser
from lfm.agents.config import CurriculumConfig, MessageEncoderConfig
from lfm.agents.decode import rerun_decoder_multiphrase_with_grad
from lfm.config.base import LFMBaseConfig
from lfm.faculty.config import FacultyConfig
from lfm.faculty.model import LanguageFaculty
from lfm.generator.config import GeneratorConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ExpressionGameConfig(LFMBaseConfig):
    """Configuration for the GRU expression game."""

    # Faculty
    embedding_dim: int = 384
    decoder_path: str = "data/vae_decoder.pt"
    spm_path: str = "data/spm.model"
    num_memory_tokens: int = 8
    max_output_len: int = 96
    vq_codebook_path: str | None = None
    vq_residual_alpha: float = 1.0

    # z-sequence generator
    z_generator: str = "gru"  # "gru" or "diffusion"
    z_hidden_dim: int = 512
    max_phrases: int = 8
    max_tokens_per_phrase: int = 48

    # Diffusion-specific (only when z_generator="diffusion")
    diffusion_steps: int = 4
    diffusion_layers: int = 4
    diffusion_heads: int = 8
    target_phrases: float = 2.5
    length_weight: float = 0.5

    # PonderNet geometric prior: lambda_p controls expected phrase count.
    # E[K] = 1/lambda_p.  lambda_p=0.5 → ~2 phrases, 0.3 → ~3.3, 0.2 → ~5.
    # Set use_halt=False to always generate all max_phrases (no halting).
    use_halt: bool = True
    lambda_p: float = 0.4
    kl_beta: float = 0.5

    # z diversity: penalize intra-expression z similarity above a data-driven
    # target (auto-computed from pretrained z distribution).  Prevents the GRU
    # from collapsing all phrases to near-identical z's while keeping them
    # close enough for linguistic coherence.
    z_diversity_weight: float = 0.0
    z_diversity_target: float | None = None
    z_distribution_weight: float = 0.0

    # Surface bottleneck: primary loss scores on token-level representations
    # (through output_head -> softmax -> token_embedding), forcing surface
    # forms to be discriminative.  Hidden-state path acts as auxiliary signal,
    # annealed from hidden_state_weight -> 0 over hidden_state_anneal_steps.
    hidden_state_weight: float = 1.0
    hidden_state_anneal_steps: int = 1000

    # Phase 2 mode: "decoder" = frozen decoder re-run (original),
    # "refinement" = lightweight diffusion denoiser (lower VRAM, bidirectional)
    phase2_mode: str = "decoder"
    refinement_layers: int = 4
    refinement_steps: int = 4

    # IPA-to-IPA receiver: score based on IPA token representations
    # instead of raw embeddings. Forces surface forms to be discriminative.
    use_ipa_receiver: bool = False
    ipa_cache_refresh: int = 0  # 0 = no refresh, N = refresh every N steps
    ipa_encoder_init_from_decoder: bool = True

    # Message encoder
    encoder: MessageEncoderConfig = MessageEncoderConfig()

    # Game
    num_distractors: int = 15
    embedding_store_dir: str = "data/embeddings"

    # Training
    batch_size: int = 256
    gradient_accumulation_steps: int = 1
    steps: int = 2000
    gru_lr: float = 1e-4
    receiver_lr: float = 3e-4
    max_grad_norm: float = 1.0
    curriculum: CurriculumConfig = CurriculumConfig()

    # Output
    checkpoint_every: int = 100
    log_every: int = 50
    output_dir: str = "data/expression_game"

    # Runtime
    device: str = "cuda"
    seed: int = 42

    def build_faculty_config(self) -> FacultyConfig:
        """Construct the ``FacultyConfig`` from game settings."""
        return FacultyConfig(
            dim=self.embedding_dim,
            generator=GeneratorConfig(
                pretrained_decoder_path=self.decoder_path,
                spm_model_path=self.spm_path,
                freeze_decoder=True,
                max_output_len=self.max_output_len,
                num_statements=1,
                vq_codebook_path=self.vq_codebook_path,
                vq_residual_alpha=self.vq_residual_alpha,
                num_memory_tokens=self.num_memory_tokens,
            ),
        )


# ---------------------------------------------------------------------------
# GRU z-sequence generator with PonderNet halting
# ---------------------------------------------------------------------------


class ZSequenceGenerator(nn.Module):
    """Autoregressive z-sequence generator with geometric-prior halting.

    z_0 is a direct projection of the input (discriminative from step 0).
    z_1..z_K are produced by a GRU conditioned on previous z's.  The
    halt distribution is regularized toward a geometric prior via KL
    divergence (PonderNet).

    Args:
        input_dim: Anchor embedding dimension.
        hidden_dim: GRU hidden dimension.
        latent_dim: Output z dimension (must match decoder).
        max_phrases: Upper bound on phrase count.
        z_mean: Pretrained z distribution mean for initialization.
        z_std: Pretrained z distribution std for initialization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        max_phrases: int = 8,
        use_halt: bool = True,
        z_mean: Tensor | None = None,
        z_std: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.max_phrases = max_phrases
        self.latent_dim = latent_dim
        self.use_halt = use_halt

        # z_0: direct projection (discriminative from step 0)
        self.z0_proj = nn.Linear(input_dim, latent_dim)

        # GRU for z_1..z_K
        self.h_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gru = nn.GRUCell(latent_dim, hidden_dim)
        self.z_proj = nn.Linear(hidden_dim, latent_dim)
        self.halt_head = nn.Linear(hidden_dim, 1)

        # Initialize gate bias to start closed (sigmoid(-3) ≈ 0.05).
        # Phrase 0 always has weight 1.0; later phrases must learn
        # to open their gates when the content demands it.
        with torch.no_grad():
            self.halt_head.bias.fill_(-3.0)

        # Scale z projections to match the pretrained z distribution
        if z_mean is not None and z_std is not None:
            with torch.no_grad():
                target_std = z_std.mean().item()
                for proj in [self.z0_proj, self.z_proj]:
                    fan_in = proj.weight.size(1)
                    current_std = proj.weight.data.std().item() * (fan_in ** 0.5)
                    scale = target_std / max(current_std, 1e-6)
                    proj.weight.data.mul_(scale)
                    proj.bias.data.copy_(z_mean)

    def forward(
        self, embedding: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Generate z sequence with per-step halt probabilities.

        Args:
            embedding: ``(batch, input_dim)`` anchor embeddings.

        Returns:
            z_sequence: ``(batch, max_phrases, latent_dim)``.
            halt_probs: ``(batch, max_phrases)`` per-step halt probs.
            z_weights: ``(batch, max_phrases)`` effective weights.
            num_phrases: ``(batch,)`` soft phrase count.
        """
        batch = embedding.size(0)
        device = embedding.device
        K = self.max_phrases

        z_seq = torch.zeros(batch, K, self.latent_dim, device=device)
        halts = torch.zeros(batch, K, device=device)
        weights = torch.ones(batch, K, device=device)

        # z_0: direct projection (discriminative from step 0)
        z = self.z0_proj(embedding)
        z_seq[:, 0] = z

        # GRU hidden state
        h = self.h_proj(embedding)

        for i in range(1, K):
            h = self.gru(z, h)
            z = self.z_proj(h)
            z_seq[:, i] = z

            if self.use_halt:
                # Independent gate: each phrase decides its own weight.
                # No cumulative product — phrase 3 can contribute
                # even if phrase 2 doesn't.
                gate = torch.sigmoid(self.halt_head(h)).squeeze(-1)
                halts[:, i] = gate
                weights[:, i] = gate

        # Soft phrase count = sum of weights
        num_phrases = weights.sum(dim=-1)

        return z_seq, halts, weights, num_phrases


def geometric_kl(halt_probs: Tensor, lambda_p: float) -> Tensor:
    """KL divergence between learned halt distribution and geometric prior.

    From PonderNet (Banino et al., "PonderNet: Learning to Ponder",
    ICML 2021 Workshop on Theoretic Foundation, Criticism, and Application
    Grounding of Deep Learning).

    The learned distribution q(k) is derived from per-step halt probs.
    The geometric prior is p(k) = lambda_p * (1 - lambda_p)^{k-1}.

    Args:
        halt_probs: ``(batch, max_phrases)`` per-step halt probabilities.
            halt_probs[:, 0] is unused (first phrase always active).
        lambda_p: Geometric prior parameter (E[K] = 1/lambda_p).

    Returns:
        Scalar KL divergence averaged over the batch.
    """
    K = halt_probs.size(1)
    eps = 1e-8

    # Build q(k): probability of halting at step k
    # q(k) = halt_probs[k] * prod(1 - halt_probs[j] for j < k)
    # For k=0: q(0) = 1 (always use first phrase... actually we model
    # halting starting from step 1)
    # Simpler: treat halt_probs[1:] as the per-step stopping probability
    q_continue = torch.ones_like(halt_probs[:, 0])
    q_halt = []

    for i in range(1, K):
        q_k = q_continue * halt_probs[:, i]
        q_halt.append(q_k)
        q_continue = q_continue * (1.0 - halt_probs[:, i])

    # Remaining probability mass (didn't halt before K)
    q_halt.append(q_continue)
    q = torch.stack(q_halt, dim=-1)  # (B, K)
    q = q.clamp(min=eps)

    # Geometric prior: p(k) = lambda * (1-lambda)^k for k=0..K-2, remainder at K-1
    p = torch.zeros(K, device=halt_probs.device)
    for k in range(K - 1):
        p[k] = lambda_p * ((1.0 - lambda_p) ** k)
    p[-1] = (1.0 - lambda_p) ** (K - 1)  # tail mass
    p = p.clamp(min=eps)
    p = p / p.sum()  # normalize

    # KL(q || p) = sum q * log(q/p)
    kl = (q * (q.log() - p.log().unsqueeze(0))).sum(dim=-1)
    return kl.mean()


# ---------------------------------------------------------------------------
# Expression game
# ---------------------------------------------------------------------------


class ExpressionGame(nn.Module):
    """Expression game with GRU z-sequence through the linguistic bottleneck.

    Args:
        config: Expression game configuration.
        faculty: Pre-built ``LanguageFaculty`` (moved to device by caller).
    """

    def __init__(
        self, config: ExpressionGameConfig, faculty: LanguageFaculty,
    ) -> None:
        super().__init__()
        self.config = config
        self.faculty = faculty

        gen = faculty.generator
        gen.eval()

        device = next(gen.parameters()).device
        with torch.no_grad():
            faculty(torch.randn(1, config.embedding_dim, device=device))

        _z_mean = gen._z_mean if gen._z_stats_initialized else None
        _z_std = gen._z_std if gen._z_stats_initialized else None

        if config.z_generator == "diffusion":
            self.z_gen = DiffusionZGenerator(
                input_dim=config.embedding_dim,
                latent_dim=gen._latent_dim,
                d_model=config.z_hidden_dim,
                max_phrases=config.max_phrases,
                num_steps=config.diffusion_steps,
                num_layers=config.diffusion_layers,
                num_heads=config.diffusion_heads,
                variable_phrases=config.use_halt,
                z_mean=_z_mean,
                z_std=_z_std,
                target_phrases=config.target_phrases,
            )
        else:
            self.z_gen = ZSequenceGenerator(
                input_dim=config.embedding_dim,
                hidden_dim=config.z_hidden_dim,
                latent_dim=gen._latent_dim,
                max_phrases=config.max_phrases,
                use_halt=config.use_halt,
                z_mean=_z_mean,
                z_std=_z_std,
            )
        hidden_dim = gen.config.decoder_hidden_dim

        # Hidden-state path (auxiliary — gradient highway)
        self.msg_encoder = MessageEncoder(
            hidden_dim, config.embedding_dim,
            num_heads=config.encoder.num_heads,
            num_layers=config.encoder.num_layers,
        )

        # Surface-token path (primary — forces discriminative surface forms)
        self.surface_encoder = MessageEncoder(
            hidden_dim, config.embedding_dim,
            num_heads=config.encoder.num_heads,
            num_layers=config.encoder.num_layers,
        )

        self.receiver = Receiver(config.embedding_dim)

        # IPA-to-IPA receiver
        if config.use_ipa_receiver:
            full_vocab = gen._full_vocab
            self.ipa_encoder = IPAEncoder(
                full_vocab, hidden_dim, config.embedding_dim,
                num_heads=config.encoder.num_heads,
                num_layers=config.encoder.num_layers,
            )
            if config.ipa_encoder_init_from_decoder:
                self.ipa_encoder.init_from_decoder(gen.token_embedding)
        else:
            self.ipa_encoder = None

        # Refinement denoiser (replaces Phase 2 decoder re-run)
        if config.phase2_mode == "refinement":
            self.refinement = RefinementDenoiser(
                hidden_dim=hidden_dim,
                num_layers=config.refinement_layers,
                num_steps=config.refinement_steps,
            )
        else:
            self.refinement = None

        # IPA cache (populated by build_ipa_cache before training)
        self._ipa_cache_tokens: Tensor | None = None
        self._ipa_cache_masks: Tensor | None = None

        if config.z_diversity_weight > 0 and gen._z_stats_initialized:
            self.z_diversity = ZDiversityLoss(
                gen._z_mean, gen._z_std,
                target=config.z_diversity_target,
            )
        else:
            self.z_diversity = None

        if config.z_distribution_weight > 0 and gen._z_stats_initialized:
            self.z_distribution = ZDistributionLoss(gen._z_mean, gen._z_std)
        else:
            self.z_distribution = None

        # Running surface diversity tracker
        self._seen_hashes: set[int] = set()
        self._seen_total: int = 0

    @property
    def gen(self):
        """Shortcut to the underlying generator."""
        return self.faculty.generator

    @torch.no_grad()
    def build_ipa_cache(self, embeddings: Tensor, batch_size: int = 64) -> None:
        """Decode all embeddings and cache token sequences for IPA receiver."""
        import logging
        logger = logging.getLogger(__name__)

        self.eval()
        total = embeddings.size(0)
        logger.info("Building IPA cache: %d embeddings, batch_size=%d ...", total, batch_size)
        tokens_list, masks_list = [], []
        for start in range(0, total, batch_size):
            batch = embeddings[start:start + batch_size]
            z_out = self.z_gen(batch)
            z_seq, z_weights = z_out[0], z_out[-2]
            tokens, mask, _ = self._multiphrase_decode(z_seq, z_weights)
            tokens_list.append(tokens.cpu())
            if (start // batch_size) % 50 == 0:
                pct = (start + batch.size(0)) / total * 100
                mb = sum(t.nelement() * t.element_size() for t in tokens_list + masks_list) / 1e6
                logger.info("  IPA cache: %d / %d (%.0f%%) %.0fMB", start + batch.size(0), total, pct, mb)
            masks_list.append(mask.cpu())
        self._ipa_cache_tokens = torch.cat(tokens_list)
        self._ipa_cache_masks = torch.cat(masks_list)
        self.train()
        logger.info("Built IPA cache: %d sequences", self._ipa_cache_tokens.size(0))

        # Persist to disk for fast reload
        from pathlib import Path
        cache_path = Path(self.config.output_dir) / "ipa_cache.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "tokens": self._ipa_cache_tokens,
            "masks": self._ipa_cache_masks,
        }, cache_path)
        logger.info("Saved IPA cache to %s", cache_path)

    def checkpoint_state(self) -> dict:
        """Return state dict for checkpointing."""
        cfg = self.config
        ckpt = {
            "z_gen": self.z_gen.state_dict(),
            "msg_encoder": self.msg_encoder.state_dict(),
            "surface_encoder": self.surface_encoder.state_dict(),
            "receiver": self.receiver.state_dict(),
            **({"ipa_encoder": self.ipa_encoder.state_dict()} if self.ipa_encoder is not None else {}),
            **({"refinement": self.refinement.state_dict()} if self.refinement is not None else {}),
            # Architectural config (must match on resume)
            "phase2_mode": cfg.phase2_mode,
            "use_ipa_receiver": cfg.use_ipa_receiver,
            "z_generator": cfg.z_generator,
            "embedding_dim": cfg.embedding_dim,
            "z_hidden_dim": cfg.z_hidden_dim,
            "max_phrases": cfg.max_phrases,
            "max_tokens_per_phrase": cfg.max_tokens_per_phrase,
            "num_memory_tokens": cfg.num_memory_tokens,
            "use_halt": cfg.use_halt,
            "lambda_p": cfg.lambda_p,
            "kl_beta": cfg.kl_beta,
            "z_diversity_weight": cfg.z_diversity_weight,
            "z_distribution_weight": cfg.z_distribution_weight,
            "hidden_state_weight": cfg.hidden_state_weight,
            "hidden_state_anneal_steps": cfg.hidden_state_anneal_steps,
            "encoder_num_layers": cfg.encoder.num_layers,
            "encoder_num_heads": cfg.encoder.num_heads,
            "num_distractors": cfg.num_distractors,
        }
        if self.z_diversity is not None:
            ckpt["z_diversity_target"] = self.z_diversity.target_sim.item()
        return ckpt

    def load_checkpoint_state(self, ckpt: dict) -> None:
        """Restore from a checkpoint dict."""
        self.z_gen.load_state_dict(ckpt["z_gen"])
        self.msg_encoder.load_state_dict(ckpt["msg_encoder"])
        if "surface_encoder" in ckpt:
            self.surface_encoder.load_state_dict(ckpt["surface_encoder"])
        self.receiver.load_state_dict(ckpt["receiver"])
        if self.ipa_encoder is not None and "ipa_encoder" in ckpt:
            self.ipa_encoder.load_state_dict(ckpt["ipa_encoder"])
        if self.refinement is not None and "refinement" in ckpt:
            self.refinement.load_state_dict(ckpt["refinement"])

    def trainable_param_groups(self) -> list[dict]:
        """Return optimizer param groups with per-group learning rates."""
        groups = [
            {"params": list(self.z_gen.parameters()), "lr": self.config.gru_lr},
            {"params": list(self.msg_encoder.parameters()), "lr": self.config.receiver_lr},
            {"params": list(self.surface_encoder.parameters()), "lr": self.config.receiver_lr},
            {"params": list(self.receiver.parameters()), "lr": self.config.receiver_lr},
        ]
        if self.ipa_encoder is not None:
            groups.append({"params": list(self.ipa_encoder.parameters()), "lr": self.config.receiver_lr})
        if self.refinement is not None:
            groups.append({"params": list(self.refinement.parameters()), "lr": self.config.receiver_lr})
        return groups

    def forward(
        self, anchor: Tensor, distractors: Tensor,
        *, step: int = 0,
        candidate_indices: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Two-phase expression game forward pass.

        Uses a dual-path architecture:
        - **Surface path** (primary): token-level representations through
          ``output_head → softmax → token_embedding`` force discriminative
          surface forms — what the downstream LLM translator will see.
        - **Hidden-state path** (auxiliary): raw decoder hidden states
          provide smooth gradients via cross-attention.  Annealed from
          ``hidden_state_weight`` → 0 over ``hidden_state_anneal_steps``.

        Args:
            anchor: ``(batch, embedding_dim)`` input embeddings.
            distractors: ``(batch, num_distractors, embedding_dim)``.
            step: Current training step (for annealing).
        """
        batch = anchor.size(0)
        device = anchor.device
        num_candidates = distractors.size(1) + 1

        # z-sequence generation (with grad)
        if isinstance(self.z_gen, DiffusionZGenerator):
            z_seq, z_weights, num_phrases = self.z_gen(anchor)
            halt_probs = None
        else:
            z_seq, halt_probs, z_weights, num_phrases = self.z_gen(anchor)

        # Phase 1: multi-phrase decode (no_grad)
        with torch.no_grad():
            tokens, gen_mask, phrase_bounds = self._multiphrase_decode(
                z_seq, z_weights,
            )

            # Surface diversity: count unique token sequences
            _eos = self.gen.eos_id
            _batch_seqs = set()
            for row, m in zip(tokens, gen_mask):
                ids = tuple(t.item() for t, v in zip(row, m) if v and t.item() != _eos)
                _batch_seqs.add(hash(ids))
            surface_unique = len(_batch_seqs) / max(tokens.size(0), 1)

            # Running global diversity
            self._seen_hashes.update(_batch_seqs)
            self._seen_total += tokens.size(0)
            surface_global = len(self._seen_hashes) / max(self._seen_total, 1)

        # Phase 2: produce hidden states with gradients to z
        if self.refinement is not None:
            # Refinement denoiser: lightweight diffusion over token embeddings
            from lfm.agents.decode import _calibrate_or_quantize
            actual_max = int(gen_mask.float().sum(dim=1).max().item())
            _tokens_trim = tokens[:, :actual_max]
            _mask_trim = gen_mask[:, :actual_max]

            tok_emb = self.gen.token_embedding(_tokens_trim).detach()

            weighted_z = z_weights.unsqueeze(-1) * z_seq
            z_flat = weighted_z.reshape(batch * z_seq.size(1), -1)
            z_dec = _calibrate_or_quantize(self.gen, z_flat)
            z_memories = self.gen.latent_to_decoder(z_dec).reshape(
                batch, z_seq.size(1) * self.gen._num_memory_tokens, -1,
            )

            hidden = self.refinement(tok_emb, z_memories, _mask_trim)
            trimmed_mask = _mask_trim
        else:
            # Original: frozen decoder re-run with gradients
            hidden = rerun_decoder_multiphrase_with_grad(
                self.gen, z_seq, z_weights, tokens, gen_mask, phrase_bounds,
            )
            trimmed_mask = gen_mask[:, :hidden.size(1)]

        # Per-position phrase weights for halt gradient flow.
        # All phrases contribute equally to the message (no attenuation),
        # but the weights are available for auxiliary losses.
        from lfm.agents.decode import _compute_phrase_assignment
        phrase_assign = _compute_phrase_assignment(
            phrase_bounds, trimmed_mask.size(1), z_seq.size(1), device,
        )
        per_pos_weight = z_weights.gather(1, phrase_assign)  # (B, T)

        cfg = self.config

        # --- IPA-to-IPA receiver path ---
        if self.ipa_encoder is not None and candidate_indices is not None:
            # Sender: encode live tokens through IPA encoder
            # Use straight-through surface tokens for gradient flow
            logits_per_pos = self.gen.output_head(hidden)
            soft_probs = F.softmax(logits_per_pos, dim=-1)
            hard_ids = logits_per_pos.argmax(dim=-1)
            hard_onehot = F.one_hot(hard_ids, logits_per_pos.size(-1)).float()
            straight_through = (hard_onehot - soft_probs).detach() + soft_probs
            sender_embedded = straight_through @ self.ipa_encoder.token_embed.weight
            sender_vec = self.ipa_encoder.encoder(
                sender_embedded * per_pos_weight.unsqueeze(-1), trimmed_mask,
            )

            # Candidates: look up cached IPA, encode (no grad on cache, grad on encoder)
            # candidate_indices: (B, 16) — indices into the IPA cache
            perm = torch.stack([
                torch.randperm(num_candidates, device=device)
                for _ in range(batch)
            ])
            perm_indices = torch.gather(candidate_indices, 1, perm)
            target_idx = (perm == 0).long().argmax(dim=1)

            # Encode candidates (batched if VRAM allows, sequential fallback)
            if batch < 32:
                # Batched: all candidates at once
                cand_tokens = self._ipa_cache_tokens[perm_indices.cpu().reshape(-1)].to(device)
                cand_masks = self._ipa_cache_masks[perm_indices.cpu().reshape(-1)].to(device)
                with torch.no_grad():
                    cand_flat = self.ipa_encoder(cand_tokens, cand_masks)
                candidate_vecs = cand_flat.reshape(batch, num_candidates, -1)
            else:
                # Sequential: one candidate position at a time
                cand_vecs = []
                for k in range(num_candidates):
                    k_idx = perm_indices[:, k].cpu()
                    k_tok = self._ipa_cache_tokens[k_idx].to(device)
                    k_msk = self._ipa_cache_masks[k_idx].to(device)
                    with torch.no_grad():
                        cand_vecs.append(self.ipa_encoder(k_tok, k_msk))
                candidate_vecs = torch.stack(cand_vecs, dim=1)

            ipa_logits = self.receiver(sender_vec, candidate_vecs)

            # Soft topology loss: IPA similarity structure should mirror
            # input embedding similarity structure.  Use KL divergence
            # between IPA logits and embedding cosine similarities.
            candidates_emb = torch.cat([anchor.unsqueeze(1), distractors], dim=1)
            perm_expanded = perm.unsqueeze(-1).expand_as(candidates_emb)
            candidates_emb = torch.gather(candidates_emb, 1, perm_expanded)
            with torch.no_grad():
                teacher_sims = F.cosine_similarity(
                    anchor.unsqueeze(1), candidates_emb, dim=-1,
                )  # (B, 16)
                teacher_probs = F.softmax(teacher_sims / 0.1, dim=-1)
            student_log_probs = F.log_softmax(ipa_logits, dim=-1)
            ipa_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

            # Accuracy: still track hard classification for logging
            surface_logits = ipa_logits
            surface_loss = ipa_loss

            # Anneal hs weight (scaffold for denoiser z learning)
            anneal = cfg.hidden_state_anneal_steps
            if anneal > 0 and cfg.hidden_state_weight > 0:
                t = min(step / anneal, 1.0)
                hs_weight = cfg.hidden_state_weight * (1.0 - t)
            else:
                hs_weight = cfg.hidden_state_weight
            if hs_weight > 0:
                hidden_message = self.msg_encoder(
                    hidden * per_pos_weight.unsqueeze(-1), trimmed_mask,
                )
                hidden_logits = self.receiver(hidden_message, candidates_emb)
                hidden_loss = F.cross_entropy(hidden_logits, target_idx)
            else:
                hidden_loss = torch.tensor(0.0, device=device)
            ipa_weight = 1.0 - hs_weight if anneal > 0 else 1.0
            receiver_loss = ipa_weight * ipa_loss + hs_weight * hidden_loss
        else:
            # --- Original embedding-based receiver path ---
            candidates = torch.cat([anchor.unsqueeze(1), distractors], dim=1)
            perm = torch.stack([
                torch.randperm(num_candidates, device=device)
                for _ in range(batch)
            ])
            perm_expanded = perm.unsqueeze(-1).expand_as(candidates)
            candidates = torch.gather(candidates, 1, perm_expanded)
            target_idx = (perm == 0).long().argmax(dim=1)

            # Hidden-state loss
            hidden_message = self.msg_encoder(
                hidden * per_pos_weight.unsqueeze(-1), trimmed_mask,
            )
            hidden_logits = self.receiver(hidden_message, candidates)
            hidden_loss = F.cross_entropy(hidden_logits, target_idx)

            # Surface loss (anneals IN as hidden-state anneals OUT)
            anneal = cfg.hidden_state_anneal_steps
            if anneal > 0 and cfg.hidden_state_weight > 0:
                t = min(step / anneal, 1.0)
                hs_weight = cfg.hidden_state_weight * (1.0 - t)
                surface_weight = t
            else:
                hs_weight = cfg.hidden_state_weight
                surface_weight = 0.0 if hs_weight > 0 else 1.0

            if surface_weight > 0:
                # Surface path: project hidden → logits → soft tokens → re-embed
                logits_per_pos = self.gen.output_head(hidden)
                soft_probs = F.softmax(logits_per_pos, dim=-1)
                hard_ids = logits_per_pos.argmax(dim=-1)
                hard_onehot = F.one_hot(hard_ids, logits_per_pos.size(-1)).float()
                straight_through = (hard_onehot - soft_probs).detach() + soft_probs
                surface_repr = straight_through @ self.gen.token_embedding.weight
                surface_message = self.surface_encoder(
                    surface_repr * per_pos_weight.unsqueeze(-1), trimmed_mask,
                )
                surface_logits = self.receiver(surface_message, candidates)
                surface_loss = F.cross_entropy(surface_logits, target_idx)
            else:
                surface_logits = hidden_logits
                surface_loss = torch.tensor(0.0, device=device)

            receiver_loss = surface_weight * surface_loss + hs_weight * hidden_loss

        # Length regularization
        total_tokens = trimmed_mask.float().sum(dim=1)
        if isinstance(self.z_gen, DiffusionZGenerator) and cfg.length_weight > 0:
            len_loss = length_distribution_loss(z_weights, cfg.target_phrases)
            loss = receiver_loss + cfg.length_weight * len_loss
            kl_loss = len_loss
        elif cfg.use_halt and cfg.kl_beta > 0:
            kl_loss = geometric_kl(halt_probs, cfg.lambda_p)
            loss = receiver_loss + cfg.kl_beta * kl_loss
        else:
            kl_loss = torch.tensor(0.0, device=device)
            loss = receiver_loss

        # z diversity regularization
        if self.z_diversity is not None:
            div_loss, z_intra_sim = self.z_diversity(z_seq, z_weights)
            loss = loss + cfg.z_diversity_weight * div_loss
        else:
            with torch.no_grad():
                _, z_intra_sim = ZDiversityLoss.compute_similarity(
                    z_seq, z_weights,
                )
            div_loss = torch.tensor(0.0, device=device)

        # z distribution matching
        if self.z_distribution is not None:
            z_all = z_seq.reshape(-1, z_seq.size(-1))
            dist_loss, z_coverage = self.z_distribution(z_all)
            loss = loss + cfg.z_distribution_weight * dist_loss
        else:
            dist_loss = torch.tensor(0.0, device=device)
            z_coverage = torch.tensor(0.0, device=device)

        with torch.no_grad():
            accuracy = (surface_logits.argmax(1) == target_idx).float().mean()

        return {
            "loss": loss,
            "accuracy": accuracy,
            "msg_lengths": total_tokens.mean().detach(),
            "logits": surface_logits,
            "target_idx": target_idx,
            "halt_cost": kl_loss.detach(),
            "num_phrases": num_phrases.mean().detach(),
            "z_intra_sim": z_intra_sim,
            "z_div_loss": div_loss.detach(),
            "z_dist_loss": dist_loss.detach(),
            "z_coverage": z_coverage,
            "hs_weight": torch.tensor(hs_weight),
            "surface_unique": torch.tensor(surface_unique),
            "surface_global": torch.tensor(surface_global),
            "_tokens": tokens.detach(),
            "_gen_mask": gen_mask.detach(),
            "surface_loss": surface_loss.detach(),
        }

    @torch.no_grad()
    def _multiphrase_decode(
        self, z_seq: Tensor, z_weights: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Phase 1: KV-cached multi-phrase decode with z-switching."""
        from lfm.generator.layers import PhraseDecoder, multiscale_causal_mask

        gen = self.gen
        cfg = self.config
        batch, K, _ = z_seq.shape
        device = z_seq.device
        max_tok_per_phrase = cfg.max_tokens_per_phrase
        max_total = max_tok_per_phrase * K

        weighted_z = z_weights.unsqueeze(-1) * z_seq
        n_mem = gen._num_memory_tokens
        hidden_dim = gen.config.decoder_hidden_dim
        memories = gen.latent_to_decoder(
            weighted_z.reshape(batch * K, -1),
        ).reshape(batch, K, n_mem, hidden_dim)

        decoder = gen.decoder
        is_linguistic = isinstance(decoder, PhraseDecoder)

        if gen._full_causal_mask is None or gen._full_causal_mask.size(1) < max_total + 1:
            gen._full_causal_mask = multiscale_causal_mask(
                max_total + 1,
                num_heads=gen.config.decoder_num_heads,
                head_windows=gen.config.attention_head_windows,
                global_every=gen.config.attention_global_every,
                device=device,
            )

        all_tokens = torch.zeros(batch, max_total, dtype=torch.long, device=device)
        all_mask = torch.zeros(batch, max_total, dtype=torch.bool, device=device)
        phrase_bounds = torch.zeros(batch, K, dtype=torch.long, device=device)

        cur_phrase = torch.zeros(batch, dtype=torch.long, device=device)
        tokens_in_phrase = torch.zeros(batch, dtype=torch.long, device=device)
        total_pos = torch.zeros(batch, dtype=torch.long, device=device)

        phrase_active = z_weights > 0.01
        finished = ~phrase_active[:, 0]

        # Extend RoPE freqs for multi-phrase decode beyond max_seq_len
        rope_freqs = gen._rope_freqs
        if rope_freqs is not None and rope_freqs.size(0) < max_total + 1:
            from lfm.generator.layers import precompute_rope_freqs
            rope_freqs = precompute_rope_freqs(
                gen.config.decoder_hidden_dim // gen.config.decoder_num_heads,
                max_total + 1, device=device,
            )

        if is_linguistic:
            kv_cache = decoder.make_kv_cache(
                batch, max_total + 1, device, dtype=torch.float16,
            )

        cur_embed = gen.token_embedding(
            torch.full((batch, 1), gen.bos_id, dtype=torch.long, device=device),
        )
        batch_idx = torch.arange(batch, device=device)

        def _gather_memory() -> Tensor:
            idx = cur_phrase.clamp(max=K - 1)
            mem = memories[batch_idx, idx]
            active_mask = phrase_active[batch_idx, idx] & ~finished
            return mem * active_mask.unsqueeze(-1).unsqueeze(-1).float()

        memory = _gather_memory()

        if is_linguistic:
            mask_row = gen._full_causal_mask[:, 0:1, 0:1]
            out = decoder.forward_cached(
                cur_embed, memory, kv_cache,
                rope_freqs=rope_freqs, tgt_mask_row=mask_row,
            )
            kv_cache.advance()
        else:
            out = decoder(cur_embed, memory)

        for t in range(max_total):
            logits = gen.output_head(out[:, -1])
            next_token = logits.argmax(dim=-1)

            # Vectorized token storage
            active = ~finished
            all_tokens[batch_idx, total_pos] = next_token * active.long()
            all_mask[batch_idx, total_pos] = active
            total_pos += active.long()
            tokens_in_phrase += active.long()

            # Vectorized z-switch
            hit_eos = (next_token == gen.eos_id) & (tokens_in_phrase >= 1)
            hit_max = tokens_in_phrase >= max_tok_per_phrase
            should_switch = (hit_eos | hit_max) & active

            cur_phrase += should_switch.long()
            tokens_in_phrase *= ~should_switch

            # Record phrase boundaries
            switched_valid = should_switch & (cur_phrase < K)
            if switched_valid.any():
                phrase_bounds[batch_idx[switched_valid], cur_phrase[switched_valid]] = (
                    total_pos[switched_valid]
                )

            # Mark finished
            clamped = cur_phrase.clamp(max=K - 1)
            next_inactive = ~phrase_active[batch_idx, clamped]
            finished = finished | (cur_phrase >= K) | (should_switch & next_inactive)

            if finished.all():
                break

            memory = _gather_memory()
            new_embed = gen.token_embedding(next_token.unsqueeze(1))

            if is_linguistic:
                seq_so_far = kv_cache.seq_len + 1
                mask_row = gen._full_causal_mask[
                    :, kv_cache.seq_len : kv_cache.seq_len + 1, :seq_so_far
                ]
                out = decoder.forward_cached(
                    new_embed, memory, kv_cache,
                    rope_freqs=rope_freqs, tgt_mask_row=mask_row,
                )
                kv_cache.advance()
            else:
                cur_ids = torch.cat([cur_ids, next_token.unsqueeze(1)], dim=1)
                all_embed = gen.token_embedding(cur_ids)
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                    cur_ids.size(1), device=device,
                )
                out = decoder(all_embed, memory, tgt_mask=tgt_mask)

        return all_tokens, all_mask, phrase_bounds
