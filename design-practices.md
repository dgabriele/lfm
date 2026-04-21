# LFM Design Practices

Mandatory guidelines for all code in this project. These are not suggestions.

## Configuration

- **Explicit mode enums over implicit zero-checks.** If a feature can be enabled/disabled or has variants (e.g., β-VAE vs DIP-VAE), use a string enum field (`vae_regularizer: "beta" | "dip" | "none"`), not a float weight that's implicitly disabled at 0. Reading `dip_weight: 0.0` doesn't communicate "DIP is off" — reading `regularizer: beta` does.
- **Config fields are self-documenting.** A new contributor should understand what the system does from the config alone, without reading the training loop.

## Code Structure

- **Small, single-responsibility functions.** Each function does one thing. If a function has multiple conceptual stages, extract them.
- **Explicit over implicit.** Don't rely on attribute existence checks (`hasattr`), truthy/falsy values, or default fallbacks to control behavior. Use explicit flags and enums.
- **DRY.** If the same logic appears in two places, extract it. If two modules share infrastructure, build a shared base.
- **Modules are removable.** Any game variant, loss term, or scoring method should be removable without touching unrelated code. No deep coupling between components.

## Naming

- **Names explain purpose, not implementation.** `memory_pooler` not `attn_pool_v2`. `contrastive_loss` not `loss_fn_3`.
- **Config field names readable without context.** Another ML engineer should understand `decoder_rerun_weight` without reading the PR that introduced it.
- **No phase1/phase2 naming.** Use descriptive names: `autoregressive_decode`, `parallel_rerun`, `memory_scoring`.

## VAE Design

- **Design for downstream use from day one.** The VAE exists to serve agent games and corpus generation. Every architectural decision (latent regularization, z stats tracking, EOS behavior) must account for this.
- **Z stats are part of the model.** Track running mean/std of z as persistent buffers during training. Downstream consumers need these.
- **Prior diagnostic during training.** Sample z ~ prior, decode, and log EOS rate + sample quality at every checkpoint. This is the ground truth for whether the VAE will work downstream.

## Process

- **Review the existing codebase before proposing new approaches.** If the project already has an implementation of something (DIP-VAE, z calibration, multi-phrase decode), use it or explain why not.
- **Don't flip-flop.** If an approach is chosen, see it through to a clear success/failure signal before switching. Premature switching wastes time and money.
- **Measure before optimizing.** Profile bottlenecks, check existing metrics, run diagnostics — don't guess.
