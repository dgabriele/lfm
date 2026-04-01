#!/bin/bash
# Wait for v5 training → visualize → update docs → expression game
set -e

echo "Waiting for v5 training to finish..."
while pgrep -f "pretrain_vae.py" > /dev/null 2>&1; do
    sleep 30
done
echo "Training finished."

# Extract decoder checkpoint
echo "Extracting decoder checkpoint..."
python -m poetry run python -c "
import torch
ckpt = torch.load('data/models/v5/vae_resume.pt', map_location='cpu', weights_only=False)
modules = ckpt['modules']
decoder_ckpt = {
    'latent_dim': 256, 'vocab_size': 8000,
    'decoder_hidden_dim': 512, 'decoder_num_layers': 4,
    'decoder_num_heads': 8, 'max_seq_len': 96,
    'num_memory_tokens': 8, 'encoder_num_layers': 2,
    'attention_head_windows': [3,3,7,7,15,15,0,0],
    'attention_global_every': 7, 'use_rope': True,
    'share_decoder_layers': True, 'encoder_pooling': 'mean',
    'latent_to_decoder': modules['latent_to_decoder'],
    'token_embedding': modules['dec_token_embedding'],
    'pos_embedding': modules['dec_pos_embedding'],
    'decoder': modules['decoder'],
    'output_head': modules['output_head'],
    'z_mean': ckpt['z_mean'], 'z_std': ckpt['z_std'],
    'train_loss': 0.0, 'val_loss': 0.0,
    'spm_hash': ckpt.get('spm_hash', ''),
}
torch.save(decoder_ckpt, 'data/models/v5/vae_decoder.pt')
print('Extracted decoder checkpoint')
"

# Run all visualizations
echo "Running all visualizations..."
python -m poetry run lfm visualize all \
  --checkpoint data/models/v5/vae_resume.pt \
  --spm-model data/models/v5/spm.model \
  --corpus-cache data/models/v5/preprocessed_cache.pt \
  --output-dir output/viz-v5 \
  --device cuda \
  2>&1 | tee data/models/v5/viz.log

# Copy images to docs
echo "Copying visualizations to docs..."
cp output/viz-v5/*.png docs/static/images/ 2>/dev/null || true
cp output/viz-v5/*.svg docs/static/images/ 2>/dev/null || true
cp output/viz-v5/*.webp docs/static/images/ 2>/dev/null || true

# Run expression game
echo "Running expression game (3000 steps)..."
mkdir -p data/expression_game_v5
python -m poetry run lfm agent expression \
  --decoder-path data/models/v5/vae_decoder.pt \
  --spm-path data/models/v5/spm.model \
  --steps 3000 --batch-size 128 \
  --output-dir data/expression_game_v5 \
  2>&1 | tee data/expression_game_v5/train.log

echo "All done. Update README/docs manually with final metrics."
