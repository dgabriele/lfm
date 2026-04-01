#!/bin/bash
# Post-epoch-2 pipeline: extract decoder → visualize → update docs → push
set -e
cd /home/daniel/projects/lfm

echo "Waiting for epoch 2 to complete..."
while true; do
    # Check if step count exceeds 2 * 20457 = 40914
    LAST_STEP=$(grep "step=" data/models/v5-leaf/train.log | tail -1 | grep -oP 'step=\K[0-9]+')
    if [ -n "$LAST_STEP" ] && [ "$LAST_STEP" -ge 40900 ]; then
        echo "Epoch 2 complete (step=$LAST_STEP)"
        break
    fi
    sleep 30
done

# Extract decoder checkpoint from latest resume
echo "Extracting decoder checkpoint..."
python -m poetry run python -c "
import torch
ckpt = torch.load('data/models/v5-leaf/vae_resume.pt', map_location='cpu', weights_only=False)
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
torch.save(decoder_ckpt, 'data/models/v5-leaf/vae_decoder.pt')
print('Done')
"

# Patch resume checkpoint with architecture metadata
python -m poetry run python -c "
import torch
ckpt = torch.load('data/models/v5-leaf/vae_resume.pt', map_location='cpu', weights_only=False)
ckpt['num_memory_tokens'] = 8
ckpt['encoder_num_layers'] = 2
ckpt['attention_head_windows'] = [3,3,7,7,15,15,0,0]
ckpt['attention_global_every'] = 7
ckpt['use_rope'] = True
ckpt['share_decoder_layers'] = True
ckpt['encoder_pooling'] = 'mean'
ckpt['decoder_hidden_dim'] = 512
ckpt['latent_dim'] = 256
torch.save(ckpt, 'data/models/v5-leaf/vae_resume.pt')
print('Patched')
"

# Run all visualizations
echo "Running visualizations..."
python -m poetry run lfm visualize all \
  --checkpoint data/models/v5-leaf/vae_resume.pt \
  --spm-model data/models/v5-leaf/spm.model \
  --corpus-cache data/models/v5-leaf/preprocessed_cache.pt \
  --output-dir output/viz-v5-leaf \
  --device cuda \
  2>&1 | tee data/models/v5-leaf/viz.log

# Copy images
echo "Copying images to docs..."
cp output/viz-v5-leaf/*.png docs/static/images/ 2>/dev/null || true

echo "DONE — images copied. Update README/docs manually, then commit."
