#!/bin/bash
# Post-training automation: organize, visualize, commit, launch game
# Run this after pretraining completes.
set -e

cd /home/daniel/projects/lfm

echo "=== Organizing model files into data/models/v1/ ==="
mkdir -p data/models/v1
for f in vae_decoder.pt vae_resume.pt spm.model spm.vocab preprocessed_cache.pt training_history.json; do
    if [ -f "data/$f" ]; then
        mv "data/$f" "data/models/v1/$f"
        ln -sf "models/v1/$f" "data/$f"
        echo "  Moved + linked: $f"
    fi
done

echo "=== Generating all visualizations ==="
poetry run lfm visualize all --checkpoint data/vae_resume.pt --max-samples 5000

echo "=== Copying images to docs ==="
cp output/viz/*.png docs/static/images/

echo "=== Committing ==="
git add docs/static/images/ data/models/v1/
git commit -m "docs: update visualizations and organize model artifacts into data/models/v1/

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
git push

echo "=== Launching agent game ==="
poetry run python scripts/run_referential_reinforce.py &
GAME_PID=$!
echo "Agent game launched (PID=$GAME_PID)"
