#!/usr/bin/env python
"""Initialize training_history.json for the currently training model.

Run once to bootstrap the history file for a model that started
training before the history tracking was implemented.
"""

from lfm.generator.pretrain import VAEPretrainConfig
from lfm.generator.training_history import TrainingHistory

# Current run config (as it was when training started)
config = VAEPretrainConfig(
    corpus_loader="leipzig",
    corpus_loader_config={"data_dir": "data/leipzig"},
    latent_dim=256,
    max_seq_len=96,
    encoder_pooling="mean",  # this run used mean, not attention
    batch_size=160,
    lr=0.001,
    lr_min=0.0001,
    num_epochs=40,
    dip_weight=0.1,
    kl_weight=0.0,
    use_amp=True,
)

history = TrainingHistory("data")
history.start_session(
    start_epoch=0,
    config=config,
    spm_hash=None,  # unknown — SPM was generated at training start
)
# Update to reflect current progress (epoch 23+, best val 0.83)
history.update_epoch(epoch=23, best_val_loss=0.8308)

print(f"Initialized {history.path}")
print(f"Sessions: {len(history.sessions)}")
