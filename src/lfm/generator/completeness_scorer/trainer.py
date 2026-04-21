"""Train the completeness scorer on real + corrupted sentences."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from lfm.generator.completeness_scorer.data import ScorerDataset
from lfm.generator.completeness_scorer.model import CompletenessConfig, CompletenessScorer

logger = logging.getLogger(__name__)


def train_scorer(
    dataset_path: str | Path,
    output_dir: str | Path,
    cfg: CompletenessConfig | None = None,
) -> CompletenessScorer:
    """Train the completeness scorer and save the best checkpoint."""
    if cfg is None:
        cfg = CompletenessConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ScorerDataset(dataset_path)
    n_val = max(int(len(ds) * 0.05), 100)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = CompletenessScorer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("CompletenessScorer: %d params (%.1fM)", n_params, n_params / 1e6)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.BCEWithLogitsLoss()
    best_acc = 0.0

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)

            scores = model(tokens, lengths)
            loss = criterion(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = (scores > 0).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / max(total, 1)
        train_loss = total_loss / max(total, 1)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch["tokens"].to(device)
                lengths = batch["lengths"].to(device)
                labels = batch["labels"].to(device)
                scores = model(tokens, lengths)
                preds = (scores > 0).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / max(val_total, 1)
        logger.info(
            "ep%d  train_loss=%.4f train_acc=%.1f%% val_acc=%.1f%%",
            epoch, train_loss, train_acc * 100, val_acc * 100,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg.model_dump(),
                "val_acc": val_acc,
            }, output_dir / "best.pt")
            logger.info("  New best val_acc=%.1f%%", val_acc * 100)

    logger.info("Training complete. Best val_acc=%.1f%%", best_acc * 100)
    return model
