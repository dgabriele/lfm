"""Phonotactic structural priors via GRU pre-training on cross-linguistic IPA data.

Pre-trains the smoothness GRU on real IPA sequences from WikiPron (3.1M
word/pronunciation pairs, 337 languages), represented as PanPhon articulatory
feature vectors (values in {-1, 0, +1}).  The pre-training task is
identical to the smoothness loss in ``SurfacePhonology`` — predict next feature
vector from preceding ones — so the GRU learns cross-linguistic phonotactic
patterns that transfer directly.

Usage::

    from lfm.phonology.priors import pretrain_phonotactic_prior, PhonotacticPriorConfig

    config = PhonotacticPriorConfig(
        wikipron_dir="path/to/wikipron/data/scrape/tsv",
        surface_dim=12,
        smoothness_hidden_dim=32,
    )
    metrics = pretrain_phonotactic_prior(config)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, random_split

from lfm.config.base import LFMBaseConfig

logger = logging.getLogger(__name__)

def _detect_panphon_dim() -> int:
    """Detect the number of articulatory features in the installed PanPhon version."""
    try:
        import panphon

        ft = panphon.FeatureTable()
        seg = ft.fts("a")
        if seg is not None:
            return len(seg.numeric())
    except Exception:
        pass
    return 24  # PanPhon >=0.20 default


PANPHON_DIM: int = 24
"""PanPhon articulatory feature count (24 in panphon >=0.20)."""


class PhonotacticPriorConfig(LFMBaseConfig):
    """Configuration for phonotactic prior pre-training.

    Attributes:
        wikipron_dir: Path to WikiPron TSV directory (``data/scrape/tsv/``).
        max_samples_per_language: Cap per language for typological balance.
        min_word_length: Minimum IPA segments per word (shorter words skipped).
        max_word_length: Maximum IPA segments per word (longer words truncated).
        surface_dim: Must match target ``PhonologyConfig.surface_dim``.
        smoothness_hidden_dim: Must match target ``PhonologyConfig.smoothness_hidden_dim``.
        batch_size: Training batch size.
        lr: Learning rate for Adam optimizer.
        num_epochs: Number of training epochs.
        val_fraction: Fraction of data held out for validation.
        seed: Random seed for reproducibility.
        device: Torch device string.
        output_path: Where to save the pre-trained checkpoint.
    """

    wikipron_dir: str
    max_samples_per_language: int = 5000
    min_word_length: int = 2
    max_word_length: int = 30
    surface_dim: int = 12
    smoothness_hidden_dim: int = 32
    batch_size: int = 256
    lr: float = 1e-3
    num_epochs: int = 10
    val_fraction: float = 0.1
    seed: int = 42
    device: str = "cuda"
    output_path: str = "data/phonotactic_prior.pt"


class WikiPronLoader:
    """Load and balance WikiPron TSV data.

    WikiPron files follow the naming convention
    ``{lang_code}_{script}_{broad|narrow}.tsv`` with tab-separated columns:
    ``word<TAB>space-segmented IPA``.  Broad (phonemic) transcriptions are
    preferred over narrow (phonetic) — we want phonotactic patterns, not
    allophonic detail.

    Args:
        wikipron_dir: Path to the WikiPron ``tsv/`` directory.
        max_samples_per_language: Per-language sample cap for typological balance.
        min_word_length: Minimum IPA segment count per word.
        max_word_length: Maximum IPA segment count per word.
    """

    def __init__(
        self,
        wikipron_dir: str,
        max_samples_per_language: int = 5000,
        min_word_length: int = 2,
        max_word_length: int = 30,
    ) -> None:
        self.wikipron_dir = Path(wikipron_dir)
        self.max_samples_per_language = max_samples_per_language
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length

    def load(self) -> list[tuple[str, list[str]]]:
        """Load WikiPron data, preferring broad transcriptions.

        Returns:
            List of ``(lang_code, [ipa_segments])`` tuples, balanced across
            languages.
        """
        tsv_dir = self.wikipron_dir
        if not tsv_dir.is_dir():
            raise FileNotFoundError(f"WikiPron directory not found: {tsv_dir}")

        # Discover TSV files, preferring broad over narrow per language
        lang_files: dict[str, Path] = {}
        for tsv_file in sorted(tsv_dir.glob("*.tsv")):
            name = tsv_file.stem  # e.g. "eng_latn_broad"
            parts = name.rsplit("_", 1)
            if len(parts) < 2:
                continue
            lang_key = parts[0]  # e.g. "eng_latn"
            transcript_type = parts[1]  # "broad" or "narrow"

            if lang_key not in lang_files or transcript_type == "broad":
                lang_files[lang_key] = tsv_file

        # Parse each file with per-language cap
        samples: list[tuple[str, list[str]]] = []
        lang_counts: dict[str, int] = defaultdict(int)

        for lang_key, tsv_file in sorted(lang_files.items()):
            lang_code = lang_key.split("_")[0]
            count = 0

            with open(tsv_file, encoding="utf-8") as f:
                for line in f:
                    if count >= self.max_samples_per_language:
                        break
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                    segments = parts[1].split()
                    if len(segments) < self.min_word_length:
                        continue
                    if len(segments) > self.max_word_length:
                        segments = segments[: self.max_word_length]
                    samples.append((lang_code, segments))
                    count += 1
                    lang_counts[lang_code] += 1

        num_languages = len(lang_counts)
        logger.info(
            "Loaded %d samples from %d languages (%d unique lang codes)",
            len(samples),
            len(lang_files),
            num_languages,
        )
        return samples


class FeatureConverter:
    """Convert IPA segments to PanPhon articulatory feature vectors.

    Uses ``panphon.FeatureTable`` to map each IPA segment to an articulatory
    feature vector with values in {-1, 0, +1}.  ``panphon`` is imported lazily
    so the rest of the phonology module works without it installed.
    """

    def __init__(self) -> None:
        try:
            import panphon
        except ImportError as e:
            raise ImportError(
                "panphon is required for phonotactic prior pre-training. "
                "Install it with: pip install panphon>=0.20"
            ) from e
        self._ft = panphon.FeatureTable()
        # Detect actual feature dimensionality from this panphon version
        self.feature_dim = _detect_panphon_dim()

    @property
    def dim(self) -> int:
        """Number of articulatory features per segment."""
        return self.feature_dim

    def segments_to_features(self, segments: list[str]) -> np.ndarray | None:
        """Convert IPA segments to a feature matrix.

        Args:
            segments: List of IPA segment strings.

        Returns:
            Array of shape ``(len(segments), feature_dim)`` with values in
            {-1, 0, +1}, or ``None`` if more than 50% of segments are
            unrecognized.
        """
        features: list[list[int]] = []
        failures = 0

        for seg in segments:
            fts = self._ft.fts(seg)
            if fts is None:
                failures += 1
                features.append([0] * self.feature_dim)
            elif hasattr(fts, "numeric"):
                features.append(fts.numeric())
            else:
                failures += 1
                features.append([0] * self.feature_dim)

        if failures > len(segments) * 0.5:
            return None

        return np.array(features, dtype=np.float32)


class PhonotacticDataset(Dataset):
    """Padded IPA feature sequences for GRU pre-training.

    Stores pre-converted feature matrices and pads them to a uniform
    ``max_word_length`` for batched training.

    Args:
        feature_arrays: List of feature arrays, each ``(word_len, feature_dim)``.
        max_word_length: Pad/truncate all sequences to this length.
        feature_dim: Number of features per segment (auto-detected from data if omitted).
    """

    def __init__(
        self,
        feature_arrays: list[np.ndarray],
        max_word_length: int = 30,
        feature_dim: int | None = None,
    ) -> None:
        self.max_word_length = max_word_length
        if feature_dim is None:
            feature_dim = feature_arrays[0].shape[1] if feature_arrays else PANPHON_DIM
        self.feature_dim = feature_dim
        self.data: list[tuple[np.ndarray, int]] = []

        for arr in feature_arrays:
            length = min(arr.shape[0], max_word_length)
            padded = np.zeros((max_word_length, feature_dim), dtype=np.float32)
            padded[:length] = arr[:length]
            self.data.append((padded, length))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        features, length = self.data[idx]
        return torch.from_numpy(features), length


class PhonotacticPretrainer:
    """Pre-train the smoothness GRU on cross-linguistic IPA data.

    The pre-training task mirrors the smoothness loss in ``SurfacePhonology``:
    predict the next surface vector from preceding ones (next-step prediction
    via GRU).  A learned ``feature_proj`` maps PanPhon feature vectors into
    ``surface_dim`` space before the GRU sees them.

    Args:
        config: Pre-training configuration.
    """

    def __init__(self, config: PhonotacticPriorConfig) -> None:
        self.config = config

    def pretrain(self) -> dict[str, float]:
        """Run the full pre-training pipeline.

        Returns:
            Metrics dict with ``train_loss``, ``val_loss``,
            ``num_languages``, and ``num_samples``.
        """
        cfg = self.config
        torch.manual_seed(cfg.seed)

        # 1. Load WikiPron data
        loader = WikiPronLoader(
            wikipron_dir=cfg.wikipron_dir,
            max_samples_per_language=cfg.max_samples_per_language,
            min_word_length=cfg.min_word_length,
            max_word_length=cfg.max_word_length,
        )
        samples = loader.load()
        if not samples:
            raise RuntimeError("No samples loaded from WikiPron directory")

        lang_codes = {s[0] for s in samples}
        num_languages = len(lang_codes)

        # 2. Convert to feature vectors
        converter = FeatureConverter()
        feature_arrays: list[np.ndarray] = []
        skipped = 0

        for _lang, segments in samples:
            features = converter.segments_to_features(segments)
            if features is not None:
                feature_arrays.append(features)
            else:
                skipped += 1

        if not feature_arrays:
            raise RuntimeError("No valid feature arrays produced from WikiPron data")

        logger.info(
            "Converted %d samples to features (%d skipped)",
            len(feature_arrays),
            skipped,
        )

        # 3. Build dataset and split
        panphon_dim = converter.dim
        dataset = PhonotacticDataset(feature_arrays, cfg.max_word_length, panphon_dim)
        val_size = max(1, int(len(dataset) * cfg.val_fraction))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.seed),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
        )

        # 4. Build model components
        device = torch.device(cfg.device)
        feature_proj = nn.Linear(panphon_dim, cfg.surface_dim).to(device)
        smoothness_gru = nn.GRU(
            cfg.surface_dim,
            cfg.smoothness_hidden_dim,
            batch_first=True,
        ).to(device)
        smoothness_head = nn.Linear(cfg.smoothness_hidden_dim, cfg.surface_dim).to(device)

        optimizer = torch.optim.Adam(
            list(feature_proj.parameters())
            + list(smoothness_gru.parameters())
            + list(smoothness_head.parameters()),
            lr=cfg.lr,
        )

        # 5. Training loop
        best_val_loss = float("inf")
        best_metrics: dict[str, float] = {}

        for epoch in range(cfg.num_epochs):
            # Train
            feature_proj.train()
            smoothness_gru.train()
            smoothness_head.train()
            train_loss_sum = 0.0
            train_count = 0

            for batch_features, batch_lengths in train_loader:
                batch_features = batch_features.to(device)  # (B, L, 21)
                batch_lengths = torch.as_tensor(batch_lengths)

                # Project to surface_dim
                projected = feature_proj(batch_features)  # (B, L, surface_dim)

                # Next-step prediction: input is positions 0..L-2, target is 1..L-1
                gru_input = projected[:, :-1, :]  # (B, L-1, surface_dim)
                target = projected[:, 1:, :]  # (B, L-1, surface_dim)

                # Pack for variable-length sequences (lengths - 1 for input)
                input_lengths = (batch_lengths - 1).clamp(min=1)
                sorted_lengths, sort_idx = input_lengths.sort(descending=True)
                gru_input_sorted = gru_input[sort_idx]
                target_sorted = target[sort_idx]

                packed = pack_padded_sequence(
                    gru_input_sorted,
                    sorted_lengths.cpu(),
                    batch_first=True,
                    enforce_sorted=True,
                )
                gru_out_packed, _ = smoothness_gru(packed)
                gru_out, _ = pad_packed_sequence(gru_out_packed, batch_first=True)

                predicted = smoothness_head(gru_out)  # (B, max_len, surface_dim)

                # Mask out padding positions
                max_len = predicted.size(1)
                mask = (
                    torch.arange(max_len, device=device).unsqueeze(0)
                    < sorted_lengths.to(device).unsqueeze(1)
                )  # (B, max_len)
                mask = mask.unsqueeze(-1).expand_as(predicted)  # (B, max_len, surface_dim)

                target_trimmed = target_sorted[:, :max_len, :]
                loss = F.mse_loss(
                    predicted * mask.float(),
                    target_trimmed * mask.float(),
                    reduction="sum",
                )
                num_valid = mask.float().sum()
                loss = loss / num_valid.clamp(min=1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * num_valid.item()
                train_count += int(num_valid.item())

            train_loss = train_loss_sum / max(train_count, 1)

            # Validate
            feature_proj.eval()
            smoothness_gru.eval()
            smoothness_head.eval()
            val_loss_sum = 0.0
            val_count = 0

            with torch.no_grad():
                for batch_features, batch_lengths in val_loader:
                    batch_features = batch_features.to(device)
                    batch_lengths = torch.as_tensor(batch_lengths)

                    projected = feature_proj(batch_features)
                    gru_input = projected[:, :-1, :]
                    target = projected[:, 1:, :]

                    input_lengths = (batch_lengths - 1).clamp(min=1)
                    sorted_lengths, sort_idx = input_lengths.sort(descending=True)
                    gru_input_sorted = gru_input[sort_idx]
                    target_sorted = target[sort_idx]

                    packed = pack_padded_sequence(
                        gru_input_sorted,
                        sorted_lengths.cpu(),
                        batch_first=True,
                        enforce_sorted=True,
                    )
                    gru_out_packed, _ = smoothness_gru(packed)
                    gru_out, _ = pad_packed_sequence(gru_out_packed, batch_first=True)

                    predicted = smoothness_head(gru_out)
                    max_len = predicted.size(1)
                    mask = (
                        torch.arange(max_len, device=device).unsqueeze(0)
                        < sorted_lengths.to(device).unsqueeze(1)
                    )
                    mask = mask.unsqueeze(-1).expand_as(predicted)

                    target_trimmed = target_sorted[:, :max_len, :]
                    loss = F.mse_loss(
                        predicted * mask.float(),
                        target_trimmed * mask.float(),
                        reduction="sum",
                    )
                    num_valid = mask.float().sum()
                    loss = loss / num_valid.clamp(min=1)

                    val_loss_sum += loss.item() * num_valid.item()
                    val_count += int(num_valid.item())

            val_loss = val_loss_sum / max(val_count, 1)

            logger.info(
                "Epoch %d/%d — train_loss=%.6f val_loss=%.6f",
                epoch + 1,
                cfg.num_epochs,
                train_loss,
                val_loss,
            )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "num_languages": float(num_languages),
                    "num_samples": float(len(feature_arrays)),
                }

                output_path = Path(cfg.output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "surface_dim": cfg.surface_dim,
                        "smoothness_hidden_dim": cfg.smoothness_hidden_dim,
                        "panphon_dim": panphon_dim,
                        "smoothness_gru": smoothness_gru.state_dict(),
                        "smoothness_head": smoothness_head.state_dict(),
                        "feature_proj": feature_proj.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "num_languages": num_languages,
                        "num_samples": len(feature_arrays),
                    },
                    output_path,
                )
                logger.info("Saved best checkpoint to %s", output_path)

        return best_metrics


def pretrain_phonotactic_prior(config: PhonotacticPriorConfig) -> dict[str, float]:
    """Convenience function: create ``PhonotacticPretrainer`` and run.

    Args:
        config: Pre-training configuration.

    Returns:
        Metrics dict with ``train_loss``, ``val_loss``,
        ``num_languages``, and ``num_samples``.
    """
    return PhonotacticPretrainer(config).pretrain()
