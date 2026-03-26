"""Dataset manifest: metadata stored alongside HDF5 files."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lfm.config.base import LFMBaseConfig

logger = logging.getLogger(__name__)


class DatasetManifest(LFMBaseConfig):
    """Metadata for a generated dataset, persisted as ``manifest.yaml``.

    The manifest is the single source of truth for what a dataset contains,
    how it was generated, and what configuration was used.
    """

    model_config = {"frozen": True, "extra": "allow", "use_enum_values": True}

    name: str
    description: str = ""
    created_at: str = ""
    format: str = "h5"
    sources: list[str] = []
    languages: dict[str, int] = {}
    total_samples: int = 0
    rejected_samples: int = 0
    sanitize: dict[str, Any] = {}
    config: dict[str, Any] = {}

    def save(self, path: Path) -> None:
        """Write manifest to YAML file."""
        import yaml

        data = self.model_dump()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info("Wrote manifest: %s", path)

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        """Read manifest from YAML file."""
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        sources: list[str],
        languages: dict[str, int],
        total_samples: int,
        rejected_samples: int,
        sanitize_config: dict[str, Any],
        generate_config: dict[str, Any],
    ) -> DatasetManifest:
        """Create a new manifest with the current timestamp."""
        return cls(
            name=name,
            description=description,
            created_at=datetime.now(UTC).isoformat(),
            format="h5",
            sources=sources,
            languages=languages,
            total_samples=total_samples,
            rejected_samples=rejected_samples,
            sanitize=sanitize_config,
            config=generate_config,
        )
