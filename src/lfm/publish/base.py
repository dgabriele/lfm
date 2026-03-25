"""Base classes for HuggingFace Hub publishing."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_RELEASES_DIR = "releases/huggingface"


class ReleaseManifest:
    """Tracks a release's arguments, metadata, and results.

    Saved as a YAML file in ``releases/huggingface/`` with a
    human-readable datetime prefix.

    Args:
        release_type: ``"model"`` or ``"dataset"``.
        args: CLI arguments used to perform the release.
    """

    def __init__(self, release_type: str, args: dict[str, Any]) -> None:
        self.release_type = release_type
        self.args = args
        self.result: dict[str, Any] = {}
        self.timestamp = datetime.now(UTC)

    def set_result(self, **kwargs: Any) -> None:
        """Record result metadata from the upload."""
        self.result.update(kwargs)

    def save(self, base_dir: str = ".") -> Path:
        """Save manifest to ``releases/huggingface/<datetime>_<type>.yaml``.

        Returns:
            Path to the saved manifest file.
        """
        releases_dir = Path(base_dir) / _RELEASES_DIR
        releases_dir.mkdir(parents=True, exist_ok=True)

        ts = self.timestamp.strftime("%Y%m%d_%H%M%S")
        # Sanitize repo name for filename
        repo = self.args.get("repo_id", "unknown").replace("/", "_")
        filename = f"{ts}_{self.release_type}_{repo}.yaml"
        path = releases_dir / filename

        data = {
            "release_type": self.release_type,
            "timestamp": self.timestamp.isoformat(timespec="seconds"),
            "args": self.args,
            "result": self.result,
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info("Saved release manifest to %s", path)
        return path


class HFPublisher(ABC):
    """Base class for HuggingFace Hub publishers.

    Subclasses implement ``_collect_files`` to gather artifacts and
    ``_build_card`` to generate a README/model card.  The ``publish``
    method handles the upload and manifest generation.

    Args:
        repo_id: HuggingFace repo ID (e.g. ``"username/lfm-decoder-v1"``).
        repo_type: ``"model"`` or ``"dataset"``.
        private: Whether the repo should be private.
        token: HuggingFace API token (or ``None`` to use cached login).
    """

    def __init__(
        self,
        repo_id: str,
        repo_type: str = "model",
        private: bool = False,
        token: str | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.private = private
        self.token = token

    def publish(self, **kwargs: Any) -> ReleaseManifest:
        """Upload to HuggingFace Hub and generate a release manifest.

        Args:
            **kwargs: Subclass-specific arguments passed to ``_collect_files``
                and ``_build_card``.

        Returns:
            ReleaseManifest with upload results.
        """
        from huggingface_hub import HfApi

        api = HfApi(token=self.token)

        # Create repo if needed
        repo_url = api.create_repo(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            private=self.private,
            exist_ok=True,
        )
        logger.info("Repo: %s", repo_url)

        # Collect files to upload
        files = self._collect_files(**kwargs)
        logger.info("Uploading %d files to %s", len(files), self.repo_id)

        # Generate and include README/card
        card_content = self._build_card(**kwargs)
        card_path = Path("/tmp") / f"lfm_hf_card_{self.repo_type}.md"
        card_path.write_text(card_content, encoding="utf-8")
        files.append(("README.md", str(card_path)))

        # Upload all files
        for remote_name, local_path in files:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_name,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
            )
            logger.info("  Uploaded: %s", remote_name)

        # Build manifest
        all_args = {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "private": self.private,
            **kwargs,
        }
        manifest = ReleaseManifest(self.repo_type, all_args)
        manifest.set_result(
            url=str(repo_url),
            files_uploaded=[name for name, _ in files],
            repo_id=self.repo_id,
        )
        manifest.save()

        logger.info("Published to %s", repo_url)
        return manifest

    @abstractmethod
    def _collect_files(self, **kwargs: Any) -> list[tuple[str, str]]:
        """Gather ``(remote_name, local_path)`` tuples to upload.

        Subclasses should return a list of files, excluding the README
        (which is generated by ``_build_card``).
        """

    @abstractmethod
    def _build_card(self, **kwargs: Any) -> str:
        """Generate the README.md / model card / dataset card content."""
