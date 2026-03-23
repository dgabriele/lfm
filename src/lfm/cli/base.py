"""Abstract base for CLI commands."""

from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
from pathlib import Path


class CLICommand(ABC):
    """Base class for all CLI subcommands."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Command name (used as subparser key)."""

    @property
    @abstractmethod
    def help(self) -> str:
        """Short help string."""

    @property
    def description(self) -> str:
        """Long description for ``--help``."""
        return self.help

    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register command-specific arguments."""

    @abstractmethod
    def execute(self, args: argparse.Namespace) -> int:
        """Run the command. Returns 0 on success."""

    @staticmethod
    def error(msg: str) -> int:
        """Print error and return non-zero exit code."""
        print(f"Error: {msg}", file=sys.stderr)
        return 1

    @staticmethod
    def validate_file_exists(path: str, label: str = "File") -> Path | None:
        """Return Path if file exists, else print error and return None."""
        p = Path(path)
        if not p.exists():
            print(f"Error: {label} not found: {p}", file=sys.stderr)
            return None
        return p
