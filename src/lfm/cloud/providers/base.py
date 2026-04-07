"""Abstract cloud provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Instance:
    """A provisioned cloud GPU instance."""

    id: str
    ip: str
    status: str
    instance_type: str
    region: str
    name: str | None = None


class CloudProvider(ABC):
    """Interface for cloud GPU providers.

    Implementations handle instance lifecycle: provision, query, terminate.
    SSH-based code upload and job execution are handled by the Deployer.
    """

    @abstractmethod
    def launch(
        self,
        instance_type: str,
        region: str,
        ssh_key_name: str,
        name: str | None = None,
    ) -> Instance:
        """Provision a new GPU instance.

        Blocks until the instance is active and SSH-ready.
        """

    @abstractmethod
    def get(self, instance_id: str) -> Instance:
        """Get current state of an instance."""

    @abstractmethod
    def terminate(self, instance_id: str) -> None:
        """Terminate an instance."""

    @abstractmethod
    def list_instances(self) -> list[Instance]:
        """List all active instances."""

    @abstractmethod
    def list_instance_types(self) -> list[dict]:
        """List available instance types with pricing."""
