"""Vast.ai cloud provider."""

from __future__ import annotations

import logging
import os
import time

import requests

from lfm.cloud.providers.base import CloudProvider, Instance

logger = logging.getLogger(__name__)

API_BASE = "https://console.vast.ai/api/v0"


class VastAIProvider(CloudProvider):
    """Vast.ai GPU cloud provider.

    Requires ``VAST_API_KEY`` environment variable.
    No minimum deposit — pay-as-you-go.

    Args:
        api_key: Vast.ai API key. Falls back to env var.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("VAST_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Vast.ai API key required. Set VAST_API_KEY env var "
                "or pass api_key= argument."
            )
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _get(self, path: str) -> dict:
        r = requests.get(f"{API_BASE}{path}", headers=self._headers)
        if not r.ok:
            logger.error("Vast API GET %s → %d: %s", path, r.status_code, r.text)
            r.raise_for_status()
        return r.json()

    def _post(self, path: str, data: dict) -> dict:
        r = requests.post(
            f"{API_BASE}{path}", headers=self._headers, json=data,
        )
        if not r.ok:
            logger.error("Vast API POST %s → %d: %s", path, r.status_code, r.text)
            r.raise_for_status()
        return r.json()

    def _put(self, path: str, data: dict) -> dict:
        r = requests.put(
            f"{API_BASE}{path}", headers=self._headers, json=data,
        )
        if not r.ok:
            logger.error("Vast API PUT %s → %d: %s", path, r.status_code, r.text)
            r.raise_for_status()
        return r.json()

    def _delete(self, path: str) -> dict:
        r = requests.delete(f"{API_BASE}{path}", headers=self._headers)
        if not r.ok:
            logger.error("Vast API DELETE %s → %d: %s", path, r.status_code, r.text)
            r.raise_for_status()
        return r.json()

    def _find_offer(
        self, gpu_name: str, min_gpu_ram: int = 0,
        min_reliability: float = 0.95,
        min_download_mbps: float = 300.0,
    ) -> dict | None:
        """Search for the cheapest offer meeting reliability + bandwidth
        thresholds.

        The cheapest-by-default offers on Vast.ai often come from hosts
        with GPU-passthrough bugs or tiny uplinks that stall multi-GB
        docker pulls before SSH is reachable.  We require ``reliability2
        ≥ min_reliability`` (Vast.ai's 0-1 stability score) and
        ``inet_down ≥ min_download_mbps`` so the image actually pulls
        in-budget, then pick the cheapest among qualifying offers.
        """
        query = {
            "verified": {"eq": True},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "num_gpus": {"eq": 1},
            "gpu_name": {"eq": gpu_name},
            "reliability2": {"gte": min_reliability},
            "inet_down": {"gte": min_download_mbps},
            "type": "on-demand",
            "order": [["dph_total", "asc"]],
            "limit": 20,
        }
        if min_gpu_ram > 0:
            query["gpu_ram"] = {"gte": min_gpu_ram * 1024}  # MB

        data = self._post("/bundles/", query)
        offers = data.get("offers", [])
        if not offers:
            return None
        return offers[0]

    def launch(
        self,
        instance_type: str,
        region: str,
        ssh_key_name: str,
        name: str | None = None,
        image: str = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
    ) -> Instance:
        """Launch a Vast.ai instance.

        Args:
            instance_type: GPU name (e.g., "A100_PCIE" or "A100_SXM4").
                Also accepts "A100-80GB" which maps to appropriate GPU name.
            region: Ignored (Vast.ai auto-selects cheapest region).
            ssh_key_name: Ignored (uses account SSH keys).
            name: Instance label.
        """
        name = name or f"lfm-{int(time.time())}"

        # Normalize GPU name — Vast.ai uses space-separated names
        gpu_name_map = {
            "A100_80GB": "A100 SXM4",
            "A100-80GB": "A100 SXM4",
            "A100_40GB": "A100 PCIE",
            "A100_SXM4": "A100 SXM4",
            "A100_PCIE": "A100 PCIE",
            "H100": "H100 SXM",
            "H100_SXM": "H100 SXM",
        }
        gpu_name = gpu_name_map.get(instance_type, instance_type)
        # Underscores to spaces for any unmapped names
        gpu_name = gpu_name.replace("_", " ")

        logger.info("Searching for %s offers...", gpu_name)
        offer = self._find_offer(gpu_name)
        if offer is None:
            raise RuntimeError(
                f"No {gpu_name} offers available. "
                f"Try a different GPU type."
            )

        offer_id = offer["id"]
        price = offer.get("dph_total", 0)
        logger.info(
            "Found offer %s: %s, $%.2f/hr, %dGB GPU RAM",
            offer_id, gpu_name, price,
            offer.get("gpu_ram", 0) // 1024,
        )

        # Create instance
        data = self._put(f"/asks/{offer_id}/", {
            "image": image,
            "label": name,
            "disk": 50,
            "runtype": "ssh_direct",
        })

        if not data.get("success"):
            raise RuntimeError(f"Launch failed: {data}")

        instance_id = str(data["new_contract"])
        logger.info("Instance %s created, waiting for SSH...", instance_id)

        return self._wait_for_active(instance_id)

    def _wait_for_active(
        self, instance_id: str, timeout: float = 600,
    ) -> Instance:
        """Poll until instance is running with SSH."""
        start = time.time()
        while time.time() - start < timeout:
            # Use list_instances — more reliable than single-instance endpoint
            for inst in self.list_instances():
                if inst.id == instance_id:
                    if inst.status == "running" and inst.ip:
                        logger.info(
                            "Instance %s running at %s", inst.id, inst.ip,
                        )
                        return inst
                    logger.debug(
                        "Instance %s: status=%s ip=%s",
                        instance_id, inst.status, inst.ip,
                    )
                    break
            time.sleep(10)
        raise TimeoutError(
            f"Instance {instance_id} not ready after {timeout}s"
        )

    def get(self, instance_id: str) -> Instance:
        data = self._get(f"/instances/{instance_id}/")

        # SSH connection info
        ssh_host = data.get("ssh_host", "")
        ssh_port = data.get("ssh_port", 22)
        ip = f"{ssh_host}:{ssh_port}" if ssh_host else ""

        status = data.get("actual_status", "unknown")

        return Instance(
            id=str(data.get("id", instance_id)),
            ip=ip,
            status=status,
            instance_type=data.get("gpu_name", ""),
            region="",
            name=data.get("label"),
        )

    def terminate(self, instance_id: str) -> None:
        logger.info("Destroying instance %s", instance_id)
        self._delete(f"/instances/{instance_id}/")

    def stop(self, instance_id: str) -> None:
        """Pause the instance (keeps storage, halts GPU billing).

        Non-destructive: the container and its mounted volumes remain on
        the host.  Call ``start`` to resume.  Prefer this over
        ``terminate`` whenever the user may want to resume work later.
        """
        logger.info("Stopping instance %s (non-destructive)", instance_id)
        self._put(f"/instances/{instance_id}/", {"state": "stopped"})

    def start(self, instance_id: str) -> None:
        """Resume a previously stopped instance."""
        logger.info("Starting instance %s", instance_id)
        self._put(f"/instances/{instance_id}/", {"state": "running"})

    def list_instances(self) -> list[Instance]:
        data = self._get("/instances/")
        instances = []
        for inst in data.get("instances", []):
            ssh_host = inst.get("ssh_host", "")
            ssh_port = inst.get("ssh_port", 22)
            ip = f"{ssh_host}:{ssh_port}" if ssh_host else ""

            instances.append(Instance(
                id=str(inst["id"]),
                ip=ip,
                status=inst.get("actual_status", "unknown"),
                instance_type=inst.get("gpu_name", ""),
                region="",
                name=inst.get("label"),
            ))
        return instances

    def list_instance_types(self) -> list[dict]:
        """List available GPU types with cheapest on-demand pricing."""
        query = {
            "verified": {"eq": True},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "num_gpus": {"eq": 1},
            "type": "on-demand",
            "order": [["dph_total", "asc"]],
            "limit": 200,
        }
        data = self._post("/bundles/", query)

        # Deduplicate by GPU name, keeping cheapest
        by_gpu: dict[str, dict] = {}
        for offer in data.get("offers", []):
            name = offer.get("gpu_name", "unknown")
            price = offer.get("dph_total", 0)
            if name not in by_gpu or price < by_gpu[name]["price_cents_per_hour"] / 100:
                by_gpu[name] = {
                    "name": name,
                    "description": name.replace("_", " "),
                    "price_cents_per_hour": int(price * 100),
                    "gpu_count": 1,
                    "gpu_memory_gb": offer.get("gpu_ram", 0) // 1024,
                    "regions": ["on-demand"],
                }

        return list(by_gpu.values())
