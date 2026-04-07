"""Lambda Labs cloud provider."""

from __future__ import annotations

import logging
import os
import time

import requests

from lfm.cloud.providers.base import CloudProvider, Instance

logger = logging.getLogger(__name__)

API_BASE = "https://cloud.lambdalabs.com/api/v1"


class LambdaLabsProvider(CloudProvider):
    """Lambda Labs GPU cloud provider.

    Requires ``LAMBDA_API_KEY`` environment variable.

    Args:
        api_key: Lambda Labs API key. Falls back to env var.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("LAMBDA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Lambda API key required. Set LAMBDA_API_KEY env var "
                "or pass api_key= argument."
            )
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _get(self, path: str) -> dict:
        r = requests.get(f"{API_BASE}{path}", headers=self._headers)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, data: dict) -> dict:
        r = requests.post(
            f"{API_BASE}{path}", headers=self._headers, json=data,
        )
        if not r.ok:
            logger.error("API %s %s → %d: %s", "POST", path, r.status_code, r.text)
            r.raise_for_status()
        return r.json()

    def _delete(self, path: str, data: dict | None = None) -> dict:
        r = requests.delete(
            f"{API_BASE}{path}", headers=self._headers, json=data,
        )
        r.raise_for_status()
        return r.json()

    def launch(
        self,
        instance_type: str,
        region: str,
        ssh_key_name: str,
        name: str | None = None,
    ) -> Instance:
        name = name or f"lfm-{int(time.time())}"
        logger.info(
            "Launching %s in %s (name=%s)", instance_type, region, name,
        )

        resp = self._post("/instance-operations/launch", {
            "instance_type_name": instance_type,
            "region_name": region,
            "ssh_key_names": [ssh_key_name],
            "name": name,
            "quantity": 1,
        })

        instance_ids = resp.get("data", {}).get("instance_ids", [])
        if not instance_ids:
            raise RuntimeError(f"Launch failed: {resp}")

        instance_id = instance_ids[0]
        logger.info("Instance %s launched, waiting for active...", instance_id)

        return self._wait_for_active(instance_id)

    def _wait_for_active(
        self, instance_id: str, timeout: float = 600,
    ) -> Instance:
        """Poll until instance is active."""
        start = time.time()
        while time.time() - start < timeout:
            inst = self.get(instance_id)
            if inst.status == "active":
                logger.info("Instance %s active at %s", inst.id, inst.ip)
                return inst
            time.sleep(10)
        raise TimeoutError(
            f"Instance {instance_id} not active after {timeout}s"
        )

    def get(self, instance_id: str) -> Instance:
        resp = self._get(f"/instances/{instance_id}")
        data = resp.get("data", {})
        return Instance(
            id=data["id"],
            ip=data.get("ip", ""),
            status=data.get("status", "unknown"),
            instance_type=data.get("instance_type", {}).get("name", ""),
            region=data.get("region", {}).get("name", ""),
            name=data.get("name"),
        )

    def terminate(self, instance_id: str) -> None:
        logger.info("Terminating instance %s", instance_id)
        self._post("/instance-operations/terminate", {
            "instance_ids": [instance_id],
        })

    def list_instances(self) -> list[Instance]:
        resp = self._get("/instances")
        return [
            Instance(
                id=d["id"],
                ip=d.get("ip", ""),
                status=d.get("status", "unknown"),
                instance_type=d.get("instance_type", {}).get("name", ""),
                region=d.get("region", {}).get("name", ""),
                name=d.get("name"),
            )
            for d in resp.get("data", [])
        ]

    def list_instance_types(self) -> list[dict]:
        resp = self._get("/instance-types")
        result = []
        for name, info in resp.get("data", {}).items():
            desc = info.get("instance_type", {})
            regions = info.get("regions_with_capacity_available", [])
            result.append({
                "name": name,
                "description": desc.get("description", ""),
                "price_cents_per_hour": desc.get(
                    "price_cents_per_hour", 0,
                ),
                "gpu_count": desc.get("specs", {}).get(
                    "gpus", 0,
                ),
                "gpu_memory_gb": desc.get("specs", {}).get(
                    "gpu_memory_gib", 0,
                ),
                "regions": [r["name"] for r in regions],
            })
        return result
