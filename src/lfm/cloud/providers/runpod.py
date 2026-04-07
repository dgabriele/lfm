"""RunPod cloud provider."""

from __future__ import annotations

import logging
import os
import time

import requests

from lfm.cloud.providers.base import CloudProvider, Instance

logger = logging.getLogger(__name__)

API_BASE = "https://api.runpod.io/graphql"


class RunPodProvider(CloudProvider):
    """RunPod GPU cloud provider.

    Requires ``RUNPOD_API_KEY`` environment variable.

    Args:
        api_key: RunPod API key. Falls back to env var.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "RunPod API key required. Set RUNPOD_API_KEY env var "
                "or pass api_key= argument."
            )

    def _query(self, query: str, variables: dict | None = None) -> dict:
        """Execute a GraphQL query."""
        r = requests.post(
            f"{API_BASE}?api_key={self.api_key}",
            json={"query": query, "variables": variables or {}},
        )
        if not r.ok:
            logger.error("RunPod API %d: %s", r.status_code, r.text)
            r.raise_for_status()
        data = r.json()
        if "errors" in data:
            raise RuntimeError(f"RunPod API error: {data['errors']}")
        return data.get("data", {})

    def launch(
        self,
        instance_type: str,
        region: str,
        ssh_key_name: str,
        name: str | None = None,
    ) -> Instance:
        name = name or f"lfm-{int(time.time())}"
        logger.info(
            "Launching RunPod %s (name=%s)", instance_type, name,
        )

        query = """
        mutation {{
            podFindAndDeployOnDemand(input: {{
                name: "{name}",
                gpuTypeId: "{gpu_type}",
                imageName: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                cloudType: "SECURE",
                gpuCount: 1,
                volumeInGb: 50,
                containerDiskInGb: 20,
                startSsh: true
            }}) {{
                id
                desiredStatus
                imageName
                machine {{
                    podHostId
                }}
                runtime {{
                    uptimeInSeconds
                    ports {{
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                        type
                    }}
                }}
            }}
        }}
        """.format(name=name, gpu_type=instance_type)

        data = self._query(query)
        pod = data.get("podFindAndDeployOnDemand", {})
        if not pod or not pod.get("id"):
            raise RuntimeError(f"Launch failed: {data}")

        pod_id = pod["id"]
        logger.info("Pod %s created, waiting for ready...", pod_id)

        return self._wait_for_active(pod_id)

    def _wait_for_active(
        self, pod_id: str, timeout: float = 600,
    ) -> Instance:
        """Poll until pod is running with SSH available."""
        start = time.time()
        while time.time() - start < timeout:
            inst = self.get(pod_id)
            if inst.status == "RUNNING" and inst.ip:
                logger.info("Pod %s running at %s", inst.id, inst.ip)
                return inst
            time.sleep(10)
        raise TimeoutError(f"Pod {pod_id} not ready after {timeout}s")

    def get(self, instance_id: str) -> Instance:
        query = """
        query {
            pod(input: { podId: "%s" }) {
                id
                name
                desiredStatus
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                        type
                    }
                }
                machine {
                    gpuDisplayName
                }
                gpuCount
            }
        }
        """ % instance_id

        data = self._query(query)
        pod = data.get("pod", {})

        # Extract SSH IP and port from runtime ports
        ip = ""
        ssh_port = 22
        runtime = pod.get("runtime") or {}
        for port_info in runtime.get("ports", []):
            if port_info.get("privatePort") == 22 and port_info.get("isIpPublic"):
                ip = port_info.get("ip", "")
                ssh_port = port_info.get("publicPort", 22)
                break

        status = pod.get("desiredStatus", "UNKNOWN")
        if runtime and runtime.get("uptimeInSeconds", 0) > 0:
            status = "RUNNING"

        gpu = pod.get("machine", {}).get("gpuDisplayName", "")

        return Instance(
            id=pod.get("id", instance_id),
            ip=f"{ip}:{ssh_port}" if ip else "",
            status=status,
            instance_type=gpu,
            region="",
            name=pod.get("name"),
        )

    def terminate(self, instance_id: str) -> None:
        logger.info("Terminating pod %s", instance_id)
        query = """
        mutation {
            podTerminate(input: { podId: "%s" })
        }
        """ % instance_id
        self._query(query)

    def list_instances(self) -> list[Instance]:
        query = """
        query {
            myself {
                pods {
                    id
                    name
                    desiredStatus
                    runtime {
                        uptimeInSeconds
                        ports {
                            ip
                            isIpPublic
                            privatePort
                            publicPort
                            type
                        }
                    }
                    machine {
                        gpuDisplayName
                    }
                }
            }
        }
        """
        data = self._query(query)
        pods = data.get("myself", {}).get("pods", [])
        instances = []
        for pod in pods:
            ip = ""
            runtime = pod.get("runtime") or {}
            for port_info in runtime.get("ports", []):
                if port_info.get("privatePort") == 22 and port_info.get("isIpPublic"):
                    ip = f"{port_info['ip']}:{port_info.get('publicPort', 22)}"
                    break

            status = pod.get("desiredStatus", "UNKNOWN")
            if runtime and runtime.get("uptimeInSeconds", 0) > 0:
                status = "RUNNING"

            instances.append(Instance(
                id=pod["id"],
                ip=ip,
                status=status,
                instance_type=pod.get("machine", {}).get("gpuDisplayName", ""),
                region="",
                name=pod.get("name"),
            ))
        return instances

    def list_instance_types(self) -> list[dict]:
        query = """
        query {
            gpuTypes {
                id
                displayName
                memoryInGb
                secureCloud
                communityCloud
                lowestPrice(input: { gpuCount: 1 }) {
                    minimumBidPrice
                    uninterruptablePrice
                }
            }
        }
        """
        data = self._query(query)
        result = []
        for gpu in data.get("gpuTypes", []):
            price_info = gpu.get("lowestPrice") or {}
            price = price_info.get("uninterruptablePrice") or 0
            available = []
            if gpu.get("secureCloud"):
                available.append("secure")
            if gpu.get("communityCloud"):
                available.append("community")
            result.append({
                "name": gpu["id"],
                "description": gpu.get("displayName", ""),
                "price_cents_per_hour": int(price * 100),
                "gpu_count": 1,
                "gpu_memory_gb": gpu.get("memoryInGb", 0),
                "regions": available,
            })
        return result
