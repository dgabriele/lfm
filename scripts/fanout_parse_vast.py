#!/usr/bin/env python
"""Fan out sentence parsing across N vast.ai 3090 instances.

Splits a local ``sentences.txt`` into N roughly-equal chunks, launches
N vast.ai 3090 instances (US/EU only), and on each in parallel:

  1. Waits for SSH.
  2. Provisions (apt: git python3.11 tmux rsync; pip: torch + lfm + stanza).
  3. Rsyncs the chunk.
  4. Runs ``parse_chunk_worker.py`` (Stanza GPU, 2 workers per card).
  5. Rsyncs the wrapped ``constituents_i.txt`` back.
  6. Stops the instance (non-destructive).

Finally merges all chunks into a single ``constituents.txt`` locally.

Assumes ``.secrets`` in the project root exports ``VAST_API_KEY``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(message)s",
)
logger = logging.getLogger(__name__)

RELIABLE_COUNTRIES = [
    "US", "CA", "GB", "DE", "NL", "FR", "SE", "CH", "NO",
    "FI", "IE", "AU", "JP", "KR", "SG", "TW", "PL", "IT", "ES", "CZ",
]


# ──────────────────────────────────────────────────────────────────────
# Vast helpers
# ──────────────────────────────────────────────────────────────────────
def vast_api() -> object:
    env_path = Path(".secrets")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    from lfm.cloud.providers.vastai import VastAIProvider
    return VastAIProvider()


def _vast_headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ['VAST_API_KEY']}",
        "Content-Type": "application/json",
    }


def _find_offers(used_offer_ids: set[int], limit: int = 20) -> list[dict]:
    import requests
    query = {
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "num_gpus": {"eq": 1},
        "gpu_name": {"eq": "RTX 3090"},
        "reliability2": {"gte": 0.98},
        "inet_down": {"gte": 400.0},
        "cpu_ram": {"gte": 24000},
        "geolocation": {"in": RELIABLE_COUNTRIES},
        "type": "on-demand",
        "order": [["dph_total", "asc"]],
        "limit": limit,
    }
    r = requests.post(
        "https://console.vast.ai/api/v0/bundles/",
        headers=_vast_headers(), json=query,
    )
    return [o for o in r.json().get("offers", []) if o["id"] not in used_offer_ids]


def launch_only(label: str, used_offers: set[int]) -> int:
    """Fire a launch request and return the new instance id immediately.

    Does NOT wait for "running" — the per-instance worker polls for
    ssh_host/port on its own, in parallel.  Caller adds the selected
    offer id to ``used_offers`` so siblings don't race on one host.
    """
    import requests
    offers = _find_offers(used_offers)
    if not offers:
        raise RuntimeError("no reliable 3090 offers available")
    offer = offers[0]
    used_offers.add(offer["id"])
    logger.info(
        "[%s] launching offer %s ($%.3f/hr, %s)",
        label, offer["id"], offer["dph_total"], offer.get("geolocation"),
    )
    r = requests.put(
        f"https://console.vast.ai/api/v0/asks/{offer['id']}/",
        headers=_vast_headers(),
        json={
            # pytorch/pytorch is vast's default image and is pre-cached on
            # most hosts — nvidia/cuda often stalls on slow/throttled pulls.
            "image": "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
            "label": label, "disk": 30, "runtype": "ssh_direct",
        },
    )
    d = r.json()
    if not d.get("success"):
        raise RuntimeError(f"launch failed: {d}")
    return int(d["new_contract"])


def get_ssh_info(iid: int, timeout: int = 1800) -> tuple[str, int]:
    """Poll vast until ``iid`` has ssh_host and ssh_port populated."""
    import requests
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(
            "https://console.vast.ai/api/v0/instances/", headers=_vast_headers(),
        )
        for inst in r.json().get("instances", []):
            if int(inst["id"]) == iid:
                host = inst.get("ssh_host")
                port = inst.get("ssh_port")
                if host and port:
                    return host, int(port)
        time.sleep(15)
    raise TimeoutError(f"instance {iid} never got ssh host/port")


def stop_instance(provider, iid: int) -> None:
    try:
        provider.stop(str(iid))
        logger.info("stopped instance %s", iid)
    except Exception as e:
        logger.warning("failed to stop %s: %s", iid, e)


# ──────────────────────────────────────────────────────────────────────
# SSH helpers
# ──────────────────────────────────────────────────────────────────────
def ssh_run(host: str, port: int, cmd: str, timeout: int = 600) -> str:
    full = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes", "-o", f"ConnectTimeout=30",
        "-p", str(port), f"root@{host}", cmd,
    ]
    r = subprocess.run(
        full, check=False, capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"ssh {host}:{port} failed ({r.returncode}): "
            f"stderr={r.stderr[-600:]}  cmd={cmd[:140]}",
        )
    return r.stdout


def wait_for_ssh(host: str, port: int, label: str, timeout: int = 3600) -> None:
    logger.info("[%s] waiting for SSH on %s:%s", label, host, port)
    start = time.time()
    while time.time() - start < timeout:
        try:
            ssh_run(host, port, "echo up", timeout=15)
            logger.info("[%s] SSH up after %.0fs", label, time.time() - start)
            return
        except Exception:
            time.sleep(20)
    raise TimeoutError(f"[{label}] SSH {host}:{port} not reachable after {timeout}s")


def rsync_push(host: str, port: int, local: Path, remote: str) -> None:
    subprocess.run(
        [
            "rsync", "-avz", "--partial",
            "-e", f"ssh -o StrictHostKeyChecking=no -p {port}",
            str(local), f"root@{host}:{remote}",
        ],
        check=True, capture_output=True,
    )


def rsync_pull(host: str, port: int, remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync", "-avz", "--partial",
            "-e", f"ssh -o StrictHostKeyChecking=no -p {port}",
            f"root@{host}:{remote}", str(local),
        ],
        check=True, capture_output=True,
    )


# ──────────────────────────────────────────────────────────────────────
# Per-instance worker flow
# ──────────────────────────────────────────────────────────────────────
PROVISION = r"""
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git python3 python3-pip tmux rsync build-essential software-properties-common 2>&1 | tail -2
add-apt-repository -y ppa:deadsnakes/ppa 2>&1 | tail -2
apt-get update -qq
apt-get install -y -qq python3.11 python3.11-dev python3.11-venv 2>&1 | tail -2
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 2>&1 | tail -1
mkdir -p /workspace && cd /workspace
if [ ! -d lfm ]; then
  git clone https://github.com/dgabriele/lfm.git
fi
cd lfm && git pull --ff-only 2>&1 | tail -1
python3.11 -m pip install --quiet --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
python3.11 -m pip install --quiet --no-cache-dir -e . 2>&1 | tail -1
python3.11 -m pip install --quiet --no-cache-dir stanza h5py sentencepiece 2>&1 | tail -1
# Pre-download Stanza English models to avoid first-use download failures
# inside multiprocessing spawn workers (which silently crash if the
# constituency model is missing).
python3.11 -c "import stanza; stanza.download('en', processors='tokenize,pos,constituency', verbose=False)"
python3.11 -c "import torch, stanza; print('ready, cuda=', torch.cuda.is_available())"
"""


def run_worker(
    iid: int, chunk_path: Path,
    out_local: Path, label: str, provider,
) -> None:
    try:
        logger.info("[%s] waiting for vast to assign ssh host/port", label)
        host, port = get_ssh_info(iid)
        logger.info("[%s] ssh info: %s:%s", label, host, port)
        wait_for_ssh(host, port, label)
        logger.info("[%s] provisioning (~5 min)", label)
        ssh_run(host, port, PROVISION, timeout=1800)
        logger.info("[%s] uploading %s", label, chunk_path.name)
        rsync_push(host, port, chunk_path, f"/workspace/lfm/{chunk_path.name}")
        logger.info("[%s] running parse worker (Stanza GPU)", label)
        parse_cmd = (
            f"cd /workspace/lfm && "
            f"python3.11 scripts/parse_chunk_worker.py "
            f"--input {chunk_path.name} "
            f"--output constituents_{label}.txt 2>&1 | tee parse_{label}.log"
        )
        ssh_run(host, port, parse_cmd, timeout=60 * 60 * 4)
        logger.info("[%s] downloading constituents", label)
        rsync_pull(
            host, port,
            f"/workspace/lfm/constituents_{label}.txt",
            out_local,
        )
        logger.info("[%s] done, stopping instance", label)
    finally:
        stop_instance(provider, iid)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", type=Path,
                    default=Path("data/datasets/english-constituents-v13/sentences.txt"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/datasets/english-constituents-v13"))
    ap.add_argument("--num-instances", type=int, default=4)
    args = ap.parse_args()

    if not args.sentences.exists():
        raise SystemExit(f"missing {args.sentences}")
    all_sents = args.sentences.read_text().splitlines()
    logger.info("total sentences: %d", len(all_sents))

    N = args.num_instances
    chunk_size = (len(all_sents) + N - 1) // N
    chunks_dir = args.out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[Path] = []
    for i in range(N):
        p = chunks_dir / f"chunk_{i}.txt"
        p.write_text("\n".join(all_sents[i * chunk_size:(i + 1) * chunk_size]))
        chunk_paths.append(p)
        logger.info("wrote %s (%d lines)", p, sum(1 for _ in p.read_text().splitlines()))

    provider = vast_api()

    # Fire all N launches up-front — each one is just an API call and
    # takes ~2s.  Per-instance workers poll for their own ssh info and
    # provision in parallel.
    used_offers: set[int] = set()
    launched: list[tuple[int, Path, Path, str]] = []
    for i, chunk in enumerate(chunk_paths):
        label = f"v13-parse-{i}"
        iid = launch_only(label, used_offers)
        out = args.out_dir / f"constituents_{label}.txt"
        launched.append((iid, chunk, out, label))
        time.sleep(3)

    logger.info("%d instances launched (polling for ssh in parallel)", len(launched))

    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
        futs = [
            ex.submit(run_worker, iid, chunk, out, label, provider)
            for iid, chunk, out, label in launched
        ]
        for f in concurrent.futures.as_completed(futs):
            try:
                f.result()
            except Exception as e:
                logger.error("worker failed: %s", e)

    # Merge
    merged = args.out_dir / "constituents.txt"
    lines = 0
    with merged.open("w") as out_f:
        for _, _, part, _ in launched:
            if part.exists():
                for line in part.read_text().splitlines():
                    out_f.write(line + "\n")
                    lines += 1
            else:
                logger.warning("missing part %s", part)
    logger.info("merged → %s (%d lines)", merged, lines)


if __name__ == "__main__":
    main()
