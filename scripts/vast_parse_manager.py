#!/usr/bin/env python
"""Self-healing vast.ai parse orchestrator.

Tracks per-chunk state (pending / launching / provisioning / parsing /
done / failed), polls each in-flight instance's health + remote log,
and auto-relaunches failed chunks on fresh hosts (with an ever-growing
blocklist of bad offers).

Intended as the proper replacement for ad-hoc ``launch + subprocess``
flows I was running manually.  Invoked as:

    poetry run python scripts/vast_parse_manager.py \\
        --chunks-dir data/datasets/english-constituents-v13/chunks \\
        --out-dir data/datasets/english-constituents-v13 \\
        --adopt-existing  # register any v13-parse-* instances already running

One process, one log file (``/tmp/vast-parse-manager.log``), one status
JSON (``/tmp/vast-parse-manager.state.json``) that survives restarts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(message)s",
    handlers=[
        logging.FileHandler("/tmp/vast-parse-manager.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def _load_secrets() -> None:
    for line in Path(".secrets").read_text().splitlines():
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))


# ── Lazy import fanout helpers (launch_only, get_ssh_info, etc.) ──
_load_secrets()
import importlib.util
_fp_spec = importlib.util.spec_from_file_location(
    "fp", Path(__file__).parent / "fanout_parse_vast.py",
)
fp = importlib.util.module_from_spec(_fp_spec)
_fp_spec.loader.exec_module(fp)


@dataclass
class ChunkJob:
    chunk_idx: int
    chunk_path: str
    output_path: str
    label: str = ""
    iid: Optional[int] = None
    host: Optional[str] = None
    port: Optional[int] = None
    state: str = "pending"   # pending|launching|provisioning|parsing|done|failed
    attempt: int = 0
    last_update: float = 0.0
    errors: list[str] = field(default_factory=list)


class VastParseManager:
    """Manages N chunks in parallel with auto-healing on failures.

    Invariant: for each chunk, at most one instance is in flight at a
    time.  On failure we blocklist the offer/host and launch a fresh
    one.  The manager exits when every chunk reaches ``done`` (or
    exceeds max attempts).
    """

    MAX_ATTEMPTS = 6
    POLL_INTERVAL = 30  # seconds between health checks
    STATE_FILE = Path("/tmp/vast-parse-manager.state.json")

    def __init__(self, jobs: list[ChunkJob]) -> None:
        self.jobs = jobs
        self.blocked_offers: set[int] = set()
        self.blocked_hosts: set[int] = set()
        self._stop_flag = threading.Event()
        self._state_lock = threading.Lock()

    # ── persistence ──
    def save_state(self) -> None:
        with self._state_lock:
            self.STATE_FILE.write_text(json.dumps({
                "jobs": [asdict(j) for j in self.jobs],
                "blocked_offers": sorted(self.blocked_offers),
                "blocked_hosts": sorted(self.blocked_hosts),
            }, indent=2))

    def adopt_running(self) -> None:
        """Attach to any running instance tagged v13-parse-* by label.

        If an existing instance has label matching one of our chunks
        (e.g. ``v13-parse-2``), mark that chunk as "parsing" so we
        monitor it instead of launching a duplicate.
        """
        import requests
        h = {"Authorization": f"Bearer {os.environ['VAST_API_KEY']}"}
        r = requests.get("https://console.vast.ai/api/v0/instances/", headers=h)
        for d in r.json().get("instances", []):
            label = d.get("label") or ""
            for job in self.jobs:
                if job.state != "pending":
                    continue
                if label == f"v13-parse-{job.chunk_idx}" or \
                   label.startswith(f"v13-parse-{job.chunk_idx}-"):
                    if d.get("actual_status") == "running":
                        job.iid = int(d["id"])
                        job.host = d.get("ssh_host")
                        port = d.get("ssh_port")
                        job.port = int(port) if port else None
                        job.label = label
                        job.state = "adopted"
                        job.last_update = time.time()
                        logger.info("adopted existing %s (iid=%s)", label, job.iid)

    # ── per-chunk state machine ──
    def launch_job(self, job: ChunkJob) -> None:
        job.attempt += 1
        job.label = f"v13-parse-{job.chunk_idx}-m{job.attempt}"
        try:
            iid = fp.launch_only(job.label, self.blocked_offers)
            job.iid = iid
            job.state = "launching"
            job.last_update = time.time()
            logger.info("[%s] launched iid=%s", job.label, iid)
        except RuntimeError as e:
            msg = str(e)
            job.errors.append(f"attempt {job.attempt}: {msg[:200]}")
            if "GPU conflict" in msg:
                logger.warning("[%s] GPU conflict, will retry", job.label)
                time.sleep(15)
            else:
                logger.error("[%s] launch failed: %s", job.label, msg[:200])
                time.sleep(30)
            job.state = "pending"

    def _ssh_sanity(self, host: str, port: int, timeout: int = 10) -> bool:
        try:
            r = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
                 "-o", f"ConnectTimeout={timeout}", "-p", str(port),
                 f"root@{host}", "echo up"],
                capture_output=True, timeout=timeout + 5,
            )
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False

    def _run_provision_and_parse(self, job: ChunkJob) -> None:
        """Runs synchronously in a worker thread: provision, upload,
        parse, download, stop.  On any failure, marks ``state=failed``.
        """
        try:
            # ssh info
            if not job.host or not job.port:
                host, port = fp.get_ssh_info(job.iid, timeout=900)
                job.host, job.port = host, port
                self.save_state()

            # ssh sanity (5 min retry)
            logger.info("[%s] waiting for ssh reachable", job.label)
            start = time.time()
            while time.time() - start < 2700:  # 45 min
                if self._ssh_sanity(job.host, job.port):
                    break
                time.sleep(20)
            else:
                raise TimeoutError("ssh unreachable 45m")

            # Provision
            job.state = "provisioning"
            job.last_update = time.time()
            self.save_state()
            logger.info("[%s] provisioning", job.label)
            fp.ssh_run(job.host, job.port, fp.PROVISION, timeout=1200)

            # Upload + parse + download
            logger.info("[%s] uploading chunk", job.label)
            chunk_name = Path(job.chunk_path).name
            fp.rsync_push(job.host, job.port, Path(job.chunk_path),
                          f"/workspace/lfm/{chunk_name}")

            job.state = "parsing"
            job.last_update = time.time()
            self.save_state()
            logger.info("[%s] running parse", job.label)
            parse_cmd = (
                f"cd /workspace/lfm && "
                f"python scripts/parse_chunk_worker.py "
                f"--input {chunk_name} "
                f"--output constituents_{job.label}.txt 2>&1 | "
                f"tee parse_{job.label}.log"
            )
            fp.ssh_run(job.host, job.port, parse_cmd, timeout=60 * 60 * 4)

            logger.info("[%s] downloading result", job.label)
            fp.rsync_pull(job.host, job.port,
                          f"/workspace/lfm/constituents_{job.label}.txt",
                          Path(job.output_path))

            # Validate output
            out = Path(job.output_path)
            if not out.exists() or out.stat().st_size < 1024:
                raise RuntimeError(f"output tiny/missing: {out.stat().st_size if out.exists() else 0} bytes")

            job.state = "done"
            job.last_update = time.time()
            logger.info("[%s] DONE (%d bytes)", job.label, out.stat().st_size)
        except Exception as e:
            msg = f"attempt {job.attempt}: {type(e).__name__}: {str(e)[:200]}"
            job.errors.append(msg)
            job.state = "failed"
            logger.error("[%s] FAILED: %s", job.label, msg)
        finally:
            # Always try to stop the instance (non-destructive pause)
            if job.iid:
                try:
                    fp.stop_instance(fp.vast_api(), job.iid)
                except Exception:
                    pass
            # If failed, blocklist the host/offer for next attempt
            if job.state == "failed":
                if job.iid:
                    self.blocked_hosts.add(job.iid)
                job.iid = None
                job.host = None
                job.port = None
                if job.attempt < self.MAX_ATTEMPTS:
                    job.state = "pending"
                else:
                    logger.error("[%s] GIVING UP after %d attempts",
                                 job.label, job.attempt)
            self.save_state()

    # ── top-level loop ──
    def run(self) -> None:
        threads: dict[int, threading.Thread] = {}
        while not self._stop_flag.is_set():
            self.save_state()
            # Start workers for pending jobs
            for job in self.jobs:
                if job.state == "pending":
                    self.launch_job(job)
                    if job.state == "launching":
                        t = threading.Thread(
                            target=self._run_provision_and_parse,
                            args=(job,),
                            name=f"job-{job.chunk_idx}",
                            daemon=True,
                        )
                        t.start()
                        threads[job.chunk_idx] = t
                elif job.state == "adopted":
                    # Just monitor, don't re-provision
                    t = threading.Thread(
                        target=self._monitor_adopted,
                        args=(job,),
                        name=f"adopt-{job.chunk_idx}",
                        daemon=True,
                    )
                    t.start()
                    threads[job.chunk_idx] = t

            # Status line
            counts: dict[str, int] = {}
            for j in self.jobs:
                counts[j.state] = counts.get(j.state, 0) + 1
            logger.info("status: %s", dict(sorted(counts.items())))

            # Done?
            if all(j.state in ("done", "failed_final") for j in self.jobs):
                logger.info("all chunks resolved")
                break

            time.sleep(self.POLL_INTERVAL)

        for t in threads.values():
            t.join(timeout=5)
        self.save_state()

    def _monitor_adopted(self, job: ChunkJob) -> None:
        """For an already-running instance: wait for the output to
        appear remotely, then rsync + stop."""
        logger.info("[%s] monitoring adopted instance", job.label)
        while True:
            # Check if remote produced the output file
            r = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
                 "-o", "ConnectTimeout=10", "-p", str(job.port),
                 f"root@{job.host}",
                 f"stat -c%s /workspace/lfm/constituents_{job.label}.txt "
                 "2>/dev/null || echo 0"],
                capture_output=True, text=True, timeout=30,
            )
            try:
                size = int(r.stdout.strip().splitlines()[-1])
            except Exception:
                size = 0
            # Also check if the parse process is still running
            r2 = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
                 "-o", "ConnectTimeout=10", "-p", str(job.port),
                 f"root@{job.host}",
                 "pgrep -f parse_chunk_worker || echo dead"],
                capture_output=True, text=True, timeout=30,
            )
            parse_dead = "dead" in r2.stdout
            if size > 1024 and parse_dead:
                # Finished — pull + stop
                fp.rsync_pull(job.host, job.port,
                              f"/workspace/lfm/constituents_{job.label}.txt",
                              Path(job.output_path))
                fp.stop_instance(fp.vast_api(), job.iid)
                job.state = "done"
                job.last_update = time.time()
                logger.info("[%s] adopted: DONE", job.label)
                self.save_state()
                return
            elif parse_dead and size <= 1024:
                job.state = "failed"
                job.errors.append("adopted parse process died with no output")
                fp.stop_instance(fp.vast_api(), job.iid)
                self.save_state()
                return
            time.sleep(60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--adopt-existing", action="store_true",
                    help="Adopt already-running v13-parse-N instances")
    ap.add_argument("--skip-done", action="store_true",
                    help="Skip chunks whose output file already exists")
    args = ap.parse_args()

    jobs: list[ChunkJob] = []
    for chunk in sorted(args.chunks_dir.glob("chunk_*.txt")):
        idx = int(chunk.stem.split("_")[1])
        out = args.out_dir / f"constituents_v13-parse-{idx}.txt"
        if args.skip_done and out.exists() and out.stat().st_size > 1024:
            logger.info("skipping chunk_%d (output exists, %d bytes)",
                        idx, out.stat().st_size)
            jobs.append(ChunkJob(
                chunk_idx=idx, chunk_path=str(chunk),
                output_path=str(out), state="done",
            ))
            continue
        jobs.append(ChunkJob(
            chunk_idx=idx, chunk_path=str(chunk), output_path=str(out),
        ))

    mgr = VastParseManager(jobs)
    if args.adopt_existing:
        mgr.adopt_running()
    mgr.run()

    # Final merge
    merged = args.out_dir / "constituents.txt"
    lines = 0
    with merged.open("w") as out_f:
        for job in jobs:
            part = Path(job.output_path)
            if part.exists():
                for line in part.read_text().splitlines():
                    out_f.write(line + "\n")
                    lines += 1
    logger.info("merged → %s (%d lines)", merged, lines)


if __name__ == "__main__":
    main()
