"""The pull loop: lease a rally chunk, download it, analyze, post results.

Runs on whatever box its operator starts it on — this repo never deploys it
anywhere (D15). The court host is the only thing it talks to.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

from .engine import AnalysisEngine

LOGGER = logging.getLogger("hotmama.worker")


class WorkerClient:
    def __init__(
        self,
        *,
        token: str,
        worker_name: str,
        engine: AnalysisEngine,
        base_url: str | None = None,
        producer: str = "cv_well",
        http: httpx.Client | None = None,
    ) -> None:
        if http is None:
            if base_url is None:
                raise ValueError("base_url required when no http client is injected")
            http = httpx.Client(base_url=base_url.rstrip("/"), timeout=120.0)
            self._owns_http = True
        else:
            self._owns_http = False
        self._http = http
        self._headers = {"Authorization": f"Bearer {token}"}
        self._worker = worker_name
        self._engine = engine
        self._producer = producer

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def run_once(self) -> bool:
        """Lease and process one job. Returns False when the queue is empty."""
        response = self._http.post(
            "/api/worker/lease", json={"worker": self._worker}, headers=self._headers
        )
        if response.status_code == 204:
            return False
        response.raise_for_status()
        job: dict[str, Any] = response.json()
        clip_id = job["clip_id"]
        LOGGER.info("leased %s (%s)", clip_id, job.get("label", ""))

        try:
            observations = self._engine.analyze(self._download(job), job)
            self._complete(job, ok=True, observations=observations)
            LOGGER.info("completed %s: %d observations", clip_id, len(observations))
        except Exception as err:  # noqa: BLE001 - report failure, keep the loop alive
            LOGGER.exception("analysis failed for %s", clip_id)
            self._complete(job, ok=False, error=str(err))
        return True

    def run_forever(self, poll_seconds: float = 5.0) -> None:
        idle_delay = poll_seconds
        while True:
            try:
                worked = self.run_once()
            except httpx.HTTPError as err:
                LOGGER.warning("court host unreachable: %s", err)
                time.sleep(min(idle_delay * 2, 60.0))
                continue
            time.sleep(0.1 if worked else poll_seconds)

    def _download(self, job: dict[str, Any]) -> Path:
        url = str(job["clip_url"])
        suffix = Path(url).suffix or ".mp4"
        response = self._http.get(url)
        response.raise_for_status()
        descriptor, name = tempfile.mkstemp(
            prefix=f"hotmama_{job['clip_id']}_", suffix=suffix
        )
        with open(descriptor, "wb") as handle:
            handle.write(response.content)
        return Path(name)

    def _complete(
        self,
        job: dict[str, Any],
        *,
        ok: bool,
        observations: list[dict[str, Any]] | None = None,
        error: str | None = None,
    ) -> None:
        response = self._http.post(
            "/api/worker/complete",
            json={
                "clip_id": job["clip_id"],
                "session_id": job["session_id"],
                "worker": self._worker,
                "ok": ok,
                "error": error,
                "producer": self._producer,
                "observations": observations or [],
            },
            headers=self._headers,
        )
        response.raise_for_status()
