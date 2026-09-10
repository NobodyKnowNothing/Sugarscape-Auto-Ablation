"""Process-parallel execution of simulation runs."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path

from judge.contract import make_env, run_model
from judge.features import Recorder

RUN_TIME_LIMIT = 300.0


@dataclass(frozen=True)
class Job:
    path: str          # materialized candidate module
    point: str
    params: tuple      # sorted (key, value) pairs
    seed: int
    steps: int
    digest: bool = False


def job(path: Path, point: str, params: dict, seed: int, steps: int, digest: bool = False) -> Job:
    return Job(str(path), point, tuple(sorted(params.items())), seed, steps, digest)


def _run(j: Job) -> dict:
    params = dict(j.params)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rec = Recorder(make_env(params), j.steps, j.seed)
            h = hashlib.sha256() if j.digest else None
            run_model(Path(j.path), params, j.seed, j.steps, rec, RUN_TIME_LIMIT, h)
            out = {"ok": True, "stats": rec.stats()}
            if h is not None:
                out["digest"] = h.hexdigest()[:16]
            return out
    except BaseException as e:  # noqa: BLE001 - any model failure is a reject reason
        tb = traceback.format_exception_only(type(e), e)[-1].strip()
        return {"ok": False, "error": tb[:300]}


_POOL = None


def _workers() -> int:
    return max(1, int(os.environ.get("JUDGE_WORKERS", (os.cpu_count() or 2) - 1)))


def pool():
    global _POOL
    if _POOL is None:
        _POOL = mp.get_context("spawn").Pool(_workers())
    return _POOL


def shutdown():
    global _POOL
    if _POOL is not None:
        _POOL.terminate()
        _POOL = None


def run_jobs(jobs: list[Job], timeout: float | None = None) -> list[dict]:
    """Run jobs in parallel and return results in order. Kills the pool on timeout."""
    if not jobs:
        return []
    if len(jobs) == 1 or _workers() == 1:
        return [_run(j) for j in jobs]
    res = pool().map_async(_run, jobs, chunksize=1)
    try:
        return res.get(timeout=timeout)
    except mp.TimeoutError:
        shutdown()
        return [{"ok": False, "error": f"batch timed out after {timeout}s"} for _ in jobs]
