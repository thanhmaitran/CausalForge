"""File-based experiment tracking (MLflow-compatible fields; mirrors to MLflow if installed)."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown (not a git checkout)"


def file_sha256(path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


@contextmanager
def track_run(
    name: str,
    params: dict | None = None,
    root: str = "artifacts/runs",
    dataset_files: list[str] | None = None,
):
    """Usage: with track_run('exp01', {...}) as run: ...; run.log_metric('rmse', v)."""
    run_id = f"{name}-{time.strftime('%Y%m%dT%H%M%S')}"
    d = Path(root) / run_id
    d.mkdir(parents=True, exist_ok=True)
    metrics: dict = {}

    class Run:
        dir = d

        @staticmethod
        def log_metric(k, v):
            metrics[k] = float(v)

        @staticmethod
        def log_artifact(path):
            (d / "artifacts.txt").open("a").write(str(path) + "\n")

    import numpy as np
    import sklearn

    env = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }
    (d / "environment.json").write_text(json.dumps(env, indent=2))
    (d / "params.json").write_text(json.dumps(params or {}, indent=2, default=str))
    (d / "git_commit.txt").write_text(_git_commit())
    (d / "dataset_manifest.json").write_text(
        json.dumps({f: file_sha256(f) for f in (dataset_files or [])}, indent=2)
    )
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        yield Run
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics.update(runtime_s=time.perf_counter() - t0, peak_python_mem_mb=peak / 1e6)
        (d / "metrics.json").write_text(json.dumps(metrics, indent=2))
        try:  # optional mirror
            import mlflow  # type: ignore[import-not-found]

            with mlflow.start_run(run_name=run_id):
                mlflow.log_params({k: str(v) for k, v in (params or {}).items()})
                mlflow.log_metrics(metrics)
        except Exception:
            pass
