"""Download and verify raw datasets. Run on a machine WITH internet access.

    python scripts/download_data.py --dataset ihdp
    python scripts/download_data.py --dataset criteo
    python scripts/download_data.py --dataset all

For every file the script records HTTP status, size and SHA-256 into
data/manifests/downloads.json. Nothing is silently substituted: if a URL fails, the script
stops and tells you to find a documented alternative source and record it in the manifest.
ACIC 2016 must be downloaded manually from Zenodo (check the record's file list first).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "criteo": [
        (
            "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
            "criteo-research-uplift-v2.1.csv.gz",
            "data/raw/criteo",
        )
    ],
    "ihdp": [
        ("https://www.fredjo.com/files/ihdp_npci_1-100.train.npz", "data/raw/ihdp"),
        ("https://www.fredjo.com/files/ihdp_npci_1-100.test.npz", "data/raw/ihdp"),
    ],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url: str, dest_dir: str) -> dict:
    dest = ROOT / dest_dir / url.rsplit("/", 1)[1]
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "causalforge-downloader/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        status = r.status
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if dest.stat().st_size == 0:
        raise RuntimeError(f"empty download: {url}")
    return {
        "url": url,
        "file": str(dest.relative_to(ROOT)),
        "http_status": status,
        "bytes": dest.stat().st_size,
        "sha256": sha256(dest),
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=[*SOURCES, "all"], default="all")
    a = ap.parse_args()
    names = list(SOURCES) if a.dataset == "all" else [a.dataset]
    manifest_path = ROOT / "data/manifests/downloads.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for n in names:
        manifest[n] = [fetch(u, d) for u, d in SOURCES[n]]
        print(n, "OK", [(m["file"], m["bytes"]) for m in manifest[n]])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
