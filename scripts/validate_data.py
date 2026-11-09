"""Validate downloaded files against data/manifests/downloads.json (size + SHA-256)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from download_data import ROOT, sha256  # noqa: E402

m = json.loads((ROOT / "data/manifests/downloads.json").read_text())
bad = 0
for name, files in m.items():
    for f in files:
        p = ROOT / f["file"]
        ok = p.exists() and p.stat().st_size == f["bytes"] and sha256(p) == f["sha256"]
        bad += not ok
        print(("OK  " if ok else "FAIL"), name, f["file"])
sys.exit(1 if bad else 0)
