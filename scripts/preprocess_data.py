"""Convert Criteo gzip CSV to partitioned files with a manifest (streaming, low memory)."""

import sys

from causalforge.data.loaders.criteo_stream import ingest

src = sys.argv[1] if len(sys.argv) > 1 else "data/raw/criteo/criteo-research-uplift-v2.1.csv.gz"
m = ingest(src, "data/processed/criteo")
print({k: m[k] for k in ("rows", "invalid_rows", "format", "sha256")})
