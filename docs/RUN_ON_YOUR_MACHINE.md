# Steps that must run on your machine (internet required)

The development sandbox had no network, so real datasets and some libraries could not be used.
Everything below is a single copy-paste sequence (PowerShell; use `source .venv/bin/activate` on Linux/macOS).

```powershell
cd D:\CausalForge
py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -e ".[dev,reference,tracking,api]"
pytest tests -v                                   # expect all tests to pass
ruff check src tests scripts ; mypy src           # not run during development

# 1) Download (records HTTP status, bytes, SHA-256 -> data/manifests/downloads.json)
python scripts/download_data.py --dataset all
python scripts/validate_data.py
```

Manual steps (verify each source first; do not substitute silently):

1. **ACIC 2016** — download the archive from the Zenodo record (DOI 10.5281/zenodo.18198601),
   extract under `data/raw/acic2016/`. The loader expects `x.csv` plus `zymu_*.csv` files with
   columns `z,y0,y1,mu0,mu1`. If the archive layout differs, adapt `load_acic` and note it.
2. **LaLonde** — obtain the Dehejia–Wahba NSW experimental file and a comparison-group file (CPS or
   PSID), then `python scripts/prepare_lalonde.py <in> data/raw/lalonde/nsw_experimental.csv` and
   `... data/raw/lalonde/comparison.csv`.
3. If an IHDP URL fails, find a documented alternative source and record it in the manifest.

```powershell
# 2) Real-data benchmarks (skips datasets that are missing; ~minutes, Criteo uses 1M rows by default)
$env:PYTHONPATH="src"
python scripts/run_real_benchmarks.py --ihdp-reps 100 --acic-max 30 --criteo-rows 1000000
python scripts/crosscheck_reference.py            # EconML / DoWhy cross-check (untested here)
python scripts/generate_report.py                 # adds real results to the benchmark report

# 3) Services
mlflow server --host 127.0.0.1 --port 5000        # tracking UI (runs also logged to artifacts/runs/)
$env:PYTHONPATH="api;src"; uvicorn causalforge_api.main:app --port 8000
cd frontend ; npm install ; npm run dev           # dashboard
docker compose -f docker/docker-compose.yml up --build
```

Then take the screenshots listed in `docs/evidence_checklist.md` and send me
`artifacts/real/*.csv` + `artifacts/real/summary.json` to fill in the research report.
Disk: Criteo is ~311 MB compressed (~3.5 GiB uncompressed); check free space on D:.
