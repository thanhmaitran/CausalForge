from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "artifacts" / "results"
OUTPUT_DIR = ROOT / "docs" / "screenshots" / "01-experiments"
OUTPUT_FILE = OUTPUT_DIR / "02-experiment-summary.html"
ENV_FILE = RESULTS_DIR / "environment.json"

def esc(value: object) -> str:
    return html.escape(str(value), quote=True)

def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)
    return columns, rows

if not RESULTS_DIR.is_dir():
    raise SystemExit(f"Results directory not found: {RESULTS_DIR}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

csv_files = sorted(RESULTS_DIR.glob("*.csv"))
if not csv_files:
    raise SystemExit(f"No CSV files found in {RESULTS_DIR}")

sections: list[str] = []
total_rows = 0
empty_files: list[str] = []

for path in csv_files:
    columns, rows = read_csv(path)
    total_rows += len(rows)

    if not rows:
        empty_files.append(path.name)
        body = (
            '<p class="empty">No data rows are present in this CSV file. '
            'The file is retained as an empty experiment output.</p>'
        )
    elif not columns:
        body = '<p class="empty">No column headers found.</p>'
    else:
        header = "".join(f"<th>{esc(col)}</th>" for col in columns)
        row_html = []
        for row in rows:
            cells = []
            for col in columns:
                value = row.get(col, "")
                shown = "—" if value is None or value.strip() == "" else value
                cells.append(f"<td>{esc(shown)}</td>")
            row_html.append("<tr>" + "".join(cells) + "</tr>")

        body = (
            '<div class="table-wrap"><table><thead><tr>'
            + header
            + "</tr></thead><tbody>"
            + "".join(row_html)
            + "</tbody></table></div>"
        )

    sections.append(
        f"""
        <section class="experiment" id="{esc(path.stem)}">
          <div class="section-heading">
            <h2>{esc(path.stem)}</h2>
            <span class="row-count">{len(rows)} rows</span>
          </div>
          <p class="filename">{esc(path.name)}</p>
          {body}
        </section>
        """
    )

environment_html = ""
if ENV_FILE.exists():
    try:
        environment = json.loads(ENV_FILE.read_text(encoding="utf-8"))
        env_text = json.dumps(environment, indent=2, ensure_ascii=False)
        environment_html = (
            '<section class="experiment"><h2>Execution environment</h2>'
            f'<pre>{esc(env_text)}</pre></section>'
        )
    except (json.JSONDecodeError, OSError) as exc:
        environment_html = (
            '<section class="experiment"><h2>Execution environment</h2>'
            f'<p class="empty">Could not read environment metadata: {esc(exc)}</p>'
            '</section>'
        )

generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
empty_text = ", ".join(empty_files) if empty_files else "None"

document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CausalForge — Experiment Results</title>
<style>
:root {{
  color-scheme: light;
  --ink: #172033;
  --muted: #64748b;
  --line: #dce3ec;
  --paper: #ffffff;
  --canvas: #f3f6fa;
  --accent: #3157a4;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--canvas);
  color: var(--ink);
  font-family: "Segoe UI", Arial, sans-serif;
  line-height: 1.5;
}}
main {{ max-width: 1440px; margin: 0 auto; padding: 38px 30px 60px; }}
.hero {{
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 30px 34px;
  margin-bottom: 22px;
}}
.eyebrow {{
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 1.5px;
  text-transform: uppercase;
}}
h1 {{ font-size: 34px; line-height: 1.15; margin: 9px 0 12px; }}
.subtitle {{ color: var(--muted); margin: 0; }}
.notice {{
  margin-top: 22px;
  padding: 13px 16px;
  background: #fff8e8;
  border-left: 4px solid #d49a2a;
  border-radius: 5px;
  color: #624819;
  font-size: 14px;
}}
.metrics {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin: 18px 0 28px;
}}
.metric {{
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 20px 22px;
}}
.metric .value {{ font-size: 29px; font-weight: 700; }}
.metric .label {{ color: var(--muted); font-size: 13px; }}
.experiment {{
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 22px;
  margin: 16px 0;
}}
.section-heading {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}}
h2 {{ font-size: 20px; margin: 0; }}
.filename {{ color: var(--muted); font-size: 12px; margin: 4px 0 14px; }}
.row-count {{
  color: var(--accent);
  background: #edf2ff;
  border-radius: 20px;
  padding: 4px 10px;
  font-size: 12px;
  white-space: nowrap;
}}
.table-wrap {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
th, td {{
  padding: 9px 11px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  white-space: nowrap;
}}
th {{
  background: #f1f4f9;
  color: #334155;
  font-weight: 700;
}}
tbody tr:nth-child(even) {{ background: #fafbfd; }}
.empty {{ color: #8a5a13; background: #fff8e8; padding: 12px; border-radius: 6px; }}
pre {{
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  background: #f5f7fa;
  padding: 14px;
  border-radius: 8px;
  font-size: 12px;
}}
footer {{ color: var(--muted); font-size: 12px; margin-top: 28px; }}
@media (max-width: 700px) {{
  main {{ padding: 18px 12px 36px; }}
  .hero {{ padding: 22px; }}
  h1 {{ font-size: 27px; }}
  .metrics {{ grid-template-columns: 1fr; }}
}}
@media print {{
  body {{ background: white; }}
  main {{ max-width: none; padding: 10px; }}
  .experiment {{ break-inside: avoid; }}
}}
</style>
</head>
<body>
<main>
  <header class="hero">
    <div class="eyebrow">Causal inference research engineering</div>
    <h1>CausalForge Experiment Results</h1>
    <p class="subtitle">Consolidated outputs from the project's experiment CSV files.</p>
    <div class="notice">
      <strong>Interpretation note:</strong> These results are from synthetic simulation
      experiments. They demonstrate controlled experimental behaviour and are not
      evidence of validation on real-world datasets.
    </div>
  </header>

  <div class="metrics">
    <div class="metric"><div class="value">{len(csv_files)}</div>
      <div class="label">CSV experiment files</div></div>
    <div class="metric"><div class="value">{len(csv_files) - len(empty_files)}</div>
      <div class="label">CSV files containing rows</div></div>
    <div class="metric"><div class="value">{total_rows}</div>
      <div class="label">Total result rows</div></div>
  </div>

  <section class="experiment">
    <h2>Dataset inventory</h2>
    <p class="subtitle">Source: artifacts/results/</p>
    <p><strong>Empty CSV files:</strong> {esc(empty_text)}</p>
    <p><strong>Generated:</strong> {esc(generated_at)}</p>
  </section>

  {''.join(sections)}
  {environment_html}

  <footer>
    CausalForge · Generated from local CSV artifacts · Original result files were not modified.
  </footer>
</main>
</body>
</html>
"""

OUTPUT_FILE.write_text(document, encoding="utf-8")
print(f"Created: {OUTPUT_FILE}")
print(f"CSV files: {len(csv_files)}")
print(f"Non-empty CSV files: {len(csv_files) - len(empty_files)}")
print(f"Total data rows: {total_rows}")
print(f"Empty CSV files: {empty_text}")