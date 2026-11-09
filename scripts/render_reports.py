from pathlib import Path
import markdown
import html
import re

ROOT = Path(__file__).resolve().parents[1]

REPORTS = [
    {
        "source": ROOT / "reports" / "benchmark_report" / "benchmark_report.md",
        "output": ROOT / "docs" / "screenshots" / "04-reports" / "01-benchmark-report.html",
        "title": "CausalForge Benchmark Report",
        "subtitle": "Experimental evaluation and benchmark evidence",
    },
    {
        "source": ROOT / "reports" / "research_report" / "causalforge_research_report.md",
        "output": ROOT / "docs" / "screenshots" / "04-reports" / "02-research-report.html",
        "title": "CausalForge Research Report",
        "subtitle": "Research methodology, experimental results and discussion",
    },
    {
        "source": ROOT / "reports" / "technical_report" / "causalforge_technical_report.md",
        "output": ROOT / "docs" / "screenshots" / "04-reports" / "03-technical-report.html",
        "title": "CausalForge Technical Report",
        "subtitle": "System architecture, implementation and testing",
    },
]

CSS = """
:root {
    color-scheme: light;
    font-family: Inter, "Segoe UI", Arial, sans-serif;
    color: #202938;
    background: #eef2f7;
}
* { box-sizing: border-box; }
body {
    margin: 0;
    padding: 42px 20px;
    line-height: 1.72;
}
.page {
    max-width: 1080px;
    margin: 0 auto;
    padding: 64px 76px;
    background: white;
    border: 1px solid #e1e7ef;
    border-radius: 12px;
    box-shadow: 0 12px 40px rgba(25, 40, 65, .08);
}
.cover {
    border-bottom: 2px solid #dce5ef;
    padding-bottom: 28px;
    margin-bottom: 38px;
}
.brand {
    color: #315c91;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.subtitle {
    color: #66758a;
    font-size: 15px;
}
h1 {
    color: #152b49;
    font-size: 30px;
    line-height: 1.25;
    margin: 16px 0;
}
h2 {
    color: #1c426c;
    font-size: 22px;
    margin-top: 42px;
    padding-bottom: 9px;
    border-bottom: 1px solid #e4eaf1;
}
h3 {
    color: #294d73;
    font-size: 18px;
    margin-top: 28px;
}
h4 { color: #355675; }
p { margin: 13px 0; }
a { color: #245d9c; overflow-wrap: anywhere; }
ul, ol { padding-left: 28px; }
li { margin: 5px 0; }
blockquote {
    margin: 20px 0;
    padding: 12px 20px;
    border-left: 4px solid #6c91bd;
    background: #f3f7fb;
    color: #42536a;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin: 22px 0;
    font-size: 13px;
}
th, td {
    padding: 10px 12px;
    border: 1px solid #dce4ed;
    text-align: left;
    vertical-align: top;
}
th {
    background: #edf3f9;
    color: #244669;
}
tr:nth-child(even) td { background: #fafbfd; }
pre {
    padding: 18px;
    border-radius: 8px;
    overflow-x: auto;
    background: #172334;
    color: #edf3fa;
    font-size: 12px;
    line-height: 1.55;
}
code {
    font-family: Consolas, "Courier New", monospace;
    font-size: .9em;
}
p code, li code, td code {
    background: #edf2f7;
    color: #234d78;
    padding: 2px 5px;
    border-radius: 4px;
}
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 22px auto;
}
hr {
    border: 0;
    border-top: 1px solid #dce4ed;
    margin: 32px 0;
}
@media print {
    body { padding: 0; background: white; }
    .page {
        max-width: none;
        border: 0;
        box-shadow: none;
        border-radius: 0;
        padding: 20mm;
    }
    h2, h3 { break-after: avoid; }
    pre, table, img { break-inside: avoid; }
}
@media (max-width: 700px) {
    body { padding: 8px; }
    .page { padding: 24px 20px; }
}
"""

def render_report(item):
    source = item["source"]
    output = item["output"]

    if not source.exists():
        raise FileNotFoundError(f"Report not found: {source}")

    text = source.read_text(encoding="utf-8-sig")

    extensions = [
        "extra",
        "tables",
        "fenced_code",
        "toc",
        "sane_lists",
    ]

    body = markdown.markdown(
        text,
        extensions=extensions,
        output_format="html5",
    )

    # Resolve relative Markdown image/link paths from the original report.
    base_uri = source.parent.as_uri() + "/"
    body = re.sub(
        r'(<(?:img|a)\b[^>]*?\b(?:src|href)=["\'])(?!https?://|file://|mailto:|#)([^"\']+)(["\'])',
        lambda match: match.group(1) + html.escape(base_uri + match.group(2), quote=True) + match.group(3),
        body,
        flags=re.IGNORECASE,
    )

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(item["title"])}</title>
<style>{CSS}</style>
</head>
<body>
<main class="page">
<header class="cover">
<div class="brand">CausalForge | Research & Engineering</div>
<h1>{html.escape(item["title"])}</h1>
<div class="subtitle">{html.escape(item["subtitle"])}</div>
</header>
<article>
{body}
</article>
</main>
</body>
</html>
"""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")

    print(f"Rendered: {output.name} ({output.stat().st_size:,} bytes)")

def main():
    for item in REPORTS:
        render_report(item)

    print("\\nAll reports rendered successfully.")

if __name__ == "__main__":
    main()