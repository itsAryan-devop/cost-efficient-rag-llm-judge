"""Reproducible builder for the independent evaluation corpus.

This script downloads a small set of public, permissively-licensed technical
documents and writes them into ``data/`` in three formats (Markdown, HTML, PDF)
so the RAG ingestion pipeline is exercised across formats.

The committed corpus files are a *pinned snapshot* produced by this script.
Re-running it may pick up upstream edits (Wikipedia is a living document), which
would change chunk text and therefore the SHA-256 chunk IDs that
``eval/test_set.json`` is labelled against. Treat the committed files as the
canonical, versioned dataset; this script documents how they were produced.

Provenance and licenses are written to ``data/SOURCES.md``.

Build-time dependencies (NOT required to run the service):
    pip install fpdf2

Usage:
    python -m data.build_corpus
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, "corpus")
ACCESS_DATE = date.today().isoformat()

WIKI_API = "https://en.wikipedia.org/w/api.php"
FASTAPI_FEATURES_URL = "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/features.md"

# Sections at/after which the trailing apparatus of an article begins.
TAIL_MARKERS = (
    "== References ==",
    "== See also ==",
    "== External links ==",
    "== Further reading ==",
    "== Notes ==",
    "== Citations ==",
    "== Sources ==",
)


def _http_get(url: str) -> bytes:
    """Fetch a URL. Prefer curl (honours system proxy reliably), fall back to urllib."""
    try:
        result = subprocess.run(
            ["curl", "-fsSL", "--max-time", "45", "-A", "rag-corpus-builder/1.0", url],
            capture_output=True,
            check=True,
        )
        if result.stdout:
            return result.stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    req = urllib.request.Request(url, headers={"User-Agent": "rag-corpus-builder/1.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:  # noqa: S310 (trusted hosts)
        return resp.read()


def fetch_wikipedia_plaintext(title: str) -> str:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "format": "json",
        "titles": title,
    }
    url = f"{WIKI_API}?{urllib.parse.urlencode(params)}"
    data = json.loads(_http_get(url).decode("utf-8"))
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("extract", "")


def trim_at(text: str, markers: tuple[str, ...]) -> str:
    """Cut everything from the earliest marker onward."""
    cut = len(text)
    for marker in markers:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].strip()


def plaintext_to_html(title: str, body: str, attribution: str) -> str:
    """Render a Wikipedia plaintext extract as clean semantic HTML."""
    out = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        f"  <title>{title}</title>",
        "</head>",
        "<body>",
        "<article>",
        f"  <h1>{title}</h1>",
    ]
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading = re.match(r"^(=+)\s*(.+?)\s*=+$", line)
        if heading:
            level = min(len(heading.group(1)) + 1, 6)  # == -> h3, === -> h4 ...
            out.append(f"  <h{level}>{heading.group(2)}</h{level}>")
        else:
            out.append(f"  <p>{line}</p>")
    out.append(f"  <footer><small>{attribution}</small></footer>")
    out.extend(["</article>", "</body>", "</html>", ""])
    return "\n".join(out)


_PDF_REPLACEMENTS = {
    "—": "-",  # em dash
    "–": "-",  # en dash
    "→": "->",  # right arrow
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    " ": " ",  # non-breaking space
    "…": "...",
    "·": "-",
}


def _ascii_sanitize(text: str) -> str:
    for bad, good in _PDF_REPLACEMENTS.items():
        text = text.replace(bad, good)
    return text.encode("latin-1", "ignore").decode("latin-1")


def plaintext_to_pdf(title: str, body: str, attribution: str, out_path: str) -> None:
    import textwrap

    from fpdf import FPDF

    # Pre-wrap to short physical lines so fpdf's word-wrap never has to scan a
    # long line (char-level wrapping is O(n^2) and stalls on big paragraphs).
    def emit(pdf: FPDF, text: str, height: float) -> None:
        for piece in textwrap.wrap(
            _ascii_sanitize(text), width=95, break_long_words=True, break_on_hyphens=False
        ) or [""]:
            pdf.multi_cell(0, height, piece, new_x="LMARGIN", new_y="NEXT")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", style="B", size=16)
    emit(pdf, title, 9)
    pdf.ln(2)

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading = re.match(r"^(=+)\s*(.+?)\s*=+$", line)
        if heading:
            pdf.ln(2)
            pdf.set_font("Helvetica", style="B", size=12)
            emit(pdf, heading.group(2), 7)
        else:
            pdf.set_font("Helvetica", size=11)
            emit(pdf, line, 6)
    pdf.ln(4)
    pdf.set_font("Helvetica", style="I", size=8)
    emit(pdf, attribution, 5)
    pdf.output(out_path)


def build() -> None:
    os.makedirs(CORPUS_DIR, exist_ok=True)

    # ---- 1. Markdown: FastAPI features (MIT) -------------------------------
    fastapi_md = _http_get(FASTAPI_FEATURES_URL).decode("utf-8")
    md_attr = (
        "\n\n---\n_Source: FastAPI documentation (features.md), "
        "https://github.com/fastapi/fastapi - MIT License. "
        f"Retrieved {ACCESS_DATE}._\n"
    )
    with open(os.path.join(CORPUS_DIR, "fastapi_features.md"), "w", encoding="utf-8") as f:
        f.write(fastapi_md.rstrip() + md_attr)

    # ---- 2. HTML: vector search & RAG (Wikipedia, CC BY-SA 4.0) -----------
    vdb = trim_at(fetch_wikipedia_plaintext("Vector database"), TAIL_MARKERS)
    nns = trim_at(fetch_wikipedia_plaintext("Nearest neighbor search"), TAIL_MARKERS)
    rag = trim_at(fetch_wikipedia_plaintext("Retrieval-augmented generation"), TAIL_MARKERS)
    combined = (
        vdb
        + "\n\n== Nearest neighbor search ==\n"
        + nns
        + "\n\n== Retrieval-augmented generation (overview) ==\n"
        + rag
    )
    html_attr = (
        "Source: Wikipedia articles 'Vector database', 'Nearest neighbor search', "
        "and 'Retrieval-augmented generation', licensed CC BY-SA 4.0. "
        f"Retrieved {ACCESS_DATE}."
    )
    html = plaintext_to_html("Vector Search and Retrieval-Augmented Generation", combined, html_attr)
    with open(os.path.join(CORPUS_DIR, "vector_search_and_rag.html"), "w", encoding="utf-8") as f:
        f.write(html)

    # ---- 3. PDF: standard HTTP status codes (Wikipedia, CC BY-SA 4.0) -----
    status = fetch_wikipedia_plaintext("List of HTTP status codes")
    # Keep only the standardized IETF codes; drop vendor-specific nonstandard ones.
    status = trim_at(status, ("== Nonstandard codes ==",) + TAIL_MARKERS)
    pdf_attr = (
        "Source: Wikipedia article 'List of HTTP status codes', licensed CC BY-SA 4.0. "
        f"Retrieved {ACCESS_DATE}."
    )
    plaintext_to_pdf(
        "HTTP Response Status Codes (Standard)",
        status,
        pdf_attr,
        os.path.join(CORPUS_DIR, "http_status_codes.pdf"),
    )

    write_sources_md()
    print("Corpus rebuilt:")
    for name in ("fastapi_features.md", "vector_search_and_rag.html", "http_status_codes.pdf"):
        path = os.path.join(CORPUS_DIR, name)
        print(f"  corpus/{name}: {os.path.getsize(path):,} bytes")


def write_sources_md() -> None:
    content = f"""# Corpus Sources and Licenses

The evaluation corpus is **independent** of this project's author: every document
is public, third-party, permissively-licensed technical material. Questions in
`eval/test_set.json` are labelled against the committed snapshot of these files.

These files are a pinned snapshot retrieved on **{ACCESS_DATE}** and regenerated
by `data/build_corpus.py`. Wikipedia is a living document, so re-running the
builder may change content (and therefore chunk IDs); the committed files are the
canonical dataset.

| File | Format | Source | License |
|---|---|---|---|
| `corpus/fastapi_features.md` | Markdown | [FastAPI docs — features.md](https://github.com/fastapi/fastapi/blob/master/docs/en/docs/features.md) | MIT |
| `corpus/vector_search_and_rag.html` | HTML | Wikipedia: [Vector database](https://en.wikipedia.org/wiki/Vector_database), [Nearest neighbor search](https://en.wikipedia.org/wiki/Nearest_neighbor_search), [Retrieval-augmented generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) | CC BY-SA 4.0 |
| `corpus/http_status_codes.pdf` | PDF | Wikipedia: [List of HTTP status codes](https://en.wikipedia.org/wiki/List_of_HTTP_status_codes) (standard codes only) | CC BY-SA 4.0 |

The evaluated corpus lives in `data/corpus/`. This `SOURCES.md` and
`build_corpus.py` stay at `data/` root and are excluded from ingestion
(`DATA_ROOT` defaults to `data/corpus`).

Attribution for the CC BY-SA 4.0 content is embedded in each file. The MIT
license text for FastAPI is reproduced at
<https://github.com/fastapi/fastapi/blob/master/LICENSE>.

The HTML file is rendered from the Wikipedia plaintext extracts as clean
semantic HTML; the PDF is rendered from the plaintext with `fpdf2`. No content
was authored by this project's author, so evaluation questions cannot be
answered "from the author's own notes".
"""
    with open(os.path.join(HERE, "SOURCES.md"), "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    build()
