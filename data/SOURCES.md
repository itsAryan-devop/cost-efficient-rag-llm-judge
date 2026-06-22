# Corpus Sources and Licenses

The evaluation corpus is **independent** of this project's author: every document
is public, third-party, permissively-licensed technical material. Questions in
`eval/test_set.json` are labelled against the committed snapshot of these files.

These files are a pinned snapshot retrieved on **2026-06-23** and regenerated
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
