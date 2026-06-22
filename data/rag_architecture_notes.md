# RAG Architecture Notes

The application uses FastAPI to expose a runnable HTTP service. The main endpoints are `/health`, `/ingest`, and `/query`.

The ingestion pipeline supports PDF, HTML, Markdown, and plain text files. It normalizes extracted text, splits content into chunks, and stores deterministic SHA-256 identifiers for documents and chunks.

Idempotent re-ingestion is handled by deleting existing rows with the same chunk IDs before inserting updated rows. This prevents duplicate vectors when the same corpus is ingested more than once.

The default chunk size is 1000 characters and the default overlap is 200 characters. These defaults are configurable through environment variables.

LanceDB stores the vector, chunk text, document ID, source file, document type, chunk index, chunk size, and chunk overlap. The query endpoint supports metadata filters such as document type or source file.

The generation step builds a grounded prompt from retrieved chunks. If no relevant chunks are retrieved, the system returns a no-context response instead of hallucinating.

Per-query telemetry is emitted as structured JSON. The telemetry includes latency in milliseconds, retrieved chunk count, token usage, provider, model, and whether the LLM call was skipped.
