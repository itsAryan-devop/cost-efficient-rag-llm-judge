"""Generate eval/test_set.json with honest, reproducible relevant-chunk labels.

Each question is authored against the *committed* corpus and references the
chunk(s) (by ``source_file`` + ``chunk_index``) that actually contain the
answer. This script resolves those references to the deterministic SHA-256
chunk IDs using ``reports/chunk_inventory.json`` (produced by
``python -m eval.export_chunks`` after ingesting ``data/corpus``), so the
64-char IDs are never hand-copied.

Out-of-corpus questions have ``relevant_chunk_ids: []`` and ``expected_refusal:
true`` — the system should decline to answer them.

Usage:
    python -m eval.export_chunks        # refresh reports/chunk_inventory.json
    python -m eval.build_test_set       # regenerate eval/test_set.json
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INVENTORY = os.path.join(ROOT, "reports", "chunk_inventory.json")
OUT = os.path.join(HERE, "test_set.json")

# (source_file, [chunk_index, ...], query, reference_answer, difficulty)
ANSWERABLE = [
    # ---- FastAPI features (Markdown) -------------------------------------
    (
        "fastapi_features.md",
        [0],
        "Which two open standards is FastAPI based on?",
        "FastAPI is based on OpenAPI (for API creation) and JSON Schema (for data model documentation).",
        "easy",
    ),
    (
        "fastapi_features.md",
        [1],
        "Which two interactive API documentation interfaces does FastAPI include by default?",
        "Swagger UI and ReDoc.",
        "easy",
    ),
    (
        "fastapi_features.md",
        [8],
        "What web framework is FastAPI built on, and what is the class relationship between them?",
        "FastAPI is built on Starlette; the FastAPI class is a subclass of Starlette.",
        "hard",
    ),
    (
        "fastapi_features.md",
        [10],
        "Which library does FastAPI use for all of its data handling and validation?",
        "Pydantic.",
        "easy",
    ),
    (
        "fastapi_features.md",
        [9],
        "What test coverage and type-annotation level does the FastAPI codebase report?",
        "A 100% test coverage and a 100% type-annotated codebase.",
        "medium",
    ),
    # ---- Vector search & RAG (HTML) --------------------------------------
    (
        "vector_search_and_rag.html",
        [0],
        "What is a vector database?",
        "A database that stores and retrieves embeddings in vector space and typically uses approximate "
        "nearest neighbor algorithms to find records that are semantically similar to a query.",
        "easy",
    ),
    (
        "vector_search_and_rag.html",
        [1],
        "What is the typical range of dimensionality for vector embeddings?",
        "From a few hundred to tens of thousands of dimensions, depending on the complexity of the data.",
        "medium",
    ),
    (
        "vector_search_and_rag.html",
        [2],
        "Name common techniques used for similarity search over high-dimensional vectors.",
        "Hierarchical Navigable Small World (HNSW) graphs, locality-sensitive hashing (LSH), product "
        "quantization (PQ), and inverted files.",
        "hard",
    ),
    (
        "vector_search_and_rag.html",
        [3],
        "In a retrieval-augmented generation system, how is the retrieval component most often implemented?",
        "As a vector database.",
        "medium",
    ),
    (
        "vector_search_and_rag.html",
        [8],
        "What is the running time of the naive linear-search approach to nearest neighbor search?",
        "O(dN), where N is the number of points and d is the dimensionality.",
        "hard",
    ),
    (
        "vector_search_and_rag.html",
        [20],
        "What are the two most well-known variants of the nearest neighbor search problem?",
        "The k-nearest neighbor search and the epsilon-approximate nearest neighbor search.",
        "medium",
    ),
    (
        "vector_search_and_rag.html",
        [23],
        "What is retrieval-augmented generation (RAG)?",
        "A technique that enables large language models to retrieve and incorporate new information from "
        "external data sources before responding to a query.",
        "easy",
    ),
    (
        "vector_search_and_rag.html",
        [25],
        "In what year was the term retrieval-augmented generation introduced?",
        "2020, in a paper combining a parametric language model with non-parametric external memory.",
        "medium",
    ),
    (
        "vector_search_and_rag.html",
        [34],
        "Why do fixed-length chunking strategies overlap consecutive chunks?",
        "Overlapping consecutive chunks helps maintain semantic context across the chunks.",
        "medium",
    ),
    (
        "vector_search_and_rag.html",
        [35],
        "What is hybrid search and what problem does it address?",
        "Combining vector search with traditional text search, used because vector search can sometimes "
        "miss key facts needed to answer a question.",
        "hard",
    ),
    (
        "vector_search_and_rag.html",
        [36],
        "Does retrieval-augmented generation eliminate hallucinations in large language models?",
        "No. RAG improves accuracy but does not prevent hallucinations; the model can still hallucinate "
        "around the source material.",
        "hard",
    ),
    # ---- HTTP status codes (PDF) -----------------------------------------
    (
        "http_status_codes.pdf",
        [0],
        "Who defines and maintains the standard HTTP status codes?",
        "They are defined by the IETF in RFC publications and maintained by the IANA.",
        "medium",
    ),
    (
        "http_status_codes.pdf",
        [3],
        "What does HTTP status code 200 OK mean?",
        "It is the standard response for a successful HTTP request.",
        "easy",
    ),
    (
        "http_status_codes.pdf",
        [12],
        "What does HTTP status code 400 Bad Request indicate?",
        "The server cannot or will not process the request due to an apparent client error.",
        "easy",
    ),
    (
        "http_status_codes.pdf",
        [12],
        "What is HTTP status code 401 Unauthorized used for?",
        "It is used when authentication is required and has either failed or not yet been provided.",
        "medium",
    ),
    (
        "http_status_codes.pdf",
        [22],
        "Which novel is HTTP status code 451 a reference to?",
        "Fahrenheit 451.",
        "easy",
    ),
    (
        "http_status_codes.pdf",
        [23],
        "What does HTTP status code 502 Bad Gateway mean?",
        "The server, acting as a gateway or proxy, received an invalid response from the upstream server.",
        "easy",
    ),
    (
        "http_status_codes.pdf",
        [23],
        "Which HTTP status code should a proxy return when it receives an invalid response from the "
        "upstream server?",
        "502 Bad Gateway.",
        "hard",
    ),
]

REFUSALS = [
    ("What is the capital of France?", "easy"),
    ("What was Tesla's total revenue in fiscal year 2023?", "easy"),
    ("How do I make a sourdough bread starter from scratch?", "easy"),
]

# Reference text used for refusal cases, kept in sync with src.generation.
NO_CONTEXT_MESSAGE = "I could not find relevant context in the indexed documents to answer this question."


def build() -> None:
    with open(INVENTORY, encoding="utf-8") as f:
        inventory = json.load(f)

    lookup: dict[tuple[str, int], str] = {}
    for chunk in inventory["chunks"]:
        lookup[(chunk["source_file"], chunk["chunk_index"])] = chunk["id"]

    dataset = []
    missing = []
    for source, indices, query, reference, difficulty in ANSWERABLE:
        ids = []
        for idx in indices:
            key = (source, idx)
            if key not in lookup:
                missing.append(key)
                continue
            ids.append(lookup[key])
        dataset.append(
            {
                "query": query,
                "relevant_chunk_ids": ids,
                "reference_answer": reference,
                "source_files": [source],
                "difficulty": difficulty,
                "expected_refusal": False,
            }
        )

    for query, difficulty in REFUSALS:
        dataset.append(
            {
                "query": query,
                "relevant_chunk_ids": [],
                "reference_answer": NO_CONTEXT_MESSAGE,
                "source_files": [],
                "difficulty": difficulty,
                "expected_refusal": True,
            }
        )

    if missing:
        raise SystemExit("Chunk references not found in inventory (re-run export_chunks?): " + str(missing))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        f.write("\n")

    answerable = sum(1 for d in dataset if not d["expected_refusal"])
    print(
        f"Wrote {len(dataset)} questions ({answerable} answerable, "
        f"{len(dataset) - answerable} refusal) to {OUT}"
    )


if __name__ == "__main__":
    build()
