# RAG Service Notes

The default chunk size is 1000 characters and the default chunk overlap is 200 characters.

The application uses LanceDB as the embedded vector store because it avoids always-on vector database infrastructure.

The service exposes an ingestion endpoint and a query endpoint through FastAPI.

Each query should log latency, retrieved chunk count, and token usage so the system can be evaluated honestly.
