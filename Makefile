# Thin wrappers over the existing module entrypoints.
# Override the interpreter for a local venv, e.g.:
#   make test PYTHON=.venv/Scripts/python.exe   (Windows)
#   make test PYTHON=.venv/bin/python           (POSIX)
PYTHON ?= python
DATA_ROOT ?= data/corpus
IMAGE ?= cost-efficient-rag

# Part 1 paths only; Problem 2 (eval/pipeline, eval/run_*pipeline*) is a separate workstream.
LINT_PATHS = src data/build_corpus.py \
	eval/build_test_set.py eval/cost_analysis.py eval/export_chunks.py \
	eval/ir_metrics.py eval/llm_judge.py eval/run.py eval/text_metrics.py eval/smoke.py

# Mock everything and use throwaway state dirs so tests never touch real keys or db/.
MOCK_ENV = EMBEDDING_PROVIDER=mock GENERATION_PROVIDER=mock JUDGE_PROVIDER=mock \
	DB_PATH=db/test_lancedb CACHE_PATH=cache/test_diskcache

.PHONY: setup test lint ingest eval cost serve docker-build docker-run

setup:
	$(PYTHON) -m pip install -r requirements-dev.txt

lint:
	$(PYTHON) -m ruff check $(LINT_PATHS)
	$(PYTHON) -m black --check $(LINT_PATHS)

test:
	$(MOCK_ENV) $(PYTHON) -m pytest -q

ingest:
	DATA_ROOT=$(DATA_ROOT) $(PYTHON) -m src.ingest

eval:
	DATA_ROOT=$(DATA_ROOT) $(PYTHON) -m eval.run

cost:
	$(PYTHON) -m eval.cost_analysis

serve:
	$(PYTHON) -m uvicorn src.api:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env $(IMAGE)
