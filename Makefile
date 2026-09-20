# The runbook. Everything here is free: no API key, no account, no GPU.
PY := ./.venv/bin/python
PIP := ./.venv/bin/pip
PORT ?= 8000

.PHONY: help setup gate test test-all notebooks run-01 record docs docs-serve rtl clean

help:
	@grep -E '^[a-z0-9-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## create .venv and install everything (uses uv when present, pip otherwise)
	@if command -v uv >/dev/null 2>&1; then \
	  echo "uv found - using it"; \
	  uv venv .venv && uv pip install --python .venv -r requirements.txt -r requirements-test.txt -r requirements-notebooks.txt; \
	else \
	  echo "uv not found - using venv + pip (works everywhere, just slower)"; \
	  python3 -m venv .venv && $(PIP) install -q --upgrade pip && \
	  $(PIP) install -r requirements.txt -r requirements-test.txt -r requirements-notebooks.txt; \
	fi

gate:  ## THE GATE: under 60s. Run this first when anything looks broken
	$(PY) scripts/gate.py

test:  ## the fast suite: pure functions, no model, no downloads
	$(PY) -m pytest tests -m "not needs_model and not slow"

test-all:  ## everything, including tests that download an embedding model
	$(PY) -m pytest

notebooks:  ## execute every chapter from recorded cassettes (no network)
	AIHE_BACKEND=replay $(PY) -m pytest --nbmake chapters/*/chapter.ipynb

run-01:  ## execute chapter 01 end to end
	AIHE_BACKEND=replay $(PY) -m pytest --nbmake chapters/01-rag/chapter.ipynb

record:  ## re-record cassettes from a real model, then review the diff
	$(PY) scripts/record_cassettes.py

rtl:  ## check the Hebrew writing rules that no stylesheet can fix
	$(PY) scripts/check_rtl.py

docs:  ## build the RTL site
	$(PY) -m mkdocs build --strict

docs-serve:  ## the RTL site, locally (PORT=8000 by default)
	$(PY) -m mkdocs serve -a 127.0.0.1:$(PORT)

clean:  ## remove build and cache directories
	$(PY) scripts/clean.py
