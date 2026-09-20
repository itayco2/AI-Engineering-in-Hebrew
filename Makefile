# The runbook. Everything here is free: no API key, no account, no GPU.
PY := ./.venv/bin/python
PIP := ./.venv/bin/pip
PORT ?= 8000

# The first interpreter on this machine that is 3.10 or newer. macOS ships `python3` as 3.9,
# which would build a venv that cannot install torch and fail ten minutes in with a message
# about "no matching distribution". Finding the right one up front turns that into a one-line
# error at second zero, or into silence.
PYTHON ?= $(shell for p in python3.13 python3.12 python3.11 python3.10 python3; do \
  if command -v $$p >/dev/null 2>&1 && $$p -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then echo $$p; break; fi; done)

.PHONY: help setup gate test test-all notebooks run-01 record docs docs-serve rtl clean

help:
	@grep -E '^[a-z0-9-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## create .venv and install everything (uses uv when present, pip otherwise)
	@if [ -z "$(PYTHON)" ]; then \
	  echo "No Python 3.10 or newer found. macOS ships 3.9 as python3, which is too old."; \
	  echo "Install one:  brew install python@3.12   (or https://www.python.org/downloads/)"; \
	  exit 1; \
	fi
	@echo "using $(PYTHON) ($$($(PYTHON) --version))"
	@if command -v uv >/dev/null 2>&1; then \
	  echo "uv found - using it"; \
	  uv venv --python $(PYTHON) .venv && $(MAKE) -s _cpu_torch PIP="uv pip install --python .venv" && \
	  uv pip install --python .venv -r requirements.txt -r requirements-test.txt -r requirements-notebooks.txt -r requirements-docs.txt; \
	else \
	  echo "uv not found - using venv + pip (works everywhere, just slower)"; \
	  $(PYTHON) -m venv .venv && $(PIP) install -q --upgrade pip && $(MAKE) -s _cpu_torch && \
	  $(PIP) install -r requirements.txt -r requirements-test.txt -r requirements-notebooks.txt -r requirements-docs.txt; \
	fi
	@$(PIP) install -q -e . --no-deps
	@echo "done. Next:  make gate"

# On Linux, plain `pip install torch` pulls the CUDA build and several gigabytes of NVIDIA
# libraries this course never uses. macOS wheels are already CPU/Metal, so nothing to do there.
_cpu_torch:
	@if [ "$$(uname -s)" = "Linux" ]; then \
	  echo "Linux: installing CPU-only torch first, to skip the CUDA download"; \
	  $(PIP) install torch --index-url https://download.pytorch.org/whl/cpu; \
	fi

gate:  ## THE GATE: under 60s. Run this first when anything looks broken
	$(PY) scripts/gate.py

test:  ## the fast suite: the package and every chapter's own claims, no model, no downloads
	$(PY) -m pytest tests chapters -m "not needs_model and not slow"

test-all:  ## everything, including tests that download an embedding model
	$(PY) -m pytest

notebooks:  ## execute every chapter from recorded cassettes (no network)
	AIHE_BACKEND=replay $(PY) -m pytest --nbmake chapters/*/chapter.ipynb

run-01:  ## execute chapter 01 end to end
	AIHE_BACKEND=replay $(PY) -m pytest --nbmake chapters/01-rag/chapter.ipynb

run-02:  ## execute chapter 02 end to end
	AIHE_BACKEND=replay $(PY) -m pytest --nbmake chapters/01-rag/chapter.ipynb

run-04:  ## execute chapter 04 end to end
	AIHE_BACKEND=replay $(PY) -m pytest --nbmake chapters/04-context/chapter.ipynb

run-05:  ## execute chapter 05 end to end
	AIHE_BACKEND=replay $(PY) -m pytest --nbmake chapters/05-agents/chapter.ipynb

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
