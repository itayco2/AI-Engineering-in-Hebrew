# Changelog

Notable changes, newest first. This project follows [semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- **Chapter 01 — RAG from scratch.** A complete retrieval pipeline measured end to end on a
  46-document, 28-question corpus written for the purpose. `recall@1` climbs 57.1% → 67.9% →
  78.6% → 82.1% across four steps; failed retrievals at `k=3` fall 23.2% → 8.9%. The chapter
  calls no language model, so it runs on any laptop.
- `aihe` package: chunking with exact character offsets, BM25 implemented from scratch,
  cosine search, Reciprocal Rank Fusion, an injected-scorer reranker, and the retrieval
  metrics — all pure functions, all covered by a suite that runs in under a second.
- One `chat()` interface with four backends: `ollama`, `llamacpp`, `replay` and `fake`.
  `replay` is the CI default and reads committed cassettes, so the build makes no network
  call and cannot flake.
- Chapter contract enforced by tests: required files, `meta.yml` completeness, no `def` or
  `class` in any notebook cell, and terminology drift checked against `TERMS.md`.
- `TERMS.md`, the terminology spine — 80 terms with the form used and the form rejected.
- RTL documentation site (MkDocs Material) with a stylesheet that bidi-isolates inline code
  inside Hebrew paragraphs, plus `scripts/check_rtl.py` for the writing rules CSS cannot fix.
- `PREFLIGHT.md` documenting every defect found while building, with the number that exposed
  each one.

### Fixed
- The reranker was silently a no-op: `cross-encoder/ms-marco-MiniLM-L-6-v2` returns `NaN` from
  its forward pass on torch 2.14, with clean parameters and no error raised. A `NaN` score is
  not a low score — every comparison against it is false, so sorting left the order untouched
  and the step reported success while changing nothing. Switched to the multilingual
  `mmarco-mMiniLMv2` reranker and made the scorer raise rather than return `NaN`.
