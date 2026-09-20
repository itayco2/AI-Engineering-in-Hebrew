# Changelog

Notable changes, newest first. This project follows [semantic versioning](https://semver.org/).

## [1.0.0] - 2026-09-20

Five chapters, written, measured and executed on every push. 341 tests in about a fifth of a
second; the whole course runs end to end in roughly seventy seconds and never touches the
network.

### Added in 1.0
- **Chapter 03 — How you know it works.** Retrieval metrics with no model at all, then
  `LLM-as-judge` measured against ground truth known by construction. A holistic judge told the
  good answer from the bad one on **6 of 12** questions — exactly what a coin gets — while a
  substring check with no model in it was right on all 24.
- **Chapter 04 — Context engineering.** Summarising saved **67%** of the characters and lost the
  planted fact **6 times out of 6**; capping tool output on write saved 47% and lost nothing.
  The mechanism is decided by comparing bytes: a cache hits only on an unchanged prefix.
- **Chapter 05 — Agents.** The same tool described in one line and in six: first-try success
  fell from **75% to 12%**. And **89% of every first-attempt failure** was the model answering
  `null` for an *optional* parameter and a strict validator refusing it. Reading `null` as
  "absent" took the wide schema from 12% to **88%**.
- `aihe.judge`, `aihe.context` and `aihe.tools`, all pure and all covered.
- Cassettes: every model reply in chapters 03-05 recorded once from a local
  `Llama-3.2-3B-Instruct-Q4_K_M`, so the chapters run offline and CI costs nothing.

### Fixed in 1.0
- Recording was not idempotent. Asking the same prompt twice in one run re-recorded it, and
  since generated text is not reproducible even at `temperature 0`, the second reply overwrote
  the first and replay walked a path that had never been recorded.
- The validator rejected correct answers: `null` for an optional key is a model declining an
  option, not an error.
- A notebook defined logic. The rule was written on day one and broken in the last chapter, by
  the person who wrote it. `tests/test_no_logic.py` caught it in under a second.

## [Unreleased]

### Added
- **Chapter 02 — RAG in Hebrew.** The same pipeline on 150 real Hebrew passages and 1,036
  human-written questions from HeQ (CC BY 4.0). Splitting Hebrew's glued prefixes lifts BM25
  from `0.866` to `0.902` MRR and cuts failed retrievals by 39%, from about fifteen lines of
  rules and no download. Hebrew costs `4.48x` more tokens than English under an
  English-trained BPE and `1.03x` under a multilingual one — the word `ובמסמכים` is 10 tokens
  under one and 3 under the other. And chapter 01's embedding model turns out to be the worst
  of four tested on Hebrew, which the chapter measures rather than asserts.
- `aihe.hebrew`: prefix splitting by rule and by model, niqqud normalisation, final-form
  folding, and token-cost measurement.
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
- The Hebrew segmentation model loaded with a randomly initialised head and raised nothing,
  segmenting `מידע` as `מ` + `ידע`. The checkpoint stores its head at the top level while the
  remote code expects it nested; `aihe.hebrew.load_segmenter` remaps the keys and now refuses
  to return a model whose head did not load. Same failure class as the reranker below, in a
  different library.
- The rule-based prefix splitter looped, turning `ובמסמכים` into three pieces because Hebrew
  stems often begin with prefix letters. It now strips at most once.
- The reranker was silently a no-op: `cross-encoder/ms-marco-MiniLM-L-6-v2` returns `NaN` from
  its forward pass on torch 2.14, with clean parameters and no error raised. A `NaN` score is
  not a low score — every comparison against it is false, so sorting left the order untouched
  and the step reported success while changing nothing. Switched to the multilingual
  `mmarco-mMiniLMv2` reranker and made the scorer raise rather than return `NaN`.
