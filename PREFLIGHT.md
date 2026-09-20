# PREFLIGHT: everything that was wrong, with the number that showed it

Every chapter in this repo claims something. This file is where the claims were checked and
where they failed first. The format is the one that works: *what was wrong, the number that
exposed it, the fix, and the number after*. A tutorial that only shows the version that
worked teaches half the job.

Re-run any row with `python scripts/measure_01.py`.

## Chapter 01: RAG from scratch

| # | what was wrong | the number that showed it | fix | the number after |
|---|---|---|---|---|
| 1 | **Document titles were never indexed.** Chunks were cut from `text` only, so a chunk from the middle of a document no longer mentioned its own subject | paraphrase questions recall@3 **0.68**; "why does the server keep refusing my credentials" missed a document actually titled *Rejected credentials* | prepend the title to every chunk before embedding and before the BM25 index, the cheapest form of Anthropic's contextual retrieval | paraphrase recall@3 **0.89**, failed@3 **23.2% → 10.7%** |
| 2 | **The reranker was silently a no-op.** `cross-encoder/ms-marco-MiniLM-L-6-v2` returns NaN from its forward pass on torch 2.14. Model parameters were clean, the bi-encoder was unaffected, and no error was raised | step 4 was **byte-identical** to step 3, including nDCG@3, which is order-sensitive, so reordering could not have produced it | switch to `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (also multilingual, which chapter 02 needs) and raise on NaN instead of sorting past it | recall@1 **0.786 → 0.821** |
| 3 | **The reranker shortlist was too wide.** A reranker can only reorder what it is handed, so handing it more gives it more chances to push a good answer down | at shortlist 20, recall@3 fell **0.911 → 0.875** while recall@1 did not move | shortlist 5 | recall@3 **0.911**, recall@1 **0.821** |
| 4 | **The corpus was too small to test hybrid fairly.** With 26 documents BM25 was already near ceiling, and fusing it into a stronger dense retriever only diluted it | hybrid made failed@3 *worse*, **8.3% → 11.1%** | 20 more documents chosen to be genuinely confusable, near-duplicate subjects sharing vocabulary | hybrid improves both: failed@3 **10.7% → 8.9%**, recall@1 **0.679 → 0.786** |

### Why NaN was the dangerous one

A NaN score is not a low score. Every comparison against it is false, so a sort leaves the
order exactly as it found it. The reranker ran, cost its full inference time, changed
nothing, and reported success. Had the chapter shipped a day earlier it would have taught,
with numbers, that reranking does not help, which is false, and which no reader could have
caught. `aihe.embeddings.cross_encoder_scorer` now raises rather than returning NaN.

## The measured staircase, chapter 01

46 documents, 105 chunks, 28 questions. Every number below comes from
`scripts/measure_01.py` on this corpus, not from a paper.

| setup | recall@1 | recall@3 | nDCG@3 | failed@3 |
|---|---|---|---|---|
| plain embeddings | 0.571 | 0.768 | 0.708 | 23.2% |
| + context prepended | 0.679 | 0.893 | 0.817 | 10.7% |
| + BM25 fused in (RRF) | 0.786 | 0.911 | 0.876 | 8.9% |
| + reranker (top 5) | **0.821** | 0.911 | **0.892** | 8.9% |
| *BM25 alone* | *0.643* | *0.732* | *0.710* | *26.8%* |

Read the nDCG column alongside recall: reranking moves nDCG@3 from `0.876` to `0.892` and
recall@1 from `0.786` to `0.821`, while leaving recall@3 and failed@3 untouched. That is the
signature of a step that reorders rather than retrieves.

The two halves of that table say different things, and the chapter says both. Prepending
context fixes **misses**, documents that were never retrieved at all. Fusing BM25 and
reranking fix **ordering**, documents that were retrieved but not first. Reading only the
failed@3 column makes the last two steps look worthless; reading only recall@1 hides that
context-prepending is by far the largest single win.

For reference, Anthropic reported the same shape on their own corpora in 2024: 5.7% failed
retrievals with plain embeddings, 3.7% with contextual embeddings, 2.9% adding BM25, 1.9%
adding a reranker. Those are their numbers on their data. The point of this chapter is that
you run it on yours.

## Chapter 02: RAG in Hebrew

| # | what was wrong | the number that showed it | fix | the number after |
|---|---|---|---|---|
| 5 | **The Hebrew segmentation model loaded with a randomly initialised head.** `dicta-il/dictabert-seg` stores its prefix head at the top level (`classifiers.*`, `transform.*`) while the current remote code expects it under `prefix.*`. The loader fills the gap with random weights and raises nothing | it segmented `מידע` (information) as `מ` + `ידע`, and left `ובמסמכים`, genuinely `וב` + `מסמכים`, whole. Confident, fluent, wrong | remap the checkpoint keys in `aihe.hebrew.load_segmenter`, and raise rather than return a model whose head did not load | `missing: 0, unexpected: 0`, and `ובמסמכים → וב+מסמכים` |
| 6 | **The rule-based prefix splitter looped.** It kept finding prefix letters inside stems, because Hebrew stems often begin with them | `ובמסמכים` came out as `וב` + `מ` + `סמכים`, `השרתים` as `ה` + `ש` + `רתים`, and `שולחן` as `ש` + `ול` + `חן` | strip at most once, handle stacked prefixes with a bounded cluster list, and raise the minimum stem to 3 | agreement with the model went from unusable to **74.9%** on 574 real words |
| 7 | **`לה` was in the prefix cluster list.** Taking both letters destroyed the stem | `להצפנה` (to the encryption) came out as `לה` + `צפנה`, where `צפנה` is not a word | remove it, so the single-letter rule handles it | `להצפנה → ל+הצפנה`, matching the model |

Defects 2 and 5 are the same bug in two different libraries, a year apart, and both were
invisible: a model that loads, runs, costs its full inference time, and is quietly wrong. Neither
raised anything. Both were caught by a number that could not be explained any other way, a
reranker that changed an order-sensitive metric by exactly zero, and a segmenter that split a
word every Hebrew speaker knows is one piece.

## Chapters 03-05

| # | what was wrong | the number that showed it | fix | the number after |
|---|---|---|---|---|
| 8 | **Recording was not idempotent.** Asking the same prompt twice in one run re-recorded it, and generated text is not reproducible even at `temperature 0`, so the second reply overwrote the first and replay then walked a path that had never been recorded | `CassetteMiss` on the fourth of four chapter 05 variants, after a run that had reported 42 calls written successfully | `aihe.models.asker` skips a request already present in the library; re-recording is what a clean `make record` is for | all four variants replay, 26 cassettes, no misses |
| 9 | **A notebook defined logic.** Chapter 05's first draft declared `run()` and `ask()` in a cell, the exact thing this repo forbids | `tests/test_no_logic.py` failed on `05-agents` while 313 other tests passed | moved to `aihe.tools.run_tasks`, which the recorder now calls too, so the prompts cannot drift apart | the rule holds across all five chapters |
| 10 | **The validator rejected correct answers.** A model answering `null` for an optional key was refused, though the schema itself said the key was optional | `89%` of all first-attempt failures, across both a one-line and a six-line tool description | `null_is_absent=True` in `aihe.tools.validate`, for optional parameters only | wide schema first-try `12%` → `88%`; narrow `75%` → `100%` |

Defect 9 is the one worth pausing on. The rule was written on day one, argued for in
`CONTRIBUTING.md`, enforced by a test, and then broken by the person who wrote it, in the last
chapter. It was caught in under a second by a test that knows nothing about what the notebook
was trying to do. That is the entire argument for spending the first day on infrastructure
rather than content.

## Found in the pre-publication audit

| # | what was wrong | the number that showed it | fix | the number after |
|---|---|---|---|---|
| 11 | **A chapter shipped a claim that was never measured.** Chapter 03's `meta.yml` headline said a holistic judge called the reply good on **100%** of the answers that had lost the fact. That figure came from the source paper and was written before the chapter was run | the measurement said **50.0%**, and the sharper number, pairs separated correctly, was `6/12` | headline rewritten to the measured result; `tests/test_contract.py` now checks every headline against the values its chapter produces | every headline quotes a number the chapter computes |
| 12 | **CI was red and the badge was green-by-luck.** On Linux `pip install torch` fetches the CUDA build and several gigabytes of NVIDIA libraries this course never uses; with five chapters of model cache that overflowed the runner's disk | `OSError: [Errno 28] No space left on device`, in both `docs` and `notebooks` | install CPU-only torch first, in CI and in `make setup` on Linux | both workflows install and run |
| 13 | **`make setup` failed on a clean Mac.** macOS ships `python3` as 3.9, which builds a venv that cannot install torch and fails ten minutes later with "no matching distribution" | a fresh clone died at `exit 2` with a shell syntax error, having also never installed the package or the docs requirements | detect the newest interpreter that is 3.10 or above and say which one; stop at second zero with the `brew` command if there is none | fresh clone: `make setup` exit 0, `make gate` 39.7s, `make test` 314 passed, `make run-01` passed |
| 14 | **`make test` skipped every chapter's own claims.** It ran `tests/` only | 273 tests instead of 314 | run `tests chapters` | 314 |

Defect 11 is the one that matters most for a repository whose whole argument is that it
measures things. The claim was plausible, it was in the right shape, it cited a real source,
and it was about this chapter's own data, which had said something else. Nothing but a test
comparing the headline to the measurement was ever going to catch it.

| # | what was wrong | the number that showed it | fix | the number after |
|---|---|---|---|---|
| 15 | **Three chapters published error tracebacks to the site.** The docs build executes every notebook, but without `AIHE_BACKEND=replay` they reached for a model that was not running. `mkdocs-jupyter` caught the exception, embedded the traceback in the page, and the build reported success | **6 error outputs** on chapter 03's published page, headed `ConnectionRefusedError`, while `mkdocs build --strict` exited 0, `nbmake` passed and every test was green | set the backend for the docs build, and add `scripts/check_site.py`, which fails if any published chapter contains a traceback or has no output at all | chapters 03-05: **0 errors**; chapter 03 went from 2 figures to 3 |
| 16 | **Every chart was displayed twice.** A figure returned as a cell's value is rendered once as `execute_result` and again as `display_data` from the inline backend | two `image/png` outputs per `viz` call, in all five chapters | a trailing semicolon suppresses the returned value | one figure per call |

Defect 15 is the most alarming in this file, because **every other check was green when it
shipped**. The tests passed, the notebooks passed under `nbmake`, the build passed under
`--strict`, and the page a reader would open was full of red tracebacks. A build that executes
something and then swallows the failure is worse than one that does not execute it at all, and
the only defence is to check the artefact you actually publish.

## Punctuation that read as machine-written

| # | what was wrong | the number that showed it | fix | the number after |
|---|---|---|---|---|
| 17 | **The prose was full of marks that read as machine-written.** Em dashes above all, plus en dashes, ellipsis characters, curly quotes and decorative emoji in the chapter headers | **234** em dashes, 63 en dashes, 33 ellipses, 14 curly quotes and 15 emoji across the authored files | replaced with commas, colons or sentence breaks, by hand where a comma could not carry the meaning; `tests/test_terms.py` now fails the build on any of them | **0** in every authored file. The corpus and the cassettes keep theirs, because they are quoted, not written |
| 18 | **The first attempt at that replacement destroyed every Python file.** A tidy-up rule collapsing runs of whitespace also collapsed every indentation level in the repository | `aihe/models.py` came back with one-space indents; the suite could not even collect | reverted through `git stash`, then rewritten to touch the punctuation characters only and never a whitespace run | 314 tests pass, all modules compile |
| 19 | **The second attempt broke the type annotations.** A rule meant to tidy `, .` into `.` also matched `tuple[str, ...]` | `SyntaxError: invalid syntax` in `chunking.py` and `tools.py` | dropped that rule | the suite collects |
| 20 | **Chapter 04's cassettes stopped matching.** The cleanup changed the truncation marker inside `cap_tool_output`, and that marker is part of the prompt that was recorded | `CassetteMiss` on the published page, caught by `scripts/check_site.py` rather than by any test | re-recorded chapter 04 against the local model | `site: every chapter rendered, no tracebacks` |

Defects 18 and 19 are worth keeping in the same table as the rest. A cleanup script that
rewrites ninety files is a change like any other, and running it without reading its diff is
how a cosmetic edit takes out an entire repository. Both were caught in seconds, by the suite
that already existed.
