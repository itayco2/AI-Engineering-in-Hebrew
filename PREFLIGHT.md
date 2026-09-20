# PREFLIGHT — everything that was wrong, with the number that showed it

Every chapter in this repo claims something. This file is where the claims were checked and
where they failed first. The format is the one that works: *what was wrong, the number that
exposed it, the fix, and the number after*. A tutorial that only shows the version that
worked teaches half the job.

Re-run any row with `python scripts/measure_01.py`.

## Chapter 01 — RAG from scratch

| # | what was wrong | the number that showed it | fix | the number after |
|---|---|---|---|---|
| 1 | **Document titles were never indexed.** Chunks were cut from `text` only, so a chunk from the middle of a document no longer mentioned its own subject | paraphrase questions recall@3 **0.68**; "why does the server keep refusing my credentials" missed a document actually titled *Rejected credentials* | prepend the title to every chunk before embedding and before the BM25 index — the cheapest form of Anthropic's contextual retrieval | paraphrase recall@3 **0.89**, failed@3 **23.2% → 10.7%** |
| 2 | **The reranker was silently a no-op.** `cross-encoder/ms-marco-MiniLM-L-6-v2` returns NaN from its forward pass on torch 2.14. Model parameters were clean, the bi-encoder was unaffected, and no error was raised | step 4 was **byte-identical** to step 3 — including nDCG@3, which is order-sensitive, so reordering could not have produced it | switch to `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (also multilingual, which chapter 02 needs) and raise on NaN instead of sorting past it | recall@1 **0.786 → 0.821** |
| 3 | **The reranker shortlist was too wide.** A reranker can only reorder what it is handed, so handing it more gives it more chances to push a good answer down | at shortlist 20, recall@3 fell **0.911 → 0.875** while recall@1 did not move | shortlist 5 | recall@3 **0.911**, recall@1 **0.821** |
| 4 | **The corpus was too small to test hybrid fairly.** With 26 documents BM25 was already near ceiling, and fusing it into a stronger dense retriever only diluted it | hybrid made failed@3 *worse*, **8.3% → 11.1%** | 20 more documents chosen to be genuinely confusable — near-duplicate subjects sharing vocabulary | hybrid improves both: failed@3 **10.7% → 8.9%**, recall@1 **0.679 → 0.786** |

### Why NaN was the dangerous one

A NaN score is not a low score. Every comparison against it is false, so a sort leaves the
order exactly as it found it. The reranker ran, cost its full inference time, changed
nothing, and reported success. Had the chapter shipped a day earlier it would have taught,
with numbers, that reranking does not help — which is false, and which no reader could have
caught. `aihe.embeddings.cross_encoder_scorer` now raises rather than returning NaN.

## The measured staircase, chapter 01

46 documents, 105 chunks, 28 questions. Every number below comes from
`scripts/measure_01.py` on this corpus — not from a paper.

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
context fixes **misses** — documents that were never retrieved at all. Fusing BM25 and
reranking fix **ordering** — documents that were retrieved but not first. Reading only the
failed@3 column makes the last two steps look worthless; reading only recall@1 hides that
context-prepending is by far the largest single win.

For reference, Anthropic reported the same shape on their own corpora in 2024: 5.7% failed
retrievals with plain embeddings, 3.7% with contextual embeddings, 2.9% adding BM25, 1.9%
adding a reranker. Those are their numbers on their data. The point of this chapter is that
you run it on yours.

## Chapter 02 — RAG in Hebrew

| # | what was wrong | the number that showed it | fix | the number after |
|---|---|---|---|---|
| 5 | **The Hebrew segmentation model loaded with a randomly initialised head.** `dicta-il/dictabert-seg` stores its prefix head at the top level (`classifiers.*`, `transform.*`) while the current remote code expects it under `prefix.*`. The loader fills the gap with random weights and raises nothing | it segmented `מידע` (information) as `מ` + `ידע`, and left `ובמסמכים` — genuinely `וב` + `מסמכים` — whole. Confident, fluent, wrong | remap the checkpoint keys in `aihe.hebrew.load_segmenter`, and raise rather than return a model whose head did not load | `missing: 0, unexpected: 0`, and `ובמסמכים → וב+מסמכים` |
| 6 | **The rule-based prefix splitter looped.** It kept finding prefix letters inside stems, because Hebrew stems often begin with them | `ובמסמכים` came out as `וב` + `מ` + `סמכים`, `השרתים` as `ה` + `ש` + `רתים`, and `שולחן` as `ש` + `ול` + `חן` | strip at most once, handle stacked prefixes with a bounded cluster list, and raise the minimum stem to 3 | agreement with the model went from unusable to **74.9%** on 574 real words |
| 7 | **`לה` was in the prefix cluster list.** Taking both letters destroyed the stem | `להצפנה` (to the encryption) came out as `לה` + `צפנה`, where `צפנה` is not a word | remove it, so the single-letter rule handles it | `להצפנה → ל+הצפנה`, matching the model |

Defects 2 and 5 are the same bug in two different libraries, a year apart, and both were
invisible: a model that loads, runs, costs its full inference time, and is quietly wrong. Neither
raised anything. Both were caught by a number that could not be explained any other way — a
reranker that changed an order-sensitive metric by exactly zero, and a segmenter that split a
word every Hebrew speaker knows is one piece.

## Chapters 03-05

| # | what was wrong | the number that showed it | fix | the number after |
|---|---|---|---|---|
| 8 | **Recording was not idempotent.** Asking the same prompt twice in one run re-recorded it, and generated text is not reproducible even at `temperature 0`, so the second reply overwrote the first and replay then walked a path that had never been recorded | `CassetteMiss` on the fourth of four chapter 05 variants, after a run that had reported 42 calls written successfully | `aihe.models.asker` skips a request already present in the library; re-recording is what a clean `make record` is for | all four variants replay, 26 cassettes, no misses |
| 9 | **A notebook defined logic.** Chapter 05's first draft declared `run()` and `ask()` in a cell — the exact thing this repo forbids | `tests/test_no_logic.py` failed on `05-agents` while 313 other tests passed | moved to `aihe.tools.run_tasks`, which the recorder now calls too, so the prompts cannot drift apart | the rule holds across all five chapters |
| 10 | **The validator rejected correct answers.** A model answering `null` for an optional key was refused, though the schema itself said the key was optional | `89%` of all first-attempt failures, across both a one-line and a six-line tool description | `null_is_absent=True` in `aihe.tools.validate`, for optional parameters only | wide schema first-try `12%` → `88%`; narrow `75%` → `100%` |

Defect 9 is the one worth pausing on. The rule was written on day one, argued for in
`CONTRIBUTING.md`, enforced by a test — and then broken by the person who wrote it, in the last
chapter. It was caught in under a second by a test that knows nothing about what the notebook
was trying to do. That is the entire argument for spending the first day on infrastructure
rather than content.
