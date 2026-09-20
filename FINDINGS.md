# FINDINGS — measurements that decided something

`PREFLIGHT.md` records what was broken. This file records what was *measured* in order to
choose, including the measurements that overturned a plan. One file, one question, re-runnable.

## Do Hebrew-specific embedders beat general multilingual ones?

**Asked before writing chapter 02, because the chapter's outline depended on the answer.**
Re-run with `python scripts/spike_02_hebrew.py`.

Instrument: **HeQ** (`Etelis/HeQ_v1`) — 30,147 human-written Hebrew questions over Wikipedia
and Geektime passages. The validation split gives 1,318 questions over 200 unique passages, so
a random guess scores `recall@5 = 0.025`.

| model | kind | recall@1 | recall@5 | MRR |
|---|---|---|---|---|
| **BM25** — no model at all | keyword | 0.790 | 0.918 | 0.846 |
| paraphrase-multilingual-MiniLM-L12-v2 | multilingual (chapter 01's) | 0.505 | 0.715 | 0.603 |
| **multilingual-e5-small** | multilingual | **0.822** | **0.942** | **0.874** |
| sentence-transformers-alephbert | Hebrew-specific | 0.525 | 0.758 | 0.630 |
| MPA/sambert | Hebrew-specific | 0.643 | 0.835 | 0.731 |

**Answer: no.** The best general multilingual model beats the best Hebrew-specific one by a
wide margin — `0.874` against `0.731` MRR. This is the chapter's finding, and it is a better
chapter than the version that assumed the opposite. It also matches what the research warned:
DictaBERT, the strongest Hebrew encoder, ships no sentence-transformers embedding model at all,
so "the Hebrew-specific option" in practice means older or smaller models.

**Caveat, stated because it changes how the BM25 row should be read.** HeQ is *extractive* QA:
every answer is a span inside its passage, so questions share vocabulary with their passage by
construction. That flatters lexical matching. BM25 beating three of four embedders here is real
but is partly an artefact of the benchmark, and chapter 02 says so rather than quoting the
number bare.

**Second finding, unplanned:** BM25 with no model, no download and no GPU beats three of the
four embedding models on this set. Any chapter that reaches for embeddings without measuring
the keyword baseline first is teaching a habit, not an engineering decision.

### What this changes

Chapter 02 is now designed around overturning a chapter 01 default with data, rather than
around confirming that Hebrew needs Hebrew-specific tools:

1. Measure chapter 01's model on Hebrew. It is the worst of the four.
2. Show that a general multilingual model wins, and that it is **asymmetric** — `multilingual-e5`
   expects `query:` and `passage:` prefixes, and omitting them silently costs accuracy. That is
   a practical trap most tutorials skip entirely.
3. Show the BM25 baseline beating most embedders, and explain why the benchmark's construction
   partly causes it.
4. Then Hebrew morphology: prefixed particles, no vowels, what `dictabert-seg` changes.

## Should chapter 01 switch to multilingual-e5-small?

**No, decided deliberately.** On chapter 01's own corpus:

| model | plain r@1 | + context r@1 | hybrid r@1 | hybrid r@3 |
|---|---|---|---|---|
| paraphrase-multilingual-MiniLM (current) | 0.571 | 0.679 | 0.786 | **0.911** |
| multilingual-e5-small | **0.661** | **0.750** | **0.821** | 0.893 |

e5 wins at rank 1 and loses at rank 3, and it needs the `query:`/`passage:` prefixes. Chapter 01
is the chapter where every piece of infrastructure gets debugged, so it keeps the simpler
symmetric model and the cleaner `recall@3` staircase. The switch becomes chapter 02's finding,
where it is earned by measurement rather than asserted in a footnote.
