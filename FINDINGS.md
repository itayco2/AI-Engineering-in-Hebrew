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

## Does splitting Hebrew prefixes improve keyword retrieval?

**Yes, and the interesting part is that a rule which is wrong a quarter of the time still
helps.** Measured on the chapter's own HeQ subset: 150 passages, 1,036 questions. Only the
tokenizer changes between rows.

| tokenizer | MRR | recall@1 | recall@5 | failed@5 |
|---|---|---|---|---|
| chapter 01's plain tokenizer | 0.866 | 0.814 | 0.931 | 6.9% |
| + normalise niqqud only | 0.868 | 0.817 | 0.931 | 6.9% |
| **+ split glued prefixes (rules)** | **0.902** | **0.856** | **0.958** | **4.2%** |

Failed retrievals fall by 39%, from about fifteen lines of rules and no download at all.

### Why a 75%-accurate rule still helps

The rule-based splitter agrees with `dictabert-seg` on **74.9%** of 574 real Hebrew words. It
wrongly splits `כלים` into `כ` + `לים`, `משתמשים` into `מ` + `שתמשים`, and `הקלטות` into
`ה` + `קלטות` — words that merely begin with a prefix letter.

It helps anyway because of how the index is built: `tokenize_hebrew` adds the stem **alongside**
the original word rather than replacing it. A wrong stem contributes a token no query will ever
ask for — noise — and never removes a match that would have been found. Replace the word instead
of adding to it and the same rule would hurt.

That generalises: design the failure mode and you can afford a worse component.

## What does Hebrew actually cost in tokens?

**It depends entirely on the tokenizer, and the spread is much larger than the folklore.**

| tokenizer | English tokens | Hebrew tokens | ratio |
|---|---|---|---|
| GPT-2 (English BPE) | 84 | 376 | **4.48x** |
| XLM-R (multilingual) | 102 | 105 | **1.03x** |

Same four sentence pairs, same content. The single word `ובמסמכים` becomes **10 tokens** of
byte fragments under GPT-2 and **3 tokens** under XLM-R — `▁וב`, `מס`, `מכים`, where the first
split happens to match the morphology.

"Hebrew is expensive" is a statement about a tokenizer, not about Hebrew.
