# FINDINGS: measurements that decided something

`PREFLIGHT.md` records what was broken. This file records what was *measured* in order to
choose, including the measurements that overturned a plan. One file, one question, re-runnable.

## Do Hebrew-specific embedders beat general multilingual ones?

**Asked before writing chapter 02, because the chapter's outline depended on the answer.**
Re-run with `python scripts/spike_02_hebrew.py`.

Instrument: **HeQ** (`Etelis/HeQ_v1`), 30,147 human-written Hebrew questions over Wikipedia
and Geektime passages. The validation split gives 1,318 questions over 200 unique passages, so
a random guess scores `recall@5 = 0.025`.

| model | kind | recall@1 | recall@5 | MRR |
|---|---|---|---|---|
| **BM25**, no model at all | keyword | 0.790 | 0.918 | 0.846 |
| paraphrase-multilingual-MiniLM-L12-v2 | multilingual (chapter 01's) | 0.505 | 0.715 | 0.603 |
| **multilingual-e5-small** | multilingual | **0.822** | **0.942** | **0.874** |
| sentence-transformers-alephbert | Hebrew-specific | 0.525 | 0.758 | 0.630 |
| MPA/sambert | Hebrew-specific | 0.643 | 0.835 | 0.731 |

**Answer: no.** The best general multilingual model beats the best Hebrew-specific one by a
wide margin, `0.874` against `0.731` MRR. This is the chapter's finding, and it is a better
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
2. Show that a general multilingual model wins, and that it is **asymmetric**, `multilingual-e5`
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
| chapter 01's plain tokenizer | 0.866 | 0.814 | 0.931 | 6.95% |
| + normalise niqqud only | 0.868 | 0.817 | 0.931 | 6.95% |
| **+ split glued prefixes (rules)** | **0.902** | **0.856** | **0.958** | **4.15%** |

Failed retrievals fall by 40%, from about fifteen lines of rules and no download at all.

### Why a 75%-accurate rule still helps

The rule-based splitter agrees with `dictabert-seg` on **74.9%** of 574 real Hebrew words. It
wrongly splits `כלים` into `כ` + `לים`, `משתמשים` into `מ` + `שתמשים`, and `הקלטות` into
`ה` + `קלטות`, words that merely begin with a prefix letter.

It helps anyway because of how the index is built: `tokenize_hebrew` adds the stem **alongside**
the original word rather than replacing it. A wrong stem contributes a token no query will ever
ask for, noise, and never removes a match that would have been found. Replace the word instead
of adding to it and the same rule would hurt.

That generalises: design the failure mode and you can afford a worse component.

## What does Hebrew actually cost in tokens?

**It depends entirely on the tokenizer, and the spread is much larger than the folklore.**

| tokenizer | English tokens | Hebrew tokens | ratio |
|---|---|---|---|
| GPT-2 (English BPE) | 54 | 234 | **4.33x** |
| XLM-R (multilingual) | 66 | 70 | **1.06x** |

The same four sentence pairs, same content. The single word `ובמסמכים` becomes **10 tokens** of
byte fragments under GPT-2 and **3 tokens** under XLM-R, `▁וב`, `מס`, `מכים`, where the first
split happens to match the morphology.

"Hebrew is expensive" is a statement about a tokenizer, not about Hebrew.

## Is an LLM-as-judge worth anything?

**Measured on 24 answers with ground truth known by construction**, twelve questions, each with
an answer that states its fact and one that does not. Judge: `Llama-3.2-3B-Instruct-Q4_K_M`.

| judge | separated the good answer from the bad one |
|---|---|
| holistic, score 1-5 | **6 / 12** |
| binary, one checkable question | 7 / 12 |
| `contains_fact`, no model at all | **24 / 24** |

The holistic judge scored `3.83` on average for answers that kept the fact and `3.42` for
answers that lost it, a gap of `0.42` on a five-point scale. Per question pair it was right
exactly half the time, which is what a coin gets.

The binary judge failed differently rather than less: `8.3%` false positives against `41.7%`
false negatives. It rarely approves a bad answer and frequently rejects a good one. One
accuracy figure would have hidden that, and the two failures need completely different fixes.

**Caveat:** a 3B judge is a weak judge, and a larger one does better. What does not change with
model size is the method, measure the judge against known answers before trusting it, and look
at both error types separately.

## Does summarising a conversation save money?

**No, and it was the only strategy that lost information.** Six planted facts, buried under tool
output, then asked for back.

| strategy | characters | saved | fact recalled | cache kept |
|---|---|---|---|---|
| keep everything | 10,397 | 0% | 4/6 | 100% |
| **summarise the old turns** | 3,433 | **67%** | **0/6** | **0%** |
| **cap tool output on write** | 5,549 | 46% | **4/6** | 100% |

The mechanism is decided by comparing bytes, not by calling a model. A prompt cache hits only
on an unchanged prefix. Appending leaves the prefix intact; summarising rewrites the beginning
of the conversation, so every cached token behind it is thrown away and the next turn pays full
price for the whole history.

Capping is the cheap win because it applies **on write**: the long output never entered the
history, so nothing that was already sent is rewritten.

**Caveat:** keeping everything recalled 4 of 6, not 6 of 6. The two misses are a 3B model
failing to find a fact that is demonstrably still in front of it. The comparison between
strategies on identical probes is what the chapter rests on, and there it is 4 against 0.

## Why do small models break tool-calling JSON?

**Mostly because of two choices we make, not because of the model.** One tool, described twice;
eight tasks; up to three attempts.

| schema | validator | first try | eventually | mean calls |
|---|---|---|---|---|
| narrow (1-line description) | strict | 75% | 88% | 1.38 |
| narrow | **lenient** | **100%** | 100% | **1.00** |
| wide (6-line description) | strict | **12%** | 100% | 1.88 |
| wide | **lenient** | **88%** | 100% | 1.12 |

**Description length.** The same tool with the same parameters and the same tasks: lengthening
the description from one line to six dropped first-try success from `75%` to `12%`. A long
description is not better documentation, it is noise the model has to get through.

**Validator strictness.** `89%` of every first-attempt failure, in both schemas, was the model
writing `null` for an **optional** parameter. The schema said the key was optional; the model
said "not this one"; the validator rejected a correct answer. Reading `null` as "absent" for
optional keys took the wide schema from `12%` to `88%`.

**The repair loop earns its place either way**: it took the wide schema to `100%` eventually
even from a `12%` start. But it pays for that in extra calls, and the worst combination costs
`1.88` calls per task against `1.00` for the best.
