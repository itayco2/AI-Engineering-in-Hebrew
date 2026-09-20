# Where this data came from

## `heq-subset.json`

A 150-passage, 1,036-question subset of **HeQ**, the Hebrew Question Answering dataset, built
by NNLP-IL and Bar-Ilan University. Human-written questions over Hebrew Wikipedia and Geektime
articles: 79 Wikipedia passages and 71 Geektime passages here.

- **Source:** [Etelis/HeQ_v1](https://huggingface.co/datasets/Etelis/HeQ_v1) on Hugging Face
- **Project:** [NNLP-IL/Hebrew-Question-Answering-Dataset](https://github.com/NNLP-IL/Hebrew-Question-Answering-Dataset)
- **Paper:** HeQ: A Dataset for Hebrew Question Answering
- **Licence:** CC BY 4.0, redistributed here with attribution, unmodified apart from taking a
  subset and renaming the fields to match this repository's shape.

The full validation split is 1,501 questions; this subset is the 1,036 whose passages fall in
the first 150 unique contexts, kept small enough to embed on a laptop in under a minute.

**A limitation worth stating, because the chapter's numbers depend on it.** HeQ is *extractive*
question answering: every answer is a span inside its passage, so the questions share vocabulary
with their passages by construction. That flatters keyword matching. When BM25 scores well here,
part of that is the benchmark rather than the method, and the chapter says so.

## `dictabert-seg.json`

The segmentation that [`dicta-il/dictabert-seg`](https://huggingface.co/dicta-il/dictabert-seg)
produces for 574 unique Hebrew words drawn from the passages above, recorded once.

The model is about 700 MB. Recording its output means the chapter can compare rules against it
without asking every reader to download it, the same reasoning as the cassettes used for
language models. Reproduce it with `aihe.hebrew.segment()`.

**Read `PREFLIGHT.md` before trusting this model in your own code.** On current transformers it
loads with a randomly initialised prefix head, raises nothing, and segments confidently and
wrongly. `aihe.hebrew.load_segmenter` fixes it and refuses to return a model whose head did not
load.
