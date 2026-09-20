# AI Engineering in Hebrew

A free, runnable course in modern AI engineering — written in Hebrew, with all code in
English. RAG, Hebrew retrieval, evals, context engineering and agents.

**[→ Read it](https://itayco2.github.io/AI-Engineering-in-Hebrew/)**

## Why it exists

Excellent material on RAG, agents and evals exists almost entirely in English. Hebrew has one
good book on classical deep learning and nothing on the 2026 stack. This fills that gap.

The prose is Hebrew because that is the gap. Everything else — code, identifiers, comments,
figure labels, technical terms — is English, so the repository still reads as engineering work
to anyone, and so a reader can search for a term five minutes later.

## What is different about it

- **Numbers are measured, not cited.** Chapter 01 reproduces the contextual-retrieval
  staircase on its own 46-document corpus: `recall@1` climbs 57.1% → 67.9% → 78.6% → 82.1%
  across four steps, computed on the reader's machine.
- **Failures are documented.** [`PREFLIGHT.md`](PREFLIGHT.md) lists every defect found while
  building, with the number that exposed it and the number after the fix. One of them: the
  reranker returned `NaN` silently, raised nothing, and became a no-op — a sort leaves NaN
  ordering untouched, so the step reported success while changing nothing.
- **Free to run is engineered, not promised.** Chapters 01–03 call no language model at all.
  Chapters that do ship recorded real responses, so CI makes no network call and costs nothing.
- **Nothing rots.** All logic lives in `aihe/`, so the fast suite runs in under a second with
  no model, and every chapter notebook is executed on every push.

## Run it

```bash
git clone https://github.com/itayco2/AI-Engineering-in-Hebrew
cd AI-Engineering-in-Hebrew
make setup && make run-01
```

`make setup` uses `uv` when present and falls back to `venv` + `pip` when it isn't.
`make gate` checks everything in under a minute and usually names the problem.

## Licence

Code is MIT. Prose, chapters and figures are CC BY-SA 4.0. See [LICENSE](LICENSE).
