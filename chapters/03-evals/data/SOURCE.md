# Where this data came from

## `judge-cases.json`

Twelve questions about the `Meridian` documentation from chapter 01, each with two answers.

- The **kept** answer states a specific fact, "sixty minutes", "five hundred records".
- The **lost** answer is fluent, on-topic, helpful-sounding, and does not state it. It is not a
  bad answer in any obvious way. It is the answer you get when retrieval quietly dropped the
  chunk that held the number.

Written for this chapter, and written this way on purpose: **the ground truth is known by
construction.** Whether an answer contains its fact is decided by `aihe.judge.contains_fact`, a
substring check with no model in it, and `tests/test_chapter.py` asserts that all twelve kept
answers contain their fact and none of the twelve lost answers do.

That matters more than it sounds. A chapter that measures judges has to be measured against
something that cannot itself be argued with, or it is just two opinions disagreeing.

**Licence:** written for this repository, CC BY-SA 4.0.

## `cassettes/`

48 real replies from `Llama-3.2-3B-Instruct-Q4_K_M`, two judges over 24 answers, recorded at
`temperature 0`, `seed 7`. Recorded once with `make record` so the chapter runs offline and CI
makes no network call and costs nothing.

They are recordings, not mocks. The judge in this chapter is genuinely mediocre, and a mock
written by the author would have been better than the real thing, which would have destroyed
the chapter's point.

**The judge is a 3B model, and the chapter says so where it matters.** A stronger judge scores
better. What does not change with a stronger judge is the method: measure the judge against
known ground truth before trusting it.
