# Where this data came from

## `probes.json`

Six planted-fact probes, written for this chapter. Each one states a distinctive fact, a
deploy code, a region, a number of days, then buries it under five turns of realistic tool
output (pagination logs and indexing reports), then asks for the fact back.

The facts are deliberately arbitrary. `TANGERINE-7741` cannot be guessed, inferred or
recovered from general knowledge, so whether the model answers correctly is a property of what
the context strategy kept and nothing else.

The filler is **tool output**, because that is what actually fills a real agent's context, and
because a cap is only allowed to shorten tool results. Filler written as user messages would
have made the capping strategy a no-op and the comparison meaningless.

The summary is a realistic one: it keeps the gist and drops the specific values. That is what
summarisation does, and it is exactly why the fact disappears.

**Licence:** written for this repository, CC BY-SA 4.0.

## `cassettes/`

18 real replies from `Llama-3.2-3B-Instruct-Q4_K_M`, six probes under three strategies,
recorded at `temperature 0`, `seed 7`.

**One honest caveat about the numbers.** Keeping everything recalls the fact on four probes out
of six, not six. The two misses are the small model failing to find a fact that is demonstrably
still in its context, which is a limitation of a 3B model rather than of the strategy. What the
chapter rests on is the comparison between strategies on identical probes, and there the
difference is total: four out of six against zero out of six.
