# Where this data came from

## `tool-tasks.json`

One tool, described twice, and eight ordinary requests.

- **`search_docs`** — one line of description, three primitive parameters.
- **`search_documentation_corpus`** — the same tool, described the way a real codebase
  describes one: six lines, five parameters. Nothing about it is unreasonable. It is simply
  longer, and length is the variable under test.

The eight tasks are the kind of thing anyone would ask a documentation agent. They are written
in plain English with no hints about JSON, because a task that tells the model how to answer
would be measuring the prompt rather than the model.

**Licence:** written for this repository, CC BY-SA 4.0.

## `cassettes/`

Real replies from `Llama-3.2-3B-Instruct-Q4_K_M` at `temperature 0`, `seed 7` — every attempt
of every repair loop, under both validators.

Both validators are recorded because they produce different rejection messages, and a different
rejection message is a genuinely different request. Recording only one leaves the other with a
cassette miss, which is how this was found.

**A bug worth knowing about if you record your own.** Recording used to overwrite a cassette
when the same prompt was asked twice in one run. Generated text is not reproducible even at
`temperature 0`, so the second reply differed, the cassette changed underneath the first
caller, and replay then walked a path that had never been recorded. `aihe.models.asker` now
skips a request it has already recorded. See `PREFLIGHT.md`.
