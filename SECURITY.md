# Security

## Reporting something

Open a [private security advisory](https://github.com/itayco2/AI-Engineering-in-Hebrew/security/advisories/new),
or email itay.cohen2907@gmail.com. Expect a reply within a week.

## What this repository is, and what that means for risk

It is teaching material: notebooks, a small pure-Python package, and recorded model
responses. It is not a service, it holds no user data, and it stores no credentials.

Two things are worth knowing anyway.

**Notebooks execute code.** Every chapter runs on your machine. Read a cell before you run it
— the same rule that applies to any notebook you did not write.

**Cassettes are committed and readable.** Recorded model responses live in each chapter's
`cassettes/` directory as plain JSON, deliberately: a recorded answer should be reviewable in
a diff. Nothing secret is ever recorded. If you record your own against a paid provider,
check the diff before committing, and note that `aihe.cassettes` stores the request as well
as the response.

## Dependencies

The package itself declares no required dependencies — the pure functions need only the
standard library. Everything heavier is an optional extra, which keeps the attack surface
proportional to what you actually asked for.
