"""Asking a model to grade another model, and why one score is not enough.

The finding chapter 03 turns on: a holistic judge rated replies "good" on 97-99% of turns while
the thing being tested had actually lost the fact on 42% of them. On the turns where the fact
was demonstrably gone, the holistic judge still said good - every single time.

A judge that agrees with you always has told you nothing. The fix is not a better judge, it is
a narrower question: one fact, one yes or no, checkable against something you already know.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

Ask = Callable[[str], str]

YES = re.compile(r"\b(yes|true|כן)\b", re.IGNORECASE)
NO = re.compile(r"\b(no|false|לא)\b", re.IGNORECASE)
SCORE = re.compile(r"\b([1-5])\b")


@dataclass
class Judgement:
    """One verdict, with the text it came from so a disagreement can be inspected."""

    value: bool | int | None
    raw: str

    @property
    def decided(self) -> bool:
        return self.value is not None


def read_yes_no(text: str) -> bool | None:
    """Pull a decision out of a reply, or return None when the model waffled.

    An undecided answer is reported rather than coerced. Treating "it depends" as a no is how a
    judge quietly acquires a bias nobody measured.
    """
    first_yes = YES.search(text)
    first_no = NO.search(text)
    if first_yes and first_no:
        return first_yes.start() < first_no.start()
    if first_yes:
        return True
    if first_no:
        return False
    return None


def read_score(text: str) -> int | None:
    match = SCORE.search(text)
    return int(match.group(1)) if match else None


BINARY_PROMPT = """You are checking one specific fact. Answer with a single word, yes or no.

Question asked: {question}
Answer given: {answer}

Does the answer state that {criterion}? Answer yes or no."""

HOLISTIC_PROMPT = """Rate this reply from 1 to 5, where 5 is an excellent, helpful reply.

Question asked: {question}
Reply: {answer}

Answer with the number only."""


def binary_judge(question: str, answer: str, criterion: str, ask: Ask) -> Judgement:
    """Ask one checkable yes-or-no question about one answer.

    The criterion must be specific enough that a person could settle it in a second. "states
    that the token is valid for sixty minutes" is checkable; "is accurate" is not.
    """
    raw = ask(BINARY_PROMPT.format(question=question, answer=answer, criterion=criterion))
    return Judgement(read_yes_no(raw), raw)


def holistic_judge(question: str, answer: str, ask: Ask) -> Judgement:
    """Ask for an overall score. Included so the chapter can show it failing."""
    raw = ask(HOLISTIC_PROMPT.format(question=question, answer=answer))
    return Judgement(read_score(raw), raw)


def agreement(judgements: Sequence[Judgement], truth: Sequence[bool]) -> dict[str, float]:
    """How often a judge matched a known answer, and how it got things wrong.

    Reported as four counts rather than one accuracy, because a judge that says yes to
    everything and a judge that is genuinely right score the same on accuracy alone when most
    answers are good.
    """
    if len(judgements) != len(truth):
        raise ValueError(f"{len(judgements)} judgements for {len(truth)} known answers")

    decided = [(j.value, t) for j, t in zip(judgements, truth, strict=True) if j.decided]
    if not decided:
        return {"n": 0, "undecided": len(judgements), "accuracy": 0.0,
                "said_yes": 0.0, "false_positive": 0.0, "false_negative": 0.0}

    correct = sum(1 for v, t in decided if bool(v) == t)
    said_yes = sum(1 for v, _ in decided if bool(v))
    false_positive = sum(1 for v, t in decided if bool(v) and not t)
    false_negative = sum(1 for v, t in decided if not bool(v) and t)
    negatives = sum(1 for _, t in decided if not t) or 1
    positives = sum(1 for _, t in decided if t) or 1

    return {
        "n": len(decided),
        "undecided": len(judgements) - len(decided),
        "accuracy": correct / len(decided),
        "said_yes": said_yes / len(decided),
        "false_positive": false_positive / negatives,
        "false_negative": false_negative / positives,
    }


def contains_fact(answer: str, fact: str) -> bool:
    """A judge with no model in it at all: is the fact literally present?

    Crude, free, instant, and impossible to flatter. Chapter 03 uses it as the ground truth
    that the model judges are measured against, which is the only honest way round.
    """
    needle = re.sub(r"\s+", " ", fact.strip().lower())
    haystack = re.sub(r"\s+", " ", answer.strip().lower())
    return needle in haystack


@dataclass
class Case:
    """One answer with a known verdict.

    `truth` is not an opinion: it is whether the answer literally contains the fact, decided by
    `contains_fact`. Every model judge in chapter 03 is scored against this.
    """

    question: str
    answer: str
    fact: str
    criterion: str
    variant: str

    @property
    def truth(self) -> bool:
        return contains_fact(self.answer, self.fact)


def build_cases(raw: Sequence[dict]) -> list[Case]:
    """Expand each record into its two variants: the answer that kept the fact and the one
    that lost it while still reading well."""
    cases: list[Case] = []
    for item in raw:
        for variant in ("kept", "lost"):
            cases.append(Case(item["question"], item[variant], item["fact"],
                              item["criterion"], variant))
    return cases


def judge_cases(cases: Sequence[Case], ask: Ask) -> list[dict]:
    """Run both judges over every case.

    The recorder and the notebook both call this, which is what makes a recording match the
    request the notebook later replays: the prompts are built in one place.
    """
    rows = []
    for case in cases:
        holistic = holistic_judge(case.question, case.answer, ask)
        binary = binary_judge(case.question, case.answer, case.criterion, ask)
        rows.append({
            "question": case.question,
            "variant": case.variant,
            "truth": case.truth,
            "holistic": holistic,
            "binary": binary,
        })
    return rows


def holistic_summary(rows: Sequence[dict], good_from: int = 4) -> dict[str, float]:
    """What a score-out-of-five judge actually told you.

    The number that matters is the last one: on the answers where the fact is demonstrably
    gone, how often did the judge still call the reply good?
    """
    scored = [r for r in rows if r["holistic"].decided]
    if not scored:
        return {"n": 0, "called_good": 0.0, "called_good_when_fact_lost": 0.0, "mean": 0.0}
    lost = [r for r in scored if not r["truth"]]
    good = sum(1 for r in scored if r["holistic"].value >= good_from)
    good_when_lost = sum(1 for r in lost if r["holistic"].value >= good_from)
    return {
        "n": len(scored),
        "called_good": good / len(scored),
        "called_good_when_fact_lost": good_when_lost / len(lost) if lost else 0.0,
        "mean": sum(r["holistic"].value for r in scored) / len(scored),
    }
