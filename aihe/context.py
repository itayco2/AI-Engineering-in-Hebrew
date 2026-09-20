"""What goes into the context window, and why rewriting it is expensive.

The counter-intuitive result chapter 04 is built on: keeping the whole history beat summarising
it on cost, on latency and on recall at the same time. The mechanism is the cache. A provider's
prompt cache hits only on an **unchanged prefix**, so a summary that rewrites the beginning of
the conversation throws away every cached token behind it.

That makes the decisive check a byte comparison rather than a model call, which is why this
module is deterministic and fully testable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    """One message in a conversation."""

    role: str
    content: str

    def render(self) -> str:
        return f"{self.role}: {self.content}\n"


def render(turns: Sequence[Turn]) -> str:
    """Serialise a conversation exactly as it would be sent."""
    return "".join(turn.render() for turn in turns)


def common_prefix_length(before: str, after: str) -> int:
    """How many leading characters two renderings share."""
    limit = min(len(before), len(after))
    for i in range(limit):
        if before[i] != after[i]:
            return i
    return limit


def prefix_is_stable(before: str, after: str) -> bool:
    """True when the earlier rendering is still an exact prefix of the later one.

    This is the whole cache question, answered without a model and without a network call. If
    it is False, every cached token is gone and the next turn pays full price for the entire
    history - which is how a summary that was supposed to save money costs more.
    """
    return after.startswith(before)


def cache_hit_ratio(before: str, after: str) -> float:
    """The share of the earlier rendering that is still cacheable."""
    if not before:
        return 1.0
    return common_prefix_length(before, after) / len(before)


# --- the three strategies ----------------------------------------------------------------


def keep_all(turns: Sequence[Turn]) -> list[Turn]:
    """Append and never rewrite. The prefix is stable by construction."""
    return list(turns)


def summarise_old(
    turns: Sequence[Turn], keep_recent: int, summary: str, role: str = "system"
) -> list[Turn]:
    """Replace everything but the last `keep_recent` turns with one summary.

    Saves tokens on paper and destroys the cached prefix in practice, because the beginning of
    the conversation is now different text. Chapter 04 measures both halves of that trade.
    """
    if keep_recent < 0:
        raise ValueError("keep_recent must not be negative")
    if keep_recent >= len(turns):
        return list(turns)
    return [Turn(role, summary), *turns[len(turns) - keep_recent:]]


def cap_tool_output(content: str, limit: int, marker: str = " …[truncated]") -> str:
    """Cut one tool result down to `limit` characters, at the moment it is written.

    The cheap win. Applying the cap **on write** means the history never contains the long
    version, so shrinking costs nothing in cache terms - unlike a summary, which shortens text
    that was already sent and already cached.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    if len(content) <= limit:
        return content
    return content[:limit] + marker


def capped_history(turns: Sequence[Turn], limit: int, tool_role: str = "tool") -> list[Turn]:
    """A conversation in which every tool result was capped as it was appended."""
    return [
        Turn(turn.role, cap_tool_output(turn.content, limit) if turn.role == tool_role else turn.content)
        for turn in turns
    ]


# --- what it costs ------------------------------------------------------------------------


def estimate_cost(
    rendered: str,
    cached_prefix: str = "",
    price_per_1k: float = 0.003,
    cached_discount: float = 0.1,
    chars_per_token: float = 4.0,
) -> float:
    """A cost estimate in which cached tokens are cheaper than fresh ones.

    The prices are illustrative and the point is the shape, not the figure: when the prefix is
    stable almost everything is cached, and when it is not, nothing is.
    """
    cached_chars = common_prefix_length(cached_prefix, rendered) if cached_prefix else 0
    fresh_chars = len(rendered) - cached_chars
    per_char = price_per_1k / 1000 / chars_per_token
    return fresh_chars * per_char + cached_chars * per_char * cached_discount


def planted_fact_probe(
    fact: str, probe: str, filler: Sequence[str], filler_role: str = "tool"
) -> list[Turn]:
    """Build the harness chapter 04 uses: state a fact, bury it, then ask for it back.

    A fact at turn zero, a pile of ordinary turns on top, and a question at the end that only
    the fact can answer. Whether the model still has it is then a property of what the context
    strategy kept, not of how well the model writes.

    The filler is tool output by default, because that is what actually fills a real agent's
    context, and it is the only part a cap is allowed to touch.
    """
    turns = [Turn("user", fact), Turn("assistant", "הבנתי.")]
    for item in filler:
        turns.append(Turn(filler_role, item))
        turns.append(Turn("assistant", "בסדר."))
    turns.append(Turn("user", probe))
    return turns
