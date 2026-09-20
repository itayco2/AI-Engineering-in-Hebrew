"""Every figure in this repo is generated here. None is ever a screenshot.

Two reasons, and the second is the important one. A generated figure cannot go stale
silently — change the data and the picture changes with it. And a figure drawn from numbers
the reader just computed proves its claim, where a screenshot of someone else's chart only
repeats it.

Labels inside figures are English. The explanation around a figure is Hebrew, but matplotlib
does no bidi shaping, so Hebrew axis labels render reversed and disconnected. English inside
the figure and Hebrew prose around it is the honest solution rather than a broken one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

ACCENT = "#c8102e"
INK = "#1b1b1b"
MUTED = "#8a8a8a"


def _axes(width: float = 7.0, height: float = 4.0):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height), dpi=120)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
    return fig, ax


def staircase(results: Mapping[str, float], k: int = 20, title: str = ""):
    """Failed retrieval rate per setup — the shape Anthropic reported in 2024.

    Stated as a failure rate rather than as recall on purpose: halving a failure rate reads
    as the improvement it is, where "recall went from 94.3% to 97.1%" does not.
    """
    import matplotlib.pyplot as plt

    fig, ax = _axes()
    names = list(results)
    values = [results[name] * 100 for name in names]
    worst = max(values) if values else 1.0

    colours = [MUTED] * max(len(values) - 1, 0) + [ACCENT]
    bars = ax.bar(names, values, color=colours[: len(values)], width=0.6)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + worst * 0.03,
            f"{value:.1f}%",
            ha="center",
            color=INK,
            fontsize=10,
        )

    ax.set_ylabel(f"failed retrievals  (1 - recall@{k})")
    ax.set_ylim(0, worst * 1.25 if worst else 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    if title:
        ax.set_title(title, color=INK)
    fig.tight_layout()
    return fig


def recall_curve(curves: Mapping[str, Sequence[float]], ks: Sequence[int], title: str = ""):
    """recall@k against k, one line per retrieval method."""
    fig, ax = _axes()
    last = len(curves) - 1
    for i, (name, values) in enumerate(curves.items()):
        ax.plot(
            list(ks),
            [v * 100 for v in values],
            marker="o",
            linewidth=2.0 if i == last else 1.4,
            color=ACCENT if i == last else MUTED,
            label=name,
        )
    ax.set_xlabel("k  (documents kept)")
    ax.set_ylabel("recall@k")
    ax.set_ylim(0, 105)
    ax.set_xticks(list(ks))
    ax.legend(frameon=False)
    if title:
        ax.set_title(title, color=INK)
    fig.tight_layout()
    return fig


def chunk_sizes(distributions: Mapping[str, Sequence[int]], title: str = ""):
    """How three chunking strategies actually cut the same corpus."""
    fig, ax = _axes()
    names = list(distributions)
    ax.boxplot(
        [list(distributions[n]) for n in names],
        tick_labels=names,
        patch_artist=True,
        boxprops={"facecolor": "#ececec", "color": MUTED},
        medianprops={"color": ACCENT, "linewidth": 2},
        whiskerprops={"color": MUTED},
        capprops={"color": MUTED},
        flierprops={"markeredgecolor": MUTED, "markersize": 3},
    )
    ax.set_ylabel("chunk length (characters)")
    if title:
        ax.set_title(title, color=INK)
    fig.tight_layout()
    return fig


def save(fig, path: str) -> str:
    """Write a figure and close it, so a long notebook does not leak memory."""
    import matplotlib.pyplot as plt

    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
