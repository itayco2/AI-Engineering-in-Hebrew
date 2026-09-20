"""Hebrew text handling: prefix splitting, normalisation, and what it costs.

All pure functions — no model, no download. The model-based segmenter is exercised in
chapter 02's own tests, where it is marked slow.
"""

from __future__ import annotations

import pytest

from aihe.hebrew import (
    NOT_PREFIXED,
    PREFIX_CLUSTERS,
    fold_finals,
    normalize,
    strip_prefixes,
    tokenize_hebrew,
)

# Checked against dicta-il/dictabert-seg, which is the reference this repo measures against.
AGREES_WITH_MODEL = {
    "ובמסמכים": ["וב", "מסמכים"],
    "שבמאגר": ["שב", "מאגר"],
    "השרתים": ["ה", "שרתים"],
    "כשהטוקן": ["כשה", "טוקן"],
    "להצפנה": ["ל", "הצפנה"],
    "שבדיסק": ["שב", "דיסק"],
    "של": ["של"],
}

# Where rules cannot win without a lexicon. Pinned so the chapter's honesty stays honest:
# if a future change "fixes" these, the chapter's measured disagreement rate is stale.
KNOWN_WRONG = {
    "מידע": ["מ", "ידע"],    # information, not "from knowledge"
    "שולחן": ["ש", "ולחן"],  # a table, not "that ..."
}


@pytest.mark.parametrize("word,expected", sorted(AGREES_WITH_MODEL.items()))
def test_rules_agree_with_the_model_where_they_can(word, expected):
    assert strip_prefixes(word) == expected


@pytest.mark.parametrize("word,wrong", sorted(KNOWN_WRONG.items()))
def test_known_failures_are_still_failures(word, wrong):
    """These are the chapter's evidence that rules are cheap and imperfect."""
    assert strip_prefixes(word) == wrong


def test_stripping_happens_at_most_once():
    """The bug that made this worth pinning: a loop turned `ובמסמכים` into three pieces by
    finding prefix letters inside the stem, because Hebrew stems often start with them."""
    for word in ("ובמסמכים", "השרתים", "שבמאגר"):
        assert len(strip_prefixes(word)) <= 2


def test_a_short_word_is_left_alone():
    """Stripping must never leave a stem too short to mean anything."""
    assert strip_prefixes("בית") == ["בית"]
    assert strip_prefixes("מים") == ["מים"]


def test_known_whole_words_are_never_split():
    for word in sorted(NOT_PREFIXED)[:8]:
        assert strip_prefixes(word) == [word]


def test_non_hebrew_is_returned_untouched():
    assert strip_prefixes("embedding") == ["embedding"]
    assert strip_prefixes("BM25") == ["BM25"]
    assert strip_prefixes("") == []


def test_clusters_are_ordered_longest_first():
    """Otherwise `כש` would win before `כשה` and the stem would keep a stray letter."""
    lengths = [len(c) for c in PREFIX_CLUSTERS]
    assert lengths == sorted(lengths, reverse=True)


def test_the_rejected_cluster_stays_rejected():
    """`לה` is not in the cluster list on purpose: `להצפנה` is `ל` + `הצפנה`, and taking both
    letters destroys the stem."""
    assert "לה" not in PREFIX_CLUSTERS
    assert strip_prefixes("להצפנה") == ["ל", "הצפנה"]


# --- normalisation ----------------------------------------------------------------------


def test_niqqud_is_removed():
    assert normalize("שָׁלוֹם") == "שלום"


def test_normalising_makes_a_vowelled_word_match_a_plain_one():
    """The whole point: otherwise the same word fails to match itself."""
    assert normalize("בְּרֵאשִׁית") == normalize("בראשית")


def test_plain_text_is_unchanged():
    assert normalize("מסמכים") == "מסמכים"
    assert normalize("hello") == "hello"


def test_final_letters_fold_to_their_ordinary_forms():
    assert fold_finals("שלום") == "שלומ"
    assert fold_finals("ארץ") == "ארצ"


# --- tokenizing -------------------------------------------------------------------------


def test_tokenizing_indexes_the_stem_as_well_as_the_word():
    """So that a query for `מסמכים` can match a document containing `ובמסמכים`."""
    tokens = tokenize_hebrew("ובמסמכים שבמאגר")
    assert "ובמסמכים" in tokens and "מסמכים" in tokens
    assert "שבמאגר" in tokens and "מאגר" in tokens


def test_the_whole_word_is_always_kept():
    """Because the stem is sometimes wrong, dropping the original would lose a real match."""
    assert "מידע" in tokenize_hebrew("מידע")


def test_splitting_can_be_turned_off():
    assert tokenize_hebrew("ובמסמכים", split=False) == ["ובמסמכים"]


def test_tokenizing_normalises_first():
    assert "שלום" in tokenize_hebrew("שָׁלוֹם")


def test_english_survives_hebrew_tokenizing():
    assert tokenize_hebrew("the API returns JSON", split=False) == ["the", "api", "returns", "json"]
