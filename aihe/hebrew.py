"""What Hebrew does to a retrieval pipeline that was built for English.

Three things break. Hebrew glues its function words onto the front of the next word, so
`ובמסמכים` is one token to any English-shaped tokenizer and four morphemes to a reader. It is
written without vowels, so one spelling covers several words. And it is expensive: the same
sentence costs far more tokens in Hebrew than in English with most tokenizers.

Two ways to split those prefixes are offered here, and chapter 02 measures both: a rule-based
splitter that needs nothing at all, and a model that needs about 700 MB. Which one wins is a
measurement, not an opinion.
"""

from __future__ import annotations

import functools
import json
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path

# The letters Hebrew glues onto the front of a word: conjunction, prepositions, the definite
# article, the relativiser. They stack, which is why `ובמסמכים` is `ו` + `ב` + `מסמכים`.
PREFIX_LETTERS = "ובלכמשה"

# Ordered longest-first so a stacked prefix is taken whole. Deliberately short: `לה` is not
# here, because `להצפנה` is `ל` + `הצפנה` (to the encryption) and stripping both letters
# destroys the stem. Every entry here was checked against the model's segmentation.
PREFIX_CLUSTERS = ("כשה", "ובה", "כש", "שב", "שה", "וה", "וב", "ול", "וכ", "ומ")

NIQQUD = re.compile(r"[֑-ׇ]")
HEBREW_WORD = re.compile(r"[א-ת]+")

FINAL_FORMS = str.maketrans("ךםןףץ", "כמנפצ")

# Frequent words that begin with a prefix letter but are not prefixed. Splitting these is the
# rule-based splitter's characteristic failure, and chapter 02 shows it rather than hiding it.
NOT_PREFIXED = {
    "של", "שם", "שנה", "שלום", "מה", "מי", "מים", "לא", "כל", "כי", "בן", "בת",
    "היה", "הוא", "היא", "הם", "הן", "ולא", "בית", "משה", "שלו", "שלה", "מעל",
    "בכל", "לכל", "מכל", "ועד", "מול", "לפי", "כמו", "בין", "ליד", "מאז", "שוב",
}


def normalize(text: str) -> str:
    """Strip niqqud and cantillation marks, and normalise the Unicode form.

    Vowel marks appear in poetry, scripture and children's books and essentially nowhere in
    the technical text a retrieval system indexes. Leaving them in means the same word fails
    to match itself.
    """
    return NIQQUD.sub("", unicodedata.normalize("NFKD", text))


def fold_finals(word: str) -> str:
    """Map the five final letter forms onto their ordinary forms.

    Hebrew writes five letters differently at the end of a word. Splitting a prefix off can
    leave a final form in the middle, where it no longer matches the same word written
    normally.
    """
    return word.translate(FINAL_FORMS)


def strip_prefixes(word: str, min_stem: int = 3) -> list[str]:
    """Split a Hebrew word into its prefix and stem, using rules and no model.

    **Strips at most once.** An earlier version looped, and it turned `ובמסמכים` into
    `וב` + `מ` + `סמכים` and `השרתים` into `ה` + `ש` + `רתים` - it kept finding prefix letters
    because the stems genuinely begin with them. Stacked prefixes are handled by the cluster
    list instead, which is bounded.

    It is still wrong on genuinely ambiguous words. `מידע` (information) is not `מ` + `ידע`,
    and `שולחן` (a table) is not `ש` + `ולחן`, but nothing in the spelling says so - you need
    a lexicon or a model. Chapter 02 measures how often it is wrong instead of claiming a rate.
    """
    if not word or not HEBREW_WORD.fullmatch(word):
        return [word] if word else []
    if word in NOT_PREFIXED:
        return [word]

    for cluster in PREFIX_CLUSTERS:
        if word.startswith(cluster) and len(word) - len(cluster) >= min_stem:
            return [cluster, word[len(cluster):]]

    if word[0] in PREFIX_LETTERS and len(word) - 1 >= min_stem:
        return [word[0], word[1:]]
    return [word]


def tokenize_hebrew(text: str, split: bool = True) -> list[str]:
    """Tokenize Hebrew for a keyword index, optionally splitting glued prefixes.

    With `split=False` this is `aihe.retrieval.tokenize` plus normalisation. With `split=True`
    each word also contributes its stem, so a query for `מסמכים` can match `ובמסמכים`.
    """
    words = re.findall(r"\w+", normalize(text).lower(), re.UNICODE)
    if not split:
        return words

    out: list[str] = []
    for word in words:
        out.append(word)
        pieces = strip_prefixes(word)
        if len(pieces) > 1:
            # Index the stem as well as the whole word, so a query for `מסמכים` matches
            # `ובמסמכים`. The word itself is kept because the stem is sometimes wrong.
            out.append(pieces[-1])
    return out


# --- the model-based splitter -----------------------------------------------------------

SEG_MODEL = "dicta-il/dictabert-seg"

_HEAD_BROKEN = """{model} loaded with a randomly initialised prefix head.

The checkpoint stores the head at the top level (`classifiers.*`, `transform.*`,
`prefix_class_embeddings`) while the current remote code expects it under `prefix.*`. The
loader fills the mismatch with random weights, raises nothing, and then segments confidently
and wrongly - `מידע` comes back as `מ` + `ידע`.

`aihe.hebrew.load_segmenter` remaps the keys. If you see this message the remap failed, so do
not trust the output."""


@functools.lru_cache(maxsize=1)
def load_segmenter(model_name: str = SEG_MODEL):
    """Load DictaBERT's prefix segmenter with its head actually attached.

    Returns `(model, tokenizer)`. Raises rather than returning a model whose head is random -
    a silently random head is worse than a missing model, because its output looks plausible.
    """
    import torch  # noqa: F401  (imported for the side effect of a clear error when absent)
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    weights = load_file(hf_hub_download(model_name, "model.safetensors"))
    remapped = {
        (key if key.startswith("bert.") else f"prefix.{key}"): value
        for key, value in weights.items()
    }
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        raise RuntimeError(_HEAD_BROKEN.format(model=model_name) + f"\n\nmissing: {missing[:4]}")

    model.eval()
    return model, tokenizer


def segment(sentences: Sequence[str], model_name: str = SEG_MODEL) -> list[list[list[str]]]:
    """Split every word in each sentence into prefixes and stem, using the model.

    Returns one list per sentence, one list of pieces per word, with the sentence markers
    dropped. Accurate, and about 700 MB more expensive than `strip_prefixes`.
    """
    model, tokenizer = load_segmenter(model_name)
    predictions = model.predict(list(sentences), tokenizer)
    return [
        [word for word in sentence if word and word[0] not in ("[CLS]", "[SEP]")]
        for sentence in predictions
    ]


# --- what Hebrew costs --------------------------------------------------------------------


def token_cost(texts: Sequence[str], tokenizer) -> dict[str, float]:
    """Characters, words and tokens for a body of text, plus tokens per word.

    Chapter 02 runs this over the same content in Hebrew and in English. The ratio is the
    number that explains why a Hebrew context window holds less than an English one.
    """
    characters = sum(len(t) for t in texts)
    words = sum(len(re.findall(r"\w+", t, re.UNICODE)) for t in texts)
    tokens = sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in texts)
    return {
        "characters": characters,
        "words": words,
        "tokens": tokens,
        "tokens_per_word": tokens / words if words else 0.0,
        "characters_per_token": characters / tokens if tokens else 0.0,
    }


def load_reference(path: str | Path) -> dict[str, list[str]]:
    """Load a recorded DictaBERT segmentation.

    The model is about 700 MB, which would break the promise that a chapter runs on any
    laptop. Its output over the chapter's own text is recorded once and committed instead -
    the same reasoning as a cassette, applied to a model that is not a language model.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))["words"]


def compare_to_reference(reference: dict[str, list[str]]) -> dict[str, object]:
    """How often the rule-based splitter matches the model, and where it does not.

    Returns the counts plus the disagreements, because the disagreements are the content:
    they show exactly which words rules cannot get right without a lexicon.
    """
    disagreements = [
        (word, expected, strip_prefixes(word))
        for word, expected in reference.items()
        if strip_prefixes(word) != expected
    ]
    total = len(reference)
    return {
        "total": total,
        "agree": total - len(disagreements),
        "rate": (total - len(disagreements)) / total if total else 0.0,
        "disagreements": disagreements,
    }
