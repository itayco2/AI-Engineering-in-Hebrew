"""Does a Hebrew-specific embedder beat a general multilingual one on Hebrew retrieval?

Chapter 02's outline depends on the answer, so it is measured before the chapter is written.
The instrument is HeQ (Etelis/HeQ_v1, 30,147 human-written Hebrew questions over Wikipedia
and Geektime passages) rather than hand-made probes: a handful of examples written by the
person running the test measures the test author, not the models.

Honest framing: if the Hebrew-specific models lose, that is the chapter's finding. A negative
result on a real Hebrew eval set is worth more than a promotional one.

Run: python scripts/spike_02_hebrew.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aihe.metrics import mean_recall_at_k, mrr
from aihe.retrieval import BM25, cosine_search, indices, tokenize

N_PASSAGES = 200

# Kept laptop-scale deliberately. BGE-M3 is the obvious strong baseline and is ~2.2 GB, which
# would break the promise that a chapter runs on any machine. If the chapter's point needs a
# 2 GB model, that is worth knowing before writing it.
MODELS = [
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "multilingual (ch01's)", None),
    ("intfloat/multilingual-e5-small", "multilingual", "e5"),
    ("imvladikon/sentence-transformers-alephbert", "Hebrew-specific", None),
    ("MPA/sambert", "Hebrew-specific", None),
]


def main() -> int:
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    rows = load_dataset("Etelis/HeQ_v1", split="validation")

    passages, questions, gold = [], [], []
    seen: dict[str, int] = {}
    for row in rows:
        context = row["Context"]
        if context not in seen:
            if len(seen) >= N_PASSAGES:
                continue
            seen[context] = len(passages)
            passages.append(context)
        questions.append(row["Question"])
        gold.append(seen[context])

    print(f"HeQ validation: {len(questions)} questions over {len(passages)} passages")
    print(f"a random guess would score recall@5 = {5 / len(passages):.3f}\n")

    # BM25 first: the baseline that needs no model at all, and the one chapter 02 argues
    # Hebrew morphology quietly breaks.
    bm25 = BM25([tokenize(p) for p in passages])
    bm_eval = [(indices(bm25.search(tokenize(q), k=20)), {g}) for q, g in zip(questions, gold, strict=True)]
    print(f"{'model':50s} {'kind':22s} {'r@1':>6s} {'r@5':>6s} {'MRR':>6s}")
    print("-" * 96)
    print(f"{'BM25 (no model at all)':50s} {'keyword baseline':22s} "
          f"{mean_recall_at_k(bm_eval,1):6.3f} {mean_recall_at_k(bm_eval,5):6.3f} {mrr(bm_eval):6.3f}")

    results = {}
    for name, kind, prefix in MODELS:
        try:
            model = SentenceTransformer(name, device="cpu")
            # e5 is trained with asymmetric prefixes; omitting them is a known way to make it
            # look worse than it is, which would make this comparison dishonest.
            docs = [f"passage: {p}" for p in passages] if prefix == "e5" else passages
            queries = [f"query: {q}" for q in questions] if prefix == "e5" else questions

            doc_vectors = model.encode(docs, normalize_embeddings=True, show_progress_bar=False,
                                       batch_size=16)
            query_vectors = model.encode(queries, normalize_embeddings=True,
                                         show_progress_bar=False, batch_size=16)
            evalset = [
                (indices(cosine_search(query_vectors[i], doc_vectors, k=20)), {gold[i]})
                for i in range(len(questions))
            ]
            r1, r5, rr = mean_recall_at_k(evalset, 1), mean_recall_at_k(evalset, 5), mrr(evalset)
            results[name] = (kind, r1, r5, rr)
            print(f"{name[:48]:50s} {kind:22s} {r1:6.3f} {r5:6.3f} {rr:6.3f}")
        except Exception as exc:
            print(f"{name[:48]:50s} {kind:22s}  FAILED {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:40]}")

    if results:
        best = max(results.items(), key=lambda kv: kv[1][3])
        hebrew = [v for v in results.values() if v[0] == "Hebrew-specific"]
        multi = [v for v in results.values() if v[0].startswith("multilingual")]
        print(f"\nbest by MRR: {best[0]} ({best[1][0]})")
        if hebrew and multi:
            bh, bm = max(h[3] for h in hebrew), max(m[3] for m in multi)
            verdict = "Hebrew-specific wins" if bh > bm else "the general multilingual model wins"
            print(f"best Hebrew-specific MRR {bh:.3f} vs best multilingual {bm:.3f} -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
