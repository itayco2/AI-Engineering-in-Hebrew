<div dir="rtl" lang="he" align="right" markdown="1">

# פרק 01: RAG מאפס

בונים `pipeline` שלם של `retrieval` ומודדים כל שלב: חיתוך לחתיכות, `embeddings`, `BM25`, מיזוג בעזרת `RRF`, ולבסוף `reranker`.

הפרק לא קורא לאף מודל שפה. אין `API key`, אין `GPU`, ואין דרישת זיכרון מיוחדת, כל מספר כאן מחושב אצלך.

בסוף תראה טבלה אחת שמסבירה למה `recall@1` עולה מ-`0.571` ל-`0.821`, ואיזה שלב תיקן מה.

</div>

```bash
make run-01
```

[Read the chapter](https://itayco2.github.io/AI-Engineering-in-Hebrew/chapters/01-rag/chapter/) · [Interview questions](interview.md) · [How the numbers were measured](../../PREFLIGHT.md)
