<div dir="rtl" lang="he" align="right" markdown="1">

# הנדסת AI בעברית

[![tests](https://github.com/itayco2/AI-Engineering-in-Hebrew/actions/workflows/tests.yml/badge.svg)](https://github.com/itayco2/AI-Engineering-in-Hebrew/actions/workflows/tests.yml)
[![notebooks](https://github.com/itayco2/AI-Engineering-in-Hebrew/actions/workflows/notebooks.yml/badge.svg)](https://github.com/itayco2/AI-Engineering-in-Hebrew/actions/workflows/notebooks.yml)
[![docs](https://github.com/itayco2/AI-Engineering-in-Hebrew/actions/workflows/docs.yml/badge.svg)](https://itayco2.github.io/AI-Engineering-in-Hebrew/)
[![license](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)

קורס שרץ בהנדסת `AI` מודרנית, בעברית. מורידים, מריצים, ורואים את המספרים נוצרים על המסך.

</div>

<p align="center">
  <img src="docs/assets/staircase.png" width="620" alt="recall@1 climbing across four retrieval steps">
</p>

<div dir="rtl" lang="he" align="right" markdown="1">

*כל ארבעת המספרים האלה מחושבים בפרק הראשון, על 46 מסמכים ו-28 שאלות, במחשב שלך.*

**מצב הקורס:** שני פרקים כתובים ורצים. שלושה מתוכננים, והם ייכתבו אחד אחרי השני. לא תמצאו כאן תיקיות ריקות שמתחזות לפרקים.

## למה זה קיים

יש חומר מצוין על `RAG`, על `agents` ועל `evals` — כמעט כולו באנגלית. בעברית יש ספר אחד טוב על למידה עמוקה קלאסית, ואין כלום על הערימה של 2026. הקורס הזה ממלא את החלק החסר.

## מה מיוחד כאן

**המספרים מחושבים, לא מצוטטים.** פרק שמשחזר תוצאה על קורפוס קטן שווה יותר מעשרה שמצטטים אותה ממאמר.

**מה שהיה שבור מתועד.** יש כאן קובץ שמפרט כל תקלה בדרך, עם המספר שחשף אותה והמספר אחרי התיקון. אחת מהן: ה-`reranker` החזיר `NaN` בשקט, לא זרק שגיאה, ופשוט לא עשה כלום — ומדריך שמראה רק את הגרסה שעבדה מלמד חצי מהעבודה.

**הסבר בעברית, קוד באנגלית.** כל מונח מקצועי נשאר באנגלית בדיוק כמו שמהנדסים בישראל כותבים אותו, כדי שאפשר יהיה לחפש אותו בגוגל חמש דקות אחר כך. יש מילון שמחייב את כל הפרקים, והבדיקות נכשלות אם פרק ממציא מונח חדש.

**הכל נבדק.** כל פונקציה במנוע נבדקת בפחות משנייה בלי שום מודל, וכל פרק מורץ מקצה לקצה בכל `push` — כדי שפרק לא יירקב בשקט.

## להתחיל

</div>

```bash
git clone https://github.com/itayco2/AI-Engineering-in-Hebrew
cd AI-Engineering-in-Hebrew
make setup && make run-01
```

<div dir="rtl" lang="he" align="right" markdown="1">

הפקודה `make setup` משתמשת ב-`uv` אם הוא מותקן, ונופלת חזרה ל-`venv` ול-`pip` אם לא. כשמשהו לא עובד, `make gate` היא הפקודה שבודקת הכל בפחות מדקה ובדרך כלל אומרת מה הבעיה.

## הפרקים

| פרק | מה בונים | צריך מודל שפה |
|---|---|---|
| [**01 — RAG מאפס**](chapters/01-rag/) | `chunking`, `embeddings`, `BM25`, `RRF`, `reranker` — ומדידה של כל שלב | לא |
| [**02 — RAG בעברית**](chapters/02-hebrew-rag/) | מורפולוגיה עברית, עלות `tokens`, ואיזה `embedding` באמת עובד | לא |
| 03 — איך יודעים שזה עובד | `recall`, `precision`, `nDCG`, ואיפה `LLM-as-judge` משקר | כן |
| 04 — הנדסת הקשר | למה לשמור הכל מנצח סיכום, ומה ה-`cache` עושה לזה | כן |
| 05 — סוכנים | הלולאה, `tool schemas`, ולמה מודלים קטנים שוברים `JSON` | כן |

## כמה זה עולה

שום דבר. שלושת הפרקים הראשונים לא קוראים לאף מודל שפה — רק מודל `embedding` קטן שרץ על המעבד. אין `API key`, אין `GPU`, ואין דרישת זיכרון חריגה.

הפרקים שכן צריכים מודל שפה עובדים גם בלעדיו, מתוך תשובות אמיתיות שהוקלטו מראש ונשמרות בתוך הקורס. מי שרוצה להריץ מול מודל אמיתי יכול, עם `Ollama` או עם `llama-cpp-python`, ושניהם חינמיים ומקומיים.

## קישורים

[הקורס באתר](https://itayco2.github.io/AI-Engineering-in-Hebrew/) · [מילון המונחים](TERMS.md) · [איך נמדדו המספרים](PREFLIGHT.md) · [לתרום](CONTRIBUTING.md) · [English](README.en.md)

</div>
