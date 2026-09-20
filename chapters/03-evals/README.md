<div dir="rtl" lang="he" align="right" markdown="1">

# פרק 03: איך יודעים שזה עובד

מערכת `RAG` היא שני חצאים, וצריך למדוד אותם בנפרד. החצי של האחזור נמדד בחשבון מדויק בלי שום מודל. החצי של התשובה קשה יותר, וכאן נכנס `LLM-as-judge`, ונכשל.

12 שאלות, שתי תשובות לכל אחת: אחת שמכילה את העובדה ואחת שלא. מה שנכון ידוע מראש, אז אפשר למדוד את השופט עצמו.

התוצאה: השופט ההוליסטי הפריד נכון בין התשובה הטובה לגרועה ב-6 מתוך 12, בדיוק הטלת מטבע.

</div>

```bash
make run-03
```

[Read the chapter](https://itayco2.github.io/AI-Engineering-in-Hebrew/chapters/03-evals/chapter/) · [Interview questions](interview.md) · [How the numbers were measured](../../FINDINGS.md)
