# TERMS — the terminology spine

Every Hebrew chapter in this repo uses the forms in the table below. Nothing else. If a term
is missing, add it here **first**, then use it. `tests/test_terms.py` fails the build when a
notebook uses a Hebrew technical term that this file does not define.

This file exists because there is no settled Hebrew register for modern AI engineering, and
without a written policy the same concept acquires three names across three chapters — which
is the fastest way for Hebrew technical writing to look amateur.

## The policy — three rules, in order

**1. English technical terms stay English, in backticks.**
Hebrew grammar wraps around them: `ה-embedding של כל chunk`, `ה-reranker מסדר מחדש`. This is
how Israeli engineers actually write and speak. It keeps terms searchable, it lets a reader
move straight to the English documentation, and — not a small thing — each backticked English
run becomes an LTR island that the RTL stylesheet can isolate, so it *helps* the bidi problem
instead of adding to it.

**2. Words that are already ordinary Hebrew stay Hebrew.**
`מודל`, `וקטור`, `מסמך`, `שאילתה`, `אימון`. Some of these arrived as loanwords decades ago and
are now simply Hebrew words. Writing `model` instead of `מודל` reads as affectation.

**3. Never coin, never transliterate.**
Do not invent a Hebrew equivalent, and do not spell an English term phonetically in Hebrew
letters. `אמבדינג` is not a word; `embedding` is. The only Hebrew forms permitted are ones a
working Israeli engineer already uses without thinking.

### Why rule 3 matters more than it looks

The Academy of the Hebrew Language's term database returns **seven** Hebrew equivalents for
"embedding" — שִׁעְבּוּד (linguistics, 2022), שִׁבּוּץ (communications, 2012), שִׁקּוּעַ
(histology/dentistry), תִּיכוּךְ (microbiology), שִׁכּוּן (chemical engineering), כְּלִיאָה
(microelectronics) — **and not one of them is the machine-learning sense.** The 2022 linguistics
entry means clause subordination. Reaching for an authority here means citing a dictionary while
writing something no Israeli AI engineer recognises.

The audience reads English professionally. Hebrew buys lower activation energy and identity, not
comprehension. So **the Hebrew does the explaining — analogy, intuition, why it breaks — and
never does the terminology.** Fighting for a pure Hebrew term spends the budget in the one place
it returns nothing.

## Stays English — in backticks, always

| English | Used as | Rejected | Why rejected |
|---|---|---|---|
| embedding | `embedding` | אמבדינג · שיכון · שיבוץ | Transliteration is not a word; Academy terms are all other fields (see above) |
| embed (verb) | `ה-embedding של` / `לחשב embedding` | לשכן | Coinage; reads as housing policy |
| chunk | `chunk` | צ'אנק · מקטע | Transliteration; מקטע is ambiguous with segment |
| chunking | `chunking` | חלוקה למקטעים | Accurate but nobody says it |
| token | `token` | טוקן · אסימון | אסימון means a physical token/coin |
| tokenizer | `tokenizer` | מפרק · מאסימן | Coinage |
| prompt | `prompt` | פרומפט · הנחיה | הנחיה loses the technical sense |
| retrieval | `retrieval` | אחזור | אחזור is correct Hebrew and acceptable in flowing prose, but the metric names and code are English — keep one form |
| retriever | `retriever` | מאחזר | Coinage |
| reranker | `reranker` | מדרג מחדש | Clumsy; the tool has an English name |
| rerank | `rerank` | דירוג מחדש | See above |
| recall | `recall` | היזכרות · כיסוי | It is a metric name; never translate a metric |
| precision | `precision` | דיוק | דיוק collides with accuracy |
| recall@k | `recall@k` | — | Metric name, never translated |
| nDCG · MRR | `nDCG` · `MRR` | — | Metric names |
| BM25 | `BM25` | — | Algorithm name |
| RRF / reciprocal rank fusion | `RRF` | מיזוג דירוגים | Algorithm name |
| hybrid search | `hybrid search` | חיפוש היברידי | Acceptable, but keep one form |
| dense / sparse retrieval | `dense` / `sparse retrieval` | צפוף · דליל | Literal translation is confusing here |
| context window | `context window` | חלון הקשר | חלון הקשר is real Hebrew (Wikipedia uses it) but engineers say the English; see note below |
| context engineering | `context engineering` | הנדסת הקשר | Used in the chapter *title* only, as a title; the English inline in prose |
| fine-tuning | `fine-tuning` | כוונון עדין | Same as context window — real Hebrew, but not spoken |
| inference | `inference` | הסקה · אינפרנס | הסקה is logic; transliteration is not a word |
| agent | `agent` | סוכן | סוכן alone is ambiguous (a sales agent). `agent` in prose; `סוכני AI` allowed in a title |
| workflow | `workflow` | תזרים עבודה | Nobody says it |
| tool call | `tool call` | קריאה לכלי | Acceptable; keep English for consistency with the code |
| tool schema | `tool schema` | סכמת כלי | Half-transliterated already; keep English |
| eval / evals | `eval` / `evals` | הערכה | הערכה is fine in flowing prose about the *activity*; `evals` for the artifacts |
| LLM-as-judge | `LLM-as-judge` | שופט | Needs the full term to be unambiguous |
| groundedness | `groundedness` | עיגון | Coinage |
| faithfulness | `faithfulness` | נאמנות | נאמנות is loyalty; wrong sense |
| hallucination | `hallucination` | הזיה | הזיה is actually good Hebrew and widely used — **permitted in prose**, `hallucination` in headings |
| cache · prefix cache | `cache` · `prefix cache` | מטמון | מטמון is correct but rare in speech |
| latency | `latency` | השהיה · זמן תגובה | זמן תגובה permitted in prose |
| TTFT | `TTFT` | — | Acronym |
| throughput | `throughput` | תפוקה | תפוקה permitted in prose |
| cassette | `cassette` | קלטת | Internal term of this repo; keep English |
| notebook | `notebook` | מחברת | מחברת is good Hebrew — **permitted** |
| repository / repo | `repo` | מאגר | מאגר permitted for a data store, `repo` for git |

## Stays Hebrew — these are ordinary Hebrew words

| English | Used as | Note |
|---|---|---|
| model | מודל | Fully absorbed |
| vector | וקטור | Fully absorbed |
| index | אינדקס | Fully absorbed |
| corpus | קורפוס | Fully absorbed |
| document | מסמך | Ordinary Hebrew |
| query | שאילתה | Standard in Israeli database/search usage |
| question | שאלה | Ordinary Hebrew |
| answer | תשובה | Ordinary Hebrew |
| training | אימון | Ordinary Hebrew |
| neural network | רשת נוירונים | Standard |
| similarity | דמיון | Standard |
| distance | מרחק | Ordinary Hebrew |
| score | ניקוד | Ordinary Hebrew. Use `score` only when naming a variable |
| tool (generic) | כלי | Ordinary Hebrew. `tool call` stays English |
| sentence | משפט | Ordinary Hebrew |
| word | מילה | Ordinary Hebrew |
| meaning | משמעות | Ordinary Hebrew |
| memory | זיכרון | Ordinary Hebrew |
| cost | עלות | Ordinary Hebrew |
| speed | מהירות | Ordinary Hebrew |
| laptop | לפטופ | Absorbed. מחשב נייד also fine |

## Chapter titles — the one place Hebrew leads

Titles carry the Hebrew, because a title is read as a name rather than as terminology:

| Chapter | Title |
|---|---|
| 01 | `RAG מאפס` |
| 02 | `RAG בעברית` |
| 03 | `איך יודעים שזה עובד` |
| 04 | `הנדסת הקשר` |
| 05 | `סוכנים: לולאה, כלים, וג'ייסון שבור` |

## The "real Hebrew but nobody says it" set

`חלון הקשר`, `כוונון עדין`, `אחזור`, `דירוג`. All four are correct, all four appear in Hebrew
Wikipedia, and all four are avoided in this repo's prose in favour of the English — because the
reader is an engineer who will type the English into a search box five minutes later. They are
listed here so the decision is visible and deliberate rather than looking like ignorance.

## Adding a term

1. Add the row here, including what you rejected and why. The rejected column is the valuable one.
2. Prefer rule 1 (English in backticks) unless the Hebrew is genuinely unremarkable speech.
3. If you cannot decide, it is rule 1.
