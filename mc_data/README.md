# Minute Cryptic Benchmark Dataset

Daily cryptic-clue puzzles scraped from [minutecryptic.com](https://www.minutecryptic.com)
for use as an AI model benchmark.

## Files
- `minute_cryptic_dataset.json` — array of puzzle objects (pretty-printed).
- `minute_cryptic_dataset.jsonl` — same data, one puzzle per line (for streaming eval).
- `raw_json/<date>.json` — untouched per-puzzle API responses (source of truth).
- `dates_par.txt` — every puzzle date + its rounded community-average par.
- `scrape.sh` / `assemble.js` — the scraper and the dataset builder (re-runnable).

## Coverage
- **813 puzzles**, dated **2023-09-07 → 2026-05-29**.
- 2 archive dates were not included:
  - `2024-03-16` — no playable puzzle rendered on the site (genuine gap).
  - `2024-06-25` — the "TOMORROW" launch-teaser entry (not a cryptic clue; has no hints). Excluded.

## Schema (per puzzle)
| field | description |
|---|---|
| `date` | puzzle date (YYYY-MM-DD) |
| `puzzleId` | site UUID |
| `clue` | full clue text |
| `enumeration` | answer letter pattern, e.g. `(7)`, `(3,4)` |
| `answer` | solution, UPPERCASE (spaces kept for multi-word) |
| `answerLength` | letter count (excludes spaces/punctuation) |
| `wordLengths` | per-word letter counts, e.g. `[3,4]` |
| `par` | **official fixed par** for the clue (the in-game scoring target) |
| `communityAveragePar` | rounded average hints used across all solvers |
| `numHints` | number of progressive hints available |
| `hints[]` | ordered hints: `{order, type, text, colour, highlighting}` |
| `definition` | the definition portion of the clue, when the site tags it |
| `wordplay` | the wordplay portion, when tagged |
| `clueSegments[]` | the clue split into `{text, type}` segments |
| `setterName` | clue setter, when credited |
| `explainerVideo` | YouTube breakdown URL, when present |
| `solveCount` | total solves (recent puzzles only; null for older) |

### Notes on `par`
The game scores like golf: lower is better, 0 hints is ideal. There are two par values:
- `par` — the **official** par the setter/site assigned (use this as the difficulty target).
- `communityAveragePar` — how the crowd actually did (rounded). These often differ.

### Hint types
Hints reveal the clue's mechanics progressively. Observed `type` values include
`definition`, `indicators`, `fodder`. `highlighting` gives `[start,end)` character
ranges into `clue` that the hint refers to.

## Suggested benchmark usage
- Primary task: given `clue` + `enumeration`, predict `answer` (exact match, uppercase).
- Score vs `par`: count hints the model "needs" if you feed hints progressively until it solves.
- Stratify difficulty by `par` (1=easiest … 5=hardest) and by `numHints`.

## Provenance / terms
Scraped from a logged-in subscriber session. minutecryptic.com's content is
copyrighted and its robots.txt declares content-signal preferences around AI use.
Fine for internal evaluation; review their Terms before redistributing the dataset.
