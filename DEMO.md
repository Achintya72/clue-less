# clue-less — demo & pitch

> **One-liner:** A Mesocosm/BenchAnything environment that benchmarks LLMs on **real
> Minute Cryptic daily clues**, scored golf-style against the puzzle's human **par**.

## The hook (say this first)

> "Everyone benchmarks LLMs on math and trivia. We asked a harder question: can they
> do **British cryptic crosswords** — the ones where every clue is a tiny logic puzzle
> of *definition + wordplay*? And not just 'can they', but **how do they stack up
> against the human solver baseline?**"

Why it's a good benchmark (1 sentence): cryptic clues can't be solved by retrieval —
the model has to *decompose wordplay* (anagrams, hidden words, charades, "love → O"),
hold the answer's letter-pattern, and decide when to ask for help. It's reasoning, not recall.

## What we built

- A real environment **deployed on the SWECC platform** (domain `minutecryptic`), not a
  local toy: `env.py` (game loop) + `adapter.py` (HTTP) + `benchanything.json` (rules + scoring).
- **813 real scraped clues** with official progressive hints, letter-reveal order, and par.
- A game loop that mirrors the site: each turn the model **guesses** or asks **HINT**;
  scoring is **golf vs. par** (expected # of hints) — beating par beats the human baseline.
- A **replay dashboard** (the landing page) that animates each model's play: the grid
  fills as it takes hints, green/red guesses, its reasoning per step, and a per-model
  **report card** (hints vs par, solve rate, hint-type mix) + a model **leaderboard**.

## Headline findings

| Model | clues | solve | avg hints | **vs par** | under-par |
|---|--:|--:|--:|--:|--:|
| **gpt-4o** | 40 | **100%** | 2.50 | −0.15 | 68% |
| **gemini-3.1-flash-lite** | 16* | 94% | **1.50** | **−1.19** | 94% |

\* *smaller sample — its shared key was rate-limited; topping up to 40 off-peak.*

**The story in one line:**
> "Both frontier models beat the human par — but in **opposite styles**. GPT-4o is the
> *completionist*: it solved **every** clue, but leans on hints (right around par).
> Gemini-flash-lite is the *efficient* one: it bails occasionally, but when it solves it's
> **over a full hint under par**. Capability vs. efficiency, on the same board."

## 2-minute demo script (click-by-click)

1. **Open the landing page** (the replay dashboard). "This is a real cryptic clue and the
   answer grid." Point at the clue + enumeration.
2. **Hit ▶ Play.** Narrate as it animates: "Watch it reason, take a hint — the grid reveals
   a letter — then lock in a guess. Green = solved." (Let one episode play.)
3. **Open 'Model reasoning'** for a step. "This is the model's actual wordplay reasoning —
   it's decomposing the clue, not guessing."
4. **Click a model in the leaderboard** (right column). "Switch models live — here's the
   report card: hints vs par, solve rate, and which hint types it reaches for."
5. **Contrast the two models** using the report cards → deliver the "opposite styles" line.
6. **Click 'Aggregate leaderboard ↗'** for the summary table. Close on the par result.

## Anticipated judge Q&A

- **"Isn't 16 gemini clues too few?"** Yes — that's an honest caveat. Its shared platform
  key was rate-limited; gpt-4o's wasn't. gpt-4o's 40/40 is the robust number; gemini's
  edge is on a smaller sample and we're extending it. (Mention `harvest_sweep.py` — we
  built a resilient driver that times out the platform's stuck episodes.)
- **"What's 'par'?"** Each Minute Cryptic clue ships an expected number of hints a human
  needs; we score golf-style against it. Negative = better than the human baseline.
- **"How is it scored?"** Terminal reward: 1.0 solved no-hints; −0.2 per hint (floor 0.2);
  partial credit for a near-miss; 0 if it never answers. Six tracked metrics.
- **"Could it cheat / memorize?"** Clues are obscure daily puzzles; the wordplay reasoning
  in the replay shows genuine decomposition, and letter-reveals guarantee termination
  without leaking the answer early.
- **"Is it really on the platform?"** Yes — `env submit` deployed our adapter to their
  sandbox; runs use SWECC's keys. The dashboard replays those real platform runs.

## Backup plan

- Have the dashboard pre-loaded on `http://localhost:8088/` (served from `showcase/`).
- If wifi/platform is down, the demo is fully **static** — `dashboard_data.json` is local.
- Record a 60s screen capture of a play + model switch as a fallback.
