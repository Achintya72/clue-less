# clue-less — a MinuteCryptic benchmark for LLMs

A [Mesocosm / BenchAnything](https://wiki.swecc.org/Sweccathon/START_HERE) environment
that measures how well language models solve British-style **cryptic crossword clues**
([minutecryptic.com](https://www.minutecryptic.com/)). Built for Sweccathon 2026 (Games track).

Each episode is one **real Minute Cryptic daily clue** (scraped into
`mc_data/`, 813 puzzles). The game loop mirrors the site:

1. **Every** observation carries the rules of cryptic crosswords plus a cheat-sheet of
   common cryptic substitutions (`love → O`, `the French → LE/LA`, Roman numerals, …).
2. The first observation gives only the clue and the answer's length.
3. Each turn the agent either **guesses** or replies **`HINT`**:
   - A wrong guess is non-terminal; the next observation lists every wrong guess so far.
   - `HINT` walks a **persistent, escalating ladder**: the official progressive hints
     first (definition → wordplay → …), then the answer's letters revealed **one at a
     time** in the site's official order. Enough letter hints reveal the whole word, so
     the agent can never get stuck.
   - The same wrong answer **3×** ends the episode (it will never get it); a `HINT`
     once everything is revealed ends it as a solve at max hints.
4. Scoring is **golf-style** against the clue's **par** (expected number of hints). We
   track total hints, the **order** they were taken in, and whether the agent finished
   **below / at / above par**. Reward is paid once, on the terminal step.

## Layout

```
benchanything.json   # domain manifest: binding vow + scoring (6 metrics)
env.py               # MinuteCrypticEnv — reset()/step() game loop
adapter.py           # HTTP server wrapping the env (port 8765)
test_env.py          # pytest suite locking in the loop + reward contract
mc_data/             # the real dataset + scraper/assembler (see mc_data/README.md)
  minute_cryptic_dataset.jsonl   # 813 clues, one JSON object per line (the env's source)
showcase/            # showcase UIs
  index.html         # ★ landing: Minute-Cryptic-style replay dashboard + per-model report cards
  leaderboard.html   # aggregate leaderboard table (reads Mesocosm `run export` JSON)
  dashboard_data.json# built by build_dashboard_data.py from results/episodes/*
  make_samples.py    # writes export-schema sample data from the real env
  data/*.json        # per-model run exports (real + reference policies)
```

## Clue dataset (`mc_data/minute_cryptic_dataset.jsonl`)

Each line is one puzzle. The env uses `clue`, `answer`, `wordLengths`, `enumeration`,
`par`, `communityAveragePar`, `hints` (ordered `{type, text}`), and `letterRevealOrder`
(the official letter-by-letter reveal order). Full schema and provenance are documented
in [`mc_data/README.md`](mc_data/README.md).

## Scoring

| Metric | How |
|---|---|
| `score` (primary) | mean episode reward (1.0 solved no-hints; −0.2 per hint, floor 0.2; 0.3×similarity partial credit) |
| `solve_rate` | fraction of clues solved |
| `avg_hints` | mean hints taken (explanatory + letter reveals) |
| `avg_guesses` | mean answer attempts |
| `avg_hints_to_par` | mean (hints − par); negative beats the human baseline |
| `under_par_rate` | fraction finishing at or under par |

## Develop locally

```bash
pip install swecc-mesocosm
python -m pytest -q                      # 17 tests

# real model run (needs Ollama)
ollama pull llama3.2
python adapter.py                        # terminal 1 — http://localhost:8765/health
mesocosm run local --episodes 5          # terminal 2
```

## Showcase UI

```bash
cd showcase && python -m http.server 8088
# open http://localhost:8088/            -> replay dashboard (index.html, the star)
# open http://localhost:8088/leaderboard.html  -> aggregate leaderboard table
```

The two views are cross-linked. The **replay dashboard** reads
`showcase/dashboard_data.json` (see below). The **leaderboard** reads the
per-model `run export` files in `showcase/data/` (add real runs with
`mesocosm run export RUN_ID -o showcase/data/<model>.json` + an entry in
`showcase/data/index.json`, or the "Load export JSON…" button). Generate sample
data with `python showcase/make_samples.py`.

## Submit to the platform

```bash
mesocosm auth login
mesocosm env submit --name "clue-less" --github-url https://github.com/Achintya72/clue-less
mesocosm run create --domain DOMAIN_ID --vow-version 1.0.0 --model gemini/gemini-3.1-flash-lite --episodes 5
```

## Benchmark real models across the dataset

`mesocosm run create` runs cloud models on SWECC's infrastructure (it uses your
session — you don't supply your own API keys). The CLI only runs the first ≤20
clues, so to cover the whole dataset `orchestrate_bench.py` drives the API
directly with explicit seed batches and tracks coverage **per clue**, automatically
re-running any episode that hits a rate limit.

```bash
mesocosm auth login                  # member session (only a member can read results)
python harvest_sweep.py              # resilient driver — recommended (see note)
# or: python orchestrate_bench.py    # full-813 driver
```

Output lands in `results/episodes/<model>/<seed>.json` (resumable — both drivers
skip/re-queue seeds). Edit `MODELS` / `TARGET` / `BATCH` at the top to change scope.

> **Use `harvest_sweep.py`.** The platform sometimes wedges an episode in
> `running` forever, so a batch never goes fully terminal — `orchestrate_bench.py`
> only harvests on a *complete* batch and will hang on that straggler.
> `harvest_sweep.py` instead saves each completed episode immediately and
> **times out a stuck run, re-queuing its seeds into a fresh run** (fresh small
> runs drain fast). Then rebuild the dashboard with `python build_dashboard_data.py`.

**Platform limits worth knowing:**
- Each run is capped at **20 episodes**; the orchestrator batches around it.
- Provider keys are **shared and rate-limited** — keep `RUN_PARALLEL` low; failed
  episodes are retried up to `MAX_ATTEMPTS`.
- Only models with a configured key run: OpenAI `gpt-4o` and the Gemini
  `flash-lite` family (Anthropic / DeepSeek / xAI have no platform key).
- **Member sessions expire (~25 min) with no refresh**, and only a member can read
  results (a guest can create runs but gets 403 on export). If a long sweep
  stalls, re-run `mesocosm auth login` and the orchestrator resumes.

## Model replay dashboard

A Minute-Cryptic-style replay of real model plays — the clue, an answer grid that
fills in as the model takes hints, its guesses (green hit / red miss), the
reasoning behind each step, and a per-model **report card**: average hints vs par
(below = negative, at = 0, above = positive), solve rate, and how often it reaches
for each hint type (definition / indicators / fodder / letter).

```bash
python build_dashboard_data.py       # results/episodes/* -> showcase/dashboard_data.json
cd showcase && python -m http.server 8088
# open http://localhost:8088/         (it's the landing page, index.html)
```

Re-run `build_dashboard_data.py` after any new runs and refresh — the dashboard
picks up the new episodes and recomputes the per-model report cards automatically.
