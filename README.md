# clue-less — a MinuteCryptic benchmark for LLMs

A [Mesocosm / BenchAnything](https://wiki.swecc.org/Sweccathon/START_HERE) environment
that measures how well language models solve British-style **cryptic crossword clues**
([minutecryptic.com](https://www.minutecryptic.com/)). Built for Sweccathon 2026 (Games track).

Each episode is one clue. The agent sees the clue text plus the answer's enumeration
(e.g. `(6)`), and replies with an answer. Wrong answers can be retried; the agent may
instead reply `HINT` to reveal a progressive hint (definition → device → first letter)
at a scoring penalty. Reward is paid once, on the terminal step.

## Layout

```
benchanything.json   # domain manifest: binding vow + scoring (4 metrics)
env.py               # MinuteCrypticEnv — reset()/step() episode logic
adapter.py           # HTTP server wrapping the env (port 8765)
test_env.py          # pytest suite locking in the reward contract
data/clues.jsonl     # the clue dataset (one JSON object per line)
showcase/            # leaderboard UI + sample-data generator
  index.html         # reads Mesocosm `run export` JSON, renders leaderboard + replays
  make_samples.py    # writes export-schema sample data from the real env
  data/*.json        # generated sample runs (4 reference policies)
```

## Clue dataset schema (`data/clues.jsonl`)

```json
{"id": "...", "clue": "Chaperone shredded corset (6)", "answer": "ESCORT",
 "accepted": [], "length": [6], "definition": "Chaperone", "device": "anagram",
 "difficulty": "easy", "human_avg_guesses": 3, "hints": ["...", "...", "..."]}
```

`accepted` (answer variants) and `hints` are optional — if `hints` is missing, the env
synthesises a definition → device → first-letter ladder. `human_avg_guesses` flows
through to the metrics so AI attempts can be compared against the human baseline.

## Scoring

| Metric | How |
|---|---|
| `score` (primary) | mean episode reward (1.0 solved no-hints; −0.2 per hint, floor 0.2; 0.3×similarity partial credit) |
| `solve_rate` | fraction of clues fully solved |
| `avg_guesses` | mean attempts (compare to `human_avg_guesses`) |
| `avg_hints` | mean hints revealed |

## Develop locally

```bash
pip install swecc-mesocosm
python -m pytest -q                      # 13 tests

# real model run (needs Ollama)
ollama pull llama3.2
python adapter.py                        # terminal 1 — http://localhost:8765/health
mesocosm run local --episodes 5          # terminal 2
```

## Leaderboard UI

```bash
python showcase/make_samples.py          # generate sample export JSON
cd showcase && python -m http.server 8088 # open http://localhost:8088/index.html
```

Point it at real data by exporting platform runs:
`mesocosm run export RUN_ID -o showcase/data/<model>.json`, then add the file to
`showcase/data/index.json` (or use the "Load export JSON…" button).

## Submit to the platform

```bash
mesocosm auth login
mesocosm env submit --name "clue-less" --github-url https://github.com/Achintya72/clue-less
mesocosm run create --domain DOMAIN_ID --vow-version 1.0.0 --model gemini/gemini-3.1-flash-lite --episodes 5
```
