#!/usr/bin/env python
"""Consolidate scraped run episodes into dashboard-ready JSON.

Pulls every episode we have on disk (per-seed sweep files + the probe exports),
joins each clue to its true answer / word structure from the mc_data dataset,
and emits showcase/dashboard_data.json for the Minute-Cryptic-style replay UI.
"""
from __future__ import annotations
import json, re, glob
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
OUT = ROOT / "showcase" / "dashboard_data.json"
DATASET = ROOT / "mc_data" / "minute_cryptic_dataset.jsonl"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


# clue -> answer / word structure
CLUES = {}
with DATASET.open(encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        CLUES[norm(d["clue"])] = {
            "answer": d["answer"], "wordLengths": d.get("wordLengths") or [d["answerLength"]],
            "par": d.get("par"), "definition": d.get("definition"),
        }


def step_view(step: dict) -> dict:
    obs = step.get("observation") or {}
    info = step.get("info") or {}
    action = str(step.get("action", ""))
    decision = info.get("decision") or ("hint" if info.get("outcome") == "hint" else "guess")
    reasoning = info.get("reasoning") or ""
    guess = info.get("guess") or ""
    if not reasoning and action:           # probe episodes (pre-structured): use raw reply
        reasoning = action[:400]
    return {
        "decision": decision,
        "guess": guess,
        "reasoning": reasoning,
        "outcome": info.get("outcome", ""),
        "hintsUsed": int(info.get("hints_used", 0) or 0),
        "letters": obs.get("revealed_letters") or "",
        "hints": obs.get("revealed_hints") or [],
        "wrong": obs.get("wrong_guesses") or [],
        "feedback": obs.get("feedback") or "",
    }


def make_episode(model: str, steps: list) -> dict | None:
    if not steps:
        return None
    o0 = steps[0].get("observation") or {}
    clue = o0.get("clue")
    if not clue:
        return None
    meta = CLUES.get(norm(clue), {})
    fin = steps[-1].get("info") or {}
    try:
        hint_order = json.loads(fin.get("hint_order") or "[]")
    except (ValueError, TypeError):
        hint_order = []
    return {
        "model": model,
        "clue": clue,
        "enumeration": o0.get("enumeration") or "",
        "answer": meta.get("answer") or fin.get("guess") or "",
        "wordLengths": o0.get("word_lengths") or meta.get("wordLengths") or [],
        "definition": meta.get("definition"),
        "par": int(fin.get("par", meta.get("par") or 0) or 0),
        "solved": fin.get("solved") == "1",
        "hintsUsed": int(fin.get("hints_used", 0) or 0),
        "guessesUsed": int(fin.get("guesses_used", 0) or 0),
        "parResult": fin.get("par_result", ""),
        "reward": float(fin.get("reward", 0) or 0),
        "hintOrder": hint_order,
        "steps": [step_view(s) for s in steps],
    }


episodes = []
seen = set()  # (model, clue) dedupe; prefer per-seed sweep files (loaded first)

# 1) per-seed sweep episodes (richest: structured reasoning/decision)
for f in glob.glob(str(RESULTS / "episodes" / "*" / "*.json")):
    d = json.loads(Path(f).read_text(encoding="utf-8"))
    ep = make_episode(d["model"], d.get("steps") or [])
    if ep:
        key = (ep["model"], norm(ep["clue"]))
        if key not in seen:
            seen.add(key); episodes.append(ep)

# 2) probe exports (clue #0 across several models) for breadth
for f in glob.glob(str(RESULTS / "exports" / "*.json")):
    doc = json.loads(Path(f).read_text(encoding="utf-8"))
    model = (((doc.get("run") or {}).get("config") or {}).get("agent_config") or {}).get("model") \
        or doc.get("model") or "unknown"
    for steps in (doc.get("replay") or {}).values():
        ep = make_episode(model, steps)
        if ep:
            key = (ep["model"], norm(ep["clue"]))
            if key not in seen:
                seen.add(key); episodes.append(ep)

episodes.sort(key=lambda e: (e["model"], e["clue"]))
OUT.write_text(json.dumps({"episodes": episodes}, indent=2), encoding="utf-8")

from collections import Counter
by_model = Counter(e["model"] for e in episodes)
print(f"wrote {OUT} with {len(episodes)} episodes")
for m, n in by_model.most_common():
    solved = sum(1 for e in episodes if e["model"] == m and e["solved"])
    print(f"  {m}: {n} episodes, {solved} solved")
