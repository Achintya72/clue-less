#!/usr/bin/env python
"""Clean full-813 sweep for gpt-4o + one Gemini, resilient to rate limits.

The platform's provider keys are shared + rate-limited, so episodes fail under
load. We therefore track coverage PER SEED (clue), not per run: each run carries
up to 20 not-yet-successful seeds; when it finishes we harvest the episodes that
succeeded and re-queue the ones that hit a rate limit, until every clue has a
clean result (or a seed exhausts its retry budget).

- One run per PROVIDER in flight (gpt-4o and Gemini are different keys -> run
  in parallel), low in-run concurrency to stay under the rate limit.
- create/export via direct API (CLI can't pass seed_set; get/episodes 403).
- Resumable: results/state2.json + per-episode files under results/episodes/.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
EPISODES = RESULTS / "episodes"
STATE = RESULTS / "state2.json"
SHOWCASE = ROOT / "showcase" / "data"
for d in (RESULTS, EPISODES, SHOWCASE):
    d.mkdir(parents=True, exist_ok=True)

BASE = "https://api.swecc.org/bench"
DOMAIN = "a42c98f2-8d77-432e-aee5-e89993bdd726"
VOW = "1.0.0"
MODELS = ["openai/gpt-4o", "gemini/gemini-3.1-flash-lite"]
N_CLUES = 813
BATCH = 20
RUN_PARALLEL = 3            # in-run episode concurrency (low -> under rate limit)
MAX_TOKENS = 800
MAX_ATTEMPTS = 5           # per-seed retry budget before giving up
CREATE_TIMEOUT = 180.0
POLL_SECS = 15

DONE = {"completed"}
TERMINAL = {"completed", "failed", "error", "cancelled"}


def slug(m: str) -> str:
    return m.replace("/", "-").replace(".", "-")


def get_token() -> str:
    return subprocess.run(["mesocosm", "auth", "token"], capture_output=True, text=True).stdout.strip()


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {m: {"pending": list(range(N_CLUES)), "done": [], "giveup": [],
                "attempts": {}, "active_run": None, "active_seeds": []}
            for m in MODELS}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st), encoding="utf-8")


def create_run(client, model, seeds):
    payload = {"domain_id": DOMAIN, "binding_vow_version": VOW,
               "agent_config": {"model": model, "temperature": 0.0, "max_tokens": MAX_TOKENS},
               "num_episodes": len(seeds), "seed_set": seeds, "max_parallel": RUN_PARALLEL}
    try:
        r = client.post("/v1/runs", json=payload, timeout=CREATE_TIMEOUT)
        if r.status_code >= 400:
            print(f"  create {model} {r.status_code}: {r.text[:90]}", flush=True)
            return None
        return r.json()["id"]
    except Exception as e:
        print(f"  create err {model}: {e}", flush=True)
        return None


def export(client, rid):
    try:
        r = client.get(f"/v1/runs/{rid}/export")
        return r.json() if r.status_code < 400 else None
    except Exception:
        return None


def harvest(model, doc, ms) -> tuple[int, int]:
    """Save successful episodes, re-queue failed seeds. Returns (n_ok, n_failed)."""
    rep = doc.get("replay") or {}
    ok = fail = 0
    epdir = EPISODES / slug(model)
    epdir.mkdir(parents=True, exist_ok=True)
    for ep in (doc.get("episodes") or []):
        seed = ep.get("seed")
        if seed is None or seed not in ms["active_seeds"]:
            continue
        if ep.get("status") == "completed":
            steps = rep.get(ep["id"]) or ep.get("steps") or []
            (epdir / f"{seed}.json").write_text(
                json.dumps({"seed": seed, "model": model, "steps": steps}), encoding="utf-8")
            if seed in ms["pending"]:
                ms["pending"].remove(seed)
            if seed not in ms["done"]:
                ms["done"].append(seed)
            ok += 1
        else:  # rate-limited / errored episode -> retry later, up to the budget
            a = ms["attempts"].get(str(seed), 0) + 1
            ms["attempts"][str(seed)] = a
            if a >= MAX_ATTEMPTS:
                if seed in ms["pending"]:
                    ms["pending"].remove(seed)
                if seed not in ms["giveup"]:
                    ms["giveup"].append(seed)
            fail += 1
    return ok, fail


def main():
    cycle = 0
    while True:
        cycle += 1
        token = get_token()
        with httpx.Client(base_url=BASE, headers={"Authorization": f"Bearer {token}"}, timeout=120.0) as c:
            st = load_state()
            for model in MODELS:
                ms = st[model]
                # poll the active run; harvest when the whole run is terminal
                if ms["active_run"]:
                    doc = export(c, ms["active_run"])
                    if doc:
                        rstatus = (doc.get("run") or {}).get("status")
                        eps = doc.get("episodes") or []
                        all_term = bool(eps) and all(e.get("status") in TERMINAL for e in eps)
                        if rstatus in TERMINAL or all_term:
                            ok, fl = harvest(model, doc, ms)
                            print(f"  {slug(model)}: run done +{ok} ok / {fl} retry "
                                  f"(coverage {len(ms['done'])}/{N_CLUES})", flush=True)
                            ms["active_run"], ms["active_seeds"] = None, []
                # launch next batch for this model if idle and work remains
                if not ms["active_run"] and ms["pending"]:
                    seeds = ms["pending"][:BATCH]
                    rid = create_run(c, model, seeds)
                    if rid:
                        ms["active_run"], ms["active_seeds"] = rid, seeds
            save_state(st)

            tot_done = sum(len(st[m]["done"]) for m in MODELS)
            tot_pend = sum(len(st[m]["pending"]) for m in MODELS)
            tot_give = sum(len(st[m]["giveup"]) for m in MODELS)
            print(f"[cycle {cycle}] done={tot_done}/{2*N_CLUES} pending={tot_pend} giveup={tot_give} "
                  + " | ".join(f"{slug(m)} {len(st[m]['done'])}/{N_CLUES}" for m in MODELS), flush=True)
            if all(not st[m]["pending"] for m in MODELS):
                break
        time.sleep(POLL_SECS)
    aggregate()


def aggregate():
    st = load_state()
    summary, index = [], []
    for model in MODELS:
        epdir = EPISODES / slug(model)
        files = sorted(epdir.glob("*.json"), key=lambda p: int(p.stem)) if epdir.exists() else []
        merged = {}
        agg = {"solved": 0, "hints": 0, "guesses": 0, "reward": 0.0,
               "to_par": 0, "under": 0, "n": 0}
        for f in files:
            d = json.loads(f.read_text(encoding="utf-8"))
            steps = d["steps"]
            if not steps:
                continue
            info = steps[-1].get("info") or {}
            merged[f"{slug(model)}-seed{d['seed']}"] = steps
            agg["n"] += 1
            agg["solved"] += int(info.get("solved", 0) or 0)
            agg["hints"] += float(info.get("hints_used", 0) or 0)
            agg["guesses"] += float(info.get("guesses_used", 0) or 0)
            agg["reward"] += float(info.get("reward", 0) or 0)
            agg["to_par"] += float(info.get("hints_to_par", 0) or 0)
            agg["under"] += int(info.get("under_or_at_par", 0) or 0)
        n = agg["n"] or 1
        summary.append({
            "model": model, "clues_scored": agg["n"],
            "score": agg["reward"] / n, "solve_rate": agg["solved"] / n,
            "avg_hints": agg["hints"] / n, "avg_guesses": agg["guesses"] / n,
            "avg_hints_to_par": agg["to_par"] / n, "under_par_rate": agg["under"] / n,
        })
        fn = f"{slug(model)}.json"
        (SHOWCASE / fn).write_text(json.dumps({
            "schema_version": "1", "domain_id": DOMAIN, "binding_vow_version": VOW,
            "model": model, "visibility": "gallery_public", "replay": merged}, indent=2) + "\n",
            encoding="utf-8")
        index.append({"file": fn, "model": model})
    for fn, mm in [("perfect-solver.json", "perfect-solver"), ("hint-spammer.json", "hint-spammer")]:
        if (SHOWCASE / fn).exists():
            index.append({"file": fn, "model": mm})
    (SHOWCASE / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== FINAL ===")
    print(f"{'model':34} {'clues':>6} {'score':>7} {'solve':>7} {'hints':>7} {'under_par':>9} {'to_par':>7}")
    for s in summary:
        print(f"{s['model']:34} {s['clues_scored']:>6} {s['score']:>7.3f} {s['solve_rate']:>7.3f} "
              f"{s['avg_hints']:>7.2f} {s['under_par_rate']:>9.3f} {s['avg_hints_to_par']:>7.2f}")


if __name__ == "__main__":
    main()
