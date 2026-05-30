#!/usr/bin/env python
"""Orchestrate full 813-clue Minute Cryptic runs across the runnable models.

The platform caps each run at 20 episodes and the CLI can't pick clues, so we
POST runs directly with explicit seed_set batches (seed i -> clue i): 41 batches
per model covering 0..812. Designed around the failure modes we hit:

- create POSTs occasionally time out -> long timeout + adopt any orphan run
  (matched by model + seed_set) instead of creating a duplicate.
- Gemini rate-limits under load -> low concurrency, retries with a cooldown.
- get/episodes 403 -> read status/scores via the export endpoint.

Resumable: state lives in results/state.json. Re-run to continue.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
STATE_FILE = RESULTS / "state.json"
SHOWCASE = ROOT / "showcase" / "data"

BASE = "https://api.swecc.org/bench"
DOMAIN = "a42c98f2-8d77-432e-aee5-e89993bdd726"
VOW = "1.0.0"
# Only the models with working platform keys (see access findings).
MODELS = [
    "openai/gpt-4o",
    "gemini/gemini-3.1-flash-lite",
    "gemini/gemini-3.1-flash-lite-preview",
    "gemini/gemini-2.5-flash-lite",
    "gemini/gemini-flash-lite-latest",
]
N_CLUES = 813
BATCH = 20
MAX_INFLIGHT = 5            # gentle: keeps concurrent Gemini calls under the rate limit
RUN_PARALLEL = 4            # episode concurrency within a run
MAX_TOKENS = 800           # room for REASONING + ACTION without truncating before the action
RETRIES = 4                 # per-batch retries (rate-limit failures need a few)
COOLDOWN = 2                # cycles to wait before retrying a failed batch
CREATE_TIMEOUT = 180.0
POLL_SECS = 15

DONE = {"completed"}
DEAD = {"failed", "error", "cancelled"}


def slug(model: str) -> str:
    return model.replace("/", "-").replace(".", "-")


def batches() -> list[list[int]]:
    return [list(range(i, min(i + BATCH, N_CLUES))) for i in range(0, N_CLUES, BATCH)]


def get_token() -> str:
    out = subprocess.run(["mesocosm", "auth", "token"], capture_output=True, text=True)
    return out.stdout.strip()


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {m: [{"seeds": b, "run_id": None, "status": "todo", "scores": None,
                 "n": len(b), "tries": 0, "cooldown": 0} for b in batches()]
            for m in MODELS}


def save_state(st: dict) -> None:
    STATE_FILE.write_text(json.dumps(st, indent=2), encoding="utf-8")


def adopted_ids(st: dict) -> set[str]:
    return {b["run_id"] for m in st for b in st[m] if b["run_id"]}


def reconcile_orphan(client, model, seeds, taken) -> str | None:
    """If a create timed out client-side, the run may exist server-side. Find it
    by matching model + seed_set among recent runs (not already adopted)."""
    try:
        r = client.get("/v1/runs")
        if r.status_code >= 400:
            return None
        for run in r.json():
            cfg = run.get("config") or {}
            if (cfg.get("agent_config", {}).get("model") == model
                    and cfg.get("seed_set") == seeds
                    and run["id"] not in taken
                    and run.get("status") not in DEAD):
                return run["id"]
    except Exception:
        return None
    return None


def create_run(client, model, seeds) -> str | None:
    payload = {
        "domain_id": DOMAIN, "binding_vow_version": VOW,
        "agent_config": {"model": model, "temperature": 0.0, "max_tokens": MAX_TOKENS},
        "num_episodes": len(seeds), "seed_set": seeds, "max_parallel": RUN_PARALLEL,
    }
    try:
        r = client.post("/v1/runs", json=payload, timeout=CREATE_TIMEOUT)
        if r.status_code >= 400:
            print(f"  create {r.status_code}: {r.text[:100]}", flush=True)
            return None
        return r.json()["id"]
    except Exception as e:
        print(f"  create timeout/err ({model} {seeds[0]}-{seeds[-1]}): {e}", flush=True)
        return None


def fetch_export(client, run_id) -> dict | None:
    try:
        r = client.get(f"/v1/runs/{run_id}/export")
        return r.json() if r.status_code < 400 else None
    except Exception:
        return None


def inflight(st) -> int:
    return sum(1 for m in st for b in st[m] if b["status"] == "running")


def main() -> None:
    for cycle in range(100000):
        token = get_token()
        with httpx.Client(base_url=BASE, headers={"Authorization": f"Bearer {token}"},
                          timeout=120.0) as client:
            st = load_state()

            # 1) poll running runs
            for m in MODELS:
                for b in st[m]:
                    if b["status"] == "running" and b["run_id"]:
                        doc = fetch_export(client, b["run_id"])
                        if not doc:
                            continue
                        rstatus = (doc.get("run") or {}).get("status")
                        if rstatus in DONE:
                            b["status"], b["scores"] = "completed", (doc.get("run") or {}).get("scores") or {}
                        elif rstatus in DEAD:
                            b["status"], b["cooldown"] = "dead", COOLDOWN

            # 2) tick cooldowns; requeue dead batches with retry budget left
            for m in MODELS:
                for b in st[m]:
                    if b["status"] == "dead":
                        if b["tries"] >= RETRIES:
                            continue
                        b["cooldown"] = max(0, b.get("cooldown", 0) - 1)
                        if b["cooldown"] == 0:
                            b["status"], b["run_id"] = "todo", None
            save_state(st)

            # 3) launch up to MAX_INFLIGHT
            slots = MAX_INFLIGHT - inflight(st)
            taken = adopted_ids(st)
            for m in MODELS:
                for b in st[m]:
                    if slots <= 0:
                        break
                    if b["status"] == "todo":
                        rid = create_run(client, m, b["seeds"]) \
                            or reconcile_orphan(client, m, b["seeds"], taken)
                        if rid:
                            b["run_id"], b["status"] = rid, "running"
                            b["tries"] += 1
                            taken.add(rid)
                            slots -= 1
                if slots <= 0:
                    break
            save_state(st)

            tot = sum(len(st[m]) for m in MODELS)
            done = sum(1 for m in st for b in st[m] if b["status"] == "completed")
            dead = sum(1 for m in st for b in st[m] if b["status"] == "dead" and b["tries"] >= RETRIES)
            print(f"[cycle {cycle}] completed {done}/{tot}  running={inflight(st)} "
                  f"todo={sum(1 for m in st for b in st[m] if b['status']=='todo')} "
                  f"giveup={dead}", flush=True)
            if done + dead >= tot:
                break
        time.sleep(POLL_SECS)

    aggregate()


def aggregate() -> None:
    st = load_state()
    SHOWCASE.mkdir(parents=True, exist_ok=True)
    summary, index = [], []
    token = get_token()
    with httpx.Client(base_url=BASE, headers={"Authorization": f"Bearer {token}"},
                      timeout=180.0) as client:
        for m in MODELS:
            metrics = ["score", "solve_rate", "avg_hints", "avg_guesses",
                       "under_par_rate", "avg_hints_to_par"]
            acc = {k: 0.0 for k in metrics}
            merged: dict = {}             # keyed by seed -> dedupes
            n_total = 0
            for b in st[m]:
                if b["status"] != "completed":
                    continue
                doc = fetch_export(client, b["run_id"])
                if not doc:
                    continue
                rep = doc.get("replay") or {}
                for ep in (doc.get("episodes") or []):
                    steps = rep.get(ep["id"]) or []
                    merged[f"{slug(m)}-seed{ep.get('seed')}"] = steps
                sc = b["scores"] or {}
                n = b["n"]
                n_total += n
                for k in metrics:
                    if sc.get(k) is not None:
                        acc[k] += sc[k] * n
            covered = len(merged)
            means = {k: (acc[k] / n_total if n_total else None) for k in metrics}
            summary.append({"model": m, "clues_scored": covered, **means})
            fn = f"{slug(m)}.json"
            (SHOWCASE / fn).write_text(json.dumps({
                "schema_version": "1", "domain_id": DOMAIN, "binding_vow_version": VOW,
                "model": m, "visibility": "gallery_public", "replay": merged,
            }, indent=2) + "\n", encoding="utf-8")
            index.append({"file": fn, "model": m})
        for fn, mm in [("perfect-solver.json", "perfect-solver"),
                       ("hint-spammer.json", "hint-spammer")]:
            if (SHOWCASE / fn).exists():
                index.append({"file": fn, "model": mm})
        (SHOWCASE / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== FINAL (mean over scored clues) ===")
    print(f"{'model':38} {'clues':>6} {'score':>7} {'solve':>7} {'hints':>7} {'under_par':>9} {'to_par':>7}")
    for s in summary:
        f = lambda x: "-" if x is None else f"{x:.3f}"
        print(f"{s['model']:38} {s['clues_scored']:>6} {f(s['score']):>7} {f(s['solve_rate']):>7} "
              f"{f(s['avg_hints']):>7} {f(s['under_par_rate']):>9} {f(s['avg_hints_to_par']):>7}")


if __name__ == "__main__":
    main()
