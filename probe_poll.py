#!/usr/bin/env python
"""Poll the 1-episode probe runs, collect per-episode detail, build a comparison."""
from __future__ import annotations
import json, subprocess, time
from pathlib import Path
import httpx

ROOT = Path(__file__).parent
RES = ROOT / "results"
EXP = RES / "exports"; EXP.mkdir(parents=True, exist_ok=True)
BASE = "https://api.swecc.org/bench"
DONE = {"completed"}; DEAD = {"failed", "error", "cancelled"}

runs = []
for line in (RES / "probe_runs.txt").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("ERR"):
        rid, model = line.split(maxsplit=1)
        runs.append({"id": rid, "model": model, "status": "running", "doc": None})

def token():
    return subprocess.run(["mesocosm","auth","token"],capture_output=True,text=True).stdout.strip()

for cycle in range(120):
    tok = token()
    with httpx.Client(base_url=BASE, headers={"Authorization":f"Bearer {tok}"}, timeout=120.0) as c:
        for r in runs:
            if r["status"] in DONE | DEAD:
                continue
            try:
                resp = c.get(f"/v1/runs/{r['id']}/export")
                if resp.status_code >= 400:
                    continue
                doc = resp.json()
                st = (doc.get("run") or {}).get("status")
                if st in DONE:
                    r["status"] = "completed"; r["doc"] = doc
                    (EXP / f"{r['id']}.json").write_text(json.dumps(doc), encoding="utf-8")
                elif st in DEAD:
                    r["status"] = st; r["doc"] = doc
            except Exception as e:
                print("poll err", r["model"], e, flush=True)
    pending = [r for r in runs if r["status"] not in DONE | DEAD]
    print(f"[cycle {cycle}] done={sum(1 for r in runs if r['status'] in DONE)} "
          f"dead={sum(1 for r in runs if r['status'] in DEAD)} pending={len(pending)}", flush=True)
    if not pending:
        break
    time.sleep(10)

# ---- collect per-episode detail ----
def first_ep(doc):
    rep = doc.get("replay") or {}
    return next(iter(rep.values()), [])

rows = []
for r in runs:
    if r["status"] != "completed" or not r["doc"]:
        rows.append({"model": r["model"], "status": r["status"]})
        continue
    steps = first_ep(r["doc"])
    fin = (steps[-1].get("info") if steps else {}) or {}
    run = r["doc"].get("run") or {}
    obs0 = (steps[0].get("observation") if steps else {}) or {}
    last_action = str(steps[-1].get("action","")) if steps else ""
    rows.append({
        "model": r["model"], "status": "completed",
        "clue": obs0.get("clue"), "enumeration": obs0.get("enumeration"),
        "solved": fin.get("solved"), "guesses": fin.get("guesses_used"),
        "hints": fin.get("hints_used"), "hint_order": fin.get("hint_order"),
        "par": fin.get("par"), "par_result": fin.get("par_result"),
        "reward": fin.get("reward"), "final_action": last_action[:90],
        "scores": run.get("scores"),
    })

(RES / "probe_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

print("\n=== 1-EPISODE PROBE RESULTS (clue #0) ===")
print(f"{'model':40} {'solved':6} {'hints':5} {'par':3} {'vs par':7} {'reward':6}")
for r in sorted(rows, key=lambda x: -float(x.get('reward') or 0)):
    if r["status"] != "completed":
        print(f"{r['model']:40} {r['status']}"); continue
    print(f"{r['model']:40} {r['solved']:>6} {r['hints']:>5} {r['par']:>3} "
          f"{r['par_result']:>7} {r['reward']:>6}")
clue = next((r.get("clue") for r in rows if r.get("clue")), None)
print("\nclue tested:", clue, next((r.get("enumeration") for r in rows if r.get("enumeration")), ""))
