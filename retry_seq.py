#!/usr/bin/env python
"""Sequentially retry the rate-limited Gemini variants (one at a time)."""
from __future__ import annotations
import json, subprocess, time
from pathlib import Path
import httpx

ROOT = Path(__file__).parent; RES = ROOT / "results"; EXP = RES / "exports"
BASE = "https://api.swecc.org/bench"
DOM = "a42c98f2-8d77-432e-aee5-e89993bdd726"
MODELS = ["gemini/gemini-2.5-flash", "gemini/gemini-3.5-flash", "gemini/gemini-flash-latest"]
DONE = {"completed"}; DEAD = {"failed", "error", "cancelled"}

def token():
    return subprocess.run(["mesocosm","auth","token"],capture_output=True,text=True).stdout.strip()

def create(model):
    out = subprocess.run(["mesocosm","run","create","--domain",DOM,"--vow-version","1.0.0",
                          "--model",model,"--episodes","1"],capture_output=True,text=True)
    try: return json.loads(out.stdout)["id"]
    except Exception:
        print("create failed", model, out.stdout[:120], out.stderr[:120]); return None

results = []
for model in MODELS:
    rid = create(model)
    if not rid:
        results.append({"model": model, "status": "create_failed"}); continue
    print(f"created {model} -> {rid[:8]}", flush=True)
    doc = None
    for _ in range(60):
        tok = token()
        with httpx.Client(base_url=BASE, headers={"Authorization":f"Bearer {tok}"}, timeout=120) as c:
            r = c.get(f"/v1/runs/{rid}/export")
            if r.status_code < 400:
                doc = r.json(); st = (doc.get("run") or {}).get("status")
                if st in DONE | DEAD:
                    break
        time.sleep(8)
    st = (doc.get("run") or {}).get("status") if doc else "timeout"
    if st == "completed":
        EXP.mkdir(parents=True, exist_ok=True)
        (EXP / f"{rid}.json").write_text(json.dumps(doc), encoding="utf-8")
        steps = next(iter((doc.get("replay") or {}).values()), [])
        fin = (steps[-1].get("info") if steps else {}) or {}
        results.append({"model": model, "status": "completed", "id": rid,
                        "solved": fin.get("solved"), "hints": fin.get("hints_used"),
                        "par": fin.get("par"), "par_result": fin.get("par_result"),
                        "reward": fin.get("reward")})
        print(f"  -> completed reward={fin.get('reward')} hints={fin.get('hints_used')}", flush=True)
    else:
        err = ""
        if doc:
            e = (doc.get("episodes") or [{}])[0]
            err = str((e.get("terminal_info") or {}).get("error",""))[:80]
        results.append({"model": model, "status": st, "error": err})
        print(f"  -> {st} {err}", flush=True)
    time.sleep(3)  # brief gap between providers

(RES / "retry_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("\nRETRY DONE:")
for r in results: print(" ", r["model"], r["status"], r.get("reward",""))
