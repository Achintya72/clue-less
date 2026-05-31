#!/usr/bin/env python
"""Resilient sweep: harvest completed episodes immediately, time out stuck runs.

The platform leaves stragglers wedged in 'running' forever, so a batch never goes
fully terminal and the original orchestrator hangs. This driver instead:
  - harvests each COMPLETED episode as soon as it appears (no wait for the batch),
  - abandons a run after RUN_TIMEOUT and re-queues its unfinished seeds into a fresh
    small run (fresh small runs complete fast),
  - stops at GLOBAL_TIMEOUT and writes whatever coverage we have.
Writes results/episodes/<model>/<seed>.json in the same shape build_dashboard_data.py reads.
"""
from __future__ import annotations
import json, time, subprocess, urllib.request, urllib.error
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent
EPDIR = ROOT / "results" / "episodes"
BASE = "https://api.swecc.org/bench"
DOMAIN = "62d97567-98b0-4f90-bbef-bfd21cea4550"
VOW = "1.0.0"
MODELS = ["gemini/gemini-3.1-flash-lite"]
TARGET = list(range(40))      # seeds (clues) per model
BATCH = 5                     # small runs complete fast
RUN_TIMEOUT = 110             # secs before abandoning a run's stuck seeds
GLOBAL_TIMEOUT = 1100         # ~18 min hard cap (fits token window)
POLL = 8

def slug(m): return m.replace("/", "-").replace(".", "-")
def token(): return subprocess.run(["mesocosm","auth","token"],capture_output=True,text=True).stdout.strip()

def api(method, path, tok, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE+path, data=data, method=method,
        headers={"Authorization":f"Bearer {tok}","Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"  api {method} {path} err: {e}", flush=True); return None

def create(model, seeds, tok):
    d = api("POST","/v1/runs",tok,{"domain_id":DOMAIN,"binding_vow_version":VOW,
        "agent_config":{"model":model,"temperature":0.0,"max_tokens":800},
        "num_episodes":len(seeds),"seed_set":seeds,"max_parallel":5})
    return d.get("id") if d else None

def main():
    t0 = time.time()
    # resumable: skip seeds already harvested on disk
    state = {}
    for m in MODELS:
        have = {int(p.stem) for p in (EPDIR/slug(m)).glob("*.json")} if (EPDIR/slug(m)).exists() else set()
        state[m] = {"pending":[s for s in TARGET if s not in have], "saved":set(have),
                    "run":None, "rseeds":[], "rstart":0}
        print(f"{m}: {len(have)} already on disk, {len(state[m]['pending'])} to fetch", flush=True)
    while time.time()-t0 < GLOBAL_TIMEOUT:
        tok = token()
        for m in MODELS:
            st = state[m]; epd = EPDIR/slug(m); epd.mkdir(parents=True,exist_ok=True)
            if st["run"]:
                doc = api("GET",f"/v1/runs/{st['run']}/export",tok)
                rep = (doc or {}).get("replay") or {}
                for ep in ((doc or {}).get("episodes") or []):
                    seed = ep.get("seed")
                    if ep.get("status")=="completed" and seed in st["rseeds"] and seed not in st["saved"]:
                        steps = rep.get(ep["id"]) or ep.get("steps") or []
                        (epd/f"{seed}.json").write_text(json.dumps({"seed":seed,"model":m,"steps":steps}))
                        st["saved"].add(seed)
                        if seed in st["pending"]: st["pending"].remove(seed)
                done_here = [s for s in st["rseeds"] if s in st["saved"]]
                if len(done_here)==len(st["rseeds"]) or time.time()-st["rstart"]>RUN_TIMEOUT:
                    st["run"]=None; st["rseeds"]=[]   # leftover seeds stay in pending -> retried
            if not st["run"] and st["pending"]:
                seeds = st["pending"][:BATCH]
                rid = create(m, seeds, tok)
                if rid: st["run"], st["rseeds"], st["rstart"] = rid, seeds, time.time()
        cov = " | ".join(f"{slug(m)} {len(state[m]['saved'])}/{len(TARGET)}" for m in MODELS)
        print(f"[t+{int(time.time()-t0):>3}s] {cov}", flush=True)
        if all(not state[m]["pending"] for m in MODELS): break
        time.sleep(POLL)
    print("SWEEP DONE", flush=True)
    for m in MODELS:
        print(f"  {m}: {len(state[m]['saved'])}/{len(TARGET)} clues", flush=True)

if __name__ == "__main__":
    main()
