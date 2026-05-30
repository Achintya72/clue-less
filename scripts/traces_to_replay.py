"""Convert Mesocosm episode traces (data/traces/*.jsonl) into the leaderboard's
export-schema JSON (showcase/data/<model>.json), one file per model.

Each trace file is one episode of records {episode_id, step, event_type, payload}.
We reconstruct, per agent step: the observation the model saw, its reasoning
(model_call text), the action, and the step reward/terminated/info.

Run:  python scripts/traces_to_replay.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT / "data" / "traces"
OUT = ROOT / "showcase" / "data"


def safe_name(model: str) -> str:
    return model.split("/")[-1]


def last_line(text: str) -> str:
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    return lines[-1] if lines else str(text).strip()


def parse_episode(path: Path):
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not records:
        return None, None, None
    episode_id = records[0].get("episode_id", path.stem)
    model = None
    by_step: dict[int, list] = defaultdict(list)
    for r in records:
        by_step[r.get("step", 0)].append(r)
        if r.get("event_type") == "model_call":
            model = r["payload"].get("model", model)

    steps = []
    for s in sorted(k for k in by_step if k >= 1):
        obs = reasoning = action_full = None
        sr = None
        for r in by_step[s]:
            et, p = r.get("event_type"), r.get("payload", {})
            if et == "observation" and p.get("phase") == "before_agent":
                obs = p.get("data")
            elif et == "model_call":
                reasoning = p.get("text", "")
            elif et == "action":
                action_full = p.get("action", "")
            elif et == "step_result":
                sr = p
        if sr is None:
            continue
        steps.append({
            "step": s,
            "observation": obs,
            "reasoning": reasoning or "",
            "action": last_line(action_full or ""),
            "reward": round(float(sr.get("reward", 0.0)), 4),
            "terminated": bool(sr.get("terminated", False)),
            "truncated": bool(sr.get("truncated", False)),
            "info": sr.get("info", {}),
        })
    return model, episode_id, steps


def main() -> None:
    if not TRACES.exists():
        raise SystemExit(f"No traces at {TRACES}. Run a model first (run local / run_claude).")

    per_model: dict[str, dict] = defaultdict(dict)
    for f in sorted(TRACES.glob("*.jsonl")):
        model, ep_id, steps = parse_episode(f)
        if not model or not steps:
            continue
        # Only count episodes that actually concluded (solved/failed/truncated).
        # Rate-limit-interrupted episodes have no terminal step — skip them so
        # infra errors aren't scored as wrong answers.
        if not (steps[-1]["terminated"] or steps[-1]["truncated"]):
            continue
        per_model[model][ep_id] = steps

    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for model, replay in sorted(per_model.items()):
        name = safe_name(model)
        doc = {
            "schema_version": "1",
            "domain_id": "minutecryptic",
            "binding_vow_version": "1.0.0",
            "model": name,
            "visibility": "gallery_public",
            "replay": replay,
        }
        (OUT / f"{name}.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        index.append({"file": f"{name}.json", "model": name})
        solved = sum(1 for ep in replay.values() if (ep[-1]["info"].get("solved") == "1"))
        print(f"wrote {name}.json — {len(replay)} episodes, {solved} solved")

    (OUT / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"wrote index.json — {len(index)} model(s): {[i['model'] for i in index]}")


if __name__ == "__main__":
    main()
