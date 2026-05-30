"""Generate Mesocosm-export-schema replay JSON from the real env, for the UI.

`mesocosm run local` only prints a summary; the rich replay JSON comes from
`mesocosm run export RUN_ID` (a platform run). Until you have that, this script
drives the actual MinuteCryptic env with a few scripted reference policies and
writes files in the SAME schema, so showcase/index.html renders today. When you
have real exports, just drop them in showcase/data/ and rebuild data/index.json.

Run:  python showcase/make_samples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))  # import the env package

import env as E  # noqa: E402
from env import MinuteCrypticEnv  # noqa: E402

DATA_DIR = HERE / "data"
MAX_STEPS = 12  # mirror benchanything.json episode.max_steps
N_EPISODES = 8  # clues each reference policy plays (keeps sample files small)


def _first_word(s: str) -> str:
    return (s or "ANSWER").split()[0].upper()


# Each policy maps (clue, step_index, hints_taken) -> (action, reasoning).
def policy_perfect(clue, i, hints):
    return clue["answer"], "I parse the wordplay and read off the answer directly."


def policy_careful(clue, i, hints):
    # Take exactly one hint, then answer.
    if i == 0:
        return "HINT", "Not certain yet — I'll spend one hint to confirm the definition."
    return clue["answer"], "The hint confirms the definition; this is the answer."


def policy_llama_local(clue, i, hints):
    # Mimics the observed 3B behaviour: echo the definition word / junk, never solves.
    guess = _first_word(clue.get("definition", ""))
    return guess, "This word seems related to the clue's surface reading."


def policy_hint_spammer(clue, i, hints):
    return "HINT", "I'm stuck, so I keep asking for hints."


POLICIES = {
    "perfect-solver": policy_perfect,
    "careful-solver": policy_careful,
    "llama3.2-local": policy_llama_local,
    "hint-spammer": policy_hint_spammer,
}


def run_policy(model: str, policy) -> dict:
    replay: dict[str, list] = {}
    for seed in range(min(N_EPISODES, len(E.CLUES))):
        env = MinuteCrypticEnv()
        obs = env.reset(seed=seed)
        clue = E.CLUES[seed]
        steps = []
        hints = 0
        for i in range(MAX_STEPS):
            action, reasoning = policy(clue, i, hints)
            result = env.step(action)
            if str(action).strip().upper().strip('"') == "HINT":
                hints += 1
            steps.append(
                {
                    "step": i + 1,
                    "observation": obs,
                    "reasoning": reasoning,
                    "action": action,
                    "reward": round(result.reward, 4),
                    "terminated": result.terminated,
                    "truncated": result.truncated,
                    "info": result.info,
                }
            )
            obs = result.observation
            if result.terminated or result.truncated:
                break
        replay[f"{model}-ep{seed}"] = steps
    return {
        "schema_version": "1",
        "domain_id": "minutecryptic",
        "binding_vow_version": "1.0.0",
        "model": model,
        "visibility": "gallery_public",
        "replay": replay,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for model, policy in POLICIES.items():
        doc = run_policy(model, policy)
        fname = f"{model}.json"
        (DATA_DIR / fname).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        index.append({"file": fname, "model": model})
        eps = len(doc["replay"])
        print(f"wrote {fname}  ({eps} episodes)")
    (DATA_DIR / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"wrote index.json  ({len(index)} models)")


if __name__ == "__main__":
    main()
