"""MinuteCryptic benchmark environment for Mesocosm / BenchAnything.

Each episode = one cryptic clue. The agent reads the clue + enumeration and
replies with an answer. Wrong answers can be retried; the agent may instead
request a progressive hint (reply exactly "HINT") at a scoring penalty.

Reward is paid exactly once, on the terminal step, so an episode's
``total_reward`` (what the scoring engine reads for ``episode_reward`` metrics)
equals the intended score for that clue:

    correct, no hints ............ 1.0
    correct after k hints ........ max(MIN_SOLVE_REWARD, 1.0 - HINT_PENALTY*k)
    wrong (attempts exhausted) ... PARTIAL_WEIGHT * string-similarity-to-answer
    truncated / never answered ... 0.0
"""

from __future__ import annotations

import json
import random
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from bench_common.env_sdk.base import BaseEnv, StepResult

DATA_PATH = Path(__file__).parent / "data" / "clues.jsonl"

MAX_GUESSES = 6          # answer attempts before the episode fails
HINT_PENALTY = 0.2       # reward lost per hint revealed
MIN_SOLVE_REWARD = 0.2   # floor for a correct answer, however many hints used
PARTIAL_WEIGHT = 0.3     # max reward for a close-but-wrong final guess

# Fallback so the env always runs even before data/clues.jsonl is populated.
_FALLBACK: list[dict[str, Any]] = [
    {
        "id": "demo-escort",
        "clue": "Chaperone shredded corset (6)",
        "answer": "ESCORT",
        "length": [6],
        "definition": "Chaperone",
        "device": "anagram",
        "human_avg_guesses": 3,
        "hints": [
            "The definition is at one end: 'Chaperone'.",
            "'shredded' is an anagram indicator for the remaining word.",
            "Rearrange the letters of CORSET.",
        ],
    },
]


def _normalize(s: str) -> str:
    """Lowercase and strip everything but a-z, so 'ICE CUBE' == 'ice-cube'."""
    return re.sub(r"[^a-z]", "", s.lower())


def _strip_wrapping_quotes(s: str) -> str:
    """Remove one layer of matching wrapping quotes a model may add, e.g. \"HINT\" -> HINT."""
    s = s.strip()
    while len(s) >= 2 and s[0] in "\"'`" and s[-1] == s[0]:
        s = s[1:-1].strip()
    return s


def _enumeration(length: list[int]) -> str:
    return "(" + ",".join(str(n) for n in length) + ")"


def _load_clues() -> list[dict[str, Any]]:
    if not DATA_PATH.exists():
        return _FALLBACK
    clues: list[dict[str, Any]] = []
    with DATA_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                clues.append(json.loads(line))
    return clues or _FALLBACK


CLUES = _load_clues()


def _hint_ladder(clue: dict[str, Any]) -> list[str]:
    """Per-clue hints, or a generic ladder synthesised from the clue's fields."""
    if clue.get("hints"):
        return list(clue["hints"])
    ladder: list[str] = []
    if clue.get("definition"):
        ladder.append(f"The definition part of the clue is: '{clue['definition']}'.")
    else:
        ladder.append("One end of the clue is a plain definition; the rest is wordplay.")
    if clue.get("device"):
        ladder.append(f"The wordplay device is: {clue['device']}.")
    norm = _normalize(clue["answer"])
    if norm:
        ladder.append(f"The answer begins with '{norm[0].upper()}'.")
    return ladder


def _answer_keys(clue: dict[str, Any]) -> list[str]:
    raw = [clue["answer"], *clue.get("accepted", [])]
    return [k for k in (_normalize(x) for x in raw) if k]


def _windows(response: str, n_words: int) -> list[str]:
    """Normalised n-word sliding windows, so 'the answer is ESCORT' still matches."""
    words = re.findall(r"[a-z]+", response.lower())
    return [_normalize("".join(words[i : i + n_words])) for i in range(len(words) - n_words + 1)]


def _is_correct(response: str, clue: dict[str, Any]) -> bool:
    keys = _answer_keys(clue)
    if _normalize(response) in keys:
        return True
    return any(w in keys for w in _windows(response, len(clue["answer"].split())))


def _similarity(response: str, clue: dict[str, Any]) -> float:
    target = _normalize(clue["answer"])
    best = SequenceMatcher(None, _normalize(response), target).ratio()
    for w in _windows(response, len(clue["answer"].split())):
        best = max(best, SequenceMatcher(None, w, target).ratio())
    return best


class MinuteCrypticEnv(BaseEnv):
    def __init__(self) -> None:
        self._clue: dict[str, Any] | None = None
        self._ladder: list[str] = []
        self._guesses = 0
        self._hints = 0
        self._rng = random.Random()

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        # Deterministic, full coverage when seeded; random when not.
        if seed is None:
            self._clue = self._rng.choice(CLUES)
        else:
            self._clue = CLUES[seed % len(CLUES)]
        self._ladder = _hint_ladder(self._clue)
        self._guesses = 0
        self._hints = 0
        return self._observe()

    def step(self, action: Any) -> StepResult:
        if self._clue is None:
            raise RuntimeError("Call reset() before step()")
        text = _strip_wrapping_quotes(str(action))

        # --- hint request (non-terminal, no reward) ---
        if text.upper() == "HINT" or text.upper().startswith("HINT "):
            if self._hints < len(self._ladder):
                hint = self._ladder[self._hints]
                self._hints += 1
            else:
                hint = "No more hints available."
            return StepResult(
                observation=self._observe(feedback=f"Hint: {hint}"),
                reward=0.0,
                terminated=False,
                truncated=False,
                info=self._info(solved=False, outcome="hint"),
                system_prompt=f"Hint: {hint}",
            )

        # --- answer attempt ---
        self._guesses += 1

        if _is_correct(text, self._clue):
            reward = max(MIN_SOLVE_REWARD, 1.0 - HINT_PENALTY * self._hints)
            return StepResult(
                observation={"result": "correct", "answer": self._clue["answer"]},
                reward=reward,
                terminated=True,
                truncated=False,
                info=self._info(solved=True, outcome="solved", reward=reward),
            )

        if self._guesses >= MAX_GUESSES:
            reward = PARTIAL_WEIGHT * _similarity(text, self._clue)
            return StepResult(
                observation={"result": "failed", "answer": self._clue["answer"]},
                reward=reward,
                terminated=True,
                truncated=False,
                info=self._info(solved=False, outcome="exhausted", reward=reward),
            )

        return StepResult(
            observation=self._observe(feedback=f"'{text}' is not correct. Try again or reply HINT."),
            reward=0.0,
            terminated=False,
            truncated=False,
            info=self._info(solved=False, outcome="wrong"),
        )

    # ------------------------------------------------------------------ helpers

    def _observe(self, feedback: str | None = None) -> dict[str, Any]:
        clue = self._clue
        assert clue is not None
        obs: dict[str, Any] = {
            "clue": clue["clue"],
            "enumeration": _enumeration(clue["length"]),
            "attempts_remaining": MAX_GUESSES - self._guesses,
            "hints_remaining": max(0, len(self._ladder) - self._hints),
            "instructions": (
                "Solve this cryptic clue. A cryptic clue has a definition (at one "
                "end of the clue) plus wordplay that spells the same answer. The "
                "number in parentheses is the answer's length. Reply with ONLY your "
                "answer. To reveal a hint instead, reply exactly: HINT"
            ),
        }
        if feedback:
            obs["feedback"] = feedback
        return obs

    def _info(self, *, solved: bool, outcome: str, reward: float = 0.0) -> dict[str, str]:
        clue = self._clue or {}
        return {
            "clue_id": str(clue.get("id", "")),
            "solved": "1" if solved else "0",
            "guesses_used": str(self._guesses),
            "hints_used": str(self._hints),
            "human_avg_guesses": str(clue.get("human_avg_guesses", "")),
            "device": str(clue.get("device", "")),
            "outcome": outcome,
            "reward": f"{reward:.4f}",
        }
