"""Convert the raw MinuteCryptic dataset into the env's data/clues.jsonl format.

Input : mc_data/minute_cryptic_dataset.json  (list of puzzle objects)
Output: data/clues.jsonl                      (one clue per line, env schema)

Run:  python scripts/build_clues.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "mc_data" / "minute_cryptic_dataset.json"
OUT = ROOT / "data" / "clues.jsonl"


def difficulty(par) -> str | None:
    if par is None:
        return None
    if par <= 2:
        return "easy"
    if par <= 4:
        return "medium"
    return "hard"


def convert(item: dict) -> dict | None:
    clue = (item.get("clue") or "").strip()
    answer = (item.get("answer") or "").strip().upper()
    lengths = item.get("wordLengths") or ([item["answerLength"]] if item.get("answerLength") else [])
    if not clue or not answer or not lengths:
        return None

    # human baseline: prefer community average, fall back to setter par.
    human = item.get("communityAveragePar")
    if human is None:
        human = item.get("par")

    # ordered hint ladder -> list[str]
    hints = [
        h["text"].strip()
        for h in sorted(item.get("hints") or [], key=lambda h: h.get("order", 0))
        if h.get("text")
    ]

    out: dict = {
        "id": item.get("puzzleId") or item.get("date") or clue[:24],
        "clue": clue,
        "answer": answer,
        "accepted": [],
        "length": lengths,
        "definition": item.get("definition"),
        "device": None,  # not labelled in source; env falls back to data hints
        "difficulty": difficulty(human),
        "human_avg_guesses": human,
        "hints": hints,
    }
    # drop None/empty optionals for a tidy file
    return {k: v for k, v in out.items() if v not in (None, [], "")}


def main() -> None:
    items = json.loads(SRC.read_text(encoding="utf-8"))
    rows = [c for c in (convert(it) for it in items) if c]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    kept, total = len(rows), len(items)
    with_hints = sum(1 for r in rows if r.get("hints"))
    with_human = sum(1 for r in rows if "human_avg_guesses" in r)
    print(f"{kept}/{total} clues -> {OUT.relative_to(ROOT)}")
    print(f"  with hints: {with_hints}   with human baseline: {with_human}")


if __name__ == "__main__":
    main()
