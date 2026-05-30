"""MinuteCryptic benchmark environment for Mesocosm / BenchAnything.

Each episode = one *real* Minute Cryptic daily clue (scraped into
``mc_data/minute_cryptic_dataset.jsonl``). The agent plays the clue the way a
human plays it on minutecryptic.com:

Game loop (per clue)
--------------------
1. Every observation carries the rules of cryptic crosswords plus a cheat-sheet
   of common cryptic substitutions.
2. The first observation gives only the clue and the answer's length.
3. On each turn the agent does ONE of two things:
     * GUESS   - reply with the answer.
     * "HINT"  - reveal the next hint.
   - A wrong guess is non-terminal; the next observation lists every wrong
     guess made so far.
   - Hints are an escalating ladder that PERSISTS for the rest of the clue:
     first the official progressive hints (definition -> wordplay -> ...),
     then the answer's letters revealed ONE at a time, in the site's official
     ``letterRevealOrder``. Taking enough letter hints eventually reveals the
     whole word, so the agent can never get stuck forever.
4. Termination:
     * correct guess                         -> solved
     * same wrong answer guessed 3 times      -> failed (it will never get it)
     * a HINT request once everything is shown -> solved at max hints (the
       worst-case guaranteed exit)

Scoring is golf-style: par is the expected number of hints for the clue. We
track the total hints taken, the ORDER they were taken in, and whether the
agent finished below / at / above par.

Reward (paid once, on the terminal step, so total_reward == intended score):

    correct, no hints ............ 1.0
    correct after k hints ........ max(MIN_SOLVE_REWARD, 1.0 - HINT_PENALTY*k)
    failed (same wrong 3x) ....... PARTIAL_WEIGHT * string-similarity-to-answer
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

DATA_PATH = Path(__file__).parent / "mc_data" / "minute_cryptic_dataset.jsonl"

SAME_WRONG_LIMIT = 3     # identical wrong answer this many times -> give up
HINT_PENALTY = 0.2       # reward lost per hint revealed
MIN_SOLVE_REWARD = 0.2   # floor for a correct answer, however many hints used
PARTIAL_WEIGHT = 0.3     # max reward for a close-but-wrong final guess

# Included in EVERY observation (point 1 of the loop).
RULES = (
    "Minute Cryptic — how to solve:\n"
    "- Every clue has TWO parts that both point to the same answer: a plain "
    "DEFINITION (always at the very start OR the very end of the clue) and "
    "WORDPLAY (the rest), which spells the answer by a hidden mechanism.\n"
    "- The number(s) in parentheses give the answer's length, e.g. (7) is a "
    "7-letter word, (3,4) is a 3-letter word then a 4-letter word.\n"
    "- Common wordplay devices: anagram (letters jumbled, flagged by an "
    "indicator like 'mixed', 'broken', 'shredded', 'wild'); hidden word "
    "(answer sits inside consecutive letters of the clue); reversal ('back', "
    "'returning', 'up' in a down clue); homophone ('we hear', 'sounds like'); "
    "charade (join shorter pieces in sequence); container ('in', 'around', "
    "'holding'); deletion ('endless', 'heartless'); and abbreviations.\n"
    "- Answers are matched ignoring case, spaces and punctuation.\n"
    "ACTIONS: each turn you either GUESS the answer or take a HINT. Hints first "
    "explain the clue, then reveal the answer's letters one at a time. Fewer "
    "hints is better — 'par' is the expected number of hints for this clue, and "
    "you are scored against it.\n"
    "RESPONSE FORMAT: reply in exactly two lines —\n"
    "REASONING: <one or two sentences on how you are decoding the clue>\n"
    "ACTION: <your answer in CAPITALS, or the single word HINT>"
)

# Cheat-sheet of standard cryptic substitutions, included in EVERY observation.
COMMON_SUBSTITUTIONS = [
    "love / nil / nothing / zero / duck -> O",
    "the (in French) -> LE, LA, LES; (German) -> DER, DIE, DAS; (Spanish) -> EL",
    "and -> N ('rock 'n' roll'); with -> W; without -> SANS",
    "about -> C, CA, RE; around -> RE; concerning -> RE",
    "Roman numerals: one -> I (or A/AN); five -> V; ten -> X; fifty -> L; "
    "hundred -> C; five hundred -> D; thousand -> M",
    "left -> L; right -> R; first -> initial letter; finally -> last letter; "
    "heart/centre -> middle letter(s)",
    "north/south/east/west -> N/S/E/W; point -> N, S, E or W",
    "doctor -> DR, MO, MB; learner / novice -> L; graduate -> BA, MA",
    "king -> K or R (rex); queen -> Q, ER or R; royal -> ER",
    "quiet -> P or SH; loud -> F (forte); very loud -> FF",
    "note (music) -> A B C D E F G, or DO RE MI FA SO LA TI",
    "sailor -> AB, TAR, RN, OS; soldier(s) -> GI, RE, TA, ANT(s); navy -> RN",
    "river -> R; lake -> L; road / street / way -> RD, ST, AVE, WAY",
    "old -> O; new -> N; good -> G; gold -> AU or OR; iron -> FE",
    "time -> T; hour -> HR; year -> Y; second -> S or MO; moment -> MO",
    "company -> CO, LTD, INC; society -> S, SOC; party -> DO, LAB, CON",
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


def _enumeration(length: list[int] | None) -> str:
    if not length:
        return ""
    return "(" + ",".join(str(n) for n in length) + ")"


_GUESS_PREFIX = re.compile(r"^\s*(?:my\s+)?(?:final\s+)?(?:answer|guess|solution)\s*[:\-=]\s*", re.I)
_REASONING_LINE = re.compile(r"^\s*(?:reasoning|thought|because|why)\s*[:\-]\s*(.*)", re.I)
_ACTION_LINE = re.compile(r"^\s*(?:action|decision|move)\s*[:\-]\s*(.*)", re.I)


def _is_hint_token(text: str) -> bool:
    """True if the decision text means 'take a hint'."""
    if re.sub(r"[^a-z]", "", text.lower()) == "hint":
        return True
    toks = re.findall(r"[A-Za-z]+", text)
    return bool(toks) and toks[-1].upper() == "HINT"


def _parse_action(raw: Any) -> tuple[str, str, str]:
    """Parse a model reply into (kind, guess, reasoning).

    Preferred format is two labelled lines:
        REASONING: ...
        ACTION: <answer | HINT>
    We fall back gracefully for chatty replies: the ACTION is taken from an
    'ACTION:' line if present, else the last non-empty line; the reasoning is
    the 'REASONING:' line if present, else everything before the decision.
    """
    s = _strip_wrapping_quotes(str(raw))
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]

    reasoning = ""
    decision = None
    decision_idx = None
    for i, ln in enumerate(lines):
        mr = _REASONING_LINE.match(ln)
        if mr and mr.group(1).strip():
            reasoning = mr.group(1).strip()
        ma = _ACTION_LINE.match(ln)
        if ma:
            decision = ma.group(1).strip()
            decision_idx = i

    if decision is None:                       # no explicit ACTION: line
        decision = lines[-1] if lines else s.strip()
        decision_idx = len(lines) - 1
    if not reasoning:                          # reasoning = everything before the decision
        pre = lines[:decision_idx] if decision_idx and decision_idx > 0 else []
        reasoning = " ".join(pre).strip()

    if _is_hint_token(decision):
        return ("hint", "", reasoning)
    guess = _strip_wrapping_quotes(_GUESS_PREFIX.sub("", decision))
    return ("guess", guess or decision, reasoning)


# --------------------------------------------------------------------------- data

def _to_clue(row: dict[str, Any]) -> dict[str, Any]:
    """Map a dataset row to the env's internal clue representation."""
    answer = str(row["answer"])
    alpha = [i for i, ch in enumerate(answer) if ch.isalpha()]
    order = row.get("letterRevealOrder") or list(range(len(alpha)))
    # Keep only valid letter indices, in the official reveal order.
    order = [k for k in order if 0 <= k < len(alpha)]
    content_hints = [
        {"type": h.get("type") or "hint", "text": h["text"]}
        for h in (row.get("hints") or [])
        if h.get("text")
    ]
    return {
        "id": row.get("date") or row.get("puzzleId") or "",
        "puzzle_id": row.get("puzzleId", ""),
        "clue": row["clue"],
        "answer": answer,
        "length": row.get("wordLengths") or [len(alpha)],
        "enumeration": row.get("enumeration") or _enumeration(row.get("wordLengths")),
        "par": int(row["par"]) if row.get("par") is not None else len(content_hints),
        "community_avg_par": row.get("communityAveragePar"),
        "content_hints": content_hints,
        "letter_order": order,        # alpha indices, official reveal order
        "n_letters": len(alpha),
        "definition": row.get("definition"),
    }


# Fallback so the env always runs even before the dataset is present.
_FALLBACK_ROW: dict[str, Any] = {
    "date": "demo-escort",
    "puzzleId": "demo-escort",
    "clue": "Chaperone shredded corset (6)",
    "answer": "ESCORT",
    "wordLengths": [6],
    "enumeration": "(6)",
    "par": 2,
    "communityAveragePar": 3,
    "letterRevealOrder": [0, 5, 1, 2, 4, 3],
    "definition": "Chaperone",
    "hints": [
        {"type": "definition", "text": "The definition is at one end: 'Chaperone'."},
        {"type": "indicators", "text": "'shredded' signals an anagram of the rest."},
        {"type": "fodder", "text": "Rearrange the letters of CORSET."},
    ],
}


def _load_clues() -> list[dict[str, Any]]:
    if not DATA_PATH.exists():
        return [_to_clue(_FALLBACK_ROW)]
    clues: list[dict[str, Any]] = []
    with DATA_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                clues.append(_to_clue(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
    return clues or [_to_clue(_FALLBACK_ROW)]


CLUES = _load_clues()


# --------------------------------------------------------------------------- matching

def _windows(response: str, n_words: int) -> list[str]:
    """Normalised n-word sliding windows, so 'the answer is ESCORT' still matches."""
    words = re.findall(r"[a-z]+", response.lower())
    if n_words < 1:
        n_words = 1
    return [_normalize("".join(words[i : i + n_words])) for i in range(len(words) - n_words + 1)]


def _is_correct(response: str, clue: dict[str, Any]) -> bool:
    key = _normalize(clue["answer"])
    if not key:
        return False
    if _normalize(response) == key:
        return True
    return any(w == key for w in _windows(response, len(clue["answer"].split())))


def _similarity(response: str, clue: dict[str, Any]) -> float:
    target = _normalize(clue["answer"])
    best = SequenceMatcher(None, _normalize(response), target).ratio()
    for w in _windows(response, len(clue["answer"].split())):
        best = max(best, SequenceMatcher(None, w, target).ratio())
    return best


def _letter_pattern(answer: str, revealed_alpha: set[int]) -> str:
    """Render the answer with revealed letters shown and the rest as '_'.

    Word breaks are shown as '/'. ``revealed_alpha`` holds alpha-only indices.
    """
    out: list[str] = []
    k = 0
    for ch in answer:
        if ch == " ":
            out.append("/")
        elif ch.isalpha():
            out.append(ch if k in revealed_alpha else "_")
            k += 1
        else:
            out.append(ch)  # punctuation shown as-is, not counted as a letter
    return " ".join(out)


# --------------------------------------------------------------------------- env

class MinuteCrypticEnv(BaseEnv):
    def __init__(self) -> None:
        self._clue: dict[str, Any] | None = None
        self._guesses = 0
        self._hints = 0
        self._content_taken = 0      # official content hints revealed
        self._letters_taken = 0      # letters revealed
        self._revealed_hints: list[str] = []   # textual hints shown so far
        self._hint_order: list[str] = []        # ordered hint types taken
        self._wrong_guesses: list[str] = []      # display strings, in order
        self._wrong_counts: dict[str, int] = {}  # normalized -> times guessed
        self._reasoning = ""                     # model's reasoning for the current step
        self._decision = ""                      # 'guess' or 'hint' for the current step
        self._guess_val = ""                     # the guessed text for the current step
        self._rng = random.Random()

    # ------------------------------------------------------------------ lifecycle

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        if seed is None:
            self._clue = self._rng.choice(CLUES)
        else:
            self._clue = CLUES[seed % len(CLUES)]
        self._guesses = 0
        self._hints = 0
        self._content_taken = 0
        self._letters_taken = 0
        self._revealed_hints = []
        self._hint_order = []
        self._wrong_guesses = []
        self._wrong_counts = {}
        self._reasoning = ""
        self._decision = ""
        self._guess_val = ""
        return self._observe()

    def step(self, action: Any) -> StepResult:
        if self._clue is None:
            raise RuntimeError("Call reset() before step()")
        kind, guess, reasoning = _parse_action(action)
        self._reasoning = reasoning
        if kind == "hint":
            self._decision, self._guess_val = "hint", ""
            return self._take_hint()
        self._decision, self._guess_val = "guess", guess
        return self._take_guess(guess, full=_strip_wrapping_quotes(str(action)))

    # ------------------------------------------------------------------ actions

    def _take_hint(self) -> StepResult:
        clue = self._clue
        assert clue is not None

        # 1) official content hints, in order
        if self._content_taken < len(clue["content_hints"]):
            h = clue["content_hints"][self._content_taken]
            self._content_taken += 1
            self._hints += 1
            self._revealed_hints.append(f"[{h['type']}] {h['text']}")
            self._hint_order.append(h["type"])
            return StepResult(
                observation=self._observe(feedback=f"Hint ({h['type']}): {h['text']}"),
                reward=0.0,
                terminated=False,
                truncated=False,
                info=self._info(solved=False, outcome="hint"),
                system_prompt=f"Hint: {h['text']}",
            )

        # 2) reveal one more letter
        if self._letters_taken < clue["n_letters"]:
            self._letters_taken += 1
            self._hints += 1
            self._hint_order.append("letter")
            pattern = _letter_pattern(clue["answer"], self._revealed_set())
            return StepResult(
                observation=self._observe(feedback=f"Letter revealed -> {pattern}"),
                reward=0.0,
                terminated=False,
                truncated=False,
                info=self._info(solved=False, outcome="hint"),
                system_prompt=f"Letters so far: {pattern}",
            )

        # 3) nothing left to reveal -> the whole answer is shown; guaranteed exit
        reward = max(MIN_SOLVE_REWARD, 1.0 - HINT_PENALTY * self._hints)
        return StepResult(
            observation={"result": "revealed", "answer": clue["answer"]},
            reward=reward,
            terminated=True,
            truncated=False,
            info=self._info(solved=True, outcome="fully_revealed", reward=reward),
        )

    def _take_guess(self, text: str, full: str | None = None) -> StepResult:
        clue = self._clue
        assert clue is not None
        self._guesses += 1

        # Match the extracted guess; fall back to the full reply so an answer
        # embedded in reasoning ("the answer is HEINOUS") is still caught.
        if _is_correct(text, clue) or (full and _is_correct(full, clue)):
            reward = max(MIN_SOLVE_REWARD, 1.0 - HINT_PENALTY * self._hints)
            return StepResult(
                observation={"result": "correct", "answer": clue["answer"]},
                reward=reward,
                terminated=True,
                truncated=False,
                info=self._info(solved=True, outcome="solved", reward=reward),
            )

        # record the wrong guess
        norm = _normalize(text) or text.strip().upper()
        disp = text.strip().upper()
        if disp not in self._wrong_guesses:
            self._wrong_guesses.append(disp)
        self._wrong_counts[norm] = self._wrong_counts.get(norm, 0) + 1

        # same wrong answer three times -> it will never get it
        if self._wrong_counts[norm] >= SAME_WRONG_LIMIT:
            reward = PARTIAL_WEIGHT * _similarity(text, clue)
            return StepResult(
                observation={"result": "failed", "answer": clue["answer"]},
                reward=reward,
                terminated=True,
                truncated=False,
                info=self._info(solved=False, outcome="repeated_wrong", reward=reward),
            )

        fb = (
            f"'{disp}' is not correct. You have guessed: "
            f"{', '.join(self._wrong_guesses)}. Try another answer, or reply HINT."
        )
        return StepResult(
            observation=self._observe(feedback=fb),
            reward=0.0,
            terminated=False,
            truncated=False,
            info=self._info(solved=False, outcome="wrong"),
        )

    # ------------------------------------------------------------------ helpers

    def _revealed_set(self) -> set[int]:
        clue = self._clue
        assert clue is not None
        return set(clue["letter_order"][: self._letters_taken])

    def _observe(self, feedback: str | None = None) -> dict[str, Any]:
        clue = self._clue
        assert clue is not None
        obs: dict[str, Any] = {
            "clue": clue["clue"],
            "enumeration": clue["enumeration"],
            "answer_length": clue["n_letters"],
            "word_lengths": clue["length"],
            "rules": RULES,
            "common_substitutions": COMMON_SUBSTITUTIONS,
            "hints_taken": self._hints,
            "par": clue["par"],
            "instructions": (
                "Reply in exactly two lines:\n"
                "REASONING: <one or two sentences on how you are decoding the clue>\n"
                "ACTION: <your answer in CAPITALS, or the single word HINT to reveal "
                "the next hint>"
            ),
        }
        # Hints persist once taken (point 3 of the loop).
        if self._revealed_hints:
            obs["revealed_hints"] = list(self._revealed_hints)
        if self._letters_taken:
            obs["revealed_letters"] = _letter_pattern(clue["answer"], self._revealed_set())
        if self._wrong_guesses:
            obs["wrong_guesses"] = list(self._wrong_guesses)
        if feedback:
            obs["feedback"] = feedback
        return obs

    def _info(self, *, solved: bool, outcome: str, reward: float = 0.0) -> dict[str, str]:
        clue = self._clue or {}
        par = int(clue.get("par", 0))
        to_par = self._hints - par
        if to_par < 0:
            par_result = "below"   # golf: fewer hints than par
        elif to_par == 0:
            par_result = "at"
        else:
            par_result = "above"
        avg = clue.get("community_avg_par")
        return {
            "clue_id": str(clue.get("id", "")),
            "puzzle_id": str(clue.get("puzzle_id", "")),
            "solved": "1" if solved else "0",
            "guesses_used": str(self._guesses),
            "hints_used": str(self._hints),
            "hint_order": json.dumps(self._hint_order),
            "par": str(par),
            "community_avg_par": "" if avg is None else str(avg),
            "hints_to_par": str(to_par),
            "under_or_at_par": "1" if to_par <= 0 else "0",
            "par_result": par_result,
            "outcome": outcome,
            "reward": f"{reward:.4f}",
            # per-step play data for the replay dashboard
            "decision": self._decision,
            "guess": self._guess_val,
            "reasoning": self._reasoning[:1000],
        }
