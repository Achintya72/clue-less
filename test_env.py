"""Scoring/behaviour tests for the MinuteCryptic env.

Run from the repo root:  pytest -q

These pin the reward contract and run against a FIXED in-test clue set
(installed via the autouse fixture) so they pass regardless of whatever
real dataset currently lives in data/clues.jsonl.
"""

from __future__ import annotations

import pytest

import env as E
from env import MinuteCrypticEnv

# A small, known clue set the tests control — independent of data/clues.jsonl.
TEST_CLUES = [
    {"id": "t-escort", "clue": "Chaperone shredded corset (6)", "answer": "ESCORT",
     "length": [6], "definition": "Chaperone", "device": "anagram", "human_avg_guesses": 3,
     "hints": ["The definition is 'Chaperone'.", "'shredded' signals an anagram.",
               "Anagram CORSET."]},
    {"id": "t-sword", "clue": "Words out of order make a weapon (5)", "answer": "SWORD",
     "length": [5], "definition": "weapon", "device": "anagram", "human_avg_guesses": 3},
    {"id": "t-news", "clue": "Information from the points of the compass (4)", "answer": "NEWS",
     "length": [4], "definition": "Information", "device": "initialism", "human_avg_guesses": 4},
    {"id": "t-patella", "clue": "Two girls, one on each knee (7)", "answer": "PATELLA",
     "length": [7], "definition": "knee", "device": "charade", "human_avg_guesses": 5},
    {"id": "t-icecube", "clue": "Die of cold (3,4)", "answer": "ICE CUBE",
     "accepted": ["ICECUBE"], "length": [3, 4], "definition": "Die", "human_avg_guesses": 6},
]
BY_ID = {c["id"]: i for i, c in enumerate(TEST_CLUES)}


@pytest.fixture(autouse=True)
def _fixed_clues(monkeypatch):
    # env.reset() reads the module-global CLUES, so patch it for every test.
    monkeypatch.setattr(E, "CLUES", TEST_CLUES)


def seed_for(clue_id: str) -> int:
    # reset() picks CLUES[seed % len(CLUES)]; the index is a valid seed.
    return BY_ID[clue_id]


def play(actions, seed):
    """Drive an episode; return (total_reward, last_StepResult)."""
    e = MinuteCrypticEnv()
    e.reset(seed=seed)
    total = 0.0
    result = None
    for a in actions:
        result = e.step(a)
        total += result.reward
        if result.terminated or result.truncated:
            break
    assert result is not None
    return total, result


def test_correct_first_try_full_reward():
    total, r = play(["ESCORT"], seed=seed_for("t-escort"))
    assert r.terminated
    assert total == 1.0
    assert r.info["solved"] == "1"
    assert r.info["guesses_used"] == "1"
    assert r.info["hints_used"] == "0"


def test_messy_phrasing_still_matches():
    total, r = play(["I think the answer is sword."], seed=seed_for("t-sword"))
    assert total == 1.0
    assert r.info["solved"] == "1"


def test_multiword_answer_with_space_and_variant():
    total, r = play(["ice cube"], seed=seed_for("t-icecube"))
    assert total == 1.0
    assert r.info["solved"] == "1"


def test_hint_costs_reward_and_is_non_terminal():
    total, r = play(["WRONG", "HINT", "NEWS"], seed=seed_for("t-news"))
    assert r.terminated
    assert r.info["solved"] == "1"
    assert r.info["hints_used"] == "1"
    assert abs(total - 0.8) < 1e-9


def test_two_hints_compound_penalty():
    total, r = play(["HINT", "HINT", "NEWS"], seed=seed_for("t-news"))
    assert r.info["hints_used"] == "2"
    assert abs(total - 0.6) < 1e-9


def test_hint_step_pays_no_reward_and_does_not_terminate():
    e = MinuteCrypticEnv()
    e.reset(seed=seed_for("t-news"))
    r = e.step("HINT")
    assert r.reward == 0.0
    assert not r.terminated
    assert r.system_prompt and r.system_prompt.startswith("Hint:")
    assert "feedback" in r.observation


def test_exhausted_guesses_give_partial_credit():
    actions = ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "PATILLA"]  # close miss last
    total, r = play(actions, seed=seed_for("t-patella"))
    assert r.terminated
    assert r.info["solved"] == "0"
    assert r.info["guesses_used"] == "6"
    assert 0.0 < total < 0.3


def test_wrong_then_correct_no_hint_full_reward():
    total, r = play(["NOPE", "ESCORT"], seed=seed_for("t-escort"))
    assert total == 1.0
    assert r.info["guesses_used"] == "2"
    assert r.info["hints_used"] == "0"


def test_solve_floor_never_below_min():
    e = MinuteCrypticEnv()
    e.reset(seed=seed_for("t-escort"))
    for _ in range(10):
        e.step("HINT")
    r = e.step("ESCORT")
    assert r.terminated and r.info["solved"] == "1"
    assert r.reward >= E.MIN_SOLVE_REWARD - 1e-9


def test_deterministic_reset_by_seed():
    a = MinuteCrypticEnv().reset(seed=42)
    b = MinuteCrypticEnv().reset(seed=42)
    assert a["clue"] == b["clue"]


def test_quoted_hint_is_treated_as_hint_request():
    e = MinuteCrypticEnv()
    e.reset(seed=seed_for("t-news"))
    r = e.step('"HINT"')
    assert not r.terminated
    assert r.info["hints_used"] == "1"
    assert r.info["guesses_used"] == "0"


def test_quoted_answer_still_solves():
    total, r = play(['"ESCORT"'], seed=seed_for("t-escort"))
    assert total == 1.0
    assert r.info["solved"] == "1"
    assert r.info["guesses_used"] == "1"


def test_terminal_info_fields_are_strings():
    _, r = play(["ESCORT"], seed=seed_for("t-escort"))
    for k in ("solved", "guesses_used", "hints_used"):
        assert isinstance(r.info[k], str)
        float(r.info[k])
