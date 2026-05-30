"""Scoring/behaviour tests for the MinuteCryptic env.

Run from the repo root:  pytest -q
These mirror the manual smoke cases and lock in the reward contract:
reward is paid once on the terminal step, so total_reward == intended score.
"""

from __future__ import annotations

import env as E
from env import MinuteCrypticEnv


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


# ---- per-clue lookup so tests don't depend on clue ordering -------------------
BY_ID = {c["id"]: i for i, c in enumerate(E.CLUES)}


def seed_for(clue_id: str) -> int:
    # reset() picks CLUES[seed % len(CLUES)]; index is a valid seed.
    return BY_ID[clue_id]


def test_correct_first_try_full_reward():
    total, r = play(["ESCORT"], seed=seed_for("placeholder-escort"))
    assert r.terminated
    assert total == 1.0
    assert r.info["solved"] == "1"
    assert r.info["guesses_used"] == "1"
    assert r.info["hints_used"] == "0"


def test_messy_phrasing_still_matches():
    total, r = play(["I think the answer is sword."], seed=seed_for("placeholder-sword"))
    assert total == 1.0
    assert r.info["solved"] == "1"


def test_multiword_answer_with_space_and_variant():
    # answer is "ICE CUBE"; agent types lowercase with a space.
    total, r = play(["ice cube"], seed=seed_for("placeholder-icecube"))
    assert total == 1.0
    assert r.info["solved"] == "1"


def test_hint_costs_reward_and_is_non_terminal():
    total, r = play(["WRONG", "HINT", "NEWS"], seed=seed_for("placeholder-news"))
    assert r.terminated
    assert r.info["solved"] == "1"
    assert r.info["hints_used"] == "1"
    # one hint => 1.0 - 0.2
    assert abs(total - 0.8) < 1e-9


def test_two_hints_compound_penalty():
    total, r = play(["HINT", "HINT", "NEWS"], seed=seed_for("placeholder-news"))
    assert r.info["hints_used"] == "2"
    assert abs(total - 0.6) < 1e-9


def test_hint_step_pays_no_reward_and_does_not_terminate():
    e = MinuteCrypticEnv()
    e.reset(seed=seed_for("placeholder-news"))
    r = e.step("HINT")
    assert r.reward == 0.0
    assert not r.terminated
    assert r.system_prompt and r.system_prompt.startswith("Hint:")
    assert "feedback" in r.observation


def test_exhausted_guesses_give_partial_credit():
    actions = ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "PATILLA"]  # close miss last
    total, r = play(actions, seed=seed_for("placeholder-patella"))
    assert r.terminated
    assert r.info["solved"] == "0"
    assert r.info["guesses_used"] == "6"
    assert 0.0 < total < 0.3  # partial = 0.3 * similarity, capped under full credit


def test_wrong_then_correct_no_hint_full_reward():
    total, r = play(["NOPE", "ESCORT"], seed=seed_for("placeholder-escort"))
    assert total == 1.0
    assert r.info["guesses_used"] == "2"
    assert r.info["hints_used"] == "0"


def test_solve_floor_never_below_min():
    # Burn every hint, then answer: reward must not drop below MIN_SOLVE_REWARD.
    e = MinuteCrypticEnv()
    e.reset(seed=seed_for("placeholder-escort"))
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
    # A model that emits the string with quotes — `"HINT"` — should still
    # request a hint, not have it scored as a (wrong) guess.
    e = MinuteCrypticEnv()
    e.reset(seed=seed_for("placeholder-news"))
    r = e.step('"HINT"')
    assert not r.terminated
    assert r.info["hints_used"] == "1"
    assert r.info["guesses_used"] == "0"


def test_quoted_answer_still_solves():
    total, r = play(['"ESCORT"'], seed=seed_for("placeholder-escort"))
    assert total == 1.0
    assert r.info["solved"] == "1"
    assert r.info["guesses_used"] == "1"


def test_terminal_info_fields_are_strings():
    # terminal_field metrics float() these, so they must be present + numeric-as-str.
    _, r = play(["ESCORT"], seed=seed_for("placeholder-escort"))
    for k in ("solved", "guesses_used", "hints_used"):
        assert isinstance(r.info[k], str)
        float(r.info[k])  # must not raise
