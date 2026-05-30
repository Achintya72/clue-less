"""Scoring/behaviour tests for the MinuteCryptic env.

Run from the repo root:  pytest -q
These lock in the game-loop contract: every observation carries the rules and
substitutions; hints escalate from explanations to letter reveals and persist;
wrong guesses are listed back; the same wrong answer 3x ends the episode; and
reward is paid once on the terminal step so total_reward == intended score.
"""

from __future__ import annotations

import json

import env as E
from env import MinuteCrypticEnv

# A known real clue drives the deterministic assertions.
CLUE_ID = "2026-05-29"   # "That man? In love with you AND me? Despicable!" -> HEINOUS
ANSWER = "HEINOUS"

BY_ID = {c["id"]: i for i, c in enumerate(E.CLUES)}


def seed_for(clue_id: str) -> int:
    # reset() picks CLUES[seed % len(CLUES)]; the index is a valid seed.
    return BY_ID[clue_id]


SEED = seed_for(CLUE_ID)
CLUE = E.CLUES[SEED]
N_CONTENT = len(CLUE["content_hints"])   # 2 for HEINOUS
N_LETTERS = CLUE["n_letters"]            # 7
PAR = CLUE["par"]                        # 3


def play(actions, seed=SEED):
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


# --- initial observation: only clue + length, plus rules + substitutions ------

def test_initial_obs_has_rules_subs_and_length_only():
    obs = MinuteCrypticEnv().reset(seed=SEED)
    assert obs["rules"] == E.RULES
    assert obs["common_substitutions"] == E.COMMON_SUBSTITUTIONS
    assert obs["clue"] == CLUE["clue"]
    assert obs["answer_length"] == N_LETTERS
    assert obs["enumeration"] == "(7)"
    # No hints/letters/wrong guesses leaked up front.
    assert "revealed_hints" not in obs
    assert "revealed_letters" not in obs
    assert "wrong_guesses" not in obs


def test_every_step_carries_rules_and_subs():
    e = MinuteCrypticEnv()
    e.reset(seed=SEED)
    r = e.step("WRONGWORD")
    assert r.observation["rules"] == E.RULES
    assert r.observation["common_substitutions"] == E.COMMON_SUBSTITUTIONS


# --- guessing -----------------------------------------------------------------

def test_correct_first_try_full_reward():
    total, r = play([ANSWER])
    assert r.terminated and total == 1.0
    assert r.info["solved"] == "1"
    assert r.info["guesses_used"] == "1"
    assert r.info["hints_used"] == "0"
    # 0 hints is below any par>=1.
    assert r.info["par_result"] == "below"
    assert r.info["under_or_at_par"] == "1"
    assert r.info["hints_to_par"] == str(0 - PAR)


def test_messy_phrasing_still_matches():
    total, r = play([f"I think the answer is {ANSWER.lower()}."])
    assert total == 1.0 and r.info["solved"] == "1"


def test_wrong_guess_is_listed_back_and_non_terminal():
    e = MinuteCrypticEnv()
    e.reset(seed=SEED)
    r = e.step("FOOBAR")
    assert not r.terminated
    assert r.observation["wrong_guesses"] == ["FOOBAR"]
    r2 = e.step("BAZQUUX")
    assert r2.observation["wrong_guesses"] == ["FOOBAR", "BAZQUUX"]


def test_same_wrong_three_times_ends_episode():
    total, r = play(["NOPE", "NOPE", "NOPE"])
    assert r.terminated
    assert r.info["solved"] == "0"
    assert r.info["outcome"] == "repeated_wrong"
    assert r.info["guesses_used"] == "3"
    assert 0.0 <= total < 0.3   # partial credit = 0.3 * similarity


def test_distinct_wrong_guesses_do_not_trigger_giveup():
    # three DIFFERENT wrong answers should not end it (only repeats do).
    e = MinuteCrypticEnv()
    e.reset(seed=SEED)
    for g in ("AAA", "BBB", "CCC", "DDD"):
        r = e.step(g)
        assert not r.terminated


# --- hints: content first, persistent ----------------------------------------

def test_hint_reveals_content_then_persists():
    e = MinuteCrypticEnv()
    e.reset(seed=SEED)
    r = e.step("HINT")
    assert not r.terminated
    assert r.reward == 0.0
    assert r.info["hints_used"] == "1"
    assert len(r.observation["revealed_hints"]) == 1
    # hint type recorded in order; HEINOUS's first official hint is "fodder".
    assert json.loads(r.info["hint_order"]) == [CLUE["content_hints"][0]["type"]]
    # persists into the next observation
    r2 = e.step("WRONG")
    assert len(r2.observation["revealed_hints"]) == 1


def test_correct_after_one_hint_applies_penalty():
    total, r = play(["HINT", ANSWER])
    assert r.info["solved"] == "1"
    assert r.info["hints_used"] == "1"
    assert abs(total - 0.8) < 1e-9
    assert r.info["hints_to_par"] == str(1 - PAR)
    assert r.info["par_result"] == ("below" if 1 < PAR else "at" if 1 == PAR else "above")


# --- hints: letter reveals as the escalation / anti-stuck mechanism -----------

def test_hints_escalate_to_letter_reveals():
    e = MinuteCrypticEnv()
    e.reset(seed=SEED)
    for _ in range(N_CONTENT):     # exhaust content hints
        e.step("HINT")
    r = e.step("HINT")             # first letter reveal
    assert "revealed_letters" in r.observation
    assert json.loads(r.info["hint_order"])[-1] == "letter"
    # exactly one alphabetic letter shown so far
    shown = [c for c in r.observation["revealed_letters"].split() if c.isalpha()]
    assert len(shown) == 1


def test_full_reveal_guarantees_termination():
    # content hints + every letter, then one more HINT -> guaranteed solved exit.
    actions = ["HINT"] * (N_CONTENT + N_LETTERS + 1)
    total, r = play(actions)
    assert r.terminated
    assert r.info["solved"] == "1"
    assert r.info["outcome"] == "fully_revealed"
    assert r.info["hints_used"] == str(N_CONTENT + N_LETTERS)
    assert total >= E.MIN_SOLVE_REWARD - 1e-9
    assert r.info["par_result"] == "above"   # way over par


def test_can_read_and_guess_after_letters_revealed():
    # reveal everything, then actually type the answer.
    actions = ["HINT"] * (N_CONTENT + N_LETTERS) + [ANSWER]
    total, r = play(actions)
    assert r.info["solved"] == "1"
    assert r.info["outcome"] == "solved"


# --- robustness ---------------------------------------------------------------

def test_quoted_hint_is_treated_as_hint_request():
    e = MinuteCrypticEnv()
    e.reset(seed=SEED)
    r = e.step('"HINT"')
    assert not r.terminated
    assert r.info["hints_used"] == "1"
    assert r.info["guesses_used"] == "0"


def test_quoted_answer_still_solves():
    total, r = play([f'"{ANSWER}"'])
    assert total == 1.0 and r.info["solved"] == "1"


def test_chatty_hint_on_last_line_is_a_hint():
    # Real models prepend reasoning; "...\n\nHINT" must register as a hint.
    e = MinuteCrypticEnv()
    e.reset(seed=SEED)
    r = e.step("The definition is 'Despicable' and this is wordplay.\n\nHINT")
    assert not r.terminated
    assert r.info["hints_used"] == "1"
    assert r.info["guesses_used"] == "0"


def test_chatty_guess_with_prefix_solves():
    total, r = play([f"I'll parse the wordplay and read it off.\n\nGUESS: {ANSWER}"])
    assert total == 1.0 and r.info["solved"] == "1"


def test_answer_embedded_in_reasoning_solves():
    total, r = play([f"After substituting love->O, the answer is {ANSWER.lower()}."])
    assert total == 1.0 and r.info["solved"] == "1"


def test_parse_action_helper():
    assert E._parse_action("HINT")[:2] == ("hint", "")
    assert E._parse_action("blah blah\nHINT")[0] == "hint"
    assert E._parse_action("Answer: HEINOUS")[:2] == ("guess", "HEINOUS")
    assert E._parse_action("reasoning\nGuess: foo")[0] == "guess"


def test_structured_reasoning_and_action_are_separated():
    # The canonical two-line format: reasoning + action captured distinctly.
    kind, guess, reasoning = E._parse_action(
        "REASONING: 'shredded' flags an anagram of CORSET.\nACTION: HINT")
    assert kind == "hint"
    assert "anagram" in reasoning

    kind, guess, reasoning = E._parse_action(
        f"REASONING: definition is Despicable.\nACTION: {ANSWER}")
    assert kind == "guess" and guess == ANSWER
    assert "Despicable" in reasoning


def test_rambling_without_action_is_not_a_garbage_guess():
    # Model truncated mid-reasoning, never emitted ACTION: -> empty guess, not
    # the whole reasoning blob, and reasoning is still captured.
    kind, guess, reasoning = E._parse_action(
        "REASONING: hmm let me think about circuit in grand prix and the letters")
    assert kind == "guess" and guess == ""
    assert "circuit" in reasoning


def test_rambling_mentioning_answer_does_not_falsely_solve():
    e = MinuteCrypticEnv()
    e.reset(seed=SEED)
    r = e.step(f"REASONING: maybe it is {ANSWER} or something, not sure")
    assert r.info["solved"] == "0"          # no ACTION committed -> not a solve
    assert r.info["decision"] == "guess"
    assert r.info["guess"] == ""


def test_reasoning_then_bare_answer_line_solves():
    total, r = play(["REASONING: it means despicable\n" + ANSWER])
    assert total == 1.0 and r.info["solved"] == "1"


def test_info_records_reasoning_and_decision_per_step():
    e = MinuteCrypticEnv()
    e.reset(seed=SEED)
    r = e.step("REASONING: I need the definition.\nACTION: HINT")
    assert r.info["decision"] == "hint"
    assert r.info["guess"] == ""
    assert r.info["reasoning"] == "I need the definition."
    r2 = e.step(f"REASONING: it means despicable.\nACTION: {ANSWER}")
    assert r2.info["decision"] == "guess"
    assert r2.info["guess"] == ANSWER
    assert r2.info["reasoning"] == "it means despicable."
    assert r2.info["solved"] == "1"


def test_deterministic_reset_by_seed():
    a = MinuteCrypticEnv().reset(seed=SEED)
    b = MinuteCrypticEnv().reset(seed=SEED)
    assert a["clue"] == b["clue"]


def test_terminal_info_numeric_fields_are_floatable():
    _, r = play([ANSWER])
    for k in ("solved", "guesses_used", "hints_used", "hints_to_par", "under_or_at_par", "par"):
        assert isinstance(r.info[k], str)
        float(r.info[k])  # must not raise


def test_dataset_loaded_not_fallback():
    # The real dataset has hundreds of clues; the fallback has one.
    assert len(E.CLUES) > 100
