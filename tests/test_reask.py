"""Re-asking recorded positions: the control condition must be the prompt that
was actually sent, and a free run must exercise the whole path."""

import importlib.util
import json
import random
from pathlib import Path

from catanatron.game import Game
from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import ActionPrompt
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.value import ValueFunctionPlayer

from arena.colors import FULL_TABLE
from arena.deciders import ScriptedDecider
from arena.personas import ROSTER
from arena.prompt import build_prompt
from arena.runner import bot_seat, llm_seat, run_match
from arena.scripted import heuristic_script
from arena.table_talk import TableTalk

spec = importlib.util.spec_from_file_location(
    "reask", Path(__file__).resolve().parent.parent / "scripts" / "reask.py")
reask = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reask)


def test_the_v1_prompt_is_the_line_format_the_first_matches_were_played_with():
    """`v1` is the control for the first four matches: if it left any of the fix
    in, or removed anything else, the comparison would measure the wrong thing.
    The expected line is the format arena/prompt.py produced before the fix."""
    fixed = ("  BLUE (you): 9 VP (8 that everyone can see + 1 from victory point cards "
             "only you can see), 4 settlements, 2 cities, 13 roads (longest continuous "
             "road 8, 1 knights played) | ports: SHEEP\n"
             "  RED (largest army): 4 VP, 2 settlements, 0 cities, 7 roads (longest "
             "continuous road 3, 4 knights played)")
    assert reask.v1(fixed) == (
        "  BLUE (you): 9 VP, 4 settlements, 2 cities, 13 roads | ports: SHEEP\n"
        "  RED (largest army): 4 VP, 2 settlements, 0 cities, 7 roads")


def test_the_v2_prompt_drops_the_table_information_and_nothing_else():
    """`v2` is the control for the overnight batch of 2026-09-18, played before
    roads, ports, dice and discards reached the prompt. Stripping one section
    too many would make the control a prompt nobody ever played."""
    game = Game(players=[RandomPlayer(c) for c in list(FULL_TABLE)], seed=5)
    for _ in range(120):
        game.play_tick()
    me = game.state.colors[0]
    game.state.current_player_index = 0
    game.state.current_prompt = ActionPrompt.DISCARD
    game.state.discard_counts[0] = 2
    # A hand to discard from. Whatever 120 ticks leave depends on the hash seed
    # (DESIGN.md, *Reproducibility*), and an empty hand renders no moves at all.
    for r in ("WOOD", "BRICK", "ORE"):
        game.state.player_state[f"P0_{r}_IN_HAND"] = 3
    current = build_prompt(game, me, generate_playable_actions(game.state), TableTalk(),
                           may_offer=False)
    old = reask.v2(current)
    for gone in ("\nROADS (", "\nPORTS (", "\nBANK:", "\nSINCE YOUR LAST", "\nDISCARD: "):
        assert gone in current and gone not in old
    removed = [l for l in current.splitlines() if l not in old.splitlines()]
    kept = [l for l in current.splitlines() if l in old.splitlines()]
    assert all(l.startswith("  ") or l.startswith(("ROADS", "PORTS", "BANK", "SINCE", "DISCARD"))
               or l == "" for l in removed)
    for header in ("TURN ", "BOARD (", "PLAYERS:", "TABLE TALK", "YOUR LEGAL MOVES"):
        assert any(l.startswith(header) for l in kept), header
    assert "\n\n\n" not in old


def test_the_v2_prompt_names_ports_the_way_the_overnight_batch_saw_them():
    """The port wording changed with the same step. Left alone it cost the v2
    control 5 tokens against the match's own count, found by `--measure`."""
    now = "  RED: 4 VP | ports: 2:1 wood, 2:1 ore, 3:1\n  BLUE: 3 VP | ports: 3:1"
    assert reask.v2(now) == "  RED: 4 VP | ports: 3:1, ORE, WOOD\n  BLUE: 3 VP | ports: 3:1"


def test_a_free_run_answers_every_job_and_writes_nothing_to_runs(tmp_path):
    """The dry run is how the batch is checked before it costs anything, so it
    has to go through position finding, prompting, scoring and the summary."""
    def scripted(seed):
        return ScriptedDecider(heuristic_script("trader", rng=random.Random(seed)),
                               label="dry-run")

    match = run_match(
        seats=[(Color.RED, llm_seat(ROSTER["Hard bargainer"], scripted(1))),
               (Color.BLUE, llm_seat(ROSTER["Cooperator"], scripted(2))),
               (Color.WHITE, bot_seat(ValueFunctionPlayer))],
        runs_dir=tmp_path / "runs", seed=9, max_offers_per_turn=3,
    )
    out = tmp_path / "answers.jsonl"
    reask.main(["--dry-run", "--samples", "1", "--controls", "5",
                "--out", str(out), match["path"]])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    plan, answers = rows[0], [r for r in rows if r["kind"] == "answer"]
    assert plan["kind"] == "plan" and rows[-1]["kind"] == "summary"
    assert len(answers) == plan["jobs"] > 0
    assert all("choice" in r["answer"] for r in answers)
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == [Path(match["path"]).name]



def test_an_offer_is_the_leaders_by_the_public_score_the_agent_was_shown():
    """The leader set is what the agent could see as the leader. Counting hidden
    victory point cards would put offers in it that no agent could have known
    came from the leader, and dilute exactly the effect being measured."""
    game = Game(players=[RandomPlayer(c) for c in list(FULL_TABLE)], seed=5)
    state = game.state
    for i in range(len(state.colors)):
        state.player_state[f"P{i}_VICTORY_POINTS"] = 3
        state.player_state[f"P{i}_ACTUAL_VICTORY_POINTS"] = 3
    state.current_trade = (0,) * 10 + (0,)
    state.player_state["P0_VICTORY_POINTS"] = state.player_state["P0_ACTUAL_VICTORY_POINTS"] = 5
    assert reask.offer_from_leader(state)
    # the rival is level in truth, through a card only it can see
    state.player_state["P1_ACTUAL_VICTORY_POINTS"] = 5
    state.player_state["P1_VICTORY_POINT_IN_HAND"] = 2
    assert reask.offer_from_leader(state)
    # level on the public score: nobody leads
    state.player_state["P1_VICTORY_POINTS"] = 5
    assert not reask.offer_from_leader(state)


def test_the_win_rule_condition_adds_one_rule_and_changes_nothing_else(tmp_path):
    """`win_rule` asks whether stating the win condition makes an agent wary of
    the leader. If it also moved the persona's style or touched the position,
    a change in the answers could not be credited to the rule."""
    match = run_match(
        seats=[(Color.RED, llm_seat(ROSTER["Hard bargainer"], ScriptedDecider(
                    heuristic_script("trader", rng=random.Random(1)), label="dry-run"))),
               (Color.BLUE, llm_seat(ROSTER["Cooperator"], ScriptedDecider(
                    heuristic_script("trader", rng=random.Random(2)), label="dry-run"))),
               (Color.WHITE, bot_seat(ValueFunctionPlayer))],
        runs_dir=tmp_path / "runs", seed=9, max_offers_per_turn=3,
    )
    ply = next(reask.detach(p) for p in reask.plies(match["path"])
               if p.chosen_by == "decision"
               and p.game.state.current_prompt == ActionPrompt.DECIDE_TRADE)
    sent = {}

    def capture(condition):
        def decider(system, user):
            sent[condition] = (system, user)
            raise reask.DecisionFormatError("captured")
        return decider

    for condition in ("as_played", "win_rule"):
        reask.ask(capture(condition), ply, None, "leader_offer", condition)
    (system, user), (ruled, ruled_user) = sent["as_played"], sent["win_rule"]
    rule = reask.CONDITIONS["win_rule"]["rules"]
    assert ruled_user == user
    assert rule not in system and ruled.count(rule) == 1
    assert ruled.replace(f"\n\n{rule}", "") == system
    assert ruled.index(rule) < ruled.index("YOUR STYLE:")
