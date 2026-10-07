"""Putting a counter-offer to an agent on a position it already played: the
position must be the one it played, and the counter must read like one."""

import importlib.util
import json
import random
from pathlib import Path

import pytest
from catanatron.models.player import Color
from catanatron.players.value import ValueFunctionPlayer
from catanatron.state_functions import get_player_freqdeck

from arena.deciders import ScriptedDecider
from arena.personas import ROSTER
from arena.rebuild import detach
from arena.runner import bot_seat, llm_seat, run_match
from arena.scripted import heuristic_script

spec = importlib.util.spec_from_file_location(
    "counter_probe", Path(__file__).resolve().parent.parent / "scripts" / "counter_probe.py")
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)


@pytest.fixture(scope="module")
def match(tmp_path_factory):
    def scripted(seed):
        return ScriptedDecider(heuristic_script("trader", rng=random.Random(seed)), label="dry-run")
    tmp = tmp_path_factory.mktemp("probe")
    return run_match(
        seats=[(Color.RED, llm_seat(ROSTER["Hard bargainer"], scripted(1))),
               (Color.BLUE, llm_seat(ROSTER["Cooperator"], scripted(2))),
               (Color.WHITE, bot_seat(ValueFunctionPlayer))],
        runs_dir=tmp / "runs", seed=9, max_offers_per_turn=2,
    )


@pytest.fixture(scope="module")
def positions(match):
    found = cp.find_positions([match["path"]])
    assert found, "the scripted match left no offer alone; pick another seed"
    return found


def test_the_worse_counter_asks_exactly_one_card_more_than_the_offer():
    """The claim under test is "I'm not paying a premium". A counter asking two
    cards more is a different question, and one asking the same is `same`."""
    offer = {"give": (0, 0, 2, 0, 0), "want": (0, 0, 0, 0, 1)}
    assert cp.premium(offer, (0, 0, 3, 0, 0)) == [0, 0, 3, 0, 0]
    # all its sheep is already offered: the extra card is the one it has most of
    assert cp.premium(offer, (1, 0, 2, 4, 0)) == [0, 0, 2, 1, 0]
    assert cp.premium(offer, (0, 0, 2, 0, 5)) is None  # ore is the card it wants


def test_the_counter_is_a_move_and_takes_the_place_of_its_authors_reply(positions):
    """Only a counter listed among the legal moves can be taken up, and one
    printed under the author's "I'll pass" is a table nobody sat at."""
    ply, offer = positions[0]
    copy = detach(ply)
    counter = cp.inject(copy, offer, "worse")
    assert counter is not None
    _, user = copy.prompt()
    assert f"take up {counter.color}'s counter-offer" in user
    talk = copy.talk.utterances
    pitch = max(i for i, u in enumerate(talk) if u.kind == "pitch" and u.color == offer["color"])
    assert [u.kind for u in talk[pitch + 1:] if u.color == counter.color] == ["counter"]
    assert sum(counter.want) == sum(offer["give"]) + 1


def test_putting_a_counter_never_changes_the_position_it_was_drawn_from(positions):
    """The counter's author may be dealt the card from the bank. On the shared
    position, that would leak into the next condition asked on it, and `same`
    would be asked on a board `worse` had already changed."""
    ply, offer = positions[0]
    before = ([tuple(get_player_freqdeck(ply.game.state, c)) for c in ply.game.state.colors],
              tuple(ply.game.state.resource_freqdeck), len(ply.talk.utterances))
    for condition in cp.CONDITIONS:
        cp.inject_ok(ply, offer, condition)
        cp.inject(detach(ply), offer, condition)
    after = ([tuple(get_player_freqdeck(ply.game.state, c)) for c in ply.game.state.colors],
             tuple(ply.game.state.resource_freqdeck), len(ply.talk.utterances))
    assert before == after


def test_a_free_run_answers_every_job_and_writes_nothing_to_runs(match, tmp_path):
    """The dry run is how the batch is checked before it costs anything."""
    out = tmp_path / "answers.jsonl"
    cp.main(["--dry-run", "--samples", "1", "--controls", "3", "--out", str(out), match["path"]])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    plan, answers = rows[0], [r for r in rows if r["kind"] == "answer"]
    assert len(answers) == plan["jobs"] > 0
    assert all("choice" in r["answer"] for r in answers)
    assert [p.name for p in Path(match["path"]).parent.iterdir()] == [Path(match["path"]).name]
