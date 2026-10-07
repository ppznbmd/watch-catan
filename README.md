# watch-catan

Several agents — one model, different personas — play Settlers of Catan against
each other and against an algorithmic bot, and negotiate in plain language while
they do it. The point is not a stronger Catan bot. It is watching how agents
behave under competition, and specifically **how they bargain**.

A match costs about **$0.26** and takes about **21 minutes** with `gpt-5.6-luna`
(mean of eight matches; range $0.17-0.41 and 14-34 minutes).

---

## What it has found so far

- **Lying when it pays.** Across eight baseline matches, 279 table-talk claims
  could be checked against the speaker's real hand, and none was a lie. So the lie trap (`scripts/lie_trap.py`) builds the moment that
  would make one worth telling: a player one card from winning, with the table
  unable to see it. DeepSeek Flash lies in 18% of the offers that win it the game
  (14 of 79), mostly with a false motive ("just rounding out my hand").
- **Lying when watched.** The same trap on 100 new positions, with one added
  sentence. Told the game is part of a study of honesty, Flash reveals the win in 39 of 54 winning offers
  (0 of 61 without the sentence) and lies in 2 (10 of 61). Telling it the game
  tests skill, or that nobody is watching, changes nothing. The full analysis is
  [`experiments/awareness-study.md`](experiments/awareness-study.md), generated
  from committed data by `scripts/awareness_study.py` and pinned by a test.

The verdicts behind those numbers (lie, mistake, spin, omission...) are one
judge's hand reading of every case, against definitions written in the scripts,
and are committed next to the answers they judge.

---

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # then put your API key in it
./scripts/check-env.sh        # confirms which keys are set, never prints a value
```

The default model is `gpt-5.6-luna`, so you need `OPENAI_API_KEY`. DeepSeek and
Anthropic keys are optional — only needed if you seat those models.

---

## Run a match

```bash
.venv/bin/python -m arena.play --bot value
```

Three personas and one baseline bot. The match narrates itself as it plays:

```
  RED     offer 2 wheat for 1 brick, 1 sheep
          "I'm holding at two wheat for one brick and one sheep. That's a
           strong return, and I'm not discounting it further."
  ORANGE  reject the offer on the table
          "I need the brick myself. Pass."
```

Useful flags:

| flag | what it does |
|---|---|
| `--dry-run` | scripted agents instead of a model — **spends nothing**, finishes in a second |
| `--seed 7` | reproducible board and dice |
| `--bot value` | add the strong baseline bot, which answers trade offers on their merits. `--bot value-stock` is Catanatron's own, which rejects every offer; `--bot weighted` a weak one. Repeatable |
| `--model X` | use model X for every seat that does not name its own |
| `--effort none` | cheapest reasoning setting. Left unset, the provider's default applies |
| `--max-offers 1` | how many trade offers a player may put on the table per turn (default 2) |
| `--quiet` | summary only, no play-by-play |

Start with `--dry-run` to see the shape of the thing without spending anything.

---

## Read the result

Every match ends with a summary:

```
winner: WHITE in 41 turns (502.5s, 36 offers made)
  RED     Hard bargainer   gpt-5.6-luna    3 VP   32 calls  0 unusable  0 fallbacks
  BLUE    Cooperator       gpt-5.6-luna    3 VP   34 calls  0 unusable  0 fallbacks
  ORANGE  Quiet builder    gpt-5.6-luna    2 VP   17 calls  0 unusable  0 fallbacks
  WHITE   ValueFunctionPlayer  bot        10 VP    0 calls
                                                          total: $0.0743
```

- **calls** — how often that seat asked the model. Most decisions never do; see
  *The gate* below.
- **unusable** — replies the model sent that could not be used (empty, wrong
  shape, truncated). Each gets one retry. **Watch this column.** A model that
  cannot hold the schema quietly becomes the fallback policy wearing a persona.
- **fallbacks** — decisions that gave up on the model and used the cheap policy.
- **calls saved** — times the engine asked a player about its own trade offer and
  we answered for it without asking the model. This counts the workaround
  *working*, not a problem: these calls were never billed. A Catanatron quirk,
  explained in DESIGN.md. Seats that sit first in the engine's shuffled order
  never hit it, so a `0` there is normal, not a broken guard.

`calls` counts every request that reached the provider, including the ones that
came back unusable — those are billed too, and a cost report that hides them
understates the bill by exactly the failure rate.

Every match also appends a JSONL file to `runs/`, one line per event, written as
it happens. That file is the replay and the dataset — every decision, its
reasoning, and everything said at the table.

What happened to every trade offer — repeated, improved, withdrawn, and whether
anyone ever changed their answer — comes from reading that file:

```bash
.venv/bin/python scripts/offers.py runs/<match>.jsonl [more.jsonl ...]
.venv/bin/python scripts/offers.py --turns runs/<match>.jsonl   # every offer, turn by turn
```

It separates a rejection a model *chose* from one it was forced into because it
could not pay, and from the bot's. Counted together, "nobody would take it" and
"nobody could take it" become the same number.

---

## Watch it

```bash
.venv/bin/python scripts/viewer.py      # then open http://127.0.0.1:8765
```

The board, every hand, and the negotiation as it was spoken. Pick a run from the
menu, scrub with the slider, or step a ply at a time with the arrow keys.

The four seats sit in the corners of the board, in the engine's turn order, so
the screen is laid out like the table it is showing — which leaves the whole
sidebar for the conversation, the part worth reading.

Each tile carries what it produces and how often — forest, hills, pasture, fields,
mountains, with the number token and its pips below. A settlement is a house with
a pitched roof and a city is a flat-topped tower beside a hall: they are told
apart by their outline, not by being slightly different sizes. Hands are shown as
cards, a trade as the cards on both sides of it, and a roll as the two dice that
were thrown. All of it is inline SVG in `viewer/icons.js`: no image files, no icon
font, nothing fetched.

**A match in progress is the same thing.** The viewer follows the file a match is
writing, so start it before, during or after a match and it catches up either
way — including a match whose process has since died. There is no streaming
server to keep alive: a run file is append-only and line-buffered, so following
is just reading it again.

What the viewer shows that the terminal cannot:

- **what was said against what was thought.** The public pitch is set in serif,
  the private reasoning sits under it in small type. They are never merged — the
  gap between them is the whole point.
- **the hands.** Everyone's cards, at every ply. The agents never see this; see
  *Hidden information* in DESIGN.md.
- **degradation.** A seat whose replies came back unusable, or fell through to
  the cheap policy, carries a count. A failing model still produces a complete,
  plausible match, so this has to be visible.

`?run=<path>&at=<ply>` opens a specific position, which is how you point someone
at the trade you are arguing about. `?theme=light` or `dark` overrides the
system setting.

---

## Seats, personas, models

A seat is a `model:persona` pair, so different models can sit at the same table:

```bash
# same persona, two models -> compares the models
.venv/bin/python -m arena.play --bot value \
    --seat gpt-5.6-luna:"Hard bargainer" \
    --seat deepseek-flash:"Hard bargainer" \
    --seat gpt-5.6-luna:"Cooperator"
```

Drop the `model:` prefix and the seat uses `--model`. Catan seats 2 to 4 players
in total, bots included.

The personas live in `arena/personas.py` and differ **only** by system prompt —
the model is held constant so that any difference in behaviour is attributable
to the prompt:

| persona | seat | in one sentence |
|---|---|---|
| Hard bargainer | RED | never takes the first offer |
| Cooperator | BLUE | trades readily, keeps score of favours |
| Quiet builder | ORANGE | rarely opens a negotiation |
| Plain | GREY | no style instructions: the control |

Plain gets the rules and nothing else. It is how to tell a persona's behaviour
from what the model does anyway. It is not in the default table; seat it with
`--seat Plain`.

A persona always sits in its own colour, whatever order the seats are given in,
because agents address each other by colour and a persona that moved would be a
different player to everyone at the table. The bot always sits on WHITE. Only a
second copy of a persona, or a second bot, takes a colour that is left over.

GREY is not one of Catanatron's colours; `arena/colors.py` adds it. A table still
seats at most four.

They separate visibly in a single match. In the run above: Cooperator made 22
offers, Hard bargainer 13, Quiet builder 1.

### Models

The provider comes from the model name, or write it explicitly as
`provider:model`:

| provider | names | API key |
|---|---|---|
| openai | `gpt-*`, `o1/o3/o4-*` | `OPENAI_API_KEY` |
| deepseek | `deepseek-*` | `DEEPSEEK_API_KEY` |
| anthropic | `claude-*` | `ANTHROPIC_API_KEY` |

Three ways to change the default, by how long it sticks:

```bash
.venv/bin/python -m arena.play --model deepseek-flash   # this run
WATCH_CATAN_MODEL=claude-haiku-4-5 .venv/bin/python ...  # this shell
# WATCH_CATAN_MODEL=... in .env                          # every run
```

An unknown model name and a missing API key both fail before the board is built,
not thirty decisions in.

---

## Tests

```bash
.venv/bin/python -m pytest tests/ -q      # no tokens spent
.venv/bin/python scripts/soak.py -n 100   # 100 games in ~12s, proves the harness
```

Nothing in the suite calls a real API. The provider path is not mocked either:
`tests/fake_provider.py` runs an OpenAI-compatible HTTP server on a loopback
socket and `test_match_over_the_wire.py` plays a whole match through it,
including a run where every third reply is deliberately broken.

One real call, with a real board, to check an API key works:

```bash
.venv/bin/python scripts/smoke.py          # costs a fraction of a cent
```

---

## How it works

**The gate** (`arena/llm_player.py`). Catan is 200+ decisions a match, most of
them forced. The model decides opening placement, what to build, the robber, and
every part of a negotiation. Catanatron's own `WeightedRandomPlayer` takes the
rest. Measured: tens of model calls per agent, not hundreds.

**Authored offers** (`arena/prompt.py`). Catanatron never lists a trade offer
among the legal moves, but a player is allowed to return one anyway. So an agent
composes any offer it likes across the full resource space, and the engine needs
no fork. Every offer is checked against the agent's real hand first.

**Table talk** (`arena/table_talk.py`). The engine moves resources; this moves
words, outside engine state, never affecting what is legal. It is public — every
agent reads every pitch and reply — and nothing said is binding. The gap between
what an agent says and what it then does is the most interesting thing here.

**Counter-offers** are emulated, because the engine has none: a responder may
only accept or reject. It rejects *with words*, the proposer reads the transcript
on its next decision in the same turn, and offers again against what it heard.

**Bad replies are expected.** The model picks an index into a rendered list, so
an index cannot be a malformed action — but it can be out of range, and an offer
can name cards the agent does not hold. Both get one retry with the complaint
attached, then fall back to the cheap policy. A match never dies on a bad reply.

Design decisions, engine archaeology and the reasoning behind all of the above
are in [DESIGN.md](DESIGN.md).

---

## Known issues

Three defects found by the first real match are fixed; [DESIGN.md](DESIGN.md)
says what they were and how they are handled. Still open:

- **The agents do not see the map.** Nodes, edges and tiles are listed by id,
  with no adjacency, so a path or a rival closing in has to be worked out from
  numbers. See DESIGN.md, *What an agent knows*.
- **Road descriptions mislead.** A road is described by the node it reaches,
  not by what it opens two steps ahead or what it does to the longest road. In
  one match this kept a seat from building for 40 turns.
- **The prompt does not say** that opening placements and Road Building roads
  are free, or when a turn's offers are spent.
- **`cost_report` overstates DeepSeek costs.** It prices a seat's summed usage
  at the peak rate; one match printed $1.87 and was billed $1.17.

---

## The data

Everything the results rest on is committed, so they can be checked without
spending anything.

- `runs/` holds every real match played, one JSONL event per line: each
  player's real hand next to what it said and its private reasoning. Open any of
  them in the viewer. They were played under prompt versions v1 to v4 and are
  not comparable across versions: the first four are v1, the next four v2, and
  from v3 on the version is in the `game_start` line as `prompt_version`.
  `runs/interrupted/` holds one match cut short by a reboot.
- `experiments/` holds the probes that ask a model about recorded positions
  instead of playing whole matches: the win probe, the lie trap and its framing
  batches, the loss probe, the counter probe and re-asks of match positions. The
  `*-verdicts*.jsonl` files are the hand-judged readings; the `.html` files are
  the review pages they were judged in.
- The `.pkl` files are Python pickles of game positions. Unpickling runs code,
  so only load ones you trust, like the ones in this repository.

---

---

## Layout

```
arena/personas.py     the personas (system prompts only) and their seat colours
arena/colors.py       adds GREY to the engine's four colours
arena/prompt.py       game state -> text, and the reply -> a legal action
arena/table_talk.py   the negotiation transcript, counters, the per-turn offer budget,
                      and what each public event moved
arena/bot.py          the baseline bot, taught to answer trade offers
arena/reflex.py       the cheap policy, wrapping Catanatron's WeightedRandomPlayer
arena/deciders.py     the model call, per provider, behind one interface
arena/llm_player.py   the agent: gate, prompt, parse, retry, fallback
arena/events.py       the JSONL sink and its subscribers
arena/runner.py       one match, ply by ply
arena/replay.py       a run file -> board positions, live or finished
arena/offers.py       a run file -> every trade offer, its answers and revisions
arena/rebuild.py      a run file -> the live engine game at every ply, verified
arena/play.py         the CLI
arena/env.py          reads .env, no dependency
viewer/               the spectator UI: board, hands, table talk
scripts/              smoke test, soak test, env check, live watcher, viewer, offer report,
                      and the probes: win_probe, lie_trap, loss_probe, counter_probe,
                      reask, lie_detector, awareness_study
tests/                195 tests, none of which spends a token
runs/                 every real match, as played
experiments/          probe answers, positions and verdicts
```

[CLAUDE.md](CLAUDE.md) is the working brief for the coding agent this project
was built with: the mistakes already made once, and what is next.

## License

GPL-3.0-or-later; see [LICENSE](LICENSE). Catanatron is GPL-3.0-or-later, so
importing it makes this project GPL on distribution.
