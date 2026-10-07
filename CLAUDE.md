# Working on watch-catan

Read [README.md](README.md) to run it and [DESIGN.md](DESIGN.md) for why it is
built this way. This file is only about how to work here, and is mostly a list of
mistakes already made once.

## The one thing to internalise

**Measure before you assert.** This project is about observing behaviour, and
every confident claim made from reasoning alone in its first session turned out
wrong:

- predicted a seat bias from the bot's value function; measured 60 matches with
  the bot rotated through every seat; **no detectable effect**
- concluded from the live log that "the bot is never consulted about offers"; it
  was consulted 17 times and simply **not logged**
- estimated match cost from token guesses; **off by 3x**
- extrapolated match duration from two samples, both from the opening phase,
  where decisions are 5x slower than the rest; **off by 2x**
- analysed output piped through `head` and nearly shipped a false finding —
  then did it again two hours later, concluding from a truncated `ls` that a file
  was missing. **Never pipe a listing or a log you are drawing a conclusion from
  through `head`.** Count it, or read all of it.

The data is cheap: `--dry-run` costs nothing, `scripts/soak.py` runs 100 games in
12 seconds, and a real match is 25 cents. There is no excuse for a guess. When you
catch yourself about to write "probably" or "should be", go measure it instead.

## Hard traps

**Catanatron comes from a git commit, not PyPI.** `pip install catanatron` gets
3.2.1, which has no player-to-player trading — the thing this project exists to
watch. `requirements.txt` pins the SHA. If trading suddenly does not exist,
check what is installed before debugging anything else.

**Read the installed package, not the repo on GitHub.** They differ. The web repo
showed `json.py`/`GameEncoder`; the installed 3.3.0 has `serialization.py` with
`web_view`/`geometry`. `.venv/lib/python3.*/site-packages/catanatron/` is the
source of truth.

**Never trust an aggregator blog for model pricing.** A wrong price from a
comparison site shipped into `PRICES` in `arena/play.py` and stayed there for
several turns. Go to the provider's own docs. A model with no price on file
prints no dollar figure, which is correct — never invent one.

**`state.colors` is not the order seats were constructed in.** `State.__init__`
shuffles. Anything that reasons about seat order must read `state.colors`.

## Secrets

Never print `.env` or any API key, not even to check it is set — a key that
reaches a terminal has to be treated as leaked, and one already was here. Use
`./scripts/check-env.sh`, which reports presence and length only.

## Tests

`.venv/bin/python -m pytest tests/ -q` — none of the tests spend a token.
Adding a test that calls a real API is a bug.

The provider path is **not** mocked: `tests/fake_provider.py` serves a real
OpenAI-compatible HTTP endpoint on a loopback socket. A new provider, a new
parameter, a new failure mode — all of them get a wire test there. Two of this
project's three real defects were found by a test that deliberately returns junk,
not by a test of the happy path.

`--dry-run` is a test surface, not a toy. The scripted agent in `arena/scripted.py`
reads the same rendered prompt the model gets and locates its moves by their
rendered description — so an unreadable prompt breaks the dry run. Keep it that
way.

## Spending money

A real v3 match is ~21 minutes and ~$0.26 with `gpt-5.6-luna` (mean of the
eight baseline matches, range 14-34 min and $0.17-0.41; v2 was 30 min and $0.26
at list price too; the very first estimate,
$0.07/10 min, was off by 3x, and the v3 estimate of $0.29/25-30 min was off the
other way). Until 2026-09-18 it read $0.23, priced at the list input rate:
GPT-5.6 writes each uncached prompt to its cache at 1.25x that rate, a match
never reuses one, and the billing dashboard caught it to the cent (DESIGN.md,
*Prompt caching*). Matches no longer write the cache, so a new one should come
back to ~$0.23; the first to be played is the check. Output is ~46% of the bill, and trade decisions
account for ~44% of the output. Before a batch, say what it will cost and
how long it will take, then ask — and offer to set up monitoring in the same
breath, not after being asked how it is going. `scripts/watch_run.py` emits
milestones from a live run; its filter must cover crashes and stalls, because
silence has to mean healthy.

`scripts/smoke.py` makes exactly one real call with a real board. Run it before
committing to anything long.

A monitor that restarts — the Monitor tool expires every 30 minutes, usually
mid-match — must recover which match it is in from what it already read, or it
ignores the result when it arrives. Two results were lost that way. Batch
scripts kept in `/tmp` do not survive a reboot, and one happened mid-batch.

## `runs/` is data. Never touch it in bulk.

A real match costs money and twenty minutes and is the only copy of what
happened. Two were destroyed here in one session:

1. `rm -f runs/*.jsonl` in a cleanup line, two messages after the file was cited
   in the docs as the dataset to design against;
2. `mv runs/2026*.jsonl runs/scratch/` — a glob run **while a real match was
   still writing**, which swept it into the directory that had just been declared
   disposable and gitignored. The command implementing "keep real matches" is
   what threw one away.

So: **never `rm`, `mv` or glob over `runs/`.** Not to clean up, not to reorganise,
not "just the old ones". Dry runs already write to `runs/scratch/`; if you need a
clean slate, clean that. Before touching any file under `runs/`, read its
`game_start` line and see whether a model played it — a real match has a `model`
on its seats and non-zero token usage.

Check `pgrep -f arena.play` first. A match in flight keeps writing through a
rename, so moving its file hides a live run rather than stopping it.

## If it is not in the event log, it did not happen

`runs/*.jsonl` is the product, not a side effect — it is the replay, the dataset,
and the only evidence of what an agent did. Every new actor or decision path must
emit. The bot seats went unlogged for a whole session and produced a wrong
conclusion about the engine.

The same applies to counters in the summary. `unusable` and `fallbacks` exist to
catch silent degradation: a model that fails often still produces a complete,
plausible-looking match, because the reflex policy fills every gap. Anything that
could hide that must surface instead.

## Voice

Comments and docstrings explain **why**, not what — the reasoning that would
otherwise be lost, especially where the code works around an engine quirk. Match
the existing density; do not add narration to obvious lines. No emoji, no
decorative headers.

Test names are sentences stating the invariant
(`test_the_proposer_is_never_asked_about_its_own_offer`), and each test's
docstring says what would break in the real world if it failed.

Deliverable documents are written in English. Chat with the user is in Portuguese.

## The viewer is checkable; check it

`scripts/viewer.py` serves `viewer/` and the frames `arena/replay.py` builds. Do
not claim a page renders without looking at it. There is no playwright package
installed, but its chromium is on disk and takes a screenshot on its own:

```
~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome --headless --disable-gpu \
  --no-sandbox --screenshot=out.png --window-size=1440,900 \
  --virtual-time-budget=6000 "http://127.0.0.1:8765/?run=<path>&at=<ply>&theme=light"
```

`?run=&at=&theme=` exist partly so a specific position can be rendered without
driving the UI. Two defects in the first pass were visible only this way: a class
`display` silently beating `hidden`, and the feed staying at ply 0 while the
board moved. Reading pixels out of the PNG beats reading the PNG by eye — an
apparently mixed light/dark render turned out to be correct when sampled.

## The journal

`JOURNAL.md` is the diary of the experiment, written in the user's first person
and meant to become a blog post. It, `ideas.txt` and `insights.md` are kept
local and gitignored, and so is `posts/`. At the end of a session with a
finding, a surprise or a change of direction, offer to add a dated entry: what
the user wanted and why (their own words are the best source), what was tried,
what turned out differently. A belief that measurement overturned goes in
*Things I believed that turned out wrong*. Numbers in it are measured, like
everywhere else here.

## What is next

State at the end of the 2026-09-18 session, and the order worth following.

**0. The v3 baseline is played.** Eight matches, 18 September 11:18-14:25, all
clean (0 unusable, 0 fallbacks): Hard bargainer 4, Cooperator 2, Quiet builder
2, bot 0. Run files `runs/20260918-1118*` through `runs/20260918-1404*`, batch
log `runs/batch-20260918-1118.log`. A ninth match, interrupted by a reboot at
turn 48, is in `runs/interrupted/` and is not part of it. The user is analysing
the eight separately; DESIGN.md, *Under v3*, has the headline numbers. Groups:
v1 (4 matches), v2 (4), v3 (8) — never pool them.

**1. Prompt v4 is in use; what it still lacks.** v4 (2026-09-24) only replaced
the `reasoning` summary "for the spectator" with a private `note`: asking for a
summary told the model it had a reader. Stored and logged as `reasoning` still.
The first four-Flash match under it (`runs/20260924-204029-1d45dd.jsonl`,
JOURNAL.md *24 Sep, evening*) showed what to fix next, in this order, each
tested on recorded positions before any match:

- road descriptions: say what a road opens two steps ahead. "Nobody may ever
  settle" at the node reached kept ORANGE from building for 40 turns (20 cards
  discarded) and read to BLUE as a hint from a test designer. ORANGE's turn 35
  is the first test case; the win probe's positions are the second.
- say that opening placements and Road Building roads are free (all four seats
  tried to pay for them; two tried to trade during the opening);
- say when the turn's offers are spent (invalid third offers, one fallback);
- stray characters in `say`. Still open by design: the agents do not see the
  node graph, and memory older than the last decision. The note they reread
  also makes a mistake stick ("Same as before").

`cost_report` in `arena/play.py` prices a seat's summed usage at DeepSeek's
peak rate: the match above printed $1.87 and cost $1.17. Price per call.

The working tree has moved on from tag `baseline-v3` (DeepSeek budget and
timeout in `arena/deciders.py`, a Terra price, model-comparison work in
`scripts/reask.py` and `scripts/win_probe.py`). For `gpt-5.6-luna` calls it is
identical, which is why the replacement eighth match was played on it.

**1b. Comparing models: `scripts/win_probe.py`.** Four value bots play fresh
games; positions where building a road, and no other kind of move, wins on the
spot are pickled (`experiments/win-positions-20260918.pkl`, 166 of them) and
asked of any model. A win is the one move whose rightness is not a judgement
call. On 150 positions × 3: deepseek-flash 61%, gpt-5.6-terra 45%, gpt-5.6-luna
33%, every gap well outside noise; `win_probe.py report` reproduces every number.
Flash's reasoning is the skill: thinking off or effort `low` lost most wins on a
5-position preview. Flash has played two matches (18 Sep, one seat; 24 Sep, all
four, v4). The prompt's road
descriptions (the node reached, "nobody may ever settle") point away from the
winning road; stating what a road does to the longest road is a v4 candidate,
to be tested here before any match.

**1c. The lie trap: `scripts/lie_trap.py`.** Flash lies in 18% of the offers
that win it the game (14 of 79 on unseen positions), mostly a false motive
("just rounding out my hand"), and deceives far more often with true words.
Verdicts are a single judge's reading (Claude's; the user left the call to it)
against definitions in the script; the rule-based `classify` found 8 of 14 and
must not be used to count. Every decision logs the provider's trace as
`thinking`; judge intent from that, never from `reasoning`.

**1d. Test awareness and framing (`ideas.txt` 10).** Written up as a study in
`experiments/awareness-study.md`, generated by `scripts/awareness_study.py`
from committed files and pinned by `tests/test_awareness_study.py`; after any
verdict or analysis change, regenerate it. Done so far (JOURNAL.md,
24 Sep evening and 25 Sep): the trap's traces suspect a test in 26 of 273 (the
old "8" was a regex), a match's in 8 of 362; with trace length held equal the
two do not differ, so suspicion follows deliberation, not the probe format.
Then `lie_trap.py ask --frames` on 100 new v4 positions
(`experiments/framing-plan.md`, written before the first call): told the game
studies honesty, Flash reveals the win in 39 of 54 winning offers (0 of 61
unframed) and lies in 2 (10 of 61); skill and "unobserved" frames change
nothing. Verdicts are Claude's hand reading of all 236 winning offers
(`experiments/lie-trap-framing-verdicts-b*.jsonl`); the classifier misses
nearly every reveal. Open: an independent judge; the control condition under
the frames (not yet asked); whether lying runs longer within a position (17 of
25, p = 0.11). When splitting positions by difficulty, measure difficulty from
answers other than the one being counted: a circular split produced a false
finding here once.

**2. Find out why the building gap moved.** In v1 the agents built less than half as much as the bot. Measured
and unexplained (DESIGN.md, *Why the agents lose to the bot*). It is **not** the
development cards: a simulation says the agents' dev-card habit wins more, not
less. Free to investigate with `arena/rebuild.py`: do their roads lead to
buildable spots, do they use the bank when a build is one trade away, are they
cut off. Under v3 the agents build 87% of what the bot does, up from 38% in v1,
but the gap closed at v2; seeing roads and ports did not add to it. The bot's
own building fell and is unexplained.

**3. Test prompt changes on recorded positions before playing matches.**
`scripts/reask.py` asks the model again about positions it already played, under
each prompt version (`as_played`, `v1`, `v2`, `current`); a run's control is
`as_played`. Bump `PROMPT_VERSION` in `arena/prompt.py` with any prompt change. Drop its `dev_buy` set — it assumed buying was a
mistake, and the simulation says otherwise. Always `--dry-run`, then `--measure`
(3 calls), before a batch.

**4. Then the ideas in `ideas.txt`,** in this order: 3 (post-match reflection,
cheap, and the material for 4), 7 (strong vs weak model), 4 (memory between
matches, with a no-memory control on the same seeds), 5 and the saboteur, the
human player last. Ideas 2 (fixed colours), 8 (offer analysis) and the Plain
persona are done.

The open research questions, in the order they are worth answering:

1. Do trades with the baseline bot leave the agents better or worse off? It never
   initiates, and from v3 it accepts whatever helps itself, with no thought for
   the proposer, so a naive table may collectively feed the leader. Measurable
   as bot win rate at `--max-offers 2` versus `0`. Before v3 the question had no
   answer: the stock bot rejected every offer by tie-break (338 of 338).
2. Does an agent's stated intention match what it then does? From v3 each agent
   rereads its own last note, which makes it more consistent by construction;
   do not compare this across the v2/v3 line. Every pitch is cheap
   talk and the log holds both halves.
3. Do the personas separate on outcomes, not just on behaviour? They separate on
   offers made in a single match; win rate needs many more.
