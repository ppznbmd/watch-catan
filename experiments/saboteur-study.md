# The saboteur study

2026-10-07 to 2026-10-08. What happens to a table when one player's goal is not
to win but to stop whoever is closest to winning?

## Design

**The persona.** `Saboteur` in `arena/personas.py`, seat colour BLACK. Its
instructions: each turn, find who leads on public points or is about to take
longest road or largest army, and pick the move that sets them back most, even
when another move would help it more. Rob them, refuse their offers, build where
they want to expand, and trade with players well behind. It may still win: "the
only result you are trying to prevent is someone else winning" (the user's
choice, following `ideas.txt` 1).

**Pairs.** Each seed is played twice: {4 Plain} as control, and {3 Plain +
Saboteur}. The seed fixes the board and the seating, so the Saboteur sits
exactly where the fourth Plain (ORANGE) sits in the control, and the two tables
differ in that seat's goal alone until the first move that differs. Every seat
is `gpt-5.6-luna` at effort `high`. Plain, not the other personas, so that no
style is mixed into the comparison.

**Outcome.** Measured in turns, not minutes: wall-clock time also follows the
API tier and the provider's load at that hour.

## Manipulation check

Before any match, the persona was checked on recorded positions
(`scripts/reask.py`, conditions `plain_high` and `saboteur_high`, one answer per
position, Luna at `high`), from the eight v3 matches and the four-Flash v4
match. The same position is asked under each persona, so any difference is the
persona's.

| decision | Plain | Saboteur, first draft | Saboteur, final |
|---|---|---|---|
| accepts an offer from the public leader | 23/72 | 0/72 | **0/72** |
| accepts an offer from anyone else | 34/72 | 9/72 | **25/72** |
| accepts an offer from the sole last-placed player | 8/14 | 3/14 | **8/14** |
| robs the leading opponent | 60/84 | **83/84** | (sentence unchanged, not re-asked) |

The first draft said only "trade with the players furthest behind" and was read
as "help nobody": it refused the last-placed player as often as the leader. The
trade sentence was rewritten ("Trades are a weapon: refuse anything that helps a
player near the lead, and accept fair offers from players well behind"), after
which it trades with trailing players as Plain does (−12 points, 95% −26 to
+1) and still refuses the leader every time. Files:
`experiments/reask-saboteur-check-20261008.jsonl` (first draft, both
conditions), `experiments/reask-saboteur-check2-20261008.jsonl` (final, Saboteur
only; same positions). Cost $0.81.

## The matches

All twelve clean: 0 unusable replies, 0 fallbacks. Pair 201 ran at the standard
tier, pairs 202-206 at flex. "4th seat VP" is ORANGE in the control and BLACK
with the Saboteur. Cost is priced call by call (the printed match summary
ignores flex and shows twice the real figure).

| seed | table | winner | turns | min | cost | calls | offers | 4th seat VP | run file(s) |
|---|---|---|---|---|---|---|---|---|---|
| 201 | control | GREY | 74 | 86 | $0.64 | 394 | 95 | 6 | `20261008-004542-1c2739` |
| 201 | saboteur | RED | 124 | 103 | $0.85 | 557 | 156 | 3 | `20261008-021121-6e3284` |
| 202 | control | GREY | 109 | 135 | $0.47 | 564 | 145 | 8 | `20261008-035637-4977d4` |
| 202 | saboteur | BLUE | 103 | 88 | $0.37 | 463 | 117 | 8 | `20261008-061124-795fe0` |
| 203 | control | RED | 68 | 55 | $0.27 | 352 | 86 | 4 | `20261008-073925-6e7c07` |
| 203 | saboteur | BLUE | 89 | 91 | $0.38 | 477 | 97 | 4 | `20261008-083417-1509e0` + `20261008-102858-f3d9be` (resumed) |
| 204 | control | BLUE | 116 | 107 | $0.44 | 580 | 136 | 9 | `20261008-105159-003fcb` |
| 204 | saboteur | GREY | 119 | 116 | $0.49 | 608 | 148 | 5 | `20261008-123841-2204b2` |
| 205 | control | ORANGE | 94 | 79 | $0.37 | 474 | 131 | 10 | `20261008-143508-807131` |
| 205 | saboteur | RED | 99 | 64 | $0.30 | 413 | 103 | 5 | `20261008-155407-ded986` |
| 206 | control | RED | 68 | 55 | $0.27 | 339 | 77 | 6 | `20261008-165808-7466e2` + `20261008-173825-5f569e` (resumed) |
| 206 | saboteur | BLUE | 101 | 86 | $0.38 | 482 | 130 | 3 | `20261008-175622-030776` |

Two matches were cut by reboots (seed 203's saboteur at turn 75, seed 206's
control at turn 55) and finished with `arena.play --resume`, which restarts at
the roll opening the last logged turn, with each seat's note and history marker
restored and fresh dice; the resumed file's `game_start` carries
`resumed_from`. Batch logs: `runs/batch-saboteur-20261008-0045-seed201.log`,
`runs/batch-saboteur-study-20261008-0200.log`,
`runs/batch-saboteur-study-20261008-1028.log` and
`runs/batch-saboteur-study-20261008-1738.log`.

## Results

**1. The Saboteur scores less than a plain player in its seat.** Mean 4.7
points against ORANGE's 7.2 in the same seat of the control: −2.5 points (95%
−4.7 to −0.3, paired t). It never scored more: lower in 4 pairs, level in 2.
The exact sign-flip test gives p = 0.125, which is the floor with 4 non-zero
pairs, so the size of the sample, not the data, limits that test.

**2. The other players do not gain from it.** The three Plain players at the
Saboteur's table average 0.4 points more than the three at the control table.
The cost of sabotage falls on the saboteur.

**3. Matches with the Saboteur look longer, not yet established.** Longer in 5
of 6 pairs, by +50, −6, +21, +3, +5, +33 turns: mean +18 (95% −4 to +40),
median +18%. Sign-flip test p = 0.06 one-sided, 0.13 two-sided. Two pairs
carry most of it; three show almost nothing.

**4. It won none of its 6 matches.** If it had a plain seat's 1 in 4, that
happens 18% of the time. Not evidence on its own.

**5. Its behaviour inside matches is the persona's.** Read from its notes in
the resumed seed-203 match: a knight played to put the robber on the ore that
fed the two leaders' cities, stealing from the leader ("Red is the most
dangerous city engine"); trades accepted from the trailing player
("strengthening a trailing player"); offers from the leaders refused; a
development card bought "to deny the leaders" one. Sometimes it only delays
the same player: on seed 206 RED led both tables, and won only the control.

**Checked and not supported:** that the Saboteur's table negotiates less per
turn. Offers per turn go down in some pairs and up in others; calls per turn
likewise.

## What it looks like

The Saboteur spends its moves holding the leader back rather than building, and
pays for it in its own score. It does not visibly lift anyone else's. The
matches it sits in tend to run longer, and the winner changes more often than
the leader would suggest, but six pairs cannot tell that from chance.

## What it would take to confirm

- **Duration.** Six more pairs that came out exactly like these would make it
  +18 turns (95% +5 to +30), permutation p = 0.013 two-sided. Real pairs will
  vary, and pair 201 (+50) may not recur; an effect of this size against a
  spread of ±20 turns between pairs needs ~15-20 pairs to show reliably. Six
  more pairs at flex: ~$4.50 and 16-18 h played one at a time.
- **Points.** Already consistent; six more pairs would firm up the size of the
  loss, and settle whether it is ~2.5 points or less.
- **Win rate.** Out of reach at this size. Telling a 10% win rate from a plain
  seat's 25% takes ~41 Saboteur matches (80% power, 5% one-sided), ~100 for
  15%. Controls are not needed for this question, only Saboteur matches: ~41 at
  ~$0.40 and ~1.5 h each, ~$16-20 and 60-75 h in sequence. Zero wins in 12
  would already be p = 0.03 against 25%.
- **Mechanism, free from the logs.** Who the Saboteur robs and refuses over a
  whole match, whether the leader at turn N wins less often at its table, and
  whether the others start refusing its offers once they notice (their
  `reasoning`; Luna returns no thinking trace).

## Caveats

- Pair 201 ran at the standard tier, the rest at flex. The model is the same
  on both; only serving differs, and the comparison is within each pair.
- The dice within a pair are shared until the two tables first draw from the
  engine's rng differently (a steal, a development card), then run on the same
  stream shifted: 30-58% of rolls coincide position by position, against ~11%
  for independent dice. This narrows the luck difference within a pair; it is
  not a fixed-dice design.
- The resumed halves of two matches have new dice from the resume point. They
  were never observed under the old dice, so nothing is distorted, but those
  matches are each one game in two files.
- Prompt version 4 throughout. Do not pool with v3 matches, and do not compare
  durations with them: effort `high` plays about 5x slower per turn than the
  default.

## Cost and time

Manipulation check $0.81; twelve matches $5.18 by our logs (the OpenAI bill ran
4-10% above our logs on earlier days). At `high`, a match is ~1 minute per turn
and $0.27-0.85; flex halved the price without slowing calls (median 8.5 s
against 12.5 s on a ten-call test at 01:00).
