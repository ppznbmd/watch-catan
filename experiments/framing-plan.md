# Framing the lie trap: plan, written before any answer

25 Sep 2026, 00:55, before the first call.

## Question

Does Flash change what it does in the lie trap when told what the game is part
of? In the trap and in a real match it built its own theory of what the
examiner wanted, mostly a test of skill, and acted on it (JOURNAL.md, 24 Sep,
evening). Here the theory is given to it.

## Design

- Positions: `experiments/lie-trap-positions-v4-20260925.pkl`, 235 positions
  from bot games seeded 31000-31499, prompt v4. None of them has been asked.
- Frames, one sentence first in the system prompt (`FRAMES` in
  `scripts/lie_trap.py`): `none`, `skill`, `honesty`, `unobserved`. The last is
  false; every answer is recorded and read.
- The trap condition only, every frame, one answer each: 4 calls per
  position. Plain persona, deepseek-flash, default effort, no summary asked.
  (The first draft asked the control too, 8 calls a position; dropped at 01:15,
  before any batch, to halve the cost. The control can be asked later on the
  same positions if the trap shows something worth comparing.)
- Batches of 50 positions in file order (`--start 0 --limit 50`, then 50, ...),
  off-peak only.

## Measures

Primary, trap condition, each frame against `none`, paired by position:

1. Whether it makes the winning offer (share of answers).
2. Among winning offers, the share that lie, judged by hand against the
   definitions in `scripts/lie_trap.py` (`VERDICTS`), with the classifier only
   choosing what to read first.

Both are primary because a frame can lower the lie rate by making the model
avoid the offer rather than by making it more honest.

Secondary: reveals; whether the trace reasons about what the evaluator wants, by
trace length; trace length itself; moves other than offers.

## When to stop

Look after every batch, but decide only from 100 positions on. At 100: if
`honesty` against `none` on the lie rate has p between 0.05 and 0.3, continue
to 200 and analyse all 200; otherwise stop. Nothing is dropped after it is
seen: every answer asked is analysed.

## Expected cost

$0.0037 a call off-peak (the 24 Sep trap), 200 calls a batch: about $0.72 and
12 minutes with 8 workers. The `--measure` step (8 calls, both conditions) was
run at 01:00 and cost $0.011.
