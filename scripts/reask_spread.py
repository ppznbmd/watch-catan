"""How much a model disagrees with itself, against how much a prompt change moves it.

    .venv/bin/python scripts/reask_spread.py experiments/reask-<batch>.jsonl

Reads the answers `scripts/reask.py` wrote. For every position, two answers
under the same condition are compared, and so are two answers under different
conditions. If the second rate is no higher than the first, the change did
nothing the model's own sampling does not already do. A rate of accepting or
taking a win can stay flat while answers churn underneath, so this looks at
the choices themselves.
"""

import collections
import itertools
import json
import sys


def choice(answer):
    if "choice" not in answer:
        return None
    return answer.get("described")


def main(path):
    rows = [json.loads(line) for line in open(path)]
    answers = [r for r in rows if r["kind"] == "answer"]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in answers:
        c = choice(r["answer"])
        if c is not None:
            by[(r["set"], r["position"])][r["condition"]].append(c)

    conditions = sorted({r["condition"] for r in answers})
    for s in sorted({k[0] for k in by}):
        same = [0, 0]
        across = collections.Counter(), collections.Counter()
        unanimous = positions = 0
        for (set_, _), d in by.items():
            if set_ != s:
                continue
            for v in d.values():
                for x, y in itertools.combinations(v, 2):
                    same[0] += x != y
                    same[1] += 1
            for a, b in itertools.combinations(conditions, 2):
                for x in d.get(a, []):
                    for y in d.get(b, []):
                        across[0][(a, b)] += x != y
                        across[1][(a, b)] += 1
            first = d.get(conditions[0], [])
            if len(first) > 1:
                positions += 1
                unanimous += len(set(first)) == 1
        print(f"{s}: two answers differ, same condition {same[0]}/{same[1]} "
              f"= {same[0] / max(same[1], 1):.0%}")
        for pair, n in across[1].items():
            print(f"  {pair[0]} vs {pair[1]}: {across[0][pair]}/{n} = {across[0][pair] / n:.0%}")
        print(f"  positions where every '{conditions[0]}' answer agreed: {unanimous}/{positions}")
    played = [r["answer"].get("same_as_played") for r in answers
              if r["condition"] == "as_played" and "choice" in r["answer"]]
    if played:
        print(f"'as_played' answers that repeat the move played in the match: "
              f"{sum(played)}/{len(played)} = {sum(played) / len(played):.0%}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
