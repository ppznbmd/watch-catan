"""Every decision, as it happens.

The JSONL sink is written first and always. It is the replay, the dataset, and
what survives when the live view has a bad day. Anything that wants events as
they happen (the SSE server, later) subscribes on top.
"""

import json
import time
from pathlib import Path
from typing import Callable, List, Optional

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


class EventLog:
    """Writes to `runs/` by default; a scratch match goes to `runs/scratch/`.

    The split exists because a real match is the only copy of something that cost
    money and ten minutes, and a dry run is disposable. Keeping both in one
    directory meant one careless `rm runs/*.jsonl` destroyed the first real match
    ever played here. Never delete `runs/*.jsonl` — only `runs/scratch/`.
    """

    def __init__(self, game_id: str, runs_dir: Optional[Path] = None,
                 scratch: bool = False):
        self.game_id = game_id
        base = Path(runs_dir) if runs_dir else (RUNS_DIR / "scratch" if scratch else RUNS_DIR)
        self.path = base / f"{game_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", buffering=1)  # line buffered: tail -f works
        self._subscribers: List[Callable[[dict], None]] = []
        self.count = 0

    def subscribe(self, fn: Callable[[dict], None]) -> None:
        self._subscribers.append(fn)

    def emit(self, kind: str, **payload) -> dict:
        event = {"seq": self.count, "at": time.time(), "kind": kind, **payload}
        self.count += 1
        self._fh.write(json.dumps(event, default=str) + "\n")
        for fn in self._subscribers:
            try:
                fn(event)
            except Exception:  # a broken spectator must never stop the game
                pass
        return event

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_run(path) -> List[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]
