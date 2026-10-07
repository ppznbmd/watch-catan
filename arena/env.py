"""Read `.env` into the environment, with no dependency.

Rules, chosen so nothing surprises you later:

- a variable already set in the real environment always wins, so an `export`
  overrides the file rather than the other way round;
- unknown lines are skipped rather than raising, but a malformed line is
  reported, because a silently ignored key looks exactly like a missing one;
- no interpolation, no `export` keyword magic beyond stripping it.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = PROJECT_ROOT / ".env"


def parse(text: str) -> Tuple[dict, List[str]]:
    """Return (values, complaints). Never raises."""
    values, complaints = {}, []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key:
            complaints.append(f"line {number}: expected KEY=value, got {raw.strip()!r}")
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values, complaints


def load_env(path: Optional[Path] = None, override: bool = False) -> List[str]:
    """Load the file if it exists. Returns the complaints, for the caller to show."""
    path = Path(path) if path else DEFAULT_PATH
    if not path.exists():
        return []
    values, complaints = parse(path.read_text())
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return complaints
