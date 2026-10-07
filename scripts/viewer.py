#!/usr/bin/env python3
"""Serve the spectator UI, and the frames it reads.

There is no streaming protocol here on purpose. A run file is append-only and
line-buffered, so following a live match is the same operation as reading a
finished one — ask again and take what is new. That removes an entire moving part
(an SSE server coupled to the running match) and means the viewer works on a match
started before it was, or on one whose process has since died.

Stdlib only, like `arena/env.py`. Nothing here is on the path of a real match.

    .venv/bin/python scripts/viewer.py
"""

import argparse
import json
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arena.replay import Replay, ReplayError  # noqa: E402

RUNS = ROOT / "runs"
STATIC = ROOT / "viewer"

_cache = {}
_lock = threading.Lock()


def list_runs():
    """Every run on disk, newest first, real matches before scratch.

    A real match cost money and is the only copy of what happened; a dry run is
    disposable. The viewer keeps that distinction visible rather than mixing them
    into one list.
    """
    live = writers()
    out = []
    for path in RUNS.rglob("*.jsonl"):
        relative = path.relative_to(RUNS)
        stat = path.stat()
        out.append({
            "path": str(relative),
            "name": path.stem,
            "scratch": relative.parts[0] == "scratch",
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "started": started_at(path),
            "status": ("live" if path.resolve() in live
                       else "finished" if ends_in_game_over(path) else "stopped"),
        })
    out.sort(key=lambda r: (r["scratch"], r["status"] != "live", -r["modified"]))
    return out


def writers() -> set:
    """Run files some process currently holds open for writing.

    This is the only reliable test for "being played right now". A file that has
    not reached `game_over` may be live or may belong to a process that died, and
    its mtime cannot tell them apart: one model call may legitimately sit silent
    for 120s x 3 attempts. The writer keeps its file open for the whole match
    (`EventLog`), so the open descriptor is the answer. Linux only; elsewhere
    nothing is reported live, which errs towards "stopped" rather than a false
    live badge.
    """
    runs = RUNS.resolve()
    found = set()
    for fd in Path("/proc").glob("[0-9]*/fd/*"):
        try:
            target = Path(fd.readlink())
            if runs not in target.parents or target.suffix != ".jsonl":
                continue
            info = (fd.parent.parent / "fdinfo" / fd.name).read_text()
        except OSError:  # another user's process, or one that exited mid-scan
            continue
        flags = next(int(line.split()[1], 8) for line in info.splitlines()
                     if line.startswith("flags:"))
        if flags & 0o3:  # O_WRONLY or O_RDWR; our own Replay only ever reads
            found.add(target)
    return found


def started_at(path: Path):
    """When the match began, from its own `game_start`, not the filesystem."""
    try:
        with path.open("rb") as fh:
            return json.loads(fh.readline()).get("at")
    except (OSError, ValueError):
        return None


def ends_in_game_over(path: Path) -> bool:
    """Whether the last complete line is `game_over`, read from the end so a
    listing does not parse megabytes per run."""
    with path.open("rb") as fh:
        end = fh.seek(0, 2)
        tail = b""
        while end > 0 and tail.rstrip(b"\n").count(b"\n") < 1:
            start = max(0, end - 65536)
            fh.seek(start)
            tail = fh.read(end - start) + tail
            end = start
    lines = tail.rstrip(b"\n").split(b"\n")
    try:
        return json.loads(lines[-1]).get("kind") == "game_over"
    except ValueError:  # a live writer mid-line
        return False


def resolve(relative: str) -> Path:
    """A path under `runs/` or nothing. The viewer is a local tool, but it still
    has no business reading anything else on the filesystem."""
    path = (RUNS / relative).resolve()
    if not path.is_file() or RUNS.resolve() not in path.parents:
        raise FileNotFoundError(relative)
    return path


def frames_since(relative: str, cursor: int) -> dict:
    """Frames after `cursor`, and everything the client needs to render them.

    The `Replay` is kept between requests so a live poll re-reads only the bytes
    that were appended, not the whole file each second.
    """
    path = resolve(relative)
    with _lock:
        replay = _cache.get(path)
        if replay is None:
            replay = _cache[path] = Replay(path)
        replay.poll()
        payload = {
            "from": cursor,
            "total": len(replay.frames),
            "complete": replay.complete,
            "frames": replay.frames[cursor:],
            "summary": replay.summary,
        }
        if cursor == 0:
            payload["meta"] = replay.meta
    return payload


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def do_GET(self):
        route = urlparse(self.path)
        if route.path == "/api/runs":
            return self.send_json({"runs": list_runs()})
        if route.path == "/api/run":
            query = parse_qs(route.query)
            relative = (query.get("path") or [""])[0]
            cursor = int((query.get("from") or ["0"])[0])
            try:
                return self.send_json(frames_since(relative, cursor))
            except FileNotFoundError:
                return self.send_json({"error": f"no run at {relative!r}"}, status=404)
            except ReplayError as exc:
                return self.send_json({"error": str(exc)}, status=409)
        return super().do_GET()

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")  # a live run changes underneath
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # a poll every second would otherwise bury everything else


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    runs = list_runs()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"watch-catan viewer  http://{args.host}:{args.port}")
    print(f"{len(runs)} run(s) on disk"
          + (f", newest {runs[0]['path']}" if runs else " — play one first"))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
