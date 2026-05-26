"""Map running processes to the session JSONL files they are writing.

Claude Code stores sessions under ``~/.claude/projects/<encoded-cwd>/<uuid>.jsonl``.
The process does NOT hold the jsonl open, so we cannot read the mapping from
the OS. Instead we encode the process cwd into the project-dir name and pick
the most-recently-modified jsonl in that directory, validating the guess against
the ``cwd`` field embedded in the file's entries.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from claude_watcher.jsonl import read_tail, stat_file
from claude_watcher.models import ProcEntry, Session, State, Status
from claude_watcher.status import derive_status

PROJECTS_DIR = Path.home() / ".claude" / "projects"

# A subagent is considered "active" if its transcript was written within this
# many seconds — used to keep a parent that is blocked on subagents showing as
# working rather than stalled.
SUBAGENT_ACTIVE_WINDOW = 25.0

# A process still using at least this much CPU is treated as working even when
# its transcript looks stalled: a long generation streams to the terminal
# without appending to the JSONL, so timestamp age alone yields false stalls.
CPU_BUSY_THRESHOLD = 1.0  # percent


def encode_project_dir(cwd: str) -> str:
    """Encode an absolute cwd into Claude's project-dir name.

    Claude replaces both '/' and '.' with '-'. The transform is lossy and
    forward-only — never try to decode it; instead validate via the in-file
    cwd field.
    """
    return cwd.replace("/", "-").replace(".", "-")


def resolve_jsonl(cwd: str, projects_dir: Path = PROJECTS_DIR) -> list[Path]:
    """JSONL files for a cwd's project dir, most-recently-modified first."""
    project = projects_dir / encode_project_dir(cwd)
    try:
        files = [p for p in project.iterdir() if p.suffix == ".jsonl" and p.is_file()]
    except OSError:
        return []
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def file_cwd(path: Path, max_lines: int = 50) -> str | None:
    """Read leading lines until one carries a non-null `cwd`."""
    try:
        with open(path, "r", errors="replace") as f:
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                cwd = obj.get("cwd")
                if cwd:
                    return cwd
    except OSError:
        return None
    return None


def subagent_dir(parent_jsonl: str | Path) -> Path:
    """The `<session-uuid>/subagents/` dir that holds a session's subagents.

    Claude stores subagent transcripts beside the parent file in a directory
    named after the session uuid, e.g.
    ``.../<project>/<uuid>.jsonl`` -> ``.../<project>/<uuid>/subagents/``.
    The returned path may not exist (a session may never spawn subagents).
    """
    parent = Path(parent_jsonl)
    return parent.with_suffix("") / "subagents"


def list_subagent_files(parent_jsonl: str | Path) -> list[Path]:
    """All `agent-*.jsonl` transcripts for a session (transcripts only, no meta)."""
    d = subagent_dir(parent_jsonl)
    try:
        return [
            p
            for p in d.iterdir()
            if p.suffix == ".jsonl" and p.name.startswith("agent-") and p.is_file()
        ]
    except OSError:
        return []


def active_subagent_count(
    parent_jsonl: str | Path,
    now_epoch: float | None = None,
    within_s: float = SUBAGENT_ACTIVE_WINDOW,
) -> int:
    """How many subagent transcripts were written within the last `within_s`."""
    import time

    now_epoch = now_epoch if now_epoch is not None else time.time()
    count = 0
    for p in list_subagent_files(parent_jsonl):
        st = stat_file(p)
        if st is not None and now_epoch - st[1] <= within_s:
            count += 1
    return count


def apply_runtime_overrides(
    status: Status, cpu: float, active_subagents: int
) -> Status:
    """Correct a tail-derived state with live process signals (in place).

    ``derive_status`` only sees the transcript, so it flags a session stalled
    whenever the file has been quiet for a while. Two runtime signals override
    that false reading: an active subagent (the parent is blocked waiting on
    it) or a process still burning CPU (it is mid-generation, not stuck).
    """
    status.active_subagents = active_subagents
    if active_subagents > 0 and status.state in (
        State.STALLED,
        State.WORKING,
        State.UNKNOWN,
    ):
        status.state = State.WORKING
    elif status.state is State.STALLED and cpu > CPU_BUSY_THRESHOLD:
        status.state = State.WORKING
    return status


def _status_for(path: Path, cpu: float) -> tuple:
    """Read the tail of a jsonl and derive (status, size, mtime_utc).

    The tail-derived status is then corrected with live process signals
    (subagent activity, CPU) via ``apply_runtime_overrides``.
    """
    st = stat_file(path)
    size = st[0] if st else None
    mtime = datetime.fromtimestamp(st[1], tz=timezone.utc) if st else None
    entries, _ = read_tail(path)
    status = derive_status(entries, datetime.now(timezone.utc)) if entries else None
    if status is not None:
        # Count against real now (not the parent's mtime), so the "(N)" badge
        # matches the active subagent rows the expanded view shows.
        active = active_subagent_count(path)
        apply_runtime_overrides(status, cpu, active)
    return status, size, mtime


def build_sessions(
    procs: list[ProcEntry], projects_dir: Path = PROJECTS_DIR
) -> list[Session]:
    """Pair each process with a JSONL session file (best effort).

    When several processes share a cwd we cannot definitively pair PID↔file, so
    we mark them ambiguous and hand out the top-N most-recent files round-robin
    so each row at least points at a distinct session.
    """
    by_cwd: dict[str | None, list[ProcEntry]] = defaultdict(list)
    for p in procs:
        by_cwd[p.cwd].append(p)

    sessions: list[Session] = []
    for cwd, group in by_cwd.items():
        if cwd is None:
            for proc in group:
                sessions.append(Session(proc=proc))
            continue

        candidates = resolve_jsonl(cwd, projects_dir)
        ambiguous = len(group) > 1
        for i, proc in enumerate(sorted(group, key=lambda p: p.pid)):
            jsonl_path = candidates[i] if i < len(candidates) else (
                candidates[0] if candidates else None
            )
            session = Session(proc=proc, ambiguous=ambiguous)
            if jsonl_path is not None:
                session.jsonl_path = str(jsonl_path)
                session.session_id = jsonl_path.stem
                session.cwd_validated = file_cwd(jsonl_path) == cwd
                status, size, mtime = _status_for(jsonl_path, proc.cpu)
                session.status = status
                session.file_size = size
                session.file_mtime = mtime
            sessions.append(session)

    sessions.sort(key=lambda s: s.proc.pid)
    return sessions
