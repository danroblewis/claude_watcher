"""Plain data types shared across the pure-logic modules.

Nothing here does I/O; everything is a dataclass or enum so the logic layer
stays trivially unit-testable without Textual or a live system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Mode(Enum):
    """How a claude process was launched."""

    HEADLESS = "headless"  # has ` -p ` / `--print`; stdout is buffered
    INTERACTIVE = "interactive"


class State(Enum):
    """Derived liveness/activity of a session."""

    WORKING = "working"  # mid-turn or running a tool
    THINKING = "thinking"  # last content item is extended thinking
    IDLE = "idle"  # turn finished (end_turn), waiting on the user
    STALLED = "stalled"  # no new activity for a while
    UNKNOWN = "unknown"  # no jsonl resolved / nothing parseable yet


@dataclass(frozen=True)
class ProcEntry:
    """A running `claude` process and its OS-level metadata."""

    pid: int
    args: str
    mode: Mode
    cwd: str | None  # None if lsof failed / permission denied
    lstart: str  # raw `ps -o lstart` start time
    etime: str  # elapsed runtime, e.g. "01:23:45" or "2-03:04:05"
    cpu: float
    mem: float
    pstate: str  # `ps` state column, e.g. "S+", "R"


@dataclass
class Status:
    """A human-readable snapshot of what a session is currently doing."""

    state: State
    tool_name: str | None = None
    tool_input_summary: str | None = None
    last_text: str | None = None  # most recent assistant text snippet
    last_ts: datetime | None = None  # UTC, from the last entry's timestamp
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    context_tokens: int | None = None  # latest prompt size (input+cache) = context fill
    model: str | None = None  # latest assistant message's model, for window sizing
    active_subagents: int = 0  # subagent files with recently-advancing mtime


@dataclass
class SubagentInfo:
    """Display snapshot of one subagent transcript under a parent session."""

    path: str
    tag: str  # short id from the filename, e.g. "a16d"
    status: Status | None = None  # derived from the subagent's own transcript
    output_tokens: int | None = None  # deduped output tokens for this subagent
    active: bool = False  # transcript written within the active window


@dataclass
class Session:
    """A process paired with the JSONL file it is (heuristically) writing."""

    proc: ProcEntry
    jsonl_path: str | None = None
    session_id: str | None = None  # jsonl filename stem
    cwd_validated: bool = False  # in-file `cwd` matched proc.cwd
    ambiguous: bool = False  # >1 process shares this cwd
    status: Status | None = None
    file_size: int | None = None
    file_mtime: datetime | None = None  # UTC
    total_output_tokens: int | None = None  # deduped sum across parent+subagents
    subagent_paths: list[str] = field(default_factory=list)  # all subagent transcripts
    subagents: list[SubagentInfo] = field(default_factory=list)  # populated when expanded


@dataclass
class FeedEvent:
    """One line in the drill-down live feed."""

    ts: datetime | None  # UTC
    kind: str  # tool_use | tool_result | text | thinking | user | meta
    text: str  # already one-line and truncated for display
    is_error: bool = False
