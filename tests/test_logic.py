"""Unit tests for the pure-logic layer (no Textual, no live system)."""

from __future__ import annotations

import json
import signal
from datetime import datetime, timezone

import pytest

from claude_watcher.jsonl import read_incremental, read_tail
from claude_watcher.transcripts import TranscriptTail
from claude_watcher.models import State
from claude_watcher.procs import classify_mode, is_watchable, terminate_proc
from claude_watcher.models import Mode
from claude_watcher.models import Status
from claude_watcher.sessions import (
    _status_for,
    active_subagent_count,
    apply_runtime_overrides,
    encode_project_dir,
    list_subagent_files,
    subagent_dir,
)
from claude_watcher.status import (
    derive_status,
    parse_entry,
    parse_timestamp,
)


# -- path encoding ------------------------------------------------------------

@pytest.mark.parametrize(
    "cwd,expected",
    [
        ("/Users/x/project", "-Users-x-project"),
        ("/Users/x/claude_watcher", "-Users-x-claude-watcher"),  # underscore -> dash
        ("/Users/x/.foo/bar", "-Users-x--foo-bar"),  # leading dot -> -, and / -> -
        ("/a/b.c.d/e", "-a-b-c-d-e"),  # every dot becomes a dash
        ("/a/b c/d", "-a-b-c-d"),  # spaces too
    ],
)
def test_encode_project_dir(cwd, expected):
    assert encode_project_dir(cwd) == expected


# -- process classification ---------------------------------------------------

def test_classify_mode():
    assert classify_mode("claude -p 'do thing'") is Mode.HEADLESS
    assert classify_mode("/x/claude --print foo") is Mode.HEADLESS
    assert classify_mode("claude --dangerously-skip-permissions -p hi") is Mode.HEADLESS
    assert classify_mode("claude -r") is Mode.INTERACTIVE
    assert classify_mode("claude") is Mode.INTERACTIVE


def test_is_watchable():
    assert is_watchable("claude -p hi", self_pid=1, pid=2) is True
    assert is_watchable("claude", self_pid=2, pid=2) is False  # self
    assert is_watchable("/Applications/Claude.app/x", self_pid=1, pid=2) is False


def test_terminate_proc_outcomes(monkeypatch):
    sent: list[tuple[int, int]] = []

    def ok(pid, sig):
        sent.append((pid, sig))

    monkeypatch.setattr("claude_watcher.procs.os.kill", ok)
    assert terminate_proc(123) == "sent"
    assert sent == [(123, signal.SIGTERM)]

    def gone(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr("claude_watcher.procs.os.kill", gone)
    assert terminate_proc(123) == "gone"

    def denied(pid, sig):
        raise PermissionError

    monkeypatch.setattr("claude_watcher.procs.os.kill", denied)
    assert terminate_proc(123) == "denied"


# -- timestamp parsing --------------------------------------------------------

def test_parse_timestamp_z_suffix():
    dt = parse_timestamp("2026-05-24T22:14:18.853Z")
    assert dt is not None and dt.tzinfo is not None
    assert dt.year == 2026 and dt.hour == 22


def test_parse_timestamp_bad():
    assert parse_timestamp(None) is None
    assert parse_timestamp("not a date") is None


# -- jsonl tailing ------------------------------------------------------------

def _write(path, lines):
    path.write_bytes(("\n".join(lines) + "\n").encode())


def test_read_tail_drops_partial_first_line(tmp_path):
    f = tmp_path / "s.jsonl"
    rows = [f'{{"i": {i}}}' for i in range(200)]
    _write(f, rows)
    # tiny window forces a mid-line start -> first fragment must be dropped
    entries, end = read_tail(f, window=40)
    assert end == f.stat().st_size
    assert all(isinstance(e, dict) and "i" in e for e in entries)
    # the very first row (i=0) cannot be in a 40-byte tail
    assert {0} not in [{e["i"]} for e in entries]


def test_read_incremental_only_new_bytes(tmp_path):
    f = tmp_path / "s.jsonl"
    _write(f, ['{"a": 1}', '{"a": 2}'])
    entries, off = read_tail(f)
    assert off == f.stat().st_size
    # nothing new yet
    new, off2 = read_incremental(f, off)
    assert new == [] and off2 == off
    # append one complete + one partial line
    with open(f, "ab") as fh:
        fh.write(b'{"a": 3}\n{"a": 4}')  # last line has no newline
    new, off3 = read_incremental(f, off2)
    assert [e["a"] for e in new] == [3]  # partial line withheld
    # complete the partial line; it should now appear
    with open(f, "ab") as fh:
        fh.write(b"\n")
    new2, off4 = read_incremental(f, off3)
    assert [e["a"] for e in new2] == [4]
    assert off4 == f.stat().st_size


def test_read_incremental_truncation_resets(tmp_path):
    f = tmp_path / "s.jsonl"
    _write(f, ['{"a": 1}', '{"a": 2}'])
    big = f.stat().st_size + 1000
    new, off = read_incremental(f, big)  # offset beyond EOF -> restart at 0
    assert [e["a"] for e in new] == [1, 2]


def test_read_tail_skips_bad_json(tmp_path):
    f = tmp_path / "s.jsonl"
    f.write_bytes(b'{"a": 1}\nnot json\n{"a": 2}\n')
    entries, _ = read_tail(f)
    assert [e["a"] for e in entries] == [1, 2]


# -- entry parsing ------------------------------------------------------------

def _assistant(content, stop_reason="tool_use", ts="2026-05-24T00:00:00Z"):
    return {
        "type": "assistant",
        "timestamp": ts,
        "message": {"role": "assistant", "stop_reason": stop_reason, "content": content},
    }


def test_parse_entry_tool_use():
    e = _assistant([{"type": "tool_use", "name": "Bash", "input": {"command": "npm test"}}])
    ev = parse_entry(e)
    assert ev.kind == "tool_use" and "Bash" in ev.text and "npm test" in ev.text


def test_parse_entry_text_and_thinking():
    assert parse_entry(_assistant([{"type": "text", "text": "hi there"}])).kind == "text"
    assert parse_entry(_assistant([{"type": "thinking", "thinking": "hmm"}])).kind == "thinking"


def test_parse_entry_tool_result_error():
    e = {
        "type": "user",
        "timestamp": "2026-05-24T00:00:00Z",
        "message": {"role": "user", "content": [{"type": "tool_result", "is_error": True, "content": "boom"}]},
    }
    ev = parse_entry(e)
    assert ev.kind == "tool_result" and ev.is_error is True


def test_parse_entry_user_string():
    e = {"type": "user", "timestamp": "2026-05-24T00:00:00Z", "message": {"role": "user", "content": "hello"}}
    assert parse_entry(e).kind == "user"


def test_parse_entry_meta_is_none():
    for t in ("ai-title", "attachment", "last-prompt", "queue-operation", "system"):
        assert parse_entry({"type": t}) is None


# -- status derivation --------------------------------------------------------

NOW = datetime(2026, 5, 24, 0, 0, 30, tzinfo=timezone.utc)


def test_derive_status_working_on_tool():
    entries = [_assistant([{"type": "tool_use", "name": "Read", "input": {"file_path": "/a.py"}}], ts="2026-05-24T00:00:20Z")]
    s = derive_status(entries, NOW)
    assert s.state is State.WORKING and s.tool_name == "Read"


def test_derive_status_thinking():
    entries = [_assistant([{"type": "thinking", "thinking": "..."}], ts="2026-05-24T00:00:20Z")]
    assert derive_status(entries, NOW).state is State.THINKING


def test_derive_status_idle_on_end_turn():
    entries = [_assistant([{"type": "text", "text": "done"}], stop_reason="end_turn", ts="2026-05-24T00:00:20Z")]
    assert derive_status(entries, NOW).state is State.IDLE


def test_derive_status_text_midturn_is_working():
    # text content but stop_reason tool_use => mid-turn, not idle
    entries = [_assistant([{"type": "text", "text": "let me..."}], stop_reason="tool_use", ts="2026-05-24T00:00:20Z")]
    assert derive_status(entries, NOW).state is State.WORKING


def test_derive_status_stalled():
    # working but last activity 5 minutes ago
    entries = [_assistant([{"type": "tool_use", "name": "Bash", "input": {}}], ts="2026-05-24T00:00:00Z")]
    later = datetime(2026, 5, 24, 0, 5, 0, tzinfo=timezone.utc)
    assert derive_status(entries, later).state is State.STALLED


def _user(text, ts="2026-05-24T00:00:30Z"):
    return {"type": "user", "timestamp": ts, "message": {"role": "user", "content": text}}


def test_derive_status_user_prompt_after_end_turn_is_working():
    # The turn ended, then the user sent a follow-up; the reply hasn't been
    # written yet, so the session is working, not idle.
    entries = [
        _assistant([{"type": "text", "text": "all done"}], stop_reason="end_turn"),
        _user("now do the next thing"),
    ]
    assert derive_status(entries, NOW).state is State.WORKING


def test_derive_status_trailing_tool_result_uses_assistant_state():
    # A tool_result is mid-turn output, not a new prompt: state comes from the
    # assistant tool_use that triggered it.
    entries = [
        _assistant([{"type": "tool_use", "name": "Bash", "input": {}}]),
        {"type": "user", "timestamp": "2026-05-24T00:00:30Z",
         "message": {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]}},
    ]
    assert derive_status(entries, NOW).state is State.WORKING


def test_derive_status_idle_not_overridden_by_stall():
    entries = [_assistant([{"type": "text", "text": "done"}], stop_reason="end_turn", ts="2026-05-24T00:00:00Z")]
    later = datetime(2026, 5, 24, 1, 0, 0, tzinfo=timezone.utc)
    assert derive_status(entries, later).state is State.IDLE


def test_derive_status_tokens_and_text():
    entries = [
        {
            "type": "assistant",
            "timestamp": "2026-05-24T00:00:20Z",
            "message": {
                "role": "assistant",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "the answer"}],
                "usage": {"input_tokens": 5, "output_tokens": 42, "cache_read_input_tokens": 100},
            },
        }
    ]
    s = derive_status(entries, NOW)
    assert s.output_tokens == 42 and s.last_text == "the answer"


def test_derive_status_context_fill():
    entries = [
        {
            "type": "assistant",
            "timestamp": "2026-05-24T00:00:20Z",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-7",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "hi"}],
                "usage": {
                    "input_tokens": 4156,
                    "cache_read_input_tokens": 102236,
                    "cache_creation_input_tokens": 11347,
                    "output_tokens": 9434,
                },
            },
        }
    ]
    s = derive_status(entries, NOW)
    # Context fill is input + both cache fields (the whole prompt size).
    assert s.context_tokens == 4156 + 102236 + 11347  # 117739
    assert s.model == "claude-opus-4-7"


# -- subagents ----------------------------------------------------------------

def test_subagent_dir():
    d = subagent_dir("/x/proj/2913abb9-uuid.jsonl")
    assert d.as_posix() == "/x/proj/2913abb9-uuid/subagents"


def test_list_and_count_subagents(tmp_path):
    import os
    import time

    parent = tmp_path / "sess.jsonl"
    parent.write_bytes(b'{"a": 1}\n')
    sub = tmp_path / "sess" / "subagents"
    sub.mkdir(parents=True)
    fresh = sub / "agent-aaaa1111.jsonl"
    stale = sub / "agent-bbbb2222.jsonl"
    meta = sub / "agent-aaaa1111.meta.json"  # not a transcript
    for f in (fresh, stale, meta):
        f.write_bytes(b"{}\n")

    # Only the two .jsonl transcripts are listed (meta.json excluded).
    assert {p.name for p in list_subagent_files(parent)} == {fresh.name, stale.name}

    # Age the stale one well past the window.
    old = time.time() - 600
    os.utime(stale, (old, old))
    assert active_subagent_count(parent, within_s=25.0) == 1  # only fresh counts


def test_count_subagents_none_when_no_dir(tmp_path):
    parent = tmp_path / "sess.jsonl"
    parent.write_bytes(b"{}\n")
    assert active_subagent_count(parent) == 0


def test_status_active_count_uses_real_now_not_parent_mtime(tmp_path):
    import os
    import time

    # Parent blocked on a Task dispatch a while ago: its own mtime is old.
    parent = tmp_path / "sess.jsonl"
    parent.write_text(
        '{"type": "assistant", "timestamp": "2026-05-24T00:00:00Z",'
        ' "message": {"role": "assistant", "stop_reason": "tool_use",'
        ' "content": [{"type": "tool_use", "name": "Task", "input": {}}]}}\n'
    )
    old = time.time() - 120
    os.utime(parent, (old, old))

    subs = tmp_path / "sess" / "subagents"
    subs.mkdir(parents=True)
    fresh = subs / "agent-aaaa1111.jsonl"
    stale = subs / "agent-bbbb2222.jsonl"
    fresh.write_bytes(b"{}\n")  # written just now -> active
    stale.write_bytes(b"{}\n")
    os.utime(stale, (old + 20, old + 20))  # 100s ago: after the parent, but long done

    status, _, _ = _status_for(parent, cpu=0.0)
    # Only the fresh subagent counts; the stale one must not be inflated in just
    # because it was written after the parent's last write.
    assert status.active_subagents == 1


# -- transcript tail (last assistant text) ------------------------------------

def _text_line(text):
    return json.dumps(
        {"type": "assistant",
         "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}
    ) + "\n"


def _tool_line():  # an assistant line with no text content
    return json.dumps(
        {"type": "assistant",
         "message": {"role": "assistant", "content": [{"type": "tool_use", "name": "Bash", "input": {}}]}}
    ) + "\n"


def test_transcript_tail_tracks_last_text_past_the_tail(tmp_path):
    parent = tmp_path / "sess.jsonl"
    # An old text message, then a lot of tool-only activity after it.
    parent.write_text(_text_line("the last thing I said") + _tool_line() * 200)
    tail = TranscriptTail()
    tail.update(str(parent))
    assert tail.last_text(str(parent)) == "the last thing I said"


def test_transcript_tail_updates_and_persists(tmp_path):
    parent = tmp_path / "sess.jsonl"
    parent.write_text(_text_line("first"))
    tail = TranscriptTail()
    tail.update(str(parent))
    assert tail.last_text(str(parent)) == "first"

    with open(parent, "a") as f:
        f.write(_text_line("second"))
    tail.update(str(parent))
    assert tail.last_text(str(parent)) == "second"

    # Appending text-free lines must not blank the remembered message.
    with open(parent, "a") as f:
        f.write(_tool_line())
    tail.update(str(parent))
    assert tail.last_text(str(parent)) == "second"


def test_transcript_tail_includes_subagents(tmp_path):
    parent = tmp_path / "sess.jsonl"
    parent.write_text(_text_line("parent says hi"))
    subs = tmp_path / "sess" / "subagents"
    subs.mkdir(parents=True)
    sub = subs / "agent-aaaa1111.jsonl"
    sub.write_text(_text_line("subagent reply"))

    tail = TranscriptTail()
    assert tail.last_text(str(sub)) is None  # not ingested yet
    tail.update(str(parent))  # ingests parent + subagents
    assert tail.last_text(str(parent)) == "parent says hi"
    assert tail.last_text(str(sub)) == "subagent reply"


def test_transcript_tail_resets_after_truncation(tmp_path):
    parent = tmp_path / "sess.jsonl"
    parent.write_text(_text_line("the original message, nice and long"))
    tail = TranscriptTail()
    tail.update(str(parent))
    assert tail.last_text(str(parent)) == "the original message, nice and long"

    parent.write_text(_text_line("fresh"))  # shorter file -> detected as truncation
    tail.update(str(parent))
    assert tail.last_text(str(parent)) == "fresh"


# -- runtime state overrides --------------------------------------------------

def test_cpu_override_keeps_busy_session_working():
    # A stalled-looking transcript whose process is still burning CPU is really
    # mid-generation, so it should read as working, not stalled.
    s = apply_runtime_overrides(Status(state=State.STALLED), cpu=8.0, active_subagents=0)
    assert s.state is State.WORKING


def test_cpu_override_leaves_truly_idle_session_stalled():
    s = apply_runtime_overrides(Status(state=State.STALLED), cpu=0.2, active_subagents=0)
    assert s.state is State.STALLED


def test_subagent_override_still_wins_and_count_recorded():
    s = apply_runtime_overrides(Status(state=State.STALLED), cpu=0.0, active_subagents=3)
    assert s.state is State.WORKING and s.active_subagents == 3


def test_cpu_override_does_not_touch_idle_state():
    # IDLE means the turn ended (waiting on the user); CPU must not flip it.
    s = apply_runtime_overrides(Status(state=State.IDLE), cpu=50.0, active_subagents=0)
    assert s.state is State.IDLE


def test_derive_status_ignores_meta_entries():
    entries = [
        {"type": "ai-title", "aiTitle": "x"},
        _assistant([{"type": "tool_use", "name": "Grep", "input": {}}], ts="2026-05-24T00:00:20Z"),
        {"type": "queue-operation", "cwd": None},
    ]
    assert derive_status(entries, NOW).state is State.WORKING
