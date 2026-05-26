"""Textual TUI: a live table of running claude sessions + a drill-down feed.

This is the only module that imports Textual or runs async. All blocking work
(pgrep/lsof/ps + file reads) is pushed off the event loop with
``asyncio.to_thread`` inside ``exclusive`` workers, so the UI never stalls and
slow polls cannot pile up.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, Label, RichLog

from claude_watcher.jsonl import read_incremental, read_tail, stat_file
from claude_watcher.models import FeedEvent, Mode, Session, State
from claude_watcher.procs import discover_procs, terminate_proc
from claude_watcher.sessions import build_sessions, list_subagent_files
from claude_watcher.status import parse_entry

PROC_INTERVAL = 2.0
FEED_INTERVAL = 0.75
SEED_EVENTS = 25  # how many recent feed lines to show when selecting a session

_HOME = str(Path.home())

_STATE_STYLE = {
    State.WORKING: "bold green",
    State.THINKING: "cyan",
    State.IDLE: "yellow",
    State.STALLED: "bold red",
    State.UNKNOWN: "grey50",
}

_KIND_STYLE = {
    "tool_use": "green",
    "tool_result": "grey62",
    "text": "white",
    "thinking": "cyan",
    "user": "yellow",
}

_KIND_SYMBOL = {
    "tool_use": "▸",
    "tool_result": "◂",
    "text": "💬",
    "thinking": "💭",
    "user": "👤",
}


def _shorten_cwd(cwd: str | None) -> str:
    if not cwd:
        return "?"
    if cwd.startswith(_HOME):
        cwd = "~" + cwd[len(_HOME):]
    return cwd


def _shorten_msg(text: str | None, width: int = 60) -> str:
    """Collapse a multi-line agent message into a single trimmed line."""
    if not text:
        return ""
    flat = " ".join(text.split())
    if len(flat) > width:
        flat = flat[: width - 1].rstrip() + "…"
    return flat


def _fmt_time(ts: datetime | None) -> str:
    if ts is None:
        return "--:--:--"
    return ts.astimezone().strftime("%H:%M:%S")


def _fmt_etime(etime: str) -> str:
    """Fold ps elapsed days into hours, e.g. "01-00:07:43" -> "24:07:43"."""
    days, sep, rest = etime.partition("-")
    if not sep:
        return etime
    try:
        h, m, s = rest.split(":")
        return f"{int(days) * 24 + int(h):02d}:{m}:{s}"
    except ValueError:
        return etime


def _agent_tag(path: str) -> str:
    """Short label for a subagent file, e.g. agent-a708f58….jsonl -> a708."""
    stem = Path(path).stem  # agent-a708f5857f3fde650
    return stem.removeprefix("agent-")[:4] or stem


def _feed_line(ev: FeedEvent, tag: str | None = None) -> Text:
    """Render one feed event. `tag` (a subagent id) indents and labels the line."""
    style = _KIND_STYLE.get(ev.kind, "white")
    if ev.kind == "tool_result" and ev.is_error:
        style = "red"
    symbol = _KIND_SYMBOL.get(ev.kind, "·")
    line = Text()
    line.append(_fmt_time(ev.ts) + "  ", style="grey42")
    if tag is not None:
        line.append(f"  └{tag} ", style="magenta")
    line.append(symbol + " ", style=style)
    line.append(ev.text, style=style)
    return line


class ConfirmKill(ModalScreen[bool]):
    """Yes/no confirmation before sending SIGTERM to a session."""

    DEFAULT_CSS = """
    ConfirmKill {
        align: center middle;
    }
    #confirm-box {
        width: auto;
        height: auto;
        max-width: 72;
        padding: 1 2;
        border: thick $error;
        background: $surface;
    }
    #confirm-msg {
        width: auto;
        margin-bottom: 1;
    }
    #confirm-buttons {
        width: auto;
        height: auto;
        align: center middle;
    }
    #confirm-buttons Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("y", "confirm", "Kill"),
        Binding("n,escape", "cancel", "Cancel"),
    ]

    def __init__(self, pid: int, cwd: str) -> None:
        super().__init__()
        self._pid = pid
        self._cwd = cwd

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Label(
                f"Send SIGTERM to claude session?\n\nPID {self._pid}\n{self._cwd}",
                id="confirm-msg",
            )
            with Horizontal(id="confirm-buttons"):
                yield Button("Kill (y)", variant="error", id="kill")
                yield Button("Cancel (n)", variant="primary", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "kill")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class ClaudeWatcherApp(App):
    """Monitor running Claude Code sessions and tail the selected one."""

    TITLE = "claude_watcher"
    SUB_TITLE = "live Claude Code sessions"

    CSS = """
    DataTable {
        height: 45%;
        border-bottom: heavy $panel;
    }
    RichLog {
        height: 1fr;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh_now", "Refresh"),
        Binding("f", "toggle_follow", "Follow"),
        Binding("k", "kill_session", "Kill"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._self_pid = os.getpid()
        self._sessions: dict[int, Session] = {}
        self.selected_pid: int | None = None
        self._feed_parent: str | None = None  # parent session jsonl path
        self._feed_offsets: dict[str, int] = {}  # path -> byte offset (parent + subagents)
        self.follow: bool = True

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield DataTable(id="procs", cursor_type="row", zebra_stripes=True)
            yield RichLog(id="feed", highlight=False, markup=False, wrap=True, auto_scroll=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#procs", DataTable)
        table.add_columns("PID", "MODE", "STATE", "TOOL", "CWD", "ETIME", "CPU%", "OUT-TOK", "MSG")
        feed = self.query_one("#feed", RichLog)
        feed.write(Text("Select a session (↑/↓) to watch its live feed.", style="grey50"))
        self.refresh_procs()
        self.set_interval(PROC_INTERVAL, self.refresh_procs)
        self.set_interval(FEED_INTERVAL, self.refresh_feed)

    # -- process/session table ------------------------------------------------

    @work(exclusive=True, group="discover")
    async def refresh_procs(self) -> None:
        sessions = await asyncio.to_thread(self._discover_blocking)
        self._sessions = {s.proc.pid: s for s in sessions}
        self._populate_table(sessions)

    def _discover_blocking(self) -> list[Session]:
        procs = discover_procs(self._self_pid)
        return build_sessions(procs)

    def _populate_table(self, sessions: list[Session]) -> None:
        table = self.query_one("#procs", DataTable)
        prev = self.selected_pid
        table.clear()
        pids: list[int] = []
        for s in sessions:
            pids.append(s.proc.pid)
            table.add_row(*self._row_cells(s), key=str(s.proc.pid))

        if not pids:
            self.selected_pid = None
            self._switch_feed(None)
            return

        target = prev if prev in pids else pids[0]
        try:
            index = pids.index(target)
            table.move_cursor(row=index, animate=False)
        except ValueError:
            pass
        self._switch_feed(target)

    def _row_cells(self, s: Session) -> list:
        st = s.status
        state = st.state if st else State.UNKNOWN
        label = state.value
        if st and st.active_subagents > 0:
            label = f"{label} ({st.active_subagents})"
        state_cell = Text(label, style=_STATE_STYLE.get(state, "white"))
        mode_cell = "-p" if s.proc.mode is Mode.HEADLESS else "tty"
        tool = (st.tool_name if st and st.tool_name else "") or ""
        cwd = _shorten_cwd(s.proc.cwd)
        if s.ambiguous:
            cwd = "*" + cwd
        if s.jsonl_path and not s.cwd_validated:
            cwd = "!" + cwd
        out_tok = ""
        if st and st.output_tokens is not None:
            out_tok = str(st.output_tokens)
        msg = _shorten_msg(st.last_text if st else None)
        return [
            str(s.proc.pid),
            mode_cell,
            state_cell,
            tool,
            cwd,
            _fmt_etime(s.proc.etime),
            f"{s.proc.cpu:.1f}",
            out_tok,
            msg,
        ]

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        key = event.row_key
        if key is None or key.value is None:
            return
        try:
            pid = int(key.value)
        except ValueError:
            return
        self._switch_feed(pid)

    # -- live feed ------------------------------------------------------------

    def _switch_feed(self, pid: int | None) -> None:
        """Point the feed at a session. Idempotent for the current selection."""
        session = self._sessions.get(pid) if pid is not None else None
        new_path = session.jsonl_path if session else None
        if pid == self.selected_pid and new_path == self._feed_parent:
            return

        self.selected_pid = pid
        self._feed_parent = new_path
        # Seed every offset at the current EOF synchronously — parent plus any
        # subagent files already present — so the tail starts at the end and
        # never replays a huge file or a finished subagent's whole history.
        self._feed_offsets = {}
        if new_path:
            st = stat_file(new_path)
            self._feed_offsets[new_path] = st[0] if st else 0
            for sa in list_subagent_files(new_path):
                sst = stat_file(sa)
                self._feed_offsets[str(sa)] = sst[0] if sst else 0

        feed = self.query_one("#feed", RichLog)
        feed.clear()
        if new_path is None:
            if session is not None:
                feed.write(Text("(no session file resolved for this process yet)", style="grey50"))
            return
        title = Text(f"── {Path(new_path).stem}", style="bold")
        if session and session.ambiguous:
            title.append("  [ambiguous: shares cwd with another session]", style="red")
        feed.write(title)
        self._seed_feed(new_path)

    @work(exclusive=True, group="feed")
    async def _seed_feed(self, path: str) -> None:
        """Seed the feed with the parent's recent events for context."""
        entries, end_offset = await asyncio.to_thread(read_tail, path)
        if path != self._feed_parent:
            return  # selection changed while reading
        feed = self.query_one("#feed", RichLog)
        events = [ev for e in entries if (ev := parse_entry(e)) is not None]
        for ev in events[-SEED_EVENTS:]:
            feed.write(_feed_line(ev))
        self._feed_offsets[path] = end_offset

    @work(exclusive=True, group="feed")
    async def refresh_feed(self) -> None:
        parent = self._feed_parent
        if not parent:
            return
        events, new_offsets = await asyncio.to_thread(
            self._collect_feed_updates, parent, dict(self._feed_offsets)
        )
        if parent != self._feed_parent:
            return  # selection changed while reading
        feed = self.query_one("#feed", RichLog)
        for ev, tag in events:
            feed.write(_feed_line(ev, tag))
        self._feed_offsets.update(new_offsets)

    def _collect_feed_updates(
        self, parent: str, offsets: dict[str, int]
    ) -> tuple[list[tuple[FeedEvent, str | None]], dict[str, int]]:
        """Read new lines from the parent and every subagent file (blocking).

        Newly-spawned subagent files (not yet tracked) are read from the start
        so their launch is captured; existing files are tailed from their
        offset. Events from all files are merged and sorted by timestamp so the
        interleaving reads chronologically.
        """
        paths = dict(offsets)
        for sa in list_subagent_files(parent):
            sp = str(sa)
            if sp not in paths:
                paths[sp] = 0  # new subagent -> capture from its beginning

        collected: list[tuple] = []
        new_offsets: dict[str, int] = {}
        for path, off in paths.items():
            entries, new_off = read_incremental(path, off)
            new_offsets[path] = new_off
            tag = None if path == parent else _agent_tag(path)
            for e in entries:
                ev = parse_entry(e)
                if ev is not None:
                    collected.append((ev.ts, ev, tag))

        # Sort by timestamp; entries lacking one go last (the bool key keeps us
        # from ever comparing a datetime against None).
        collected.sort(key=lambda t: (t[0] is None, t[0]))
        return [(ev, tag) for _, ev, tag in collected], new_offsets

    # -- actions --------------------------------------------------------------

    def action_refresh_now(self) -> None:
        self.refresh_procs()

    def action_toggle_follow(self) -> None:
        self.follow = not self.follow
        feed = self.query_one("#feed", RichLog)
        feed.auto_scroll = self.follow
        self.notify(f"Follow {'on' if self.follow else 'off'}")

    def action_kill_session(self) -> None:
        pid = self.selected_pid
        if pid is None:
            self.notify("No session selected", severity="warning")
            return
        session = self._sessions.get(pid)
        cwd = _shorten_cwd(session.proc.cwd) if session else "?"

        def on_result(confirmed: bool | None) -> None:
            if confirmed:
                self._kill(pid)

        self.push_screen(ConfirmKill(pid, cwd), on_result)

    def _kill(self, pid: int) -> None:
        outcome = terminate_proc(pid)
        if outcome == "sent":
            self.notify(f"Sent SIGTERM to PID {pid}")
        elif outcome == "gone":
            self.notify(f"PID {pid} already exited", severity="warning")
        else:  # "denied"
            self.notify(f"Not permitted to kill PID {pid}", severity="error")
        self.refresh_procs()


def main() -> None:
    ClaudeWatcherApp().run()


if __name__ == "__main__":
    main()
