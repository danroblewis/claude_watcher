"""Textual TUI: a live table of running claude sessions + a drill-down feed.

This is the only module that imports Textual or runs async. All blocking work
(pgrep/lsof/ps + file reads) is pushed off the event loop with
``asyncio.to_thread`` inside ``exclusive`` workers, so the UI never stalls and
slow polls cannot pile up.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.coordinate import Coordinate
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, Label, RichLog

from claude_watcher.jsonl import read_incremental, read_tail, stat_file
from claude_watcher.models import FeedEvent, Mode, Session, State, SubagentInfo
from claude_watcher.procs import discover_procs, terminate_proc
from claude_watcher.sessions import (
    SUBAGENT_ACTIVE_WINDOW,
    build_sessions,
    list_subagent_files,
)
from claude_watcher.status import context_percent, derive_status, parse_entry
from claude_watcher.tokens import TokenLedger

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


def _fmt_tokens(n: int | None) -> str:
    """Compact token count: 9070 -> "9.1k", 6284048 -> "6.3m"."""
    if n is None:
        return ""
    if n < 1000:
        return str(n)
    if n < 999_500:  # below the point where rounding would print "1000.0k"
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.1f}m"


def _ctx_cell(st) -> Text:
    """Color-coded context-fill percentage: green < 50, yellow < 80, red above."""
    pct = context_percent(st.context_tokens, st.model) if st else None
    if pct is None:
        return Text("")
    style = "green" if pct < 50 else "yellow" if pct < 80 else "bold red"
    return Text(f"{pct:.0f}%", style=style)


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


def _subagent_key(pid: int, path: str) -> str:
    """Table row key for a subagent row: parses back via ``_parse_subagent_key``."""
    return f"sub:{pid}:{path}"


def _parse_subagent_key(key: str) -> tuple[int, str] | None:
    """(pid, subagent_path) for a subagent row key, or None if it isn't one."""
    if not key.startswith("sub:"):
        return None
    _, pid_str, path = key.split(":", 2)  # path may contain ':'; maxsplit keeps it whole
    return int(pid_str), path


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
        Binding("right", "expand", "Expand", priority=True),
        Binding("left", "collapse", "Collapse", priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._self_pid = os.getpid()
        self._ledger = TokenLedger()
        self._sessions: dict[int, Session] = {}
        self._expanded: set[int] = set()  # pids whose subagents are shown
        self._table_keys: list[str] = []  # ordered row keys currently in the table
        self.selected_pid: int | None = None  # owning process of the selected row
        self._selected_key: str | None = None  # full key of the selected table row
        self._feed_parent: str | None = None  # parent session jsonl path
        self._feed_focus: str | None = None  # a subagent path to show alone, or None
        self._feed_offsets: dict[str, int] = {}  # path -> byte offset
        self._feed_gen: int = 0  # bumped on every selection change; stale workers bail
        self.follow: bool = True

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield DataTable(id="procs", cursor_type="row", zebra_stripes=True)
            yield RichLog(id="feed", highlight=False, markup=False, wrap=True, auto_scroll=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#procs", DataTable)
        table.add_columns(
            "PID", "MODE", "STATE", "TOOL", "CWD", "ETIME", "CPU%", "OUT-TOK", "CTX", "MSG"
        )
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
        sessions = build_sessions(procs)
        # Cumulative token accounting needs to scan whole files (parent +
        # subagents), so it lives in the stateful ledger rather than the
        # tail-window status. Runs here, off the event loop.
        for s in sessions:
            if not s.jsonl_path:
                continue
            s.total_output_tokens = self._ledger.output_tokens(s.jsonl_path)
            s.subagent_paths = [str(p) for p in sorted(list_subagent_files(s.jsonl_path))]
            if s.proc.pid in self._expanded:
                s.subagents = self._build_subagent_infos(s)
        return sessions

    def _build_subagent_infos(self, session: Session) -> list[SubagentInfo]:
        """Tail each subagent transcript for an expanded session (off-loop)."""
        now = datetime.now(timezone.utc)
        now_epoch = time.time()
        infos: list[SubagentInfo] = []
        for path in session.subagent_paths:
            entries, _ = read_tail(path)
            status = derive_status(entries, now) if entries else None
            st = stat_file(path)
            active = bool(st and now_epoch - st[1] <= SUBAGENT_ACTIVE_WINDOW)
            infos.append(
                SubagentInfo(
                    path=path,
                    tag=_agent_tag(path),
                    status=status,
                    output_tokens=self._ledger.file_output_tokens(path),
                    active=active,
                )
            )
        return infos

    def _build_rows(self, sessions: list[Session]) -> list[tuple[str, list]]:
        """Ordered (row_key, cells) for the table: each session, plus its
        subagent rows when expanded."""
        rows: list[tuple[str, list]] = []
        for s in sessions:
            rows.append((str(s.proc.pid), self._row_cells(s)))
            if s.proc.pid in self._expanded:
                for info in s.subagents:
                    rows.append((_subagent_key(s.proc.pid, info.path), self._subagent_row_cells(info)))
        return rows

    def _populate_table(self, sessions: list[Session]) -> None:
        table = self.query_one("#procs", DataTable)
        rows = self._build_rows(sessions)
        new_keys = [k for k, _ in rows]

        # Fast path: the same rows in the same order (the steady-state poll).
        # Update cells in place rather than clear()+re-add, which would blank
        # the table for a frame and reset the scroll position — the flicker.
        if new_keys and new_keys == self._table_keys:
            for i, (_, cells) in enumerate(rows):
                for j, cell in enumerate(cells):
                    table.update_cell_at(Coordinate(i, j), cell, update_width=True)
            return

        # Structure changed (rows added/removed/reordered, or expand/collapse):
        # rebuild and restore the cursor onto the previously selected row.
        prev = self._selected_key
        table.clear()
        for key, cells in rows:
            table.add_row(*cells, key=key)
        self._table_keys = new_keys
        if not new_keys:
            self._apply_selection(None)
            return
        target = prev if prev in new_keys else new_keys[0]
        table.move_cursor(row=new_keys.index(target), animate=False)
        self._apply_selection(target)

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
        out_tok = _fmt_tokens(s.total_output_tokens)
        msg = _shorten_msg(st.last_text if st else None)
        # Disclosure marker when a session has subagents to expand into.
        marker = ""
        if s.subagent_paths:
            marker = "▾ " if s.proc.pid in self._expanded else "▸ "
        return [
            f"{marker}{s.proc.pid}",
            mode_cell,
            state_cell,
            tool,
            cwd,
            _fmt_etime(s.proc.etime),
            f"{s.proc.cpu:.1f}",
            out_tok,
            _ctx_cell(st),
            msg,
        ]

    def _subagent_row_cells(self, info: SubagentInfo) -> list:
        st = info.status
        if info.active:
            state = st.state if st else State.UNKNOWN
            state_cell = Text(state.value, style=_STATE_STYLE.get(state, "white"))
        else:
            state_cell = Text("done", style="grey50")
        tool = (st.tool_name if st and st.tool_name else "") or ""
        return [
            Text(f"  └ {info.tag}", style="magenta"),
            "sub",
            state_cell,
            tool,
            "",  # CWD — subagents share the parent's
            "",  # ETIME — not a process
            "",  # CPU% — not a process
            _fmt_tokens(info.output_tokens),
            _ctx_cell(st),
            _shorten_msg(st.last_text if st else None),
        ]

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        key = event.row_key
        if key is not None and key.value is not None:
            self._apply_selection(key.value)

    def _apply_selection(self, key: str | None) -> None:
        """Point the feed at whatever row `key` names (parent or subagent)."""
        self._selected_key = key
        if key is None:
            self.selected_pid = None
            self._switch_feed(None, None)
            return
        sub = _parse_subagent_key(key)
        if sub is not None:
            pid, sub_path = sub
            self.selected_pid = pid  # kill still targets the owning process
            session = self._sessions.get(pid)
            self._switch_feed(session.jsonl_path if session else None, sub_path)
            return
        try:
            pid = int(key)
        except ValueError:
            return
        self.selected_pid = pid
        session = self._sessions.get(pid)
        self._switch_feed(session.jsonl_path if session else None, None)

    # -- live feed ------------------------------------------------------------

    def _feed_read_paths(self) -> list[str]:
        """Files the feed currently tails: one subagent if focused, else the
        whole session (parent + subagents)."""
        if self._feed_focus is not None:
            return [self._feed_focus]
        if self._feed_parent is None:
            return []
        return [self._feed_parent] + [str(p) for p in list_subagent_files(self._feed_parent)]

    def _switch_feed(self, parent: str | None, focus: str | None) -> None:
        """Point the feed at a session (focus=None) or one subagent. Idempotent."""
        if parent == self._feed_parent and focus == self._feed_focus:
            return

        self._feed_parent = parent
        self._feed_focus = focus
        self._feed_gen += 1
        gen = self._feed_gen
        # Seed every offset at the current EOF so the tail starts at the end and
        # never replays a huge file or a finished subagent's whole history.
        self._feed_offsets = {}
        for p in self._feed_read_paths():
            st = stat_file(p)
            self._feed_offsets[p] = st[0] if st else 0

        feed = self.query_one("#feed", RichLog)
        feed.clear()
        if parent is None:
            if self.selected_pid is not None:
                feed.write(Text("(no session file resolved for this process yet)", style="grey50"))
            return
        if focus is not None:
            feed.write(Text(f"── {_agent_tag(focus)}  [subagent]", style="bold magenta"))
            self._seed_feed(focus, gen)
        else:
            feed.write(Text(f"── {Path(parent).stem}", style="bold"))
            self._seed_feed(parent, gen)

    @work(exclusive=True, group="feed")
    async def _seed_feed(self, path: str, gen: int) -> None:
        """Seed the feed with a file's recent events for context."""
        entries, end_offset = await asyncio.to_thread(read_tail, path)
        if gen != self._feed_gen:
            return  # selection changed while reading
        feed = self.query_one("#feed", RichLog)
        events = [ev for e in entries if (ev := parse_entry(e)) is not None]
        for ev in events[-SEED_EVENTS:]:
            feed.write(_feed_line(ev))
        self._feed_offsets[path] = end_offset

    @work(exclusive=True, group="feed")
    async def refresh_feed(self) -> None:
        if self._feed_parent is None:
            return
        gen = self._feed_gen
        events, new_offsets = await asyncio.to_thread(
            self._collect_feed_updates,
            self._feed_parent,
            self._feed_focus,
            dict(self._feed_offsets),
        )
        if gen != self._feed_gen:
            return  # selection changed while reading
        feed = self.query_one("#feed", RichLog)
        for ev, tag in events:
            feed.write(_feed_line(ev, tag))
        self._feed_offsets.update(new_offsets)

    def _collect_feed_updates(
        self, parent: str, focus: str | None, offsets: dict[str, int]
    ) -> tuple[list[tuple[FeedEvent, str | None]], dict[str, int]]:
        """Read new lines from the tracked files (blocking).

        Focused mode tails a single subagent. Session mode tails the parent and
        every subagent; newly-spawned subagent files (not yet tracked) are read
        from the start so their launch is captured. Events are merged and sorted
        by timestamp so the interleaving reads chronologically.
        """
        if focus is not None:
            tracked = [focus]
        else:
            tracked = [parent] + [str(p) for p in list_subagent_files(parent)]
        paths = dict(offsets)
        for p in tracked:
            paths.setdefault(p, 0)  # new file -> capture from its beginning

        collected: list[tuple] = []
        new_offsets: dict[str, int] = {}
        for path, off in paths.items():
            entries, new_off = read_incremental(path, off)
            new_offsets[path] = new_off
            tag = None if focus is not None or path == parent else _agent_tag(path)
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

    def _current_row_key(self) -> str | None:
        table = self.query_one("#procs", DataTable)
        if table.row_count == 0:
            return None
        try:
            return table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        except Exception:
            return None

    def action_expand(self) -> None:
        if len(self.screen_stack) > 1:  # a modal is up; leave arrows to it
            return
        key = self._current_row_key()
        if key is None or _parse_subagent_key(key) is not None:
            return
        try:
            pid = int(key)
        except ValueError:
            return
        session = self._sessions.get(pid)
        if session and session.subagent_paths and pid not in self._expanded:
            self._expanded.add(pid)
            self.refresh_procs()  # re-discover so subagent rows get populated

    def action_collapse(self) -> None:
        if len(self.screen_stack) > 1:
            return
        key = self._current_row_key()
        if key is None:
            return
        sub = _parse_subagent_key(key)
        if sub is not None:  # on a subagent row: collapse parent, select it
            pid = sub[0]
            self._expanded.discard(pid)
            self._selected_key = str(pid)
            self.refresh_procs()
            return
        try:
            pid = int(key)
        except ValueError:
            return
        if pid in self._expanded:
            self._expanded.discard(pid)
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
