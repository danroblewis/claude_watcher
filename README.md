# claude_watcher

A minimal terminal UI that shows what your running **Claude Code** (`claude`) CLI
sessions are doing right now — including background `claude -p` runs whose stdout
is buffered and otherwise invisible until they finish.

It finds live `claude` processes, maps each to the session JSONL file Claude Code
writes under `~/.claude/projects/`, and shows a live status per session plus a
drill-down feed you can watch stream in real time.

## Run

```bash
uv run claude-watcher
# or
uv run python -m claude_watcher
```

## What you see

- **Top table** — one row per running `claude` process:
  - `PID`, `MODE` (`-p` = headless/buffered, `tty` = interactive)
  - `STATE` — `working` (running a tool / mid-turn), `thinking`, `idle`
    (turn finished), `stalled` (no activity for >30s), `unknown`. When the
    session is blocked on subagents it shows `working (N subagents)` instead of
    going stalled.
  - `TOOL` — the tool currently in use
  - `CWD` — working directory (`*` = shares a cwd with another session, so the
    file pairing is a best-effort guess; `!` = the session file's recorded cwd
    didn't validate)
  - `ETIME`, `CPU%`, `OUT-TOK` (output tokens)
- **Bottom feed** — the live event stream for the highlighted session: tool calls
  (`▸`), tool results (`◂`), assistant text (`💬`), thinking (`💭`), user input (`👤`).
  Subagents (Task/Agent) are tailed too: their events are merged in chronologically
  and indented with a short agent tag (`└a708`), so you can watch parallel
  subagents work in real time.

### Subagents

Claude Code writes each subagent's transcript to its own file under
`~/.claude/projects/<project>/<session-uuid>/subagents/agent-*.jsonl`, separate
from the parent session file. The monitor discovers these for the selected
session, tails the active ones live (existing ones start at EOF; newly-spawned
ones are shown from their first line), and counts the ones currently being
written to drive the `working (N subagents)` status.

## Keys

| key | action |
|-----|--------|
| ↑ / ↓ | highlight a session (its feed loads below) |
| `f` | toggle follow (auto-scroll the feed) |
| `r` | refresh the process list now |
| `q` | quit |

## How it works (macOS)

- Processes: `pgrep -f 'claude( |$)'`, excluding the Desktop app; mode from `-p`/`--print`.
- cwd of each PID: `lsof -a -p <pid> -d cwd` (no `/proc` on macOS).
- Session file: the cwd is encoded into the project-dir name (`/` and `.` → `-`)
  and the most-recently-modified `*.jsonl` is read, validated against the `cwd`
  field stored inside the file.
- Files can be very large, so they are never read whole: a 64 KB tail gives the
  status, and a byte offset is tracked so the feed only reads newly-appended lines.

The logic layer (`procs`, `sessions`, `jsonl`, `status`, `models`) is pure and
unit-tested; `app` is the only Textual/async module.

## Develop

```bash
uv run pytest        # unit tests for the logic layer
```
