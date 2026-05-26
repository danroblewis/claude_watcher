"""Discover running `claude` processes and their OS metadata (macOS).

Uses only `pgrep`, `lsof`, and `ps` via subprocess with list argv (never a
shell). Every helper tolerates a process disappearing mid-poll by returning
None / [] rather than raising, since PIDs are racy by nature.
"""

from __future__ import annotations

import os
import signal
import subprocess

from claude_watcher.models import Mode, ProcEntry

# Match a process whose command starts with `claude` followed by a space or
# end-of-arg, so `claude`, `claude -p ...`, `claude -r` all match but paths
# like `/foo/claudebot` do not get caught by the word boundary.
_PGREP_PATTERN = r"claude( |$)"

# Desktop app processes also contain "claude" but live under the .app bundle.
_DESKTOP_MARKER = "/Applications/Claude.app"

_SUBPROCESS_TIMEOUT = 5.0


def _run(argv: list[str]) -> str | None:
    """Run a command, returning stripped stdout, or None on failure/timeout.

    A non-zero exit code is treated as "no answer" (None) for `lsof`/`ps`,
    which is the correct behaviour when a PID has already exited.
    """
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def list_claude_pids() -> list[int]:
    """Return PIDs of processes matching the claude command pattern.

    `pgrep` exits 1 when there are no matches; that is not an error, just an
    empty list.
    """
    try:
        proc = subprocess.run(
            ["pgrep", "-f", _PGREP_PATTERN],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    # rc 0 = matches, rc 1 = no matches, rc >1 = real error.
    if proc.returncode not in (0, 1):
        return []
    pids: list[int] = []
    for line in proc.stdout.split():
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return pids


def proc_args(pid: int) -> str | None:
    """Full command line for a PID via `ps -o args=`."""
    out = _run(["ps", "-o", "args=", "-p", str(pid)])
    if out is None:
        return None
    out = out.strip()
    return out or None


def classify_mode(args: str) -> Mode:
    """Headless if the command line carries a print/headless flag."""
    if " -p " in args or args.endswith(" -p") or "--print" in args:
        return Mode.HEADLESS
    return Mode.INTERACTIVE


def is_watchable(args: str, self_pid: int, pid: int) -> bool:
    """Exclude the Desktop app and our own process."""
    if pid == self_pid:
        return False
    if _DESKTOP_MARKER in args:
        return False
    return True


def proc_cwd(pid: int) -> str | None:
    """Current working directory of a PID via `lsof`.

    macOS has no /proc, so lsof is the only portable route. Output lines look
    like::

        2.1.150 43257 user cwd DIR 1,14 96 37323136 /Users/user/project

    We take the last whitespace-separated field of the last data line. Paths
    with spaces would be mangled by naive splitting, so we instead take
    everything after the NODE column. To keep it robust we split on runs of
    whitespace but rejoin the tail.
    """
    out = _run(["lsof", "-a", "-p", str(pid), "-d", "cwd"])
    if not out:
        return None
    data_lines = [ln for ln in out.splitlines() if ln.strip()]
    # Drop the header line if present (starts with "COMMAND").
    data_lines = [ln for ln in data_lines if not ln.startswith("COMMAND")]
    if not data_lines:
        return None
    # The NAME column is the 9th field; everything from there on is the path
    # (handles spaces in the path). Fall back to last field if the layout is
    # unexpected.
    parts = data_lines[-1].split(None, 8)
    if len(parts) == 9:
        return parts[8] or None
    return data_lines[-1].split()[-1] or None


def proc_metadata(pid: int) -> dict | None:
    """Start time / elapsed / cpu / mem / state for a PID via `ps`.

    Returns a dict with keys: lstart, etime, cpu (float), mem (float), pstate.
    None if the process is gone.
    """
    # lstart is multi-word ("Sat May 24 16:03:02 2026"), so request it last and
    # split the rest off the front. We request the wordy field separately to
    # avoid ambiguity.
    etime = _run(["ps", "-o", "etime=", "-p", str(pid)])
    if etime is None:
        return None
    lstart = _run(["ps", "-o", "lstart=", "-p", str(pid)])
    cpu = _run(["ps", "-o", "%cpu=", "-p", str(pid)])
    mem = _run(["ps", "-o", "%mem=", "-p", str(pid)])
    pstate = _run(["ps", "-o", "state=", "-p", str(pid)])

    def _to_float(v: str | None) -> float:
        try:
            return float((v or "").strip())
        except ValueError:
            return 0.0

    return {
        "lstart": (lstart or "").strip(),
        "etime": etime.strip(),
        "cpu": _to_float(cpu),
        "mem": _to_float(mem),
        "pstate": (pstate or "").strip(),
    }


def discover_procs(self_pid: int) -> list[ProcEntry]:
    """Find all watchable claude processes and assemble ProcEntry records.

    PIDs that vanish between discovery and metadata collection are silently
    skipped.
    """
    entries: list[ProcEntry] = []
    for pid in list_claude_pids():
        args = proc_args(pid)
        if args is None:
            continue
        if not is_watchable(args, self_pid, pid):
            continue
        meta = proc_metadata(pid)
        if meta is None:
            continue  # exited mid-poll
        entries.append(
            ProcEntry(
                pid=pid,
                args=args,
                mode=classify_mode(args),
                cwd=proc_cwd(pid),
                lstart=meta["lstart"],
                etime=meta["etime"],
                cpu=meta["cpu"],
                mem=meta["mem"],
                pstate=meta["pstate"],
            )
        )
    entries.sort(key=lambda e: e.pid)
    return entries


def terminate_proc(pid: int) -> str:
    """Send SIGTERM to ``pid``; return "sent", "gone", or "denied".

    Like the rest of this module, a PID that has already exited is a normal
    outcome ("gone"), not an error.
    """
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "gone"
    except PermissionError:
        return "denied"
    return "sent"
