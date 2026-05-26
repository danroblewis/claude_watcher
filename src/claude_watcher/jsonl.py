"""Read Claude Code session JSONL files without loading them whole.

Session files are append-only, one JSON object per line, and can grow to tens
or hundreds of MB. So we never read the whole file: we seek near the end for a
status snapshot, and track a byte offset to tail only newly-appended bytes.

This module is pure aside from file reads; all state (the per-file offset) is
owned by the caller and threaded through the function signatures.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

TAIL_WINDOW = 64 * 1024  # bytes to look back for a status snapshot


def stat_file(path: str | Path) -> tuple[int, float] | None:
    """Return (size_bytes, mtime_epoch) or None if the file is gone."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_size, st.st_mtime


def _parse_lines(blob: bytes) -> list[dict]:
    """Parse complete JSON lines from a byte blob, skipping junk."""
    entries: list[dict] = []
    for raw in blob.split(b"\n"):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(obj, dict):
            entries.append(obj)
    return entries


def read_tail(
    path: str | Path, window: int = TAIL_WINDOW
) -> tuple[list[dict], int]:
    """Read the last `window` bytes and parse the complete lines in them.

    Returns (entries, end_offset) where end_offset is the file size (use it to
    seed an incremental tail so we only see *new* lines afterwards). If we did
    not start at byte 0, the first fragment is a partial line and is dropped.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            start = max(0, size - window)
            f.seek(start)
            blob = f.read()
    except OSError:
        return [], 0

    if start > 0:
        # Drop everything up to and including the first newline: it is the
        # tail of a line whose start we skipped.
        nl = blob.find(b"\n")
        blob = blob[nl + 1 :] if nl != -1 else b""

    return _parse_lines(blob), size


def read_incremental(
    path: str | Path, last_offset: int
) -> tuple[list[dict], int]:
    """Read bytes appended since `last_offset` and parse complete lines.

    Returns (entries, new_offset). A trailing fragment with no newline is an
    in-progress write: it is not parsed, and `new_offset` is positioned at its
    start so it is re-read once complete. If the file shrank (truncation or
    rotation), we restart from offset 0.
    """
    st = stat_file(path)
    if st is None:
        return [], last_offset
    size, _ = st

    if size < last_offset:
        last_offset = 0  # truncated / rotated
    if size == last_offset:
        return [], last_offset

    try:
        with open(path, "rb") as f:
            f.seek(last_offset)
            blob = f.read()
    except OSError:
        return [], last_offset

    if blob.endswith(b"\n"):
        new_offset = last_offset + len(blob)
        complete = blob
    else:
        # Keep the trailing partial line for next time.
        nl = blob.rfind(b"\n")
        if nl == -1:
            return [], last_offset  # nothing complete yet
        complete = blob[: nl + 1]
        new_offset = last_offset + nl + 1

    return _parse_lines(complete), new_offset
