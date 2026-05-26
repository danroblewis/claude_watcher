"""Incremental per-file tracking of each transcript's last assistant text.

``derive_status`` only sees a 64KB tail, so an agent that has been running
tools for a while has no recent text there and its MSG goes blank. Reading
from the start — and persisting across polls — keeps the last real message
available however long ago it was sent. Each file advances a byte offset so a
poll parses only newly-appended bytes (one full read the first time a file is
seen), mirroring the feed's tail offsets. Held across polls by the caller.
"""

from __future__ import annotations

from claude_watcher.jsonl import read_incremental, stat_file
from claude_watcher.sessions import list_subagent_files

_LAST_TEXT_CAP = 200  # store at most this many chars; the UI trims further


def _assistant_text(content: object) -> str | None:
    """The last text content block in an assistant message, flattened, or None."""
    if not isinstance(content, list):
        return None
    for item in reversed(content):
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                return " ".join(text.split())[:_LAST_TEXT_CAP]
    return None


class TranscriptTail:
    """Remembers the most recent assistant text per transcript file."""

    def __init__(self) -> None:
        self._offsets: dict[str, int] = {}  # path -> bytes already consumed
        self._last_text: dict[str, str] = {}  # path -> most recent assistant text

    def update(self, parent_path: str) -> None:
        """Ingest any new bytes from the session and its subagent files."""
        for path in [parent_path] + [str(p) for p in list_subagent_files(parent_path)]:
            self._ingest(path)

    def last_text(self, path: str) -> str | None:
        """Most recent assistant text seen anywhere in `path`, or None."""
        return self._last_text.get(path)

    def _ingest(self, path: str) -> None:
        st = stat_file(path)
        if st is None:
            return
        size = st[0]
        offset = self._offsets.get(path, 0)
        if size < offset:  # truncated/rotated -> re-read from the start
            self._last_text.pop(path, None)
            offset = 0
        entries, new_offset = read_incremental(path, offset)
        self._offsets[path] = new_offset
        for e in entries:
            if e.get("type") != "assistant":
                continue
            text = _assistant_text((e.get("message") or {}).get("content"))
            if text:
                self._last_text[path] = text  # persists until a newer text arrives
