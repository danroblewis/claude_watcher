"""Cumulative output-token accounting for a session and its subagents.

The naive "tokens" reading in ``derive_status`` is wrong for two reasons that
only show up on real transcripts:

1. ``usage.output_tokens`` is **per assistant message**, not a running total,
   so the last entry's value alone undercounts a long session (and freezes
   while a parent waits on subagents).
2. A single assistant message is logged across several JSONL lines (one per
   content block) that each repeat the *same* ``usage``. Summing per line
   overcounts ~Nx, so we key by ``message.id`` and keep one value per id.

A session's true output is therefore the deduped sum across the parent file
**and every subagent transcript**. We accumulate it incrementally — each file
advances a byte offset so only newly-appended bytes are parsed per poll (one
full read the first time a file is seen) — mirroring the feed's tail offsets.
This object is stateful and is meant to be held across polls by the caller.
"""

from __future__ import annotations

from claude_watcher.jsonl import read_incremental, stat_file
from claude_watcher.sessions import list_subagent_files


class TokenLedger:
    """Tracks deduped cumulative output tokens per session file tree."""

    def __init__(self) -> None:
        self._offsets: dict[str, int] = {}  # path -> bytes already consumed
        self._by_msg: dict[str, dict[str, int]] = {}  # path -> {message_id: out}

    def output_tokens(self, parent_path: str) -> int:
        """Ingest any new bytes and return the session's total output tokens."""
        total = 0
        paths = [parent_path] + [str(p) for p in list_subagent_files(parent_path)]
        for path in paths:
            self._ingest(path)
            total += sum(self._by_msg.get(path, {}).values())
        return total

    def file_output_tokens(self, path: str) -> int | None:
        """Output tokens for a single already-ingested file, or None if unseen.

        Call ``output_tokens`` for the parent first; it ingests every subagent
        file, so their per-file totals are then available here.
        """
        msgs = self._by_msg.get(path)
        return sum(msgs.values()) if msgs is not None else None

    def _ingest(self, path: str) -> None:
        st = stat_file(path)
        if st is None:
            return
        size = st[0]
        offset = self._offsets.get(path, 0)
        if size < offset:  # truncated/rotated -> recount from the start
            self._by_msg.pop(path, None)
            offset = 0
        entries, new_offset = read_incremental(path, offset)
        self._offsets[path] = new_offset
        if not entries:
            return
        msgs = self._by_msg.setdefault(path, {})
        for e in entries:
            if e.get("type") != "assistant":
                continue
            msg = e.get("message") or {}
            usage = msg.get("usage") or {}
            mid = msg.get("id")
            out = usage.get("output_tokens")
            if mid is None or out is None:
                continue
            msgs[mid] = out  # one value per message id (dedupes repeated lines)
