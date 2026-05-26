"""Turn raw JSONL entries into feed lines and a derived session Status.

Key nuance learned from real session files: a single assistant turn is split
across several JSONL lines that *all* carry ``stop_reason == "tool_use"`` while
each individual line's content item is ``thinking``, ``text``, or ``tool_use``.
So state is decided by the latest assistant entry's *content item type*, not by
``stop_reason`` alone.
"""

from __future__ import annotations

from datetime import datetime, timezone

from claude_watcher.models import FeedEvent, State, Status

# Entry types that carry conversation content. Everything else
# (ai-title, attachment, last-prompt, queue-operation, file-history-snapshot,
# system/turn_duration, ...) is metadata noise for our purposes.
_MEANINGFUL = {"assistant", "user"}

_SNIPPET = 100  # max chars for a one-line feed entry / status text


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse an ISO-8601 UTC timestamp (`...Z`) into an aware datetime."""
    if not value:
        return None
    try:
        # fromisoformat handles the trailing Z natively on 3.11+, but be
        # explicit for older shapes.
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _truncate(text: str, limit: int = _SNIPPET) -> str:
    text = " ".join(text.split())  # collapse newlines/runs of whitespace
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _content_items(entry: dict) -> list:
    """Return the list of content items from a message entry.

    Assistant content is always a list; user content may be a plain string
    (the initial prompt) or a list (tool results).
    """
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, list):
        return content
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return []


def _tool_input_summary(name: str, tool_input: dict) -> str:
    """A short, tool-appropriate one-liner for a tool_use input."""
    if not isinstance(tool_input, dict):
        return ""
    for key in ("command", "file_path", "path", "pattern", "query", "prompt", "url"):
        val = tool_input.get(key)
        if isinstance(val, str) and val:
            return _truncate(val, 60)
    # Fallback: first short string value.
    for val in tool_input.values():
        if isinstance(val, str) and val:
            return _truncate(val, 60)
    return ""


def parse_entry(entry: dict) -> FeedEvent | None:
    """Map one JSONL object to a FeedEvent, or None if it is noise."""
    etype = entry.get("type")
    if etype not in _MEANINGFUL:
        return None
    ts = parse_timestamp(entry.get("timestamp"))
    role = (entry.get("message") or {}).get("role")

    if etype == "user":
        items = _content_items(entry)
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "tool_result":
                is_err = bool(item.get("is_error"))
                return FeedEvent(
                    ts=ts,
                    kind="tool_result",
                    text=f"result ({'err' if is_err else 'ok'})",
                    is_error=is_err,
                )
        # Plain user prompt.
        for item in items:
            if isinstance(item, dict) and item.get("type") == "text":
                return FeedEvent(ts=ts, kind="user", text=_truncate(item.get("text", "")))
        return None

    # assistant: report the most salient content item.
    items = _content_items(entry)
    # Prefer the last tool_use; else last text; else thinking.
    last_tool = None
    last_text = None
    last_thinking = None
    for item in items:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t == "tool_use":
            last_tool = item
        elif t == "text":
            last_text = item
        elif t == "thinking":
            last_thinking = item

    if last_tool is not None:
        name = last_tool.get("name", "tool")
        summary = _tool_input_summary(name, last_tool.get("input") or {})
        text = f"{name} {summary}".strip()
        return FeedEvent(ts=ts, kind="tool_use", text=text)
    if last_text is not None:
        return FeedEvent(ts=ts, kind="text", text=_truncate(last_text.get("text", "")))
    if last_thinking is not None:
        return FeedEvent(ts=ts, kind="thinking", text=_truncate(last_thinking.get("thinking", "")))
    return None


def derive_status(
    entries: list[dict],
    now_utc: datetime,
    stall_after_s: float = 30.0,
) -> Status:
    """Derive a Status from a window of recent entries (oldest→newest order).

    Walks newest→oldest. The first meaningful assistant entry determines the
    state via its content item type. Token usage and last text are pulled from
    the most recent entries that carry them.
    """
    status = Status(state=State.UNKNOWN)

    # Token usage + last_text: scan from the end for the first that has them.
    for entry in reversed(entries):
        if entry.get("type") != "assistant":
            continue
        usage = (entry.get("message") or {}).get("usage") or {}
        if status.context_tokens is None and usage:
            # Context fill = the whole prompt sent on the latest request, which
            # is the new input plus everything served from / written to cache.
            status.context_tokens = (
                (usage.get("input_tokens") or 0)
                + (usage.get("cache_read_input_tokens") or 0)
                + (usage.get("cache_creation_input_tokens") or 0)
            )
            status.model = (entry.get("message") or {}).get("model")
        if status.input_tokens is None and usage:
            status.input_tokens = usage.get("input_tokens")
            status.output_tokens = usage.get("output_tokens")
            status.cache_read_tokens = usage.get("cache_read_input_tokens")
        if status.last_text is None:
            for item in _content_items(entry):
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                    status.last_text = _truncate(item["text"])
                    break
        if status.input_tokens is not None and status.last_text is not None:
            break

    # Last timestamp across all entries.
    for entry in reversed(entries):
        ts = parse_timestamp(entry.get("timestamp"))
        if ts is not None:
            status.last_ts = ts
            break

    # State: from the most recent meaningful entry. A user prompt newer than
    # the last assistant reply means a new turn is in flight (the reply isn't
    # written yet) -> the session is working, not idle.
    for entry in reversed(entries):
        etype = entry.get("type")
        if etype == "user":
            items = _content_items(entry)
            is_tool_result = any(
                isinstance(it, dict) and it.get("type") == "tool_result" for it in items
            )
            if is_tool_result:
                continue  # mid-turn tool output; let the assistant entry decide
            status.state = State.WORKING  # a fresh prompt awaits a reply
            break
        if etype != "assistant":
            continue
        stop_reason = (entry.get("message") or {}).get("stop_reason")
        items = _content_items(entry)
        kinds = {item.get("type") for item in items if isinstance(item, dict)}
        if "tool_use" in kinds:
            status.state = State.WORKING
            for item in items:
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    status.tool_name = item.get("name")
                    status.tool_input_summary = _tool_input_summary(
                        status.tool_name or "", item.get("input") or {}
                    )
                    break
        elif "thinking" in kinds:
            status.state = State.THINKING
        elif "text" in kinds:
            # end_turn => done with the turn; tool_use stop_reason => mid-turn.
            status.state = State.IDLE if stop_reason == "end_turn" else State.WORKING
        else:
            status.state = State.WORKING
        break

    # Stall override: if the last activity is old, flag it regardless of state
    # (unless we never determined a state at all).
    if status.last_ts is not None and status.state != State.UNKNOWN:
        age = (now_utc - status.last_ts).total_seconds()
        if age > stall_after_s and status.state != State.IDLE:
            status.state = State.STALLED

    return status
