from __future__ import annotations
from typing import Optional
from .types import BaseEvent, ChatMessageEvent
from .codec import format_event_line


class StdoutEventSink:
    """Writes normalized tagged lines to stdout.

    Intentionally does NOT duplicate the legacy [ChatMessage] format yet.
    We can emit *both* if backward compatibility is required.
    """

    def __init__(self, also_chat_compat: bool = True):
        self.also_chat_compat = also_chat_compat

    def __call__(self, event: BaseEvent) -> None:
        try:
            import sys
            line = format_event_line(event)
            sys.stdout.write(line + "\n")
            if self.also_chat_compat and isinstance(event, ChatMessageEvent):
                # Preserve existing consumer expectations
                compat = f"[ChatMessage] {{\"role\": \"{event.role}\", \"content\": {json_escape(event.content)} }}"
                sys.stdout.write(compat + "\n")
            sys.stdout.flush()
        except Exception:
            pass


def json_escape(s: str) -> str:
    # Minimal JSON string escaping without full json.dumps overhead for small hot path
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n') + '"'
