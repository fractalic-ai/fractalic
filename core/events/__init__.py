"""Minimal passive events layer.

Provides:
- EventType (enum)
- Base event dataclasses
- EventBus (in-process, synchronous)
- StdoutEventSink (optional line emission)
- parse_event_line / format_event_line helpers

Design goals:
- Zero external dependencies
- No server coupling or storage side-effects
- Type clarity & extendability

The server may read stdout lines and convert back to Python events;
HTTP streaming can reuse the same serialization.
"""
from .types import EventType, BaseEvent, ChatMessageEvent, ExecutionEvent, ErrorEvent
from .bus import EventBus, GlobalEventBus
from .sinks import StdoutEventSink
from .codec import format_event_line, parse_event_line

__all__ = [
    "EventType",
    "BaseEvent",
    "ChatMessageEvent",
    "ExecutionEvent",
    "ErrorEvent",
    "EventBus",
    "GlobalEventBus",
    "StdoutEventSink",
    "format_event_line",
    "parse_event_line",
]
