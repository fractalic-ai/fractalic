"""Event types for Fractalic event system.

Provides:
- EventType (enum) - all event type constants
- Base event dataclasses (for backward compatibility if needed)

All events are sent via HTTP to server.py using emit_event() from core.event_emitters.
"""
from .types import EventType, BaseEvent, ChatMessageEvent, ExecutionEvent, ErrorEvent

__all__ = [
    "EventType",
    "BaseEvent",
    "ChatMessageEvent",
    "ExecutionEvent",
    "ErrorEvent",
]
