"""Event types for Fractalic event system.

Provides EventType enum with all event type constants.
All events are sent via HTTP to server.py using emit_event() from core.event_emitters.

The emit_event() function is type-agnostic and accepts EventType + any payload fields.
"""
from .types import EventType

__all__ = ["EventType"]
