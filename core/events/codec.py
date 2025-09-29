from __future__ import annotations
import json
from typing import Optional, Tuple, Type
from .types import (
    EventType,
    BaseEvent,
    ChatMessageEvent,
    ExecutionEvent,
    ErrorEvent,
)

_TAG = "[Event]"

_event_type_map = {
    EventType.CHAT_MESSAGE.value: ChatMessageEvent,
    EventType.EXECUTION.value: ExecutionEvent,
    EventType.ERROR.value: ErrorEvent,
}


def format_event_line(event: BaseEvent) -> str:
    return f"{_TAG} " + json.dumps(event.to_dict(), ensure_ascii=False)


def parse_event_line(line: str) -> Optional[BaseEvent]:
    line = line.rstrip("\n")
    if not line.startswith(_TAG):
        return None
    payload = line[len(_TAG):].strip()
    if not payload:
        return None
    try:
        data = json.loads(payload)
        etype = data.get("type")
        cls: Optional[Type[BaseEvent]] = _event_type_map.get(etype)
        if not cls:
            return None
        # Rehydrate by matching constructor signature; simpler to unpack and ignore unknowns
        if cls is ChatMessageEvent:
            obj = ChatMessageEvent(role=data.get("role","assistant"), content=data.get("content",""))
        elif cls is ExecutionEvent:
            obj = ExecutionEvent(phase=data.get("phase","start"), target=data.get("target"))
        elif cls is ErrorEvent:
            obj = ErrorEvent(message=data.get("message",""), details=data.get("details"))
        else:
            return None
        # Core fields
        obj.timestamp = data.get("timestamp", obj.timestamp)
        obj.event_id = data.get("event_id", obj.event_id)
        obj.execution_id = data.get("execution_id")
        return obj
    except Exception:
        return None
