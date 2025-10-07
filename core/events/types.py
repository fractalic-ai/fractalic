from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
import time
import uuid


class EventType(str, Enum):
    CHAT_MESSAGE = "chat_message"
    EXECUTION = "execution"
    WORKFLOW_COMPLETE = "workflow_complete"  # Завершение выполнения @run операции (агента)
    ERROR = "error"
    AST_UPDATE = "ast_update"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    LLM_CHUNK = "llm_chunk"
    TOKEN_USAGE = "token_usage"
    BLOCK_PROCESSING = "block_processing"


def _now() -> float:
    return time.time()


def _gen_id() -> str:
    return uuid.uuid4().hex


@dataclass
class BaseEvent:
    # "type" is assigned by subclasses in __post_init__
    type: EventType = field(init=False)
    timestamp: float = field(default_factory=_now)
    event_id: str = field(default_factory=_gen_id)
    execution_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:  # minimal stable schema
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "event_id": self.event_id,
            "execution_id": self.execution_id,
        }


@dataclass
class ChatMessageEvent(BaseEvent):
    role: str = "assistant"
    content: str = ""

    def __post_init__(self):
        self.type = EventType.CHAT_MESSAGE  # enforce

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({"role": self.role, "content": self.content})
        return base


@dataclass
class ExecutionEvent(BaseEvent):
    phase: str = "start"  # start|complete|error
    target: Optional[str] = None  # path or description

    def __post_init__(self):
        self.type = EventType.EXECUTION

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({"phase": self.phase, "target": self.target})
        return base


@dataclass
class WorkflowCompleteEvent(BaseEvent):
    target: Optional[str] = None  # workflow file path
    return_content: Optional[str] = None  # content returned by @return
    status: str = "success"  # success|error

    def __post_init__(self):
        self.type = EventType.WORKFLOW_COMPLETE

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({"target": self.target, "return_content": self.return_content, "status": self.status})
        return base


@dataclass
class ErrorEvent(BaseEvent):
    message: str = ""
    details: Optional[str] = None

    def __post_init__(self):
        self.type = EventType.ERROR

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({"message": self.message, "details": self.details})
        return base
