import pytest
from core.events import ChatMessageEvent, format_event_line, parse_event_line


def test_chat_event_roundtrip():
    ev = ChatMessageEvent(role='assistant', content='hello')
    line = format_event_line(ev)
    parsed = parse_event_line(line)
    assert parsed is not None
    assert parsed.type.value == 'chat_message'
    assert parsed.role == 'assistant'
    assert parsed.content == 'hello'
