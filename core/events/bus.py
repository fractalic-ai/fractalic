from __future__ import annotations
from typing import Callable, List, Iterable
from threading import RLock
from .types import BaseEvent

Subscriber = Callable[[BaseEvent], None]


class EventBus:
    """Synchronous in-process event dispatcher.

    Goals: tiny, predictable, no async context switching.
    """

    def __init__(self):
        self._subs: List[Subscriber] = []
        self._lock = RLock()

    def subscribe(self, fn: Subscriber) -> None:
        with self._lock:
            if fn not in self._subs:
                self._subs.append(fn)

    def unsubscribe(self, fn: Subscriber) -> None:
        with self._lock:
            if fn in self._subs:
                self._subs.remove(fn)

    def emit(self, event: BaseEvent) -> None:
        # Snapshot to protect against modifications while iterating
        with self._lock:
            subs: Iterable[Subscriber] = list(self._subs)
        for fn in subs:
            try:
                fn(event)
            except Exception:
                # Intentionally swallow to avoid cascading failures. A future
                # debug mode could log these.
                pass


GlobalEventBus = EventBus()
