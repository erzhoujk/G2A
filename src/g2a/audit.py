"""Tamper-evident in-memory audit events for safety decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class AuditEvent:
    sequence: int
    timestamp: str
    event_type: str
    payload: Mapping[str, Any]
    previous_hash: str
    event_hash: str


class AuditLog:
    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        return tuple(self._events)

    def append(self, event_type: str, payload: Mapping[str, Any]) -> AuditEvent:
        previous_hash = self._events[-1].event_hash if self._events else "0" * 64
        sequence = len(self._events)
        timestamp = datetime.now(timezone.utc).isoformat()
        canonical = json.dumps(
            {
                "sequence": sequence,
                "timestamp": timestamp,
                "event_type": event_type,
                "payload": payload,
                "previous_hash": previous_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        event_hash = hashlib.sha256(canonical.encode()).hexdigest()
        event = AuditEvent(
            sequence, timestamp, event_type, dict(payload), previous_hash, event_hash
        )
        self._events.append(event)
        return event

    def verify(self) -> bool:
        previous_hash = "0" * 64
        for event in self._events:
            canonical = json.dumps(
                {
                    "sequence": event.sequence,
                    "timestamp": event.timestamp,
                    "event_type": event.event_type,
                    "payload": event.payload,
                    "previous_hash": previous_hash,
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            expected = hashlib.sha256(canonical.encode()).hexdigest()
            if event.previous_hash != previous_hash or event.event_hash != expected:
                return False
            previous_hash = event.event_hash
        return True

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(asdict(event), sort_keys=True) for event in self._events)
