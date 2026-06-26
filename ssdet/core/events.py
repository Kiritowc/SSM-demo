import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class EventEnvelope:
    category: str
    stage: str
    payload: Dict
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self):
        return {
            "category": self.category,
            "stage": self.stage,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


class JsonlEventSink:
    def __init__(self, target_path: str):
        self.target_path = target_path
        parent = os.path.dirname(target_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def write(self, envelope: EventEnvelope):
        with open(self.target_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(envelope.to_dict(), ensure_ascii=False) + "\n")


class EventBus:
    def __init__(self, sinks: Iterable[JsonlEventSink] = ()):
        self.sinks: List[JsonlEventSink] = list(sinks)

    def emit(self, category: str, stage: str, payload: Dict):
        envelope = EventEnvelope(category=category, stage=stage, payload=payload)
        for sink in self.sinks:
            sink.write(envelope)
        return envelope
