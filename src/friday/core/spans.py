"""Per-turn latency span recorder.

One turn = one JSONL line appended to ~/.friday/spans.jsonl containing:
  - turn id
  - turn kind (reflex/state_query/reasoning)
  - wall clock start (time.time(), for human correlation)
  - a map of stage name -> nanosecond offset from turn start (monotonic,
    via time.perf_counter_ns())

Not every turn hits every stage; missing stages are simply absent from
the map.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Optional

STAGES = (
    "wake_detected",
    "stt_subscription_created",
    "speech_started",
    "stt_connect_started",
    "stt_connected",
    "stt_first_partial",
    "speech_ended_vad",
    "stt_final",
    "transcript_normalized",
    "intent_classified",
    "ack_audible",
    "context_ready",
    "llm_sent",
    "first_token",
    "first_tool_call",
    "tool_done",
    "tts_started",
    "first_content_audio",
    "task_complete",
)

DEFAULT_SPANS_PATH = Path.home() / ".friday" / "spans.jsonl"


class TurnSpan:
    """Records nanosecond-offset stage marks for a single interaction turn."""

    def __init__(self, turn_kind: str, turn_id: Optional[str] = None,
                 path: Path = DEFAULT_SPANS_PATH) -> None:
        self.turn_kind = turn_kind
        self.turn_id = turn_id or uuid.uuid4().hex
        self.path = path
        self.wall_start = time.time()
        self._start_ns = time.perf_counter_ns()
        self.stages: dict[str, int] = {}

    def mark(self, stage: str) -> int:
        """Record the current monotonic offset (ns) for `stage` and return it."""
        offset = time.perf_counter_ns() - self._start_ns
        self.stages[stage] = offset
        return offset

    def to_record(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "turn_kind": self.turn_kind,
            "wall_start": self.wall_start,
            "stages": dict(self.stages),
        }

    def write(self) -> None:
        """Append this turn's record as one JSONL line."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(self.to_record(), sort_keys=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def __enter__(self) -> "TurnSpan":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.write()
        return None


def start_turn(turn_kind: str, turn_id: Optional[str] = None,
               path: Path = DEFAULT_SPANS_PATH) -> TurnSpan:
    """Convenience factory matching `with start_turn(...) as t: t.mark(...)`."""
    return TurnSpan(turn_kind=turn_kind, turn_id=turn_id, path=path)
