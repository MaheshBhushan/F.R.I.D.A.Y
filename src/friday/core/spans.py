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

import hashlib
import json
import threading
import time
import uuid
from pathlib import Path

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


def active_spans_dir(path: Path) -> Path:
    return path.with_name(path.name + ".active")


class TurnSpan:
    """Records nanosecond-offset stage marks for a single interaction turn."""

    def __init__(self, turn_kind: str, turn_id: str | None = None,
                 path: Path = DEFAULT_SPANS_PATH) -> None:
        self.turn_kind = turn_kind
        self.turn_id = turn_id or uuid.uuid4().hex
        self.path = path
        self.wall_start = time.time()
        self._start_ns = time.perf_counter_ns()
        self.stages: dict[str, int] = {}
        self.durations_ns: dict[str, int] = {}
        self._checkpoint_path = active_spans_dir(path) / (hashlib.sha256(
            self.turn_id.encode()).hexdigest() + ".json")
        self._checkpoint_lock = threading.Lock()
        self._written = False

    def add_duration(self, name: str, elapsed_ns: int) -> None:
        """Accumulate disjoint work intervals across all tool/model rounds."""
        self.durations_ns[name] = self.durations_ns.get(name, 0) + elapsed_ns

    def mark(self, stage: str) -> int:
        """Record the current monotonic offset (ns) for `stage` and return it."""
        offset = time.perf_counter_ns() - self._start_ns
        self.stages[stage] = offset
        if stage in {"intent_classified", "llm_sent", "tts_started"} or (
            stage in {"ack_audible", "first_content_audio", "task_complete"}
            and self._checkpoint_path.exists()
        ):
            self._checkpoint()
        return offset

    def _checkpoint(self) -> None:
        """Publish an atomic snapshot so stalled turns remain observable."""
        with self._checkpoint_lock:
            if self._written:
                return
            try:
                self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self._checkpoint_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(self.to_record()), encoding="utf-8")
                temporary.replace(self._checkpoint_path)
            except OSError as exc:
                from friday.core import events
                events.emit("metrics", "checkpoint failed", error=str(exc))

    def to_record(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "turn_kind": self.turn_kind,
            "wall_start": self.wall_start,
            "stages": dict(self.stages),
            "durations_ns": dict(self.durations_ns),
        }

    def write(self) -> None:
        """Append this turn's record as one JSONL line."""
        with self._checkpoint_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(self.to_record(), sort_keys=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._written = True
            try:
                self._checkpoint_path.unlink(missing_ok=True)
            except OSError as exc:
                from friday.core import events
                events.emit("metrics", "checkpoint cleanup failed", error=str(exc))

    def __enter__(self) -> TurnSpan:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.write()
        return None


def start_turn(turn_kind: str, turn_id: str | None = None,
               path: Path = DEFAULT_SPANS_PATH) -> TurnSpan:
    """Convenience factory matching `with start_turn(...) as t: t.mark(...)`."""
    return TurnSpan(turn_kind=turn_kind, turn_id=turn_id, path=path)
