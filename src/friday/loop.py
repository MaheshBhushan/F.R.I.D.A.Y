"""The assembled voice loop: wake -> STT -> route -> ack -> brain -> TTS.

Everything before this module was verified as a component. This is the piece
that makes FRIDAY a process rather than a collection of scripts.

Design notes that are not obvious from the call order:

* **The STT subscription is created at detection.** This freezes capture's
  pre-roll and starts retaining pending frames before the task gets a chance to
  await Deepgram, so connection latency cannot open a gap in the command.

* **Wake events are accepted only while idle.** Detector inference may keep
  running, but scores produced during acknowledgements, reasoning, or TTS are
  ignored. This prevents buffered wake audio and FRIDAY's own playback from
  cancelling the response that the wake phrase just requested.

* **The ack is awaited before speech, not overlapped with it.** Two audio
  streams on one speaker means two voices at once. The ack costs ~0.7s and
  reasoning TTFT is 1.0-2.7s, so the ack has normally finished before the first
  token exists -- overlapping synthesis would buy nothing and risk a collision.

* **A turn that raises must not kill the loop.** Every failure is caught per
  turn, reported, and the loop re-arms. A voice assistant that exits on one bad
  turn is worse than one that occasionally mishears.

Transports arrive as factories so the whole loop runs offline against fakes.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from friday import brain, router
from friday.core import events
from friday.core.spans import DEFAULT_SPANS_PATH, TurnSpan
from friday.router import Tier
from friday.tiers import state_query
from friday.voice import ack as ack_mod
from friday.audio import (
    AudioCaptureService,
    AudioResourceManager,
    AudioSubscription,
    MicState,
)
from friday.voice import indicator, stt, tts, wake

# Ack chosen per tier. Reflex turns get a short confirmation; reasoning turns
# get one that licenses a wait, since that is exactly what follows.
ACK_REFLEX = "okay"
ACK_REASONING = "checking"

# Tier 1 actions that are answered entirely by playing an ack.
_ACK_ONLY_ACTIONS = {"noop_ack": "yes", "pause_turn": "one_moment"}

# Wake-free follow-up conversation is disabled for now. A positive injected
# value still enables it for tests or an explicit future opt-in.
CONVERSATION_SECONDS = 0.0
_WAKE_PREFIX = re.compile(
    r"^\s*(?:(?:hey|okay)\s+)?friday[\s,.:;!?-]*", re.IGNORECASE
)


class AudioTurnState(Enum):
    IDLE = "idle"
    WAKE_DETECTED = "wake_detected"
    STT_CONNECTING = "stt_connecting"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    SUSPENDED = "suspended"


def _normalize_wake_transcript(transcript: str) -> str:
    """Strip one leading wake phrase, including Deepgram's `fridaywhat`."""
    return _WAKE_PREFIX.sub("", transcript, count=1).strip()


@dataclass
class Turn:
    """What one turn did. Returned for tests and the CLI, not for the user."""

    turn_id: str
    transcript: str = ""
    tier: Optional[str] = None
    action: Optional[str] = None
    reply: str = ""
    spoke: bool = False
    interrupted: bool = False
    escalated: bool = False
    preempted: bool = False
    error: Optional[str] = None
    stages: dict = field(default_factory=dict)


class VoiceLoop:
    """Owns the mic, the turn lifecycle, and nothing else.

    `stt_factory` / `tts_factory` return an *un-entered* async context manager
    per turn; the loop enters and exits it, because `run_utterance` explicitly
    does not own transport lifecycle.
    """

    def __init__(
        self,
        *,
        detector: Optional[wake.WakeWordDetector] = None,
        stt_factory: Optional[Callable[[], Any]] = None,
        tts_factory: Optional[Callable[[], Any]] = None,
        llm_transport: Optional[brain.Transport] = None,
        assembler: Optional[brain.ContextAssembler] = None,
        memory: Optional[Any] = None,
        session_memory: Optional[Any] = None,
        approve: Optional[Any] = None,
        audio: Optional[AudioResourceManager] = None,
        capture: Optional[AudioCaptureService] = None,
        play_ack: Callable[..., None] = ack_mod.play_ack,
        audio_output: Optional[Any] = None,
        spans_path: Path = DEFAULT_SPANS_PATH,
        speak_enabled: bool = True,
        conversation_seconds: float = CONVERSATION_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._detector = detector
        self._stt_factory = stt_factory
        self._tts_factory = tts_factory
        self._llm = llm_transport
        self._assembler = assembler
        self._memory = memory
        self._session_memory = session_memory
        self._approve = approve
        self._audio = audio
        self._capture = capture
        self._play_ack = play_ack
        self._audio_output = audio_output
        self._spans_path = spans_path
        self._speak_enabled = speak_enabled
        self._conversation_seconds = conversation_seconds
        self._clock = clock

        self._mic_gate = tts.MicGate()
        self._speaker: Optional[tts.TTSSpeaker] = None
        self._turn_task: Optional[asyncio.Task] = None
        self._stt_subscription: Optional[AudioSubscription] = None
        self.mic_state: "MicState" = MicState.AVAILABLE
        self.audio_state = AudioTurnState.IDLE
        self.invalidated = 0
        self.turns: list[Turn] = []
        # Serialises the gateway's text-driven entry points. Two `ask` calls
        # arriving together would otherwise interleave their TTS into one
        # garbled stream and race on the shared speaker.
        self._text_lock = asyncio.Lock()

    # -- audio in ---------------------------------------------------------

    async def run(self, stop: Optional[asyncio.Event] = None) -> None:
        """Handle turns for as long as the Audio Resource Manager allows.

        This method does not own the microphone and knows nothing about who
        else wants it: `manager.capture()` simply stops producing frames while
        FRIDAY is suspended and resumes afterwards. Everything outside the mic
        subsystem -- brain, memory, world state, coding agents -- is untouched
        by a suspension.
        """
        await self._memory_call("start")
        if self._detector is None:
            self._detector = wake.WakeWordDetector()
        stop = stop or asyncio.Event()

        manager = self._audio or (
            self._capture.manager if self._capture is not None else None
        )
        if manager is None:
            manager = AudioResourceManager()
        self._audio = manager
        manager.set_callbacks(
            on_state=self._on_mic_state,
            on_preempt=self._invalidate_turn,
            on_forget=self._forget_audio,
            on_open=self._arm_detector,
        )
        capture = self._capture
        if capture is None:
            capture = AudioCaptureService(manager)
            self._capture = capture
        await manager.start()
        prewarm_stt = getattr(self._stt_factory, "prewarm", None)
        if prewarm_stt is not None:
            prewarm_stt()
        capture_task = asyncio.create_task(capture.run(stop))
        events.emit("audio", "capture-started")
        try:
            while not stop.is_set() and not capture_task.done():
                await self._pump(capture.wake_chunks(), stop)
            if capture_task.done():
                await capture_task
        finally:
            metrics = capture.metrics
            stop.set()
            capture.reset()
            if not capture_task.done():
                capture_task.cancel()
            await asyncio.gather(capture_task, return_exceptions=True)
            task = self._turn_task
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await manager.stop()
            close_stt = getattr(self._stt_factory, "close", None)
            if close_stt is not None:
                await close_stt()
            await self._memory_call("close")
            events.emit("audio", "capture-stopped", frames=metrics.frames_captured,
                        bytes=metrics.bytes_captured)
            indicator.clear()

    # -- audio resource manager callbacks --------------------------------

    def _set_audio_state(self, state: AudioTurnState, *, turn: Optional[str] = None) -> None:
        if state is self.audio_state:
            return
        previous = self.audio_state
        self.audio_state = state
        events.emit("audio-state", frm=previous.value, to=state.value, turn=turn)

    def _on_mic_state(self, state: "MicState", owners: list) -> None:
        """Mirror the microphone state machine into the visible indicator."""
        self.mic_state = state
        if state is MicState.PREEMPTING:
            self._set_audio_state(AudioTurnState.SUSPENDED)
        elif state is MicState.SUSPENDED:
            self._set_audio_state(AudioTurnState.SUSPENDED)
            detail = self._audio.describe() if self._audio is not None else ""
            indicator.set_state(indicator.State.SUSPENDED, detail=detail)
            events.emit("mic", "paused", who=detail)
        elif state is MicState.FRIDAY_LISTENING:
            self._set_audio_state(AudioTurnState.IDLE)
            indicator.set_state(indicator.State.IDLE)
            events.emit("mic", "listening")

    def _invalidate_turn(self, owners: list) -> "Optional[asyncio.Task[None]]":
        """Discard the in-flight turn's incomplete utterance.

        A turn interrupted mid-sentence must never be resumed: "tell Codex to
        delete the old..." finished twenty minutes after a call ends is worse
        than useless. Conversation context and memory survive; this audio turn
        does not.
        """
        # Stop speaking as well as listening. If a call owns the microphone,
        # anything FRIDAY says out loud is picked up and transmitted into that
        # call -- so a preemption has to silence her, not just deafen her.
        if self._speaker is not None and self._speaker.is_speaking:
            self._speaker.stop()
        task = self._turn_task
        if task is not None and not task.done():
            self.invalidated += 1
            task.cancel()
        if self._stt_subscription is not None:
            self._stt_subscription.close()
            self._stt_subscription = None
        self._set_audio_state(AudioTurnState.SUSPENDED)

        async def _finish() -> None:
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            if self._capture is not None:
                self._capture.reset()
            self._arm_detector()

        if task is not None and not task.done():
            return asyncio.create_task(_finish())
        if self._capture is not None:
            self._capture.reset()
        self._arm_detector()
        return None

    def _arm_detector(self) -> None:
        """Arm the wake detector's warmup for a newly opened capture stream.

        Same call as _forget_audio, opposite intent: that one drops retained
        audio on the way out, this one refuses to trust the first ~1.2s on the
        way in. Kept separate so the two reasons stay legible at the call site.
        """
        if self._detector is not None:
            with contextlib.suppress(Exception):
                self._detector.begin_stream()

    def _forget_audio(self) -> None:
        """Drop retained audio once the stream is closed.

        `begin_stream()` clears the wake-word pre-roll ring, so a rolling
        1.5s buffer of the room is not kept while someone else is on a call.
        It also re-arms warmup, which the next session needs anyway: every
        freshly opened capture stream drops a frame and that discontinuity
        scores as a wake word about a second later.
        """
        if self._capture is not None:
            self._capture.reset()
        if self._detector is not None:
            with contextlib.suppress(Exception):
                self._detector.begin_stream()

    async def _pump(self, frames: "Any", stop: asyncio.Event) -> None:
        """Feed captured frames through the detector until the source ends.

        Takes an async iterator rather than owning a device, so it can be
        driven from a test or from the Audio Resource Manager identically.
        """
        async for chunk in frames:
            if stop.is_set():
                break
            # Wake-trigger handling belongs to IDLE only.  In particular, do
            # not treat detector output during an acknowledgement or TTS as
            # barge-in: the detector is still seeing buffered microphone
            # audio at that point and can score the just-spoken wake phrase a
            # second time, cancelling the answer before it becomes audible.
            # Follow-up turns use their own live STT subscription and do not
            # require another wake event.
            if self._capture is not None and self.audio_state is not AudioTurnState.IDLE:
                events.debug("wake", "ignored", state=self.audio_state.value)
                continue
            detect = getattr(self._detector, "detect_chunk", self._detector.feed_chunk)
            detection = detect(chunk)
            if detection is None:
                continue
            previous_task = self._turn_task
            # The turn runs as a task, NOT awaited here. Awaiting it stops
            # this loop pumping frames, so feed_chunk never forwards audio
            # to the live queue the turn is reading -- the turn then waits
            # for audio that only this loop can deliver and hangs forever.
            # Observed live on the very first real detection.
            span = TurnSpan("pending", turn_id=detection.turn_id or None,
                            path=self._spans_path)
            span.mark("wake_detected")
            subscription = (
                self._capture.subscribe_stt() if self._capture is not None else None
            )
            span.mark("stt_subscription_created")
            self._set_audio_state(AudioTurnState.WAKE_DETECTED, turn=span.turn_id)
            if subscription is None:
                # Preserve the injected/legacy entry point used by offline
                # detector tests and file-fed callers.
                turn_task = asyncio.create_task(self.handle_detection(detection))
            else:
                async def _start_turn() -> Turn:
                    if previous_task is not None and not previous_task.done():
                        await asyncio.gather(previous_task, return_exceptions=True)
                    return await self.handle_detection(
                        detection, subscription=subscription, span=span)

                turn_task = asyncio.create_task(_start_turn())
            self._turn_task = turn_task

    # -- one turn ---------------------------------------------------------

    async def handle_detection(
        self,
        detection: wake.WakeDetection,
        *,
        subscription: Optional[AudioSubscription] = None,
        span: Optional[TurnSpan] = None,
    ) -> Turn:
        """Run one full turn. Never raises: a failed turn is recorded, not fatal."""
        span = span or TurnSpan("pending", turn_id=detection.turn_id or None,
                                path=self._spans_path)
        span.mark("speech_started")
        turn = Turn(turn_id=span.turn_id)
        indicator.set_state(indicator.State.LISTENING)
        source = subscription or detection
        self._stt_subscription = subscription
        follow_up = False
        try:
            turn.transcript = await self._transcribe(source, span)
            events.emit("stt", events.quote(turn.transcript), turn=span.turn_id)
            normalized = _normalize_wake_transcript(turn.transcript)
            span.mark("transcript_normalized")
            events.emit("normalize", original=events.quote(turn.transcript),
                        normalized=events.quote(normalized), turn=span.turn_id)
            turn.transcript = normalized
            self._set_audio_state(AudioTurnState.PROCESSING, turn=span.turn_id)
            if normalized.strip():
                await self._respond(normalized, span, turn)
            else:
                events.emit("turn", "wake-only", turn=span.turn_id)
            follow_up = True
        except asyncio.CancelledError:
            # Microphone preemption (or shutdown). The utterance is incomplete
            # and is discarded, never resumed: finishing "...delete the old"
            # after a 20-minute call would be actively dangerous.
            turn.preempted = True
            turn.error = "preempted: microphone taken by a higher priority app"
            events.emit("turn", "preempted", turn=span.turn_id)
            raise
        except Exception as exc:  # noqa: BLE001 - one bad turn must not end the loop
            turn.error = repr(exc)
            events.emit("turn", "failed", turn=span.turn_id, error=repr(exc))
        finally:
            if subscription is not None:
                subscription.close()
                if self._stt_subscription is subscription:
                    self._stt_subscription = None
            else:
                # Compatibility for file-fed and injected legacy detections.
                with contextlib.suppress(Exception):
                    self._detector.end_handoff(detection.live)
            if self.audio_state is not AudioTurnState.SUSPENDED:
                self._set_audio_state(AudioTurnState.IDLE, turn=span.turn_id)
                indicator.set_state(indicator.State.IDLE)
            turn.stages = dict(span.stages)
            if turn.reply:
                events.emit("reply", events.quote(turn.reply),
                            tier=turn.tier, spoke=turn.spoke)
            span.write()
            self.turns.append(turn)
        if (follow_up and self._capture is not None
                and self._stt_factory is not None
                and self._conversation_seconds > 0):
            await self._follow_up_window()
            self._arm_detector()
        elif subscription is not None:
            self._arm_detector()
        return turn

    async def _transcribe(
        self,
        source: "wake.WakeDetection | AudioSubscription",
        span: TurnSpan,
    ) -> str:
        """Drain one utterance into a final transcript, prewarming on interims."""
        if self._stt_factory is None:
            return ""
        final_parts: list[str] = []
        best_interim = ""
        self._set_audio_state(AudioTurnState.STT_CONNECTING, turn=span.turn_id)
        span.mark("stt_connect_started")
        events.emit("stt-connection", "connecting", turn=span.turn_id)
        try:
            async with self._stt_factory() as transport:
                span.mark("stt_connected")
                events.emit("stt-connection", "connected", turn=span.turn_id)
                self._set_audio_state(AudioTurnState.LISTENING, turn=span.turn_id)
                async for event in stt.run_utterance(source, transport, span=span):
                    if event.text.strip() and "speech_started" not in span.stages:
                        span.mark("speech_started")
                    if event.is_final:
                        if event.text.strip():
                            final_parts.append(event.text.strip())
                        events.emit("stt-final", events.quote(event.text),
                                    turn=span.turn_id,
                                    speech_final=event.speech_final,
                                    utterance_end=event.utterance_end)
                        continue
                    if event.text.strip():
                        if "stt_first_partial" not in span.stages:
                            span.mark("stt_first_partial")
                        events.debug("stt-partial", events.quote(event.text),
                                     turn=span.turn_id)
                        best_interim = event.text
                        if self._assembler is not None:
                            # Interims are the only free lead time there is.
                            self._assembler.prewarm(event.text, span=span)
        finally:
            events.emit("stt-connection", "closed", turn=span.turn_id)
        # Fall back to the last interim. `run_utterance` closes the turn at
        # MAX_WAIT_MS past VAD speech-end, which can land before Deepgram's
        # final for a short or pause-heavy utterance -- observed live, with a
        # complete interim transcript and no final at all. Requiring is_final
        # would drop the turn silently: heard, understood, ignored.
        return " ".join(final_parts) or best_interim

    async def _follow_up_window(self) -> None:
        """Listen for wake-free turns until the active conversation expires."""
        deadline = self._clock() + self._conversation_seconds
        while self.audio_state is not AudioTurnState.SUSPENDED:
            remaining = deadline - self._clock()
            if remaining <= 0:
                break
            assert self._capture is not None
            subscription = self._capture.subscribe_live()
            self._stt_subscription = subscription
            span = TurnSpan("pending", path=self._spans_path)
            span.mark("stt_subscription_created")
            self._set_audio_state(AudioTurnState.STT_CONNECTING, turn=span.turn_id)
            try:
                transcript = await asyncio.wait_for(
                    self._transcribe(subscription, span), timeout=remaining
                )
            except asyncio.TimeoutError:
                events.emit("conversation", "timeout", turn=span.turn_id)
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                events.emit("turn", "failed", turn=span.turn_id, error=repr(exc))
                break
            finally:
                subscription.close()
                if self._stt_subscription is subscription:
                    self._stt_subscription = None
            transcript = _normalize_wake_transcript(transcript)
            span.mark("transcript_normalized")
            if not transcript.strip():
                continue
            turn = Turn(turn_id=span.turn_id, transcript=transcript)
            self._set_audio_state(AudioTurnState.PROCESSING, turn=span.turn_id)
            try:
                await self._respond(transcript, span, turn)
            except asyncio.CancelledError:
                turn.preempted = True
                turn.error = "preempted: microphone taken by a higher priority app"
                raise
            except Exception as exc:  # noqa: BLE001
                turn.error = repr(exc)
                events.emit("turn", "failed", turn=span.turn_id, error=repr(exc))
            finally:
                turn.stages = dict(span.stages)
                if turn.reply:
                    events.emit("reply", events.quote(turn.reply),
                                tier=turn.tier, spoke=turn.spoke)
                span.write()
                self.turns.append(turn)
            deadline = self._clock() + self._conversation_seconds
        if self.audio_state is not AudioTurnState.SUSPENDED:
            self._set_audio_state(AudioTurnState.IDLE)
            indicator.set_state(indicator.State.IDLE)

    async def _respond(self, transcript: str, span: TurnSpan, turn: Turn) -> None:
        decision = router.classify_and_mark(transcript, span)
        tier = decision.tier or Tier.REASONING
        span.turn_kind = tier.value
        turn.tier = tier.value
        events.emit("route", tier=tier.value, turn=span.turn_id)

        if tier is Tier.REFLEX:
            await self._do_reflex(decision, span, turn)
            return

        memory_turn = await self._memory_call(
            "begin_turn", transcript, turn_id=turn.turn_id, route_tier=turn.tier
        )
        try:
            handled = False
            if tier is Tier.STATE_QUERY:
                answer = state_query.answer(transcript, span=span)
                if not answer.escalate:
                    turn.reply = answer.text
                    await self._say_text(answer.text, span, turn)
                    span.mark("task_complete")
                    handled = True
                else:
                    # The snapshot could not honestly answer it; don't guess.
                    turn.escalated = True
                    events.emit("route", "escalated", frm="state_query", to="reasoning",
                                why=answer.reason if hasattr(answer, "reason") else None)
                    span.turn_kind = Tier.REASONING.value
                    turn.tier = Tier.REASONING.value

            if not handled:
                await self._do_reasoning(transcript, span, turn)
        except asyncio.CancelledError as exc:
            await self._memory_call("fail_turn", memory_turn, repr(exc))
            raise
        except Exception as exc:
            await self._memory_call("fail_turn", memory_turn, repr(exc))
            raise

        if turn.reply.strip():
            await self._memory_call(
                "complete_turn", memory_turn, turn.reply,
                route_tier=turn.tier or "", interrupted=turn.interrupted,
                metadata={"spoke": turn.spoke, "escalated": turn.escalated},
            )
            # Preserve the pre-existing relevance memory independently of the
            # structured hot-session context. Reflexes never reach this path.
            if self._memory is not None:
                with contextlib.suppress(Exception):
                    self._memory.record(
                        f"user: {transcript}\nfriday: {turn.reply}", kind="episodic"
                    )
        else:
            await self._memory_call("fail_turn", memory_turn, "empty assistant response")

    async def _do_reflex(self, decision, span: TurnSpan, turn: Turn) -> None:
        action = router.dispatch_tier1(decision)
        turn.action = action
        if action in ("stop_playback", "cancel_last_action"):
            if self._speaker is not None and self._speaker.is_speaking:
                self._speaker.stop()
                turn.interrupted = True
        name = _ACK_ONLY_ACTIONS.get(action, ACK_REFLEX)
        await self._play(name, span)
        span.mark("task_complete")

    async def _do_reasoning(self, transcript: str, span: TurnSpan, turn: Turn) -> None:
        if self._llm is None:
            return
        # Ack first so the 1.0-2.7s TTFT is covered by a voice, not silence.
        ack_task = asyncio.create_task(self._play(ACK_REASONING, span))
        result = brain.TurnResult()
        history = await self._memory_call("context_messages") or []
        stream = brain.complete(
            transcript,
            self._llm,
            assembler=self._assembler,
            history=history,
            approve=self._approve,
            span=span,
            result=result,
        )
        try:
            await self._speak_stream(stream, span, turn, ack_task)
        finally:
            if not ack_task.done():
                ack_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ack_task
            turn.reply = result.text

    # -- audio out --------------------------------------------------------

    async def _play(self, name: str, span: TurnSpan) -> None:
        """Play an ack off the event loop; play_ack blocks on time.sleep."""
        previous = self.audio_state
        self._set_audio_state(AudioTurnState.SPEAKING, turn=span.turn_id)
        try:
            await asyncio.to_thread(self._play_ack, name, span)
        finally:
            if self.audio_state is AudioTurnState.SPEAKING:
                self._set_audio_state(previous, turn=span.turn_id)

    async def _say_text(self, text: str, span: TurnSpan, turn: Turn) -> None:
        async def _one() -> Any:
            yield text

        await self._speak_stream(_one(), span, turn, None)

    async def _speak_stream(self, stream, span: TurnSpan, turn: Turn,
                            ack_task: Optional[asyncio.Task]) -> None:
        if not self._speak_enabled or self._tts_factory is None:
            # Still drain the stream: it is what drives tool calls and spans.
            async for _ in stream:
                pass
            if ack_task is not None:
                await ack_task
            return
        if ack_task is not None:
            # Serialise: two streams on one speaker is two voices at once.
            await ack_task
        self._set_audio_state(AudioTurnState.SPEAKING, turn=span.turn_id)
        async with self._tts_factory() as transport:
            speaker = tts.TTSSpeaker(
                transport, mic_gate=self._mic_gate, output=self._audio_output
            )
            self._speaker = speaker
            try:
                await speaker.speak(stream, span=span)
            finally:
                self._speaker = None
                if self.audio_state is AudioTurnState.SPEAKING:
                    self._set_audio_state(AudioTurnState.PROCESSING,
                                          turn=span.turn_id)
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise asyncio.CancelledError
        turn.spoke = True
        turn.interrupted = turn.interrupted or speaker.stopped

    # -- memory -----------------------------------------------------------

    # -- text entry points (gateway / CLI) --------------------------------

    async def ask(self, transcript: str, *, speak: bool = True) -> Turn:
        """Run one full turn from text, as if it had just been transcribed.

        Skips wake word and STT but goes through the real router, brain and
        TTS, so it exercises everything downstream of the microphone. This is
        what makes the loop testable without a human in the room.

        The turn is appended to `self.turns` and its span written, exactly as a
        spoken turn would be -- a text turn that stayed invisible to the
        latency records would quietly corrupt the percentiles.
        """
        async with self._text_lock:
            span = TurnSpan("pending")
            span.mark("speech_started")
            span.mark("stt_final")
            turn = Turn(turn_id=span.turn_id, transcript=transcript)
            previous = self._speak_enabled
            self._speak_enabled = speak and previous
            try:
                await self._respond(transcript, span, turn)
            except asyncio.CancelledError:
                turn.preempted = True
                turn.error = "preempted: microphone taken by a higher priority app"
                raise
            except Exception as exc:  # noqa: BLE001
                turn.error = repr(exc)
            finally:
                self._speak_enabled = previous
                if turn.reply:
                    events.emit("reply", events.quote(turn.reply),
                                tier=turn.tier, spoke=turn.spoke)
                span.write()
                self.turns.append(turn)
            return turn

    async def say(self, text: str) -> None:
        """Speak `text` out loud with no routing and no LLM.

        Separate from `ask` because notifications are not turns: they have no
        transcript, they must not enter memory as if the user had said
        something, and they should not land in the latency records.
        """
        async with self._text_lock:
            span = TurnSpan("notify")
            turn = Turn(turn_id=span.turn_id, transcript="")
            await self._say_text(text, span, turn)

    async def _memory_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call optional session memory without letting it break a turn."""
        if self._session_memory is None:
            return None
        function = getattr(self._session_memory, method, None)
        if function is None:
            return None
        try:
            return await function(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - memory is non-critical context
            events.emit("memory-error", method, error=events.quote(repr(exc)))
            return None


# -- construction ---------------------------------------------------------


def build_live(*, speak: bool = True) -> VoiceLoop:
    """Wire the real transports. Raises if the keys aren't exported."""
    dg = os.environ.get("DEEPGRAM_API_KEY")
    anthropic = os.environ.get("ANTHROPIC_API_KEY")
    missing = [n for n, v in (("DEEPGRAM_API_KEY", dg),
                              ("ANTHROPIC_API_KEY", anthropic)) if not v]
    if missing:
        raise SystemExit(
            "error: missing " + ", ".join(missing)
            + ". Run: set -a; . ~/.friday/env; set +a"
        )

    from friday.memory import Memory
    from friday.session_memory import SessionMemory

    memory = Memory()
    session_memory = SessionMemory()
    assembler = brain.ContextAssembler(memory=memory)

    async def _approve(request) -> bool:
        # Destructive actions are spoken-adjacent but confirmed on the terminal:
        # a voice "yes" is not a safe authorisation channel for rm.
        print(f"\n[approval needed] {request.risk.value}: {request.action}",
              file=sys.stderr)
        answer = await asyncio.to_thread(input, "approve? [y/N] ")
        return answer.strip().lower() == "y"

    return VoiceLoop(
        stt_factory=stt.FluxTransportPool(dg),
        tts_factory=lambda: tts.DeepgramSpeakTransport(dg),
        llm_transport=brain.AnthropicTransport(anthropic),
        assembler=assembler,
        memory=memory,
        session_memory=session_memory,
        approve=_approve,
        speak_enabled=speak,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m friday.loop")
    parser.add_argument("--text", metavar="TRANSCRIPT",
                        help="skip wake+STT and run one turn from this text")
    parser.add_argument("--no-speak", action="store_true",
                        help="run the turn but don't synthesize audio")
    args = parser.parse_args(argv)

    loop = build_live(speak=not args.no_speak)

    async def _run() -> None:
        if args.text:
            span = TurnSpan("pending")
            span.mark("speech_started")
            span.mark("stt_final")
            turn = Turn(turn_id=span.turn_id, transcript=args.text)
            t0 = time.perf_counter()
            try:
                await loop._respond(args.text, span, turn)
            finally:
                span.write()
            print(f"\ntier={turn.tier} spoke={turn.spoke} "
                  f"elapsed={time.perf_counter() - t0:.2f}s")
            if turn.reply:
                print(f"reply: {turn.reply}")
            return
        stop = asyncio.Event()
        print("[friday] listening -- say the wake word (Ctrl-C to stop)",
              file=sys.stderr)
        await loop.run(stop)

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_run())
    indicator.clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
