"""The FRIDAY gateway: one WebSocket control plane in front of the voice loop.

Why this exists. Before it, "running FRIDAY" meant a foreground process
attached to a terminal, and the only way to ask it anything was to speak. That
makes three ordinary things impossible: checking whether she is alive without
watching a tty, driving a turn from a script or a phone, and restarting the
process under a supervisor without losing the ability to observe it. The
gateway is the seam that fixes all three.

Shape, borrowed from openclaw's gateway:

  * One WebSocket endpoint, request/response with an explicit `connect`
    handshake before anything else is allowed.
  * Protocol negotiated as a range, so client and daemon upgrade independently.
  * Token auth with a per-peer failure brake.
  * Unsolicited events pushed to every connected client.
  * A `health` method that a supervisor or smoke test can poll.

Deliberate choices:

  * Binds 127.0.0.1 by default. This process holds Deepgram and Anthropic keys
    and can run tools; it does not belong on a LAN interface without the user
    asking for it, and "it's behind my router" is not an authorisation model.
  * The voice loop is a *sibling*, not a child. The gateway never owns it, so a
    client storm cannot stall a turn and a crashed gateway does not take the
    microphone down with it. They communicate through the loop's public state
    and the indicator's observer hook.
  * Every handler is small and synchronous-ish where it can be. Anything that
    blocks the dispatch loop delays every other client's health check, which is
    how a monitoring endpoint ends up lying about liveness.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from friday.gateway import auth as gauth
from friday.core import events
from friday.gateway.protocol import (
    PROTOCOL_MAX,
    PROTOCOL_MIN,
    ErrorCode,
    ProtocolError,
    Request,
    err,
    event,
    negotiate,
    ok,
    parse_request,
)
from friday.voice import indicator

DEFAULT_HOST = os.environ.get("FRIDAY_GATEWAY_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("FRIDAY_GATEWAY_PORT", "8765"))

# A client that connects and then says nothing is either broken or probing.
# Holding the slot open forever lets an unauthenticated peer pin resources, so
# the handshake is on a clock.
HANDSHAKE_TIMEOUT_S = 10.0

# Upper bound on waiting for connection handlers to drain at shutdown. Short on
# purpose: dropping a client's last frame is a far smaller harm than delaying
# the microphone release and the echo-cancel unload behind it.
CLOSE_TIMEOUT_S = 2.0

SERVER_NAME = "friday-gateway"


@dataclass
class Client:
    """One connected peer, and the identity it claimed at connect time."""

    id: int
    peer: str
    send: Callable[[str], Awaitable[None]]
    connected: bool = False
    protocol: int = 0
    info: dict[str, Any] = field(default_factory=dict)
    connected_at: float = 0.0


class Gateway:
    """Serves the control plane. Owns no audio and no LLM state of its own.

    `loop_ref` is a zero-argument callable returning the live VoiceLoop, or
    None when the voice loop has not started (missing credentials, say). It is
    a callable rather than the object itself because the gateway is
    constructed before the voice loop exists, and holding a stale None forever
    would make `health` permanently report a dead loop after recovery.
    """

    def __init__(
        self,
        *,
        loop_ref: Callable[[], Any] = lambda: None,
        token: Optional[str] = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        limiter: Optional[gauth.RateLimiter] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._loop_ref = loop_ref
        self._token = token if token is not None else gauth.load_or_create_token()
        self.host = host
        self.port = port
        self._limiter = limiter or gauth.RateLimiter()
        self._clock = clock
        self._clients: dict[int, Client] = {}
        self._next_id = 0
        self._started_at = clock()
        self._server: Any = None
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._turns_served = 0

        self._methods: dict[str, Callable[[Client, dict], Awaitable[dict]]] = {
            "health": self._m_health,
            "state": self._m_state,
            "mic.owners": self._m_mic_owners,
            "spans.recent": self._m_spans_recent,
            "say": self._m_say,
            "ask": self._m_ask,
        }

    # ---------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Bind the port. Raises if the address is taken -- that is not
        recoverable by retrying, it means another FRIDAY is already running,
        and silently continuing would give the user two daemons fighting over
        one microphone."""
        import websockets

        self._async_loop = asyncio.get_running_loop()
        indicator.subscribe(self._on_indicator)
        self._server = await websockets.serve(
            self._handle, self.host, self.port, ping_interval=20, ping_timeout=20
        )
        events.emit("gateway", "listening", url=f"ws://{self.host}:{self.port}")

    async def stop(self) -> None:
        indicator.unsubscribe(self._on_indicator)
        if self._server is not None:
            self._server.close()
            # Bounded. wait_closed() waits for in-flight connection handlers to
            # finish, which makes shutdown depend on a remote peer's
            # cooperation: a client that died mid-close-handshake can hold this
            # open indefinitely. Shutdown must never be blockable from the
            # network -- the process is holding the microphone and, on this
            # machine, an echo-cancel module whose cleanup runs after this
            # point. Leaking that module renumbers every other application's
            # device list, which is the bug that silently broke Chrome.
            with contextlib.suppress(Exception, asyncio.TimeoutError):
                await asyncio.wait_for(self._server.wait_closed(),
                                       timeout=CLOSE_TIMEOUT_S)
            self._server = None

    async def serve_forever(self, stop: asyncio.Event) -> None:
        await self.start()
        try:
            await stop.wait()
        finally:
            await self.stop()

    # ------------------------------------------------------------------- events

    def _on_indicator(self, state: "indicator.State", detail: str) -> None:
        """Indicator observer. Runs on the *caller's* thread, which may not be
        the event loop's, so the actual send is hopped across with
        call_soon_threadsafe rather than touched directly."""
        loop = self._async_loop
        if loop is None or loop.is_closed():
            return
        payload = {"state": state.value, "detail": detail}
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self.broadcast("state.changed", payload))
            )

    async def broadcast(self, name: str, data: dict[str, Any]) -> None:
        """Push an event to every *connected* client.

        Sends run concurrently and failures are ignored per-client: one peer
        with a full send buffer must not delay or drop the others' events.
        Only clients past the handshake receive anything -- an unauthenticated
        socket has no business learning FRIDAY's state.
        """
        frame = event(name, data)
        targets = [c for c in list(self._clients.values()) if c.connected]
        if not targets:
            return
        await asyncio.gather(
            *(self._safe_send(c, frame) for c in targets), return_exceptions=True
        )

    @staticmethod
    async def _safe_send(client: Client, frame: str) -> None:
        with contextlib.suppress(Exception):
            await client.send(frame)

    # ---------------------------------------------------------------- connection

    async def _handle(self, socket: Any) -> None:
        self._next_id += 1
        client = Client(
            id=self._next_id,
            peer=_peer_of(socket),
            send=socket.send,
        )
        self._clients[client.id] = client
        try:
            await self._serve_client(client, socket)
        finally:
            self._clients.pop(client.id, None)

    async def _serve_client(self, client: Client, socket: Any) -> None:
        try:
            first = await asyncio.wait_for(socket.recv(), timeout=HANDSHAKE_TIMEOUT_S)
        except asyncio.TimeoutError:
            with contextlib.suppress(Exception):
                await client.send(
                    err("-", ErrorCode.BAD_REQUEST, "handshake timed out")
                )
            return
        except Exception:
            return

        if not await self._dispatch(client, first):
            return  # failed connect: close rather than leaving a dead socket

        async for raw in socket:
            await self._dispatch(client, raw)

    async def _dispatch(self, client: Client, raw: Any) -> bool:
        """Handle one frame. Returns False when the connection should close."""
        try:
            request = parse_request(raw)
        except ProtocolError as exc:
            await self._safe_send(client, err("-", exc.code, exc.message))
            return client.connected

        try:
            if request.method == "connect":
                result = await self._m_connect(client, request.params)
            elif not client.connected:
                raise ProtocolError(
                    ErrorCode.NOT_CONNECTED,
                    f"call connect before {request.method}",
                )
            else:
                handler = self._methods.get(request.method)
                if handler is None:
                    raise ProtocolError(
                        ErrorCode.UNKNOWN_METHOD, f"no such method: {request.method}"
                    )
                result = await handler(client, request.params)
        except ProtocolError as exc:
            await self._safe_send(client, err(request.id, exc.code, exc.message))
            return client.connected
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # Detail withheld from the wire on purpose: an unexpected traceback
            # can carry file paths and, in this process, fragments of prompts.
            # It goes to the log, where the operator can see it, not to the peer.
            events.emit("gateway", "handler failed", method=request.method,
                        error=repr(exc))
            await self._safe_send(
                client, err(request.id, ErrorCode.INTERNAL, "internal error")
            )
            return client.connected

        await self._safe_send(client, ok(request.id, result))
        return True

    # ------------------------------------------------------------------ methods

    async def _m_connect(self, client: Client, params: dict) -> dict:
        if client.connected:
            raise ProtocolError(
                ErrorCode.ALREADY_CONNECTED, "connect called twice on one socket"
            )

        # Order matters: brake, then protocol, then token. Checking the brake
        # first means a throttled peer cannot use protocol errors as a free
        # oracle, and negotiating before auth lets an old client learn it must
        # upgrade rather than being told its (valid) token is wrong.
        self._limiter.check(client.peer)

        try:
            protocol = negotiate(
                params.get("min_protocol", PROTOCOL_MIN),
                params.get("max_protocol", PROTOCOL_MAX),
            )
            auth_block = params.get("auth") or {}
            gauth.verify(
                auth_block.get("token") if isinstance(auth_block, dict) else None,
                self._token,
            )
        except ProtocolError:
            self._limiter.record_failure(client.peer)
            raise

        self._limiter.reset(client.peer)
        client.connected = True
        client.protocol = protocol
        client.connected_at = self._clock()
        raw_info = params.get("client")
        client.info = raw_info if isinstance(raw_info, dict) else {}

        return {
            "protocol": protocol,
            "server": {
                "name": SERVER_NAME,
                "protocol_min": PROTOCOL_MIN,
                "protocol_max": PROTOCOL_MAX,
            },
            "methods": sorted(self._methods) + ["connect"],
            "state": indicator.current().value,
        }

    async def _m_health(self, client: Client, params: dict) -> dict:
        """Liveness for a supervisor or smoke test.

        `voice_loop` is reported separately from `ok` because the gateway can
        be perfectly healthy while the voice loop is down -- that is exactly
        the situation when credentials are missing, and collapsing the two
        into one boolean would make it undiagnosable from outside.
        """
        lp = self._loop_ref()
        return {
            "ok": True,
            "uptime_s": round(self._clock() - self._started_at, 3),
            "state": indicator.current().value,
            "voice_loop": lp is not None,
            "clients": sum(1 for c in self._clients.values() if c.connected),
            "turns": len(getattr(lp, "turns", []) or []) if lp else 0,
            "invalidated": getattr(lp, "invalidated", 0) if lp else 0,
        }

    async def _m_state(self, client: Client, params: dict) -> dict:
        lp = self._loop_ref()
        manager = getattr(lp, "_audio", None) if lp else None
        return {
            "indicator": indicator.current().value,
            "mic": getattr(getattr(manager, "state", None), "value", "unknown"),
            "mic_detail": manager.describe() if manager is not None else "",
        }

    async def _m_mic_owners(self, client: Client, params: dict) -> dict:
        """Who currently outranks FRIDAY for the microphone."""
        lp = self._loop_ref()
        manager = getattr(lp, "_audio", None) if lp else None
        if manager is None:
            raise ProtocolError(ErrorCode.UNAVAILABLE, "audio manager not running")
        owners = await manager.refresh()
        return {
            "owners": [
                {
                    "label": o.label,
                    "tier": o.tier,
                    "priority": int(o.priority),
                    "preempts_friday": o.preempts_friday,
                }
                for o in owners
            ]
        }

    async def _m_spans_recent(self, client: Client, params: dict) -> dict:
        """Recent turn latency records, newest last.

        Bounded because an unbounded reply is a memory amplifier: a client
        asking for everything after a long session would have the daemon build
        a multi-megabyte frame in one allocation.
        """
        limit = params.get("limit", 20)
        if not isinstance(limit, int) or not 1 <= limit <= 200:
            raise ProtocolError(ErrorCode.BAD_REQUEST, "limit must be 1..200")
        lp = self._loop_ref()
        turns = list(getattr(lp, "turns", []) or []) if lp else []
        recent = turns[-limit:]
        return {
            "turns": [
                {
                    "transcript": getattr(t, "transcript", ""),
                    "tier": getattr(t, "tier", "") or "",
                    "reply": getattr(t, "reply", ""),
                    "error": getattr(t, "error", "") or "",
                    "preempted": bool(getattr(t, "preempted", False)),
                }
                for t in recent
            ],
            "total": len(turns),
        }

    async def _m_say(self, client: Client, params: dict) -> dict:
        """Speak text out loud, bypassing wake word and STT.

        Refuses while the microphone is held by a higher-priority app: FRIDAY's
        voice is picked up by whatever owns the mic, so speaking during a call
        transmits her into that call. Same rule the preemption path enforces.
        """
        text = params.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ProtocolError(ErrorCode.BAD_REQUEST, "'text' must be a non-empty string")
        lp = self._loop_ref()
        if lp is None:
            raise ProtocolError(ErrorCode.UNAVAILABLE, "voice loop not running")
        manager = getattr(lp, "_audio", None)
        if manager is not None and not manager.free.is_set():
            raise ProtocolError(
                ErrorCode.FORBIDDEN,
                f"microphone held by a higher priority app: {manager.describe()}",
            )
        await lp.say(text)
        return {"spoken": text}

    async def _m_ask(self, client: Client, params: dict) -> dict:
        """Run a full turn from text, as if it had been transcribed.

        This is the harness's real end-to-end probe: it exercises routing, the
        brain and TTS without needing a microphone or a human, which is what
        makes the gateway testable in CI.
        """
        text = params.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ProtocolError(ErrorCode.BAD_REQUEST, "'text' must be a non-empty string")
        speak = bool(params.get("speak", False))
        lp = self._loop_ref()
        if lp is None:
            raise ProtocolError(ErrorCode.UNAVAILABLE, "voice loop not running")
        self._turns_served += 1
        turn = await lp.ask(text, speak=speak)
        return {
            "tier": getattr(turn, "tier", "") or "",
            "reply": getattr(turn, "reply", ""),
            "error": getattr(turn, "error", "") or "",
        }


def _peer_of(socket: Any) -> str:
    """Stable rate-limit key for a socket.

    Falls back to a constant rather than the object id: an unidentifiable peer
    should share one throttle bucket, not get a fresh unlimited one per
    connection, which would make the brake trivially bypassable by reconnecting.
    """
    remote = getattr(socket, "remote_address", None)
    if isinstance(remote, (tuple, list)) and remote:
        return str(remote[0])
    return "unknown"
