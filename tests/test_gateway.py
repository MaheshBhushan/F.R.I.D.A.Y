"""Gateway tests: protocol, auth, and a real socket round trip.

The round-trip tests bind a real ephemeral port rather than mocking the
websockets layer. Mocking it would prove the handlers work and leave the part
that actually breaks -- framing, handshake ordering, close behaviour -- untested.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat

import pytest

from friday.gateway import auth as gauth
from friday.gateway import protocol as proto
from friday.gateway.client import GatewayClient, smoke
from friday.gateway.server import Gateway
from friday.voice import indicator

TOKEN = "test-token-not-a-real-secret"


# --------------------------------------------------------------- protocol

def test_parse_rejects_a_frame_with_no_id():
    # A reply to an id-less request would be uncorrelatable, so it must be
    # refused at the door rather than dispatched and silently dropped.
    with pytest.raises(proto.ProtocolError) as exc:
        proto.parse_request(json.dumps({"method": "health"}))
    assert exc.value.code is proto.ErrorCode.BAD_REQUEST


def test_parse_rejects_non_json_and_non_objects():
    for bad in ("not json", json.dumps([1, 2]), json.dumps("x")):
        with pytest.raises(proto.ProtocolError):
            proto.parse_request(bad)


def test_negotiate_picks_the_highest_shared_version():
    assert proto.negotiate(1, 99) == proto.PROTOCOL_MAX


def test_negotiate_refuses_a_disjoint_window_instead_of_downgrading():
    # Silently answering in a version the client never offered produces frames
    # it cannot parse, and the failure surfaces far from its cause.
    with pytest.raises(proto.ProtocolError) as exc:
        proto.negotiate(proto.PROTOCOL_MAX + 5, proto.PROTOCOL_MAX + 9)
    assert exc.value.code is proto.ErrorCode.UNSUPPORTED_PROTOCOL


def test_negotiate_rejects_an_inverted_window():
    with pytest.raises(proto.ProtocolError):
        proto.negotiate(9, 1)


# ------------------------------------------------------------------- auth

def test_token_file_is_created_0600(tmp_path, monkeypatch):
    monkeypatch.delenv(gauth.TOKEN_ENV, raising=False)
    target = tmp_path / "gateway-token"
    token = gauth.load_or_create_token(target)
    assert token and target.read_text().strip() == token
    mode = target.stat().st_mode
    assert not mode & (stat.S_IRGRP | stat.S_IROTH), "token file must not be readable by others"


def test_token_is_stable_across_calls(tmp_path, monkeypatch):
    monkeypatch.delenv(gauth.TOKEN_ENV, raising=False)
    target = tmp_path / "t"
    assert gauth.load_or_create_token(target) == gauth.load_or_create_token(target)


def test_empty_token_file_is_replaced_not_trusted(tmp_path, monkeypatch):
    # An empty token file authenticates nobody while reading as "configured".
    monkeypatch.delenv(gauth.TOKEN_ENV, raising=False)
    target = tmp_path / "t"
    target.write_text("   \n")
    assert gauth.load_or_create_token(target).strip()


def test_verify_rejects_non_strings_as_unauthorized_not_bad_request():
    # Distinguishing the two would let a prober skip straight to guessing.
    for bad in (None, 42, b"x", ""):
        with pytest.raises(proto.ProtocolError) as exc:
            gauth.verify(bad, TOKEN)
        assert exc.value.code is proto.ErrorCode.UNAUTHORIZED


def test_rate_limiter_trips_then_expires():
    now = [1000.0]
    limiter = gauth.RateLimiter(max_failures=3, window_seconds=60.0, now=lambda: now[0])
    for _ in range(3):
        limiter.check("peer")
        limiter.record_failure("peer")
    with pytest.raises(proto.ProtocolError) as exc:
        limiter.check("peer")
    assert exc.value.code is proto.ErrorCode.RATE_LIMITED
    now[0] += 61.0
    limiter.check("peer")  # window has slid; must not raise


def test_rate_limiter_success_clears_the_history():
    limiter = gauth.RateLimiter(max_failures=2)
    limiter.record_failure("peer")
    limiter.reset("peer")
    limiter.record_failure("peer")
    limiter.check("peer")  # one typo must not be sticky


# ------------------------------------------------------- live socket tests

class FakeTurn:
    tier = "reflex"
    reply = "done"
    error = None
    preempted = False
    transcript = "hello"


class FakeLoop:
    """Stands in for VoiceLoop: only the surface the gateway actually reads."""

    def __init__(self):
        self.turns = [FakeTurn()]
        self.invalidated = 0
        self.said: list[str] = []
        self.asked: list[str] = []
        self._audio = None

    async def ask(self, text, *, speak=True):
        self.asked.append(text)
        return FakeTurn()

    async def say(self, text):
        self.said.append(text)


async def _serve(loop_obj=None):
    """Start a gateway on an ephemeral port and return (gateway, url)."""
    gateway = Gateway(loop_ref=lambda: loop_obj, token=TOKEN, host="127.0.0.1", port=0)
    await gateway.start()
    port = gateway._server.sockets[0].getsockname()[1]
    return gateway, f"ws://127.0.0.1:{port}"


def test_health_requires_connect_first():
    async def _run():
        gateway, url = await _serve()
        try:
            async with GatewayClient(url, TOKEN) as client:
                reply = await client.request("health")
                assert reply["ok"] is False
                assert reply["error"]["code"] == proto.ErrorCode.NOT_CONNECTED.value
        finally:
            await gateway.stop()


    asyncio.run(_run())
def test_wrong_token_is_rejected_and_counted():
    async def _run():
        gateway, url = await _serve()
        try:
            async with GatewayClient(url, "wrong-token") as client:
                reply = await client.connect()
                assert reply["ok"] is False
                assert reply["error"]["code"] == proto.ErrorCode.UNAUTHORIZED.value
        finally:
            await gateway.stop()


    asyncio.run(_run())
def test_connect_health_and_state_round_trip():
    async def _run():
        fake = FakeLoop()
        gateway, url = await _serve(fake)
        try:
            async with GatewayClient(url, TOKEN) as client:
                reply = await client.connect()
                assert reply["ok"], reply
                assert reply["result"]["protocol"] == proto.PROTOCOL_MAX
                assert "health" in reply["result"]["methods"]

                health = (await client.request("health"))["result"]
                assert health["ok"] is True
                assert health["voice_loop"] is True
                assert health["turns"] == 1

                state = (await client.request("state"))["result"]
                assert state["indicator"] in {s.value for s in indicator.State}
        finally:
            await gateway.stop()


    asyncio.run(_run())
def test_health_reports_a_missing_voice_loop_without_failing():
    async def _run():
        # The gateway being healthy and the loop being down are different facts.
        # Collapsing them into one boolean makes a credentials problem invisible.
        gateway, url = await _serve(None)
        try:
            async with GatewayClient(url, TOKEN) as client:
                await client.connect()
                health = (await client.request("health"))["result"]
                assert health["ok"] is True
                assert health["voice_loop"] is False
        finally:
            await gateway.stop()


    asyncio.run(_run())
def test_ask_drives_a_turn_through_the_loop():
    async def _run():
        fake = FakeLoop()
        gateway, url = await _serve(fake)
        try:
            async with GatewayClient(url, TOKEN) as client:
                await client.connect()
                reply = await client.request("ask", {"text": "what branch am i on"})
                assert reply["ok"], reply
                assert reply["result"]["tier"] == "reflex"
                assert fake.asked == ["what branch am i on"]
        finally:
            await gateway.stop()


    asyncio.run(_run())
def test_ask_rejects_empty_text():
    async def _run():
        gateway, url = await _serve(FakeLoop())
        try:
            async with GatewayClient(url, TOKEN) as client:
                await client.connect()
                reply = await client.request("ask", {"text": "   "})
                assert reply["ok"] is False
                assert reply["error"]["code"] == proto.ErrorCode.BAD_REQUEST.value
        finally:
            await gateway.stop()


    asyncio.run(_run())
def test_unknown_method_is_named_in_the_error():
    async def _run():
        gateway, url = await _serve(FakeLoop())
        try:
            async with GatewayClient(url, TOKEN) as client:
                await client.connect()
                reply = await client.request("nope")
                assert reply["error"]["code"] == proto.ErrorCode.UNKNOWN_METHOD.value
                assert "nope" in reply["error"]["message"]
        finally:
            await gateway.stop()


    asyncio.run(_run())
def test_connect_twice_on_one_socket_is_refused():
    async def _run():
        gateway, url = await _serve(FakeLoop())
        try:
            async with GatewayClient(url, TOKEN) as client:
                assert (await client.connect())["ok"]
                reply = await client.connect()
                assert reply["error"]["code"] == proto.ErrorCode.ALREADY_CONNECTED.value
        finally:
            await gateway.stop()


    asyncio.run(_run())
def test_indicator_transitions_are_pushed_as_events():
    async def _run():
        gateway, url = await _serve(FakeLoop())
        try:
            async with GatewayClient(url, TOKEN) as client:
                await client.connect()
                indicator.set_state(indicator.State.TALKING, detail="unit-test")
                # Give the threadsafe hop and the broadcast a turn of the loop.
                for _ in range(50):
                    await asyncio.sleep(0.01)
                    if any(name == "state.changed" for name, _ in client.events):
                        break
                names = [n for n, _ in client.events]
                assert "state.changed" in names, f"no event arrived; saw {names}"
                data = dict(client.events)["state.changed"]
                assert data["state"] == "talking"
                assert data["detail"] == "unit-test"
        finally:
            indicator.set_state(indicator.State.IDLE)
            await gateway.stop()


    asyncio.run(_run())
def test_unauthenticated_sockets_receive_no_events():
    async def _run():
        # An unauthenticated peer must not be able to observe FRIDAY's state
        # simply by holding a socket open.
        gateway, url = await _serve(FakeLoop())
        try:
            async with GatewayClient(url, TOKEN) as client:
                indicator.set_state(indicator.State.THINKING)
                for _ in range(20):
                    await asyncio.sleep(0.01)
                assert client.events == []
        finally:
            indicator.set_state(indicator.State.IDLE)
            await gateway.stop()


    asyncio.run(_run())
def test_smoke_returns_2_when_nothing_is_listening():
    async def _run():
        # Port 1 is privileged and never served; this is the "daemon down" path.
        assert await smoke("ws://127.0.0.1:1", TOKEN, ask=None) == 2


    asyncio.run(_run())
def test_smoke_returns_3_on_a_bad_token():
    async def _run():
        gateway, url = await _serve(FakeLoop())
        try:
            assert await smoke(url, "wrong", ask=None) == 3
        finally:
            await gateway.stop()


    asyncio.run(_run())
def test_smoke_passes_every_stage_against_a_live_gateway():
    async def _run():
        gateway, url = await _serve(FakeLoop())
        try:
            assert await smoke(url, TOKEN, ask="what branch am i on") == 0
        finally:
            await gateway.stop()

    asyncio.run(_run())

class EmptyReplyLoop(FakeLoop):
    """A loop whose turn routes fine, errors not at all, and says nothing."""

    async def ask(self, text, *, speak=True):
        class Silent:
            tier = "reasoning"
            reply = ""
            error = None
        return Silent()


def test_smoke_fails_when_the_reply_is_empty():
    """An empty reply is a failure, not a pass.

    A turn that routes, reports no error and says nothing is the worst outcome
    to call green: the user asked and FRIDAY silently ignored them. This
    happened for real -- the daemon's working directory left it outside any git
    repo, so the state query escalated to the LLM and produced nothing, and
    smoke reported "all stages passed".
    """
    async def _run():
        gateway, url = await _serve(EmptyReplyLoop())
        try:
            assert await smoke(url, TOKEN, ask="what branch am i on") == 6
        finally:
            await gateway.stop()

    asyncio.run(_run())


def test_smoke_wait_polls_until_the_gateway_appears():
    """--wait must tolerate a gateway that is not listening yet.

    Under Type=simple, systemd runs ExecStartPost the moment ExecStart forks --
    seconds before the port is bound. Without this, every service start failed
    its own readiness check and Restart=always turned that into a crash loop.
    """
    async def _run():
        gateway_holder = {}

        async def _start_late():
            await asyncio.sleep(0.8)
            gateway, url = await _serve(FakeLoop())
            gateway_holder["gateway"] = gateway
            gateway_holder["port"] = url.rsplit(":", 1)[1]

        # Bind a port, learn its number, release it, then have the gateway
        # claim it after smoke has already started polling.
        import socket
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()

        async def _serve_on_port():
            await asyncio.sleep(0.8)
            gw = Gateway(loop_ref=lambda: FakeLoop(), token=TOKEN,
                         host="127.0.0.1", port=port)
            await gw.start()
            gateway_holder["gateway"] = gw

        later = asyncio.create_task(_serve_on_port())
        try:
            code = await smoke(f"ws://127.0.0.1:{port}", TOKEN, ask=None, wait=10.0)
            assert code == 0, "smoke should have waited for the late gateway"
        finally:
            await later
            gw = gateway_holder.get("gateway")
            if gw is not None:
                await gw.stop()

    asyncio.run(_run())


def test_smoke_without_wait_fails_fast_on_a_dead_port():
    # The default must not silently retry: a supervisor asking "is it up?" wants
    # an answer now, not in a minute.
    async def _run():
        import time as _t
        t0 = _t.monotonic()
        assert await smoke("ws://127.0.0.1:1", TOKEN, ask=None) == 2
        assert _t.monotonic() - t0 < 5.0
    asyncio.run(_run())
