"""Minimal gateway client, and the smoke harness built on it.

The client is deliberately small enough to be obviously correct: one socket,
one pending-request table, one reader task. It is what the smoke test, the
CLI and any future UI share, so a bug here would otherwise be reimplemented
three times.

Run the harness with:

    python -m friday.gateway.client --smoke

Exit codes are staged, not boolean, because "it's broken" is useless to a
supervisor or a CI log. Each stage gets its own code so the failure is
diagnosed by the exit status alone:

    0  everything passed
    1  usage / no token
    2  could not open the socket        -- daemon down or wrong port
    3  connect rejected                 -- bad token or protocol mismatch
    4  health failed                    -- daemon up but unhealthy
    5  state failed                     -- control plane partially broken
    6  ask failed                       -- the loop itself is broken

That ordering matters: each stage is a strict prerequisite of the next, so the
first non-zero code is always the root cause rather than a downstream symptom.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
from typing import Any, Optional

from friday.gateway.protocol import PROTOCOL_MAX, PROTOCOL_MIN

DEFAULT_URL = os.environ.get("FRIDAY_GATEWAY_URL", "ws://127.0.0.1:8765")
REQUEST_TIMEOUT_S = 30.0


class GatewayClient:
    """One connection to a gateway. Not thread-safe; use one per task."""

    def __init__(self, url: str = DEFAULT_URL, token: Optional[str] = None) -> None:
        self.url = url
        self._token = token
        self._socket: Any = None
        self._pending: dict[str, asyncio.Future] = {}
        self._events: list[tuple[str, dict]] = []
        self._reader: Optional[asyncio.Task] = None
        self._next = 0

    async def __aenter__(self) -> "GatewayClient":
        await self.open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def open(self) -> None:
        import websockets

        self._socket = await websockets.connect(self.url)
        self._reader = asyncio.create_task(self._read_loop())

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader
            self._reader = None
        if self._socket is not None:
            with contextlib.suppress(Exception):
                await self._socket.close()
            self._socket = None

    async def _read_loop(self) -> None:
        """Demultiplex responses from events.

        On socket close every pending future is failed rather than left hanging
        -- a caller awaiting a reply that can never arrive is a hang, and a
        hang in a health check is worse than an error.
        """
        try:
            async for raw in self._socket:
                try:
                    payload = json.loads(raw)
                except ValueError:
                    continue
                if "event" in payload:
                    self._events.append((payload["event"], payload.get("data") or {}))
                    continue
                future = self._pending.pop(str(payload.get("id")), None)
                if future is not None and not future.done():
                    future.set_result(payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            for future in list(self._pending.values()):
                if not future.done():
                    future.set_exception(ConnectionError("gateway closed the socket"))
            self._pending.clear()

    async def request(self, method: str, params: Optional[dict] = None,
                      timeout: float = REQUEST_TIMEOUT_S) -> dict:
        """Send one request and await its reply. Returns the raw envelope."""
        self._next += 1
        ident = str(self._next)
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[ident] = future
        await self._socket.send(
            json.dumps({"id": ident, "method": method, "params": params or {}})
        )
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(ident, None)
            raise

    async def connect(self, *, name: str = "friday-client") -> dict:
        from friday.gateway import auth as gauth

        token = self._token if self._token is not None else gauth.load_or_create_token()
        return await self.request("connect", {
            "min_protocol": PROTOCOL_MIN,
            "max_protocol": PROTOCOL_MAX,
            "client": {"name": name, "version": "dev"},
            "auth": {"token": token},
        })

    @property
    def events(self) -> list[tuple[str, dict]]:
        return list(self._events)


async def smoke(url: str, token: Optional[str], *, ask: Optional[str],
                wait: float = 0.0) -> int:
    """Walk the stages in dependency order, reporting the first failure.

    `wait` polls for the socket to appear before starting. Needed because a
    readiness check that races the thing it is checking is worse than no check
    at all: as an ExecStartPost under `Type=simple`, systemd runs this the
    instant ExecStart forks, which is seconds before the gateway binds. That
    turned every single service start into a failure and a restart loop.
    """
    client = GatewayClient(url, token)
    deadline = time.monotonic() + max(wait, 0.0)
    while True:
        try:
            await client.open()
            break
        except Exception as exc:  # noqa: BLE001
            if time.monotonic() >= deadline:
                print(f"smoke: cannot reach {url}: {exc}", file=sys.stderr)
                return 2
            await asyncio.sleep(0.5)
            client = GatewayClient(url, token)

    try:
        reply = await client.connect(name="gateway-smoke")
        if not reply.get("ok"):
            print(f"smoke: connect rejected: {reply.get('error')}", file=sys.stderr)
            return 3
        print(f"ok   connect   protocol={reply['result']['protocol']}")

        reply = await client.request("health")
        if not reply.get("ok"):
            print(f"smoke: health failed: {reply.get('error')}", file=sys.stderr)
            return 4
        health = reply["result"]
        print(f"ok   health    uptime={health['uptime_s']}s "
              f"state={health['state']} voice_loop={health['voice_loop']} "
              f"turns={health['turns']}")

        reply = await client.request("state")
        if not reply.get("ok"):
            print(f"smoke: state failed: {reply.get('error')}", file=sys.stderr)
            return 5
        st = reply["result"]
        print(f"ok   state     indicator={st['indicator']} mic={st['mic']}"
              + (f" ({st['mic_detail']})" if st.get("mic_detail") else ""))

        if ask:
            reply = await client.request("ask", {"text": ask, "speak": False})
            if not reply.get("ok"):
                print(f"smoke: ask failed: {reply.get('error')}", file=sys.stderr)
                return 6
            res = reply["result"]
            if res.get("error"):
                print(f"smoke: ask returned an error: {res['error']}", file=sys.stderr)
                return 6
            if not (res.get("reply") or "").strip():
                # An empty reply is a failure, not a pass. A turn that routes,
                # reports no error and says nothing is the worst outcome to
                # treat as green: the user asked and FRIDAY silently ignored
                # them. This caught a real case where the daemon's working
                # directory left it outside any git repo.
                print("smoke: ask produced an empty reply "
                      f"(tier={res.get('tier')!r})", file=sys.stderr)
                return 6
            print(f"ok   ask       tier={res['tier']} reply={res['reply'][:80]!r}")

        print("smoke: all stages passed")
        return 0
    finally:
        await client.close()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m friday.gateway.client")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--token", default=None,
                        help="defaults to the token file / FRIDAY_GATEWAY_TOKEN")
    parser.add_argument("--smoke", action="store_true", help="run the staged smoke test")
    parser.add_argument("--ask", metavar="TEXT",
                        help="also drive one full turn through the loop")
    parser.add_argument("--wait", type=float, default=0.0, metavar="SECONDS",
                        help="poll for the gateway to come up before testing")
    parser.add_argument("--method", help="call a single method and print its reply")
    parser.add_argument("--params", default="{}", help="JSON params for --method")
    args = parser.parse_args(argv)

    if args.smoke:
        return asyncio.run(smoke(args.url, args.token, ask=args.ask,
                                 wait=args.wait))

    if not args.method:
        parser.print_usage(sys.stderr)
        print("nothing to do: pass --smoke or --method", file=sys.stderr)
        return 1

    async def _one() -> int:
        async with GatewayClient(args.url, args.token) as client:
            reply = await client.connect()
            if not reply.get("ok"):
                print(json.dumps(reply, indent=2))
                return 3
            reply = await client.request(args.method, json.loads(args.params))
            print(json.dumps(reply, indent=2))
            return 0 if reply.get("ok") else 4

    return asyncio.run(_one())


if __name__ == "__main__":
    raise SystemExit(main())
