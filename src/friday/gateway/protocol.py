"""Wire format for the FRIDAY gateway.

Three message shapes, all JSON objects, all distinguished by which key is
present -- so a reader never has to guess from context:

    request   {"id": "7", "method": "health", "params": {}}
    response  {"id": "7", "ok": true,  "result": {...}}
              {"id": "7", "ok": false, "error": {"code": "...", "message": "..."}}
    event     {"event": "state.changed", "data": {...}}

Requests carry an opaque `id` chosen by the client and echoed back verbatim.
Events never carry an `id`: they are unsolicited, so there is nothing to
correlate them with, and a client that tries to match them against a pending
request would hang.

The protocol is versioned as a *range*, not a number. A client sends the
window it can speak (`min_protocol`..`max_protocol`) and the server picks the
highest value inside both windows. This is what lets an old phone client and a
freshly-built one talk to the same daemon during a rollout: neither side has
to be upgraded in lockstep, they just have to overlap somewhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# Bump MAX when adding a method or an event. Bump MIN only when dropping
# support for an old shape -- that is the breaking change, and raising MIN is
# how a client learns it must upgrade instead of failing in some subtler way
# halfway through a session.
PROTOCOL_MIN = 1
PROTOCOL_MAX = 1


class ErrorCode(str, Enum):
    """Stable, machine-readable failure reasons.

    Strings rather than integers on purpose: these show up in logs and smoke
    output, and `unauthorized` is self-describing where `4001` needs a table.
    """

    BAD_REQUEST = "bad_request"
    UNAUTHORIZED = "unauthorized"
    RATE_LIMITED = "rate_limited"
    UNSUPPORTED_PROTOCOL = "unsupported_protocol"
    NOT_CONNECTED = "not_connected"
    ALREADY_CONNECTED = "already_connected"
    UNKNOWN_METHOD = "unknown_method"
    FORBIDDEN = "forbidden"
    UNAVAILABLE = "unavailable"
    INTERNAL = "internal"


class ProtocolError(Exception):
    """A failure that maps cleanly onto an error response.

    Handlers raise this instead of returning error dicts so the dispatch loop
    has exactly one place that formats failures. Anything else that escapes a
    handler is a bug and becomes `internal` with its detail withheld.
    """

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Request:
    id: str
    method: str
    params: dict[str, Any] = field(default_factory=dict)


def parse_request(raw: str | bytes) -> Request:
    """Decode one client frame, or raise ProtocolError(BAD_REQUEST).

    Deliberately strict about `id` and `method` being non-empty strings. A
    request with a missing id cannot be answered -- the reply would be
    uncorrelatable -- so it is rejected at the door rather than dispatched and
    silently dropped later.
    """
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ProtocolError(ErrorCode.BAD_REQUEST, f"not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(ErrorCode.BAD_REQUEST, "frame must be a JSON object")

    ident = payload.get("id")
    method = payload.get("method")
    if not isinstance(ident, str) or not ident:
        raise ProtocolError(ErrorCode.BAD_REQUEST, "missing 'id'")
    if not isinstance(method, str) or not method:
        raise ProtocolError(ErrorCode.BAD_REQUEST, "missing 'method'")

    params = payload.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ProtocolError(ErrorCode.BAD_REQUEST, "'params' must be an object")
    return Request(id=ident, method=method, params=params)


def ok(ident: str, result: Optional[dict[str, Any]] = None) -> str:
    return json.dumps({"id": ident, "ok": True, "result": result or {}})


def err(ident: str, code: ErrorCode, message: str) -> str:
    return json.dumps(
        {"id": ident, "ok": False, "error": {"code": code.value, "message": message}}
    )


def event(name: str, data: Optional[dict[str, Any]] = None) -> str:
    return json.dumps({"event": name, "data": data or {}})


def negotiate(client_min: Any, client_max: Any) -> int:
    """Pick the highest protocol both sides speak, or raise.

    Returns the overlap's upper bound. Raising here rather than falling back
    to PROTOCOL_MIN is intentional: a silent downgrade to a version the client
    did not offer produces messages it cannot parse, and the resulting failure
    surfaces far from its cause.
    """
    if not isinstance(client_min, int) or not isinstance(client_max, int):
        raise ProtocolError(
            ErrorCode.BAD_REQUEST, "min_protocol and max_protocol must be integers"
        )
    if client_min > client_max:
        raise ProtocolError(
            ErrorCode.BAD_REQUEST,
            f"inverted protocol window: {client_min} > {client_max}",
        )
    chosen = min(client_max, PROTOCOL_MAX)
    if chosen < max(client_min, PROTOCOL_MIN):
        raise ProtocolError(
            ErrorCode.UNSUPPORTED_PROTOCOL,
            f"client speaks {client_min}..{client_max}, "
            f"server speaks {PROTOCOL_MIN}..{PROTOCOL_MAX}",
        )
    return chosen
