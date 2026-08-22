"""FRIDAY's gateway: a WebSocket control plane in front of the voice loop.

Kept as its own package, next to `friday.audio` rather than inside
`friday.voice`, for the same reason: it is a boundary the rest of the system
talks *through*, not a detail of how speech works. The voice loop must remain
runnable with no gateway at all.
"""

from friday.gateway.auth import RateLimiter, load_or_create_token
from friday.gateway.client import GatewayClient, smoke
from friday.gateway.protocol import (
    PROTOCOL_MAX,
    PROTOCOL_MIN,
    ErrorCode,
    ProtocolError,
    Request,
    parse_request,
)
from friday.gateway.server import DEFAULT_HOST, DEFAULT_PORT, Gateway

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "ErrorCode",
    "Gateway",
    "GatewayClient",
    "PROTOCOL_MAX",
    "PROTOCOL_MIN",
    "ProtocolError",
    "RateLimiter",
    "Request",
    "load_or_create_token",
    "parse_request",
    "smoke",
]
