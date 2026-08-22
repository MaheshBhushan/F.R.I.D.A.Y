"""Permission tiers for tool execution: an explicit allowlist, default-deny.

Every tool the LLM can call is mapped to exactly one `Risk` tier in `RISK`.
Anything not in that table -- a hallucinated tool name, a renamed tool, a
tool added without a tier -- resolves to `Risk.DESTRUCTIVE`, so an
unrecognised call needs explicit approval rather than being waved through.

Read-only and safe-reversible tiers auto-execute. Machine-modifying and
destructive tiers go to an `ApprovalCallback` the caller supplies, and the
callback is handed the exact action string before anything runs.

Not a policy engine: a dict, a lookup, and one async check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Optional


class Risk(Enum):
    READ_ONLY = "read_only"
    SAFE_REVERSIBLE = "safe_reversible"
    MACHINE_MODIFYING = "machine_modifying"
    DESTRUCTIVE = "destructive"


# The allowlist. A tool absent from this table is treated as DESTRUCTIVE.
RISK: dict[str, Risk] = {
    "read_file": Risk.READ_ONLY,
    "list_processes": Risk.READ_ONLY,
    "read_log": Risk.READ_ONLY,
    "run_readonly_command": Risk.READ_ONLY,
    "open_app": Risk.SAFE_REVERSIBLE,
    "install_package": Risk.MACHINE_MODIFYING,
    "delete_path": Risk.DESTRUCTIVE,
}

AUTO_EXECUTE = frozenset({Risk.READ_ONLY, Risk.SAFE_REVERSIBLE})

# Tiers where the approval prompt must show the exact action verbatim.
EXPLICIT_APPROVAL = frozenset({Risk.DESTRUCTIVE})


class PermissionDenied(Exception):
    """Raised when a tool call is refused: unapproved, unrecognised, or with
    arguments that failed sanitisation. Never a silent downgrade."""


@dataclass(frozen=True)
class ApprovalRequest:
    """What the user is being asked to approve. `action` is the exact,
    fully-rendered action -- it is what must be shown before anything runs."""

    tool: str
    risk: Risk
    arguments: dict
    action: str

    @property
    def requires_explicit_approval(self) -> bool:
        return self.risk in EXPLICIT_APPROVAL


# Supplied by the caller; returns True to allow the call.
ApprovalCallback = Callable[[ApprovalRequest], Awaitable[bool]]


def risk_of(tool: str) -> Risk:
    """Tier for `tool`, defaulting to DESTRUCTIVE for anything unrecognised."""
    return RISK.get(tool, Risk.DESTRUCTIVE)


def render_action(tool: str, arguments: dict) -> str:
    """Human-readable, unambiguous rendering of the call to be shown for
    approval. Sorted keys so the same call always renders identically."""
    args = ", ".join(f"{k}={json.dumps(v)}" for k, v in sorted(arguments.items()))
    return f"{tool}({args})"


async def authorize(
    tool: str,
    arguments: dict,
    approve: Optional[ApprovalCallback] = None,
) -> ApprovalRequest:
    """Gate one tool call. Returns the ApprovalRequest that was evaluated, or
    raises PermissionDenied. Unknown tools and a missing callback both deny."""
    risk = risk_of(tool)
    request = ApprovalRequest(
        tool=tool,
        risk=risk,
        arguments=dict(arguments),
        action=render_action(tool, arguments),
    )
    if tool not in RISK:
        raise PermissionDenied(
            f"unrecognised tool {tool!r}: refused as destructive (default deny)"
        )
    if risk in AUTO_EXECUTE:
        return request
    if approve is None:
        raise PermissionDenied(
            f"{risk.value} tool {tool!r} needs approval but no approval callback "
            f"was supplied; refused: {request.action}"
        )
    if not await approve(request):
        raise PermissionDenied(f"approval refused for: {request.action}")
    return request
