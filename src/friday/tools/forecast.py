"""Structured time-series forecast queries and independently checked summaries."""

from __future__ import annotations

import json
import math
import re

from friday.tools import mcp

MAX_HORIZON = 512
_SERIES_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")

SPEC = {
    "name": "query_forecast",
    "description": "Query a configured MCP time-series forecast by series ID and horizon. "
    "Returns validated values, units and locally computed count/min/max/mean. "
    "Use these numerical summaries when answering forecast questions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "series_id": {"type": "string", "pattern": _SERIES_ID.pattern},
            "horizon_steps": {"type": "integer", "minimum": 1, "maximum": MAX_HORIZON},
        },
        "required": ["series_id", "horizon_steps"],
        "additionalProperties": False,
    },
}


def _number(value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError("forecast numbers must be finite, excluding booleans and strings")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError("forecast number exceeds floating-point range") from None
    if not math.isfinite(number):
        raise ValueError("forecast numbers must be finite")
    return number


def validate_forecast(payload: dict, series_id: str, horizon_steps: int) -> dict:
    """Reject malformed, mismatched or inconsistent results; discard extra text."""
    if payload.get("series_id") != series_id:
        raise ValueError("forecast series_id does not match the query")
    unit = payload.get("unit")
    if (not isinstance(unit, str) or not unit.strip() or len(unit) > 32
            or any(ord(c) < 32 for c in unit)):
        raise ValueError("forecast unit must be a nonempty string of at most 32 characters")
    raw_values = payload.get("values")
    if not isinstance(raw_values, list) or len(raw_values) != horizon_steps:
        raise ValueError("forecast values must match horizon_steps exactly")
    values = [_number(v) for v in raw_values]
    summary = {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": _number(math.fsum(v / len(values) for v in values)),
    }
    if "summary" in payload:
        claimed = payload["summary"]
        if not isinstance(claimed, dict) or set(claimed) != set(summary):
            raise ValueError("forecast summary requires count, min, max and mean")
        for key, expected in summary.items():
            if key == "count":
                valid = type(claimed[key]) is int and claimed[key] == expected
            else:
                valid = math.isclose(_number(claimed[key]), expected, rel_tol=1e-9, abs_tol=1e-12)
            if not valid:
                raise ValueError(f"forecast summary {key} disagrees with values")
    return {"series_id": series_id, "unit": unit, "values": values, "summary": summary}


async def query_forecast(series_id: str, horizon_steps: int) -> str:
    """READ_ONLY. Accept identifiers and a bounded horizon, never raw local data."""
    if not isinstance(series_id, str) or not _SERIES_ID.fullmatch(series_id):
        raise ValueError("invalid forecast series_id")
    if type(horizon_steps) is not int or not 1 <= horizon_steps <= MAX_HORIZON:
        raise ValueError(f"horizon_steps must be an integer from 1 to {MAX_HORIZON}")
    payload = await mcp.call_tool(
        "query_forecast", {"series_id": series_id, "horizon_steps": horizon_steps}
    )
    return json.dumps(validate_forecast(payload, series_id, horizon_steps), allow_nan=False)
