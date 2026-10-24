"""Run fixed, synthetic checks against FRIDAY's hosted and local transports."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

from friday.brain import AnthropicTransport, build_request
from friday.local_transformers import TransformersTransport
from friday.tools import TOOL_SPECS

TASKS = (
    {"id": "arithmetic", "prompt": 'Return only JSON: calculate 8 * 12.5 + 3 * 40, then 19% VAT. Keys net, vat, gross.', "expected": {"net": 220, "vat": 41.8, "gross": 261.8}},
    {"id": "instruction-following", "prompt": 'Return only JSON with keys owner and due_date. Notes: Mira will send the quote by 2026-09-16.', "expected": {"owner": "Mira", "due_date": "2026-09-16"}},
    {"id": "tool-route-processes", "prompt": "Use the available tool to list running processes. Do not answer from memory.", "tool": "list_processes"},
    {"id": "tool-route-forecast", "prompt": "Use the available forecast tool to check the weather in Berlin.", "tool": "query_forecast"},
)


async def observe(transport, task: dict) -> dict:
    request = build_request(task["prompt"], tools=TOOL_SPECS)
    request["max_tokens"] = 512
    text, calls = [], []
    started = time.perf_counter()
    async for event in transport.stream(request):
        if event.kind == "text":
            text.append(event.text)
        elif event.kind == "turn_end":
            calls.extend(call.name for call in event.tool_calls)
    elapsed = time.perf_counter() - started
    answer = "".join(text)
    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError:
        parsed = None
    quality = all(parsed.get(k) == v for k, v in task["expected"].items()) if task.get("expected") and isinstance(parsed, dict) else None
    routing = sorted(calls) == [task["tool"]] if task.get("tool") else None
    return {"latency_seconds": round(elapsed, 4), "quality_pass": quality, "expected_tool": task.get("tool"), "tool_calls": calls, "routing_pass": routing, "response": answer}


async def compare(local_model: str) -> dict:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is required to compare with the hosted baseline")
    transports = {"hosted": AnthropicTransport(key), "local": TransformersTransport(local_model)}
    results = {}
    for task in TASKS:
        results[task["id"]] = {}
        for name, transport in transports.items():
            results[task["id"]][name] = await observe(transport, task)
    return {"suite_version": 1, "metrics_are_observed": True, "tasks": results}


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("compare-backends", help="measure hosted vs local model quality, latency and tool routing")
    parser.add_argument("--local-model", default=os.environ.get("FRIDAY_LOCAL_MODEL"))
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    if not args.local_model:
        print("error: provide --local-model or set FRIDAY_LOCAL_MODEL", file=__import__("sys").stderr)
        return 1
    try:
        print(json.dumps(asyncio.run(compare(args.local_model)), indent=2))
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"backend comparison: {exc}", file=__import__("sys").stderr)
        return 1
    return 0
