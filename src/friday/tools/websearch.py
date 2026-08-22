"""Exa web search: the first tool that sends data to a third party.

Exa's `/search` endpoint, verified against Exa's public API reference
(docs.exa.ai/reference/search, checked 2026-08-22):
  - request:  POST https://api.exa.ai/search, header `x-api-key: <key>`,
              JSON body `{"query": str, "numResults": int, "contents": {"text": true}}`
  - response: `{"requestId": ..., "results": [{"title", "url", "id", "text", ...}]}`

Fields relied on here: request `query`, `numResults`, `contents.text`;
response `results[].title`, `results[].url`, `results[].text`.

Because this is the only tool that leaves the machine, `safe_query()` in
`friday.tools.sanitize` refuses anything that looks like a credential or a
local-data dump -- rather than a search phrase -- before a request is ever
built. The HTTP call itself sits behind the `HTTPTransport` Protocol below
(same shape as `friday.voice.stt.Transport`), so tests never touch the
network.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Protocol

import httpx

from friday.tools.sanitize import safe_query

EXA_URL = "https://api.exa.ai/search"
DEFAULT_NUM_RESULTS = 5
MAX_RESULTS = 10
MAX_SNIPPET_CHARS = 200
MAX_OUTPUT_CHARS = 1500


class HTTPTransport(Protocol):
    """Minimal async HTTP POST seam, implemented by HttpxTransport (real)
    and a fake in tests. Injected, never branched on internally."""

    async def post(self, url: str, *, headers: dict, json: dict) -> Any: ...


class HttpxTransport:
    """Real transport: one POST via httpx.AsyncClient."""

    async def post(self, url: str, *, headers: dict, json: dict) -> Any:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, headers=headers, json=json)
            response.raise_for_status()
            return response.json()


def _require_api_key() -> str:
    key = os.environ.get("EXA_API_KEY")
    if not key:
        raise RuntimeError(
            "EXA_API_KEY is not set. Export EXA_API_KEY=<your key> and retry."
        )
    return key


def _format_results(payload: dict, num_results: int) -> str:
    results = (payload.get("results") or [])[:num_results]
    if not results:
        return "[no results]"
    lines = []
    for i, r in enumerate(results, start=1):
        title = (r.get("title") or "untitled").strip()
        url = (r.get("url") or "").strip()
        text = (r.get("text") or "").strip().replace("\n", " ")
        if len(text) > MAX_SNIPPET_CHARS:
            text = text[:MAX_SNIPPET_CHARS] + "..."
        lines.append(f"{i}. {title} - {url}\n   {text}")
    out = "\n".join(lines)
    if len(out) > MAX_OUTPUT_CHARS:
        out = out[:MAX_OUTPUT_CHARS] + f"\n...[clipped, {len(out)} chars total]"
    return out


async def web_search(
    query: str,
    num_results: int = DEFAULT_NUM_RESULTS,
    *,
    transport: Optional[HTTPTransport] = None,
) -> str:
    """READ_ONLY. Search the web via Exa and return a compact ranked list of
    titles, URLs and short snippets."""
    clean_query = safe_query(query)
    n = max(1, min(int(num_results), MAX_RESULTS))
    api_key = _require_api_key()
    transport = transport or HttpxTransport()
    payload = await transport.post(
        EXA_URL,
        headers={"x-api-key": api_key, "content-type": "application/json"},
        json={"query": clean_query, "numResults": n, "contents": {"text": True}},
    )
    return _format_results(payload, n)
