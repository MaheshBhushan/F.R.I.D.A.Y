# Async MCP forecast bridge

Install the optional MCP Python SDK with `uv sync --dev --extra mcp`. Configure
`FRIDAY_MCP_COMMAND` as a JSON array containing a trusted stdio server executable
and its arguments. Use absolute paths for executables and server scripts:

```sh
export FRIDAY_MCP_COMMAND='["/absolute/path/to/python", "/absolute/path/to/forecast_server.py"]'
```

The client uses the [official Python SDK's stdio client](https://github.com/modelcontextprotocol/python-sdk/tree/v1.26.0).
The server must implement `query_forecast(series_id: str, horizon_steps: int)`.
The model can request this tool through FRIDAY's existing reasoning loop.
The command runs directly without a shell. Server setup belongs to the operator;
model arguments cannot choose a server, command or remote tool name. The SDK
passes its minimal default environment to the server. Configure provider access
inside the server without placing secrets in command arguments.

The local permission table classifies `query_forecast` as read-only. Both the
tool dispatcher and MCP bridge enforce that table. The bridge only binds this
one tool. Remote tool discovery cannot extend the allowlist or change risk tiers.
Existing tools retain their approval requirements.

Requests contain a series identifier and a horizon of 1–512 steps. Identifiers
contain 1–128 ASCII letters, digits, underscores, periods, colons or hyphens and
start with a letter or digit. The forecast server owns the series data and step
cadence. FRIDAY does not upload local history or infer a time interval.

The response must contain this MCP `structuredContent` object:

```json
{
  "series_id": "house.load",
  "unit": "kW",
  "values": [1.0, 2.0, 3.0],
  "summary": {"count": 3, "min": 1.0, "max": 3.0, "mean": 2.0}
}
```

The summary is optional. FRIDAY computes count, minimum, maximum and arithmetic
mean from the values. A supplied summary must contain all four fields and agree
with the computed numbers. Numeric comparisons use relative tolerance `1e-9`
and absolute tolerance `1e-12`; count must be an exact integer. Values must be
finite numbers; booleans and numeric strings fail validation. The series must
match the request, the number of values must equal the horizon, and the unit
must contain 1–32 characters without ASCII control characters. FRIDAY discards
extra response fields and unstructured text. The numbers above illustrate the
contract; they do not represent a measured forecast.

Each call owns an asyncio SDK session and subprocess. A 15-second timeout covers
connection, initialization and execution. SDK context cleanup closes the session
and reaps the process after completion, failure, timeout or cancellation. Cleanup
can extend the deadline while the SDK terminates the server. Timeouts and invalid
results become error outcomes in the tool loop. Cancellation propagates to the
caller. Missing configuration or optional dependencies produce explicit errors.
No live forecast service or credentials ship with FRIDAY.

Turn records now include accumulated `durations_ns.llm_wait` and
`durations_ns.tools`. `llm_wait` measures time awaiting model stream events across
every reasoning round. It excludes tools and time spent consuming tokens, such
as speech playback. `tools` measures each concurrent tool round's wall time,
including authorization, connection, execution and cleanup. Overlapping calls
count once per round. `friday stats` reports these durations separately from the
existing stage offsets. Older records remain readable.

Run the offline bridge checks with:

```sh
uv run --extra mcp pytest -q tests/test_mcp.py tests/test_tool_timing.py tests/test_permissions.py tests/test_brain.py tests/test_stats.py
```

The integration tests launch a local fixture MCP server. They require no API
keys or network service and check that server processes exit after each call.
