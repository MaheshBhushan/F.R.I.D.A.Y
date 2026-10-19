# Runtime metrics and alerts

Install the optional Flask dashboard and run it alongside FRIDAY:

```bash
uv sync --extra dashboard --dev
uv run python -m friday.dashboard
```

Open <http://127.0.0.1:8766/>. The dashboard refreshes every five seconds and
exposes the same data at `/api/metrics`. It shows the latest 200 turns, stage
offsets, stage deltas, accumulated durations, p50/p90/p99, and individual traces.
It reads `~/.friday/spans.jsonl` and the adjacent `spans.jsonl.active` directory.
It requires no API keys or running voice services to inspect recorded traces.

The independent monitor polls every two seconds and writes `[alert]` and
`[alert-cleared]` events to stderr. It runs without an open browser. Repeated
polls suppress duplicate alerts during that monitor process. Restarting the
monitor reports existing alerts again. Alerts clear when the record recovers
or leaves the selected window. Local events are the notification channel.

```bash
uv run python -m friday.dashboard --slow-ms 1500 --missing-ms 8000 --last 500 --interval 1
```

The response timer starts at `speech_ended_vad`. Older traces fall back to
`stt_final`, `intent_classified`, `llm_sent`, or `tts_started`. A response uses
`first_content_audio`, a reflex's `ack_audible`, or `task_complete` for turns
without content audio. A reasoning acknowledgement alone does not count as a
response. Slow alerts fire strictly above `--slow-ms` (default 3000 ms).
Missing alerts fire at or above `--missing-ms` (default 10000 ms) when no response
mark exists. Wake-only and STT-only traces do not trigger response alerts.

FRIDAY publishes atomic active checkpoints when routing, model requests, or
speech synthesis establish a response obligation. It updates them at response
milestones and removes them after appending the final trace. A crashed process
can leave a checkpoint so that its missing response remains visible. Final
records override matching checkpoints. Existing recorded traces remain usable;
older runtimes cannot expose a stalled turn before writing its final record.

Replay any recorded JSONL file without modifying it:

```bash
uv run python -m friday.dashboard --path /path/to/recorded-spans.jsonl
uv run python -m friday.dashboard --path /path/to/recorded-spans.jsonl --check
```

`--check` prints one JSON snapshot. Exit codes are 0 for no alerts, 1 for alerts,
and 2 for a source read error or invalid configuration. Missing-response ages
use the current wall clock during replay. Slow-response timings use recorded
monotonic stage offsets. The dashboard reports malformed or partially written
lines and skips them. A missing log produces an empty dashboard. Other source
errors return HTTP 503 and generate an alert event.

The selected window bounds displayed turns and JSONL memory use. Each poll
streams the log; large histories increase scan time. The dashboard reports
recorded timing data and configured alert decisions. It makes no claim about
support productivity.

The command binds to loopback and disables Flask's debugger and reloader. It
provides a local dashboard through Flask's development server. External hosting
requires a suitable WSGI server and access controls; run the alert monitor
separately if another server imports `create_app()`.

## Offline verification

`tests/fixtures/runtime_traces.jsonl` contains synthetic healthy, delayed,
missing, and wake-only traces. These fixtures are test inputs. They contain no
production measurements. The delayed-response tests advance an injected clock
and exercise active checkpoints, exact thresholds, recovery, and final replay.

```bash
uv run python -m pytest -q tests/test_dashboard.py tests/test_stats.py tests/test_router.py
uv run python -m friday.dashboard --path tests/fixtures/runtime_traces.jsonl --check
```

The replay command returns 1 because the fixture contains two alerts.
