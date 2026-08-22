# FRIDAY — Project Handout

## Project at a glance

**FRIDAY** is a voice-first personal AI assistant for Linux. It runs as a local Python service, listens for a wake word, converts speech to text, chooses the fastest suitable response path, and speaks the result. It can also inspect local machine state, use approved tools, remember prior context, delegate work to coding-agent sessions, and accept authenticated commands through a local WebSocket gateway.

| Item | Details |
|---|---|
| Version | 0.1.0 |
| Language | Python 3.11+ |
| Platform | Linux desktop with PipeWire/PulseAudio compatibility |
| Wake-word engine | openWakeWord |
| Speech services | Deepgram streaming STT and Aura TTS |
| Reasoning model | Anthropic Claude Opus 5 |
| Local storage | SQLite with FTS5 full-text search |
| Control interface | CLI and authenticated WebSocket gateway |
| Test status | **367 tests passing** on 23 August 2026 |

## Aim

The project is designed to make an AI assistant feel immediate and useful on a developer workstation. Its central design choice is not to send every utterance to an LLM. Simple commands and machine-state questions are handled locally, while only open-ended tasks use the reasoning model.

## How a voice turn works

```text
Microphone
    ↓
Wake-word detection (currently "alexa")
    ↓
Streaming speech-to-text + local voice activity detection
    ↓
Three-tier intent router
    ├── Tier 1: reflex action ──────────────→ immediate local action
    ├── Tier 2: machine-state question ────→ local snapshot and answer
    └── Tier 3: reasoning task
             ↓
       acknowledgement sound
             ↓
       context + memory + Claude + gated tools
             ↓
       streaming text-to-speech
```

### Response tiers

1. **Reflex:** Commands such as “stop,” “mute,” or “never mind” are mapped directly to hard-coded actions. No model or network call is required.
2. **State query:** Questions such as “what branch am I on?” or “what’s running?” are answered from local Git, process, port, tmux, battery, disk, and memory information.
3. **Reasoning:** General requests use Claude. FRIDAY plays a pre-rendered acknowledgement immediately, assembles relevant context and memory, then streams the answer to speech sentence by sentence.

## Main capabilities

- Wake-word activation with startup warm-up protection against false triggers.
- Streaming Deepgram transcription, local VAD, and interim-transcript fallback.
- Deepgram Aura speech synthesis with pre-rendered low-latency acknowledgements.
- Barge-in support: a new wake word can pre-empt current playback.
- Local machine awareness, including Git state, active processes, listening ports, tmux sessions, and system resources.
- Long-term memory stored in SQLite and retrieved using FTS5 relevance ranking.
- Read-only file, process, log, command, and web-search tools.
- Coding-agent delegation through owned tmux sessions.
- Permission tiers for safe, machine-modifying, and destructive actions.
- Microphone priority management so calls, meetings, and recording applications pre-empt FRIDAY.
- Echo-cancellation integration and microphone gating during playback.
- Authenticated local WebSocket API for health checks, text turns, speech, and state events.
- Daemon lifecycle management through either direct mode or a systemd user service.
- Structured turn spans and percentile-based latency measurements.

## Architecture and source map

| Area | Main files | Responsibility |
|---|---|---|
| Application | `core/app.py`, `loop.py`, `daemon.py` | Supervision, lifecycle, and the assembled voice loop |
| Voice | `voice/wake.py`, `stt.py`, `tts.py`, `ack.py` | Wake detection, transcription, synthesis, and acknowledgements |
| Audio | `audio/manager.py`, `priority.py`, `echocancel.py` | Device ownership, pre-emption, and echo cancellation |
| Routing | `router.py`, `tiers/state_query.py` | Three-tier classification and fast local answers |
| Intelligence | `brain.py`, `tools/`, `permissions.py` | Model streaming, tool execution, sanitisation, and approval gates |
| Context | `state.py`, `memory.py`, `attention.py` | Machine snapshot, persistent memory, and proactive event scoring |
| Agents | `agents.py` | Safe management of FRIDAY-owned tmux agent sessions |
| Gateway | `gateway/` | Authenticated WebSocket protocol and clients |
| Operations | `cli.py`, `deploy/friday.service`, `GOLIVE.md` | User commands, service deployment, and live runbook |
| Verification | `tests/` | Unit and integration-style tests with injected fake transports |

## Safety and privacy

FRIDAY uses an explicit, default-deny tool policy:

| Risk level | Behaviour |
|---|---|
| Read-only | Runs automatically after argument sanitisation |
| Safe and reversible | Runs automatically |
| Machine-modifying | Requires an approval callback |
| Destructive or unknown | Requires explicit approval; unknown tools are denied by default |

Secrets are loaded from `~/.friday/env`, outside the repository. File and command inputs are sanitised, sensitive `/proc` paths are denied, and web-search queries that resemble credentials or local-data dumps are rejected before leaving the machine. The gateway binds to `127.0.0.1` by default and requires token authentication.

## Setup

### Requirements

- Python 3.11 or newer
- A working Linux audio session (PipeWire or PulseAudio-compatible)
- Deepgram and Anthropic API keys
- `git`, `tmux`, and common Linux process/audio utilities for the corresponding features
- Optional: an Exa API key for web search

### Install the project

```bash
python -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install pytest
```

The project defines its development dependencies as a dependency group, so environments using `uv` can instead run:

```bash
uv sync --dev
```

Create the credentials file and restrict its permissions:

```bash
mkdir -p ~/.friday
printf 'export DEEPGRAM_API_KEY=your_key\nexport ANTHROPIC_API_KEY=your_key\n' > ~/.friday/env
chmod 600 ~/.friday/env
```

Add `EXA_API_KEY` to the same file only if web search is needed. Do not commit this file.

## Everyday commands

```bash
friday doctor                         # diagnose credentials, audio, and gateway
friday start                          # start in the background
friday start --foreground             # run interactively for debugging
friday status                         # show health, voice-loop, and mic state
friday ask "what branch am I on"      # submit a text turn
friday ask --speak "what is running"  # submit and speak the response
friday say "the build is green"       # synthesize text directly
friday hear --seconds 20              # inspect mic level and wake score
friday logs -f                        # follow service logs
friday stop                           # stop cleanly
```

To install and immediately start the supplied systemd user service:

```bash
friday install --now
```

The checked-in service file contains machine-specific absolute paths. Update it before using it from another checkout or computer.

## Testing and diagnostics

Run the automated suite:

```bash
.venv/bin/pytest -q
```

Current result:

```text
367 passed in 59.79s
```

Useful additional checks:

```bash
python -m friday --selftest
python -m friday.scripts.bench_ack
python src/friday/scripts/percentiles.py
friday smoke
friday doctor
```

Tests isolate external services behind injectable transports, so the normal suite does not require live Deepgram, Anthropic, or Exa requests. Hardware and credential-dependent checks are documented in `GOLIVE.md`.

## Measured results

Live measurements recorded in the project runbook include:

| Measurement | Recorded result |
|---|---:|
| Wake inference | 2.4 ms plus an 80 ms audio chunk |
| Local VAD | 0.31 ms |
| Ack audible, p99 | 111.9 ms |
| Local state query | 36.9 ms |
| First Aura content audio | 226 ms |
| Claude reasoning TTFT | approximately 1.0–2.7 s |
| Clean shutdown with turn in flight | 183 ms |

The LLM time-to-first-token misses the original 300–600 ms goal. The current user experience masks much of that delay with an acknowledgement; a smaller model would be required for materially faster reasoning responses.

## Current limitations

- The wake phrase is **“alexa”**, because no trained “friday” openWakeWord model is included.
- Focused-window detection is not implemented on KDE Wayland, so questions like “what am I looking at?” cannot yet be answered.
- Dynamic spoken output depends on Deepgram Aura; only acknowledgement clips are local files.
- Full-duplex interruption phrases are not available during playback; barge-in requires saying the wake word.
- The echo-cancellation attenuation target still needs a valid speaker-to-microphone hardware measurement.
- Test or CI failure questions cannot always be answered from the current local state snapshot and are escalated rather than guessed.
- The systemd unit is tailored to the current checkout path and is not portable without editing.

## Suggested demonstration

1. Run `friday doctor` and show that credentials, audio, and the gateway are healthy.
2. Start FRIDAY and show `friday status`.
3. Ask “alexa, what branch am I on?” to demonstrate a fast, model-free state query.
4. Ask an open-ended task to demonstrate acknowledgement, reasoning, tools, and streamed speech.
5. Start a higher-priority microphone application to demonstrate automatic audio pre-emption.
6. Finish with the automated test result and the latency table.

## Summary

FRIDAY is an operational prototype of a low-latency Linux desktop assistant. Its strongest engineering feature is the separation of immediate local actions, local state queries, and expensive reasoning turns. That architecture improves responsiveness, reduces unnecessary API use, and creates clear safety boundaries around tool execution while preserving a natural voice interface.
