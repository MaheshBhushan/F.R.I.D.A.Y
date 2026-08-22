# FRIDAY — go-live checklist

> **STATUS 2026-08-22: all items below verified live.** Keys are in
> `~/.friday/env` (0600, outside any git repo); `set -a; . ~/.friday/env; set +a`
> to load. Measured results are recorded inline. Kept as a runbook and because
> the findings matter.

Everything in `src/friday/` is built and gated offline. These are the gates that
could NOT be measured without credentials or hardware access. Run them in order.

## 1. Export the keys

```sh
export DEEPGRAM_API_KEY=...
export ANTHROPIC_API_KEY=...
```

Both are read from the environment only. No key is stored in the repo.

> Note: `read_file` refuses `/proc/*/environ` and `/proc/*/cmdline` precisely so
> the model cannot read these two variables back out of FRIDAY's own process.
> Do not relax that (`src/friday/tools/sanitize.py`, `DENIED_NAME_RE`).

## 2. Re-render the ack bank with the real voice

The current bank is espeak-ng placeholder audio, marked in
`src/friday/voice/acks/SOURCE`. Latency was measured with it; only the voice
changes.

```sh
.venv/bin/python -m friday.scripts.render_acks --engine aura
```

Then confirm `SOURCE` no longer says `placeholder`, and re-run the ack bench —
Aura files may differ in length/sample rate, which shifts the p99:

```sh
.venv/bin/python -m friday.scripts.bench_ack
.venv/bin/python src/friday/scripts/percentiles.py
```

Gate: `ack_audible - speech_ended_vad` p99 < 150 ms. (Placeholder bank: 41.8 ms.)

## 3. Measure prompt caching — do this FIRST, it is the likeliest silent failure

The cacheable prefix is ~1.1k CHARACTERS of system prompt + 7 tool definitions.
Anthropic's minimum cacheable prefix is ~1024 TOKENS. The prefix is therefore
probably too short to cache, and it will fail SILENTLY — full input price and
full TTFT on every turn, with no error.

Check `usage.cache_read_input_tokens` on the second turn:

- non-zero  -> caching works, done.
- zero      -> either pad the stable prefix past the minimum with content worth
               having (a fuller system prompt, more tool definitions), or accept
               no caching and stop paying the ordering complexity for it.

Gate: `cache_read_input_tokens > 0` on turn 2, or a decision recorded here.

**RESULT: PASS.** The prefix is 1682 tokens (not the ~275 estimated from its
character count -- tool JSON schemas are far denser than prose, and there are 10
tools, not 7). Turn 1 `cache_creation=1607`, turn 2 `cache_read=1607`.

## 4. Live latency budget

Run real voice turns, then the percentile table:

```sh
.venv/bin/python src/friday/scripts/percentiles.py
```

Gates, from the design budget:

| stage | target | offline status |
|---|---|---|
| wake word detection | < 100 ms | 2.4 ms inference + 80 ms chunk |
| VAD speech-end | 30-50 ms | 0.31 ms |
| ack audible | < 150 ms | 41.8 ms |
| STT final after speech-end | ~300 ms | UNMEASURED (fake transport) |
| LLM time-to-first-token | 300-600 ms | UNMEASURED (fake transport) |
| first content audio | ~1.1-1.5 s | UNMEASURED |
| state-query end-to-end | < 100 ms | 36.9 ms (real, no LLM) |

**LIVE RESULTS.** ack 111.9 ms p99 (real Aura bank, speaker-pinned);
`stt_final - speech_ended_vad` = **0 ms** (Deepgram's final arrives before local
VAD declares speech-end); Aura first content audio 226 ms; reasoning-turn
**TTFT ~1.0-2.7 s**, which BLOWS the 300-600 ms budget.

The TTFT miss is inherent to `claude-opus-5`, not a bug and not thinking
overhead -- `thinking={"type":"disabled"}` measured 1059 ms vs 1056 ms default,
i.e. no difference. The design already absorbs it: the ack plays at 112 ms, so
she says "Checking." while the model works. If you ever want genuinely fast
content, that needs a smaller model for simple reasoning turns, not tuning.

Report percentiles, never averages. `percentiles.py` refuses to print averages
by design.

## 5. Tune `utterance_end_ms`, not `endpointing`

The design doc's "conservative endpointing = ~500 ms tax" is wrong about the
parameter. Deepgram's `endpointing` default is 10 ms; `stt.py` sets it to 100 ms
deliberately (avoid finalizing on micro-breaths). The ~1000 ms default that
actually costs you is `utterance_end_ms`, which is NOT currently set. If turns
feel sluggish to close, that is the knob.

This does not affect the ack path — local VAD owns it and never waits on
Deepgram (proven with `transport=None`).

## 6. Echo cancellation (needs hardware access, not a key)

Config is live at `~/.config/pipewire/pipewire.conf.d/99-echo-cancel.conf` and
follows the default sink/source. The >=20 dB attenuation measurement was never
run: it needs the headphone jack EMPTY, because with headphones there is no
acoustic path from speaker to mic and any measurement is meaningless.

```sh
wpctl get-volume @DEFAULT_AUDIO_SINK@   # save it, restore it after
# play ~3 s through the echo-cancel sink while recording the echo-cancel source
# repeat through the raw sink/source as a control; compare RMS via
# ffmpeg -i rec.wav -af volumedetect -f null -
```

Gate: echo-cancelled recording >= 20 dB below the raw control.

Until this passes, mic gating during playback (`tts.py`, `MicGate`) is the only
thing preventing self-interruption. Do not assume the mic is clean.

## 7. Verify the real SDK surfaces

Every Deepgram/Anthropic call path was written against the INSTALLED SDK source
but never executed. Expect to fix small shape mismatches here, not redesigns:

- `stt.py` — `listen.v1.connect`, real interim/final message decode
- `tts.py` — `speak.v1.connect`, `SpeakV1Flushed` sentence boundary detection
- `brain.py` — `messages.stream`, `stop_reason == "tool_use"`, whether >1 tool
  block really arrives in one message, thinking-block round-trip

## 8. Decide about thinking output

`claude-opus-5` has thinking on by default. For a voice assistant that means a
silent pause before speech. `brain.py` does not set `thinking`/`output_config`.
Decide once you can measure the pause.

## Live-only bugs found during go-live (all fixed, all now regression-tested)

The fake transports could not expose any of these. All three were in `stt.py`:

1. **Zero-length chunk killed the stream.** Deepgram reads an empty binary frame
   as end-of-stream and closes the socket. `audio_source()` forwarded an empty
   preroll unconditionally, presenting as a mid-turn network failure.
2. **`run_utterance` deadlocked when audio ended without a silence tail.**
   `closer()` awaited local VAD's speech-end forever. A clipped buffer or closed
   mic stream would hang the entire voice loop permanently.
3. **Pump/recv exceptions were swallowed** by `gather(return_exceptions=True)`,
   so a dead audio pump degraded into a silent empty turn -- FRIDAY would just
   stop hearing you, with nothing in the logs.

Also fixed: the Aura ack bank arrived with streaming placeholder WAV sizes
(`data` = 0x7fff0000), so `wave` reported 1,073,709,056 frames and `ack.py`
would have slept ~12 hours per ack. `render_acks.py` now rewrites the headers.

## Not built (deliberately)

- Wake word is **"alexa"** (an openWakeWord pretrained model). There is no
  pretrained "friday"; that needs a trained model, which is a real task.
- Speculative read-only tool execution — skipped, see T8 report.
- `focused_window` in `state.py` is always `None`; KDE Wayland has no cheap
  focus query. "What am I looking at?" is unanswerable until that is added.
- Local dynamic TTS. Acks are pre-rendered files; content TTS is Aura only.

## Speaking indicator (added 2026-08-22)

`src/friday/voice/indicator.py`. Four states -- `idle ○`, `listening ◉`,
`thinking ◌`, `talking ◆` -- published two ways:

* an in-place ANSI status line on stderr (only when stderr is a tty)
* a one-word state file at `~/.friday/status`, written atomically

Not a Qt/GTK tray icon: that needs its own event loop inside a single-asyncio-loop
process and would be invisible on a headless/SSH run. Disable with `FRIDAY_INDICATOR=0`.

Wired at: `capture_loop` (LISTENING on detection), `brain.complete` (THINKING),
`tts.TTSSpeaker.speak` (TALKING, reset in `finally`), `ack.play_ack` (TALKING).

Two placement constraints, both deliberate:

* The transition in `ack.play_ack` is around the stream, never inside the
  PortAudio callback -- that runs on the audio thread, where a file write can
  cause a dropout.
* LISTENING is set in `capture_loop`, not in `feed_chunk`, because `run_bench()`
  times `feed_chunk` for the wake-latency gate.

Measured: `set_state` costs 0.067ms p50 / 0.222ms p99. Live ack run showed
`idle -> talking` at t+0.183s and back to `idle` at t+0.948s, matching the
0.765s clip. 248 tests still pass.

External widget hook:

    # waybar custom module / tmux status / Plasma command plasmoid
    cat ~/.friday/status

## The assembled loop (2026-08-22)

`src/friday/loop.py` -- `VoiceLoop`. Wake -> STT -> route -> ack -> brain -> TTS
in one process. `core/app.py`'s `voice_loop` placeholder now runs it.

    set -a; . ~/.friday/env; set +a
    friday/.venv/bin/python -m friday.loop                 # live mic
    friday/.venv/bin/python -m friday.loop --text "..."    # skip wake+STT
    friday/.venv/bin/python -m friday.loop --text "..." --no-speak

### Live results

| check | result |
|---|---|
| state-query turn, real Aura out loud | `You're on feat/session-overlay-ids in MK-solutions (dirty).` |
| reasoning turn, real tools + Aura | answered, and refused to guess a tmux count it couldn't verify |
| closed loop (Aura -> real Deepgram -> route -> Aura) | transcript exact, spoken, 6.87s to task_complete |
| live mic, 30s + 10s | 0 phantom detections after the warmup fix |
| shutdown with a turn in flight | clean, 183ms |
| tests | 262 pass |

Closed-loop span: speech_ended_vad 415ms, stt_final 1116ms,
intent_classified 1233ms, first_content_audio 1935ms, task_complete 6869ms.

### Four bugs the assembly exposed

1. **The pump awaited the turn.** `handle_detection` was awaited inline in the
   capture loop, so frames stopped being pumped the moment a turn began --
   `feed_chunk` never forwarded audio to the live queue the turn was reading,
   and the turn waited forever for audio only the pump could deliver. The first
   real detection hung. Turns now run as a task. Regression test drives `_pump`
   directly with a blocked turn.
2. **Interim-only transcripts were discarded.** `run_utterance` closes at
   MAX_WAIT_MS past VAD speech-end, which can land before Deepgram's final:
   observed live with a complete interim (`what branch am i on`) and no final at
   all. Requiring `is_final` lost the turn silently -- heard, understood,
   ignored. Now falls back to the last interim.
3. **Wake-word phantom on every startup.** A freshly opened capture stream drops
   its second frame (chunk 1 came back at rms 4.7 between neighbours at 302 and
   362) and openWakeWord's ~1s feature window turns that discontinuity into a
   0.58-0.88 activation 0.72-1.12s later. Measured 3 phantom turns in 10s.
   Fixed with `begin_stream()` + `WARMUP_CHUNKS = 15`: the model is fed during
   warmup but its verdict ignored until the window clears. Deliberately armed
   per capture stream, NOT in the constructor, so file-fed callers (tests,
   `run_bench`, `--file`) stay exact and a wake word in the first 1.2s still fires.
4. **`end_handoff()` was unscoped.** A barge-in overlaps two turns; the old
   turn's release closed the *new* turn's live queue and starved it. Now takes
   the queue and no-ops for a stale caller.

### Barge-in

Saying the wake word during playback hard-preempts her. `MicGate`'s interrupt
phrases need a live STT stream and there is none while she speaks -- that would
be full duplex, which is not built.

### What acoustic loopback cannot verify

Playing a wake word out FRIDAY's speaker to test FRIDAY's mic does not work, and
not because of a defect: the echo canceller is referenced against that speaker,
so it correctly treats the test audio as FRIDAY's own voice and cancels it.
Measured decay across successive runs as the filter adapted: best "alexa" score
0.964 -> 0.211 -> 0.145, while a blocking read of the same mic before adaptation
scored 0.941. Bypassing the AEC to the raw mic did not fire either.

So one link is verified only by a human voice: **live mic -> wake word -> turn.**
Everything either side of it is verified (detector on real speech at the T2 gate;
30s of live capture with 0 phantoms; a real detection driving a complete turn
through live Deepgram/Anthropic/Aura). Say "alexa, what branch am I on".

## Audio Resource Manager (2026-08-22)

`src/friday/audio/` -- deliberately NOT inside `friday.voice`. The voice modules
consume audio; this package decides who may hold a device. Intended home for
mic ownership, speaker routing, echo cancellation, Bluetooth/headset switching,
call detection and listening state.

    python -m friday.audio.manager      # tiers + who is capturing right now

### Priority tiers (lower wins)

    P0  system / emergency capture
    P1  calls and meetings          zoom, teams, discord, slack, browsers
    P2  explicit recording          obs, audacity, ffmpeg, parecord
    P3  other interactive apps      <- unknown applications land here
    P4  FRIDAY

Unknown apps get P3 and still preempt: FRIDAY is least-privileged by design.
Browsers sit at P1, not P3, because a browser holding the mic is in practice a
call. Override via `~/.friday/mic-priority.json`, e.g. `{"obs": 1}`; setting an
app to 4 lets FRIDAY keep listening alongside it.

### State machine

    AVAILABLE -> FRIDAY_LISTENING -> PREEMPTING -> SUSPENDED -> AVAILABLE -> ...

PREEMPTING invalidates the in-flight turn, stops speech, closes the stream and
drops retained audio. Preemption is immediate: the higher-priority app never
waits for FRIDAY to finish, and FRIDAY never fights for or retries acquisition
-- she waits to be told the mic is free. The 5s re-check is read-only insurance
against a dead `pactl subscribe`, never an acquisition attempt.

SUSPENDED is genuinely deaf, not "listening locally but not uploading": the
capture stream is closed, so no wake-word inference runs and no STT stream
exists, and `on_forget` clears the wake-word pre-roll ring. A rolling 1.5s
buffer of the room kept while someone else is on a call is exactly what a voice
assistant must not do.

`manager.capture()` owns the stream and simply stops producing frames across a
suspension, so `_pump` needs no knowledge of ownership. FRIDAY's mic subsystem
is disposable; brain, memory, world state and coding agents run untouched.

### Two consequences worth stating

* **An interrupted utterance is discarded, never resumed.** "Friday, tell Codex
  to delete the old..." must not complete itself after a 20-minute call.
  Conversation context and memory survive; that turn does not, and it is not
  written to memory.
* **Preemption stops speech too.** If a call owns the microphone, anything
  FRIDAY says out loud is picked up and transmitted into that call, so being
  deaf is not enough -- she is silenced as well.

Failure mode is deliberately fail-OPEN: if `pactl` is missing or broken, FRIDAY
keeps listening. Being stranded permanently deaf is the worse outcome and much
harder to diagnose.

### Live results

| stage | state | indicator | pre-roll ring | streams |
|---|---|---|---|---|
| listening | `friday_listening` | `idle` | 1.52s | echo-cancel + friday |
| P2 recorder takes mic | `suspended` | `suspended` | **0.00s** | echo-cancel + pacat |
| recorder exits | `friday_listening` | `idle` | 1.52s | echo-cancel + friday |

`in use by pacat [P2_RECORDING]`. FRIDAY's own stream disappears from
`pactl list source-outputs` while suspended -- the device is released, not
merely ignored. 0 phantom detections after the reopen, because `on_forget`
re-arms wake-word warmup and every freshly opened stream needs it. 277 tests.

### Still in `voice/`, not moved

`voice/devices.py` (speaker/mic pinning, `PULSE_SINK`/`PULSE_SOURCE`) belongs
under `audio/` by this design but was left in place: moving it is churn with no
behaviour change, and the manager already calls it.

## Echo cancellation is loaded on demand (2026-08-22)

`src/friday/audio/echocancel.py`. The Audio Resource Manager loads
`module-echo-cancel` in `start()` and unloads it in `stop()`. The permanent
config was moved to `~/.friday/fallback/99-echo-cancel.conf`.

**Why, concretely.** Modules loaded from `pipewire.conf.d` load BEFORE the ALSA
devices exist, so they get lower node IDs and sort to the FRONT of the device
list -- every pre-existing capture device shifts down by two. Any application
that remembers its microphone by list index then records the wrong thing. This
happened for real: VoiceWin had `InputDeviceNumber: 1`, that position became
`echo-cancel-sink.monitor` (an output monitor, permanently silent), and it
reported "No audio captured" while the mic was perfectly healthy.

Loading at runtime inverts it -- higher IDs, appended at the END:

| | Mic1 index | echo-cancel-source | sources |
|---|---|---|---|
| friday not running | 5 | absent | 6 |
| friday running | **5** | 6 | 8 |
| after friday exits | 5 | absent | 6 |

Verified live: `device list restored, no renumbering`.

Master pinning is carried over from the config and asserted in tests
(`source_master=...Mic1`, `sink_master=...Speaker`). An unpinned AEC follows the
DEFAULT sink and silently cancels against HDMI, where there is no acoustic loop.

Ownership is tracked: an already-present `echo-cancel-source` (leftover config,
or a second FRIDAY) is used but never unloaded. A module that loads yet never
produces its node is unloaded, so restarts cannot leak half-built modules.
Failure is soft -- no AEC beats no assistant, and `devices.resolve()` already
reports the raw-mic path as degraded.

VoiceWin is now on index 5 (raw Mic1), stable whether FRIDAY runs or not. It
still stores an index rather than a device name, so plugging in a monitor (a new
sink adds a monitor source ahead of Mic1) will still shift it -- but FRIDAY no
longer causes that.

## Test hang, fixed (2026-08-22)

`tests/test_attention.py::test_dev_server_death_preempts_speech_and_names_the_process`
ran for five hours and reported nothing. Two separate faults:

1. **Unbounded `subprocess.Popen.wait()`** in `_dead_dev_server_event`. Other
   tests in this suite drive children through asyncio, whose child watcher
   installs a SIGCHLD handler; a Popen child reaped out from under a blocking
   `waitpid()` has nothing left to wake it. Now `wait(timeout=30)` with a kill
   fallback -- a child running `pass` that hasn't exited in 30s is a broken
   environment, not something to keep waiting on.
2. **My own debug wrapper made it un-killable.** I had invoked pytest under
   `timeout 20 python -X faulthandler -c "faulthandler.register(SIGTERM); ..."`.
   `faulthandler.register` defaults to `exit=False`, so SIGTERM dumped a
   traceback and *continued*. `timeout` fired and did nothing. Use
   `register(..., exit=True)`, or `timeout -s KILL`.

### Suite-wide watchdog

`tests/conftest.py` -- autouse SIGALRM, 60s default, `FRIDAY_TEST_TIMEOUT`
overrides (0 disables). SIGALRM rather than a plugin: no new dependency, and it
fires even when the main thread is blocked in C (`waitpid`, `read`, PortAudio),
which is exactly where the hangs that matter happen. A thread-based timer cannot
interrupt those. The slowest legitimate test here is 8.35s, so 60s catches hangs
without policing performance.

Verified: a test blocked in `waitpid` fails in 3.07s instead of hanging, and
`tests/test_watchdog_selfcheck.py` keeps that guarantee honest by running pytest
in a bounded child on a deliberately hanging test.

The original test now passes 8/8 under three spinning CPU hogs, in 0.21-0.42s.
285 tests.

## Gateway harness

Before this, "running FRIDAY" meant a foreground process on a tty, and the only
way to ask her anything was to speak. Three ordinary things were impossible:
checking liveness without watching a terminal, driving a turn from a script,
and running under a supervisor while still being able to observe her.

The gateway is one WebSocket control plane in front of the voice loop, modelled
on openclaw's: explicit `connect` handshake, protocol negotiated as a *range*
so client and daemon upgrade independently, token auth with a per-peer failure
brake, and unsolicited events pushed to connected clients.

    python -m friday                      # daemon: voice + agent + gateway
    python -m friday.gateway --smoke      # staged health check
    python -m friday.gateway --method state

The voice loop is a **sibling** of the gateway, not a child. A client storm
cannot stall a turn, and a crashed gateway does not take the microphone down
with it. `health` reports `ok` and `voice_loop` as separate facts, because the
gateway being fine while the loop is down is exactly what a missing credential
looks like — collapsing them into one boolean makes it undiagnosable.

### Methods

| method | purpose |
|---|---|
| `connect` | handshake; negotiates protocol, authenticates |
| `health` | uptime, state, `voice_loop`, turn and invalidation counts |
| `state` | indicator state, mic state, who holds the mic |
| `mic.owners` | live priority-tier listing of mic owners |
| `spans.recent` | recent turn records (bounded 1..200) |
| `say` | speak text; refused while a higher-priority app holds the mic |
| `ask` | run a full turn from text — routing, brain and TTS, no microphone |

`ask` is what makes the loop testable end to end without a human in the room.

### Staged exit codes

The smoke harness fails with a distinct code per stage, in dependency order, so
the *first* non-zero code is always the root cause rather than a symptom:

    0 pass   2 socket unreachable   3 connect rejected
    4 health failed   5 state failed   6 ask failed

### Security

Binds `127.0.0.1` only. This process holds Deepgram and Anthropic keys and can
run tools, so there is no anonymous mode even on loopback — a default-open
localhost port is reachable from every process on the box, including a browser
tab via DNS rebinding. The token is 256 bits in a 0600 file rather than an
environment variable, because env vars leak through `/proc/<pid>/environ`,
which FRIDAY's own `read_file` tool already refuses for this reason. Comparison
is constant-time and wrong guesses are throttled to 5 per peer per minute.

### It immediately found a real bug

The first thing `health` reported was `turns=3` after twelve seconds of
silence. Phantom wake-word fires — the startup bug I had already "fixed" — were
still happening, invisibly.

Cause: `begin_stream()` arms the wake warmup, and it was only ever called from
the preemption path. The **first** capture stream of the process was never
armed — and that is precisely the stream that drops its second frame, which
openWakeWord scores as a wake word about a second later. So FRIDAY answered a
phantom on every startup; the earlier fix only covered streams reopened after
a preemption.

Fixed with an `on_open` callback on the audio manager, symmetric with
`on_forget`. Measured after: **0 turns across 30s of silence**, where it had
been 3–4. Regression-tested in `test_audio_manager.py`.

That is the argument for the gateway in one paragraph: the bug was always
there, and nothing could see it.

### Running under systemd

`deploy/friday.service` is a **user** unit — the mic and speaker are reached
through this user's PipeWire session, and a system unit would need an explicit
seat handoff to touch audio at all.

    mkdir -p ~/.config/systemd/user
    cp deploy/friday.service ~/.config/systemd/user/
    systemctl --user daemon-reload
    systemctl --user enable --now friday

Details worth knowing:

* `ExecStartPost` runs the smoke test, so systemd only calls the start
  successful once she actually answers. A process that booted but cannot serve
  `health` is not "running" in any sense the user cares about.
* `KillSignal=SIGINT`, because the app's handler unwinds the capture stream and
  releases the mic. A hard kill can leak the echo-cancel module, and a leaked
  virtual source renumbers everyone's device list — the bug that broke Chrome.
* `RequiresMountsFor` the NTFS volume: the project is not under `$HOME`, so
  without the mount the interpreter does not exist and systemd retries until
  the crash-loop brake trips.
* Credentials come from a 0600 `EnvironmentFile`, not `Environment=` lines —
  the latter are readable via `systemctl show`.

## The `friday` command

    friday start | stop | restart | status | logs
    friday ask "what branch am i on"      # one turn, no microphone
    friday say "the build is green"
    friday doctor                          # check what actually breaks
    friday install [--now] | uninstall     # systemd user unit
    friday token [--rotate]

Installed as a console entry point (`[project.scripts]`), so it works from
anywhere once the venv is on `PATH`.

### Two design decisions worth knowing

**It detects who owns the lifecycle.** The systemd unit sets `Restart=always`.
If `friday stop` signalled the process directly while systemd owned it, systemd
would resurrect it in three seconds and the command would look broken. So every
lifecycle operation asks first, and delegates to `systemctl --user` when the
unit is installed — even if it is currently stopped, because an installed unit
means the user's intent is systemd, and starting a competing bare process would
give them two daemons fighting over one microphone with only one visible to
`systemctl status`.

**"Running" means answering, not existing.** Liveness is the gateway answering
`health`; the PID is only a fallback for when the gateway is disabled. The
interesting failure — wedged, holding the mic, answering nothing — looks
perfectly alive to a PID check. `status` reports `running` and `healthy`
separately for that reason, and `start` distinguishes "failed to start" (exit 1)
from "started but not healthy" (exit 3), because the remedy differs: one wants a
retry, the other wants the log.

Exit codes are scriptable: `0` fine, `1` failed, `2` not running, `3` up but
unhealthy.

### PID recycling

Every PID read is verified against `/proc/<pid>/cmdline` before anything is
signalled. A pidfile surviving a reboot can name an unrelated process, and
`kill` does not ask whether you meant it.

### Shutdown is the dangerous part

`stop` sends **SIGINT first, always** — the app's handler unwinds the capture
stream and releases the microphone. Escalation is SIGINT → SIGTERM → SIGKILL.

If it ever reaches a hard kill, the daemon never ran its own cleanup, so the CLI
runs it: `reap_echo_cancel()` unloads any leftover `module-echo-cancel`. This is
not tidiness. A leaked echo-cancel source inserts itself at the front of every
application's capture device list, and the symptom appears hours later in an
unrelated app with no visible connection to FRIDAY — that is exactly how Chrome
and VoiceWin silently lost their microphones for nine days. Telling the user to
go unload a module by hand is not a fix; the crash path repairs itself while the
cause is still obvious.

Related hardening: the gateway's shutdown wait is now **bounded at 2s**.
`wait_closed()` waits for in-flight connection handlers, which made shutdown
depend on a remote peer's cooperation — a client that died mid-close-handshake
could hold it open indefinitely, delaying the mic release and the module unload
behind it. Shutdown must never be blockable from the network.

### `friday doctor`

Every check corresponds to a failure this project has really had, ordered by
dependency so the first ✗ is the one to fix: credentials present and 0600, the
default source being a real input rather than a monitor, the source unmuted, no
leaked echo-cancel module, token file permissions, daemon and gateway health,
voice loop up, and which supervisor is in charge.

The echo-cancel check is state-dependent: FRIDAY loads that module on demand
herself, so it is only a *leak* when she is not running. Flagging her own
working module would train the user to ignore the line, which is worse than not
checking at all.

### Installing it — three bugs the first real install found

`friday install --now` is what you run. Getting there exposed three failures
that only appear outside a developer shell:

**1. The command was not on PATH.** The entry point lands in `.venv/bin/`, so it
only worked because I had the venv on `PATH`. Symlinked to `~/.local/bin/friday`,
which is already on the login PATH.

**2. Credentials never arrived — twice, for two different reasons.** Started
from a fresh terminal, the daemon inherited a shell that had never sourced
`~/.friday/env`, so it came up with `voice_loop=false`. Fixed by having the app
load the file itself at startup, so `friday start`, `--foreground`,
`systemctl start` and a bare `python -m friday` all behave identically.

The unit's `EnvironmentFile=` would *not* have covered this. That directive
cannot parse `export KEY=value`, which is the format the file uses — systemd
reads the variable name as `"export KEY"` and drops it silently. Measured
directly: a file containing `export FOO=bar` yields an empty `$FOO`. So the
directive was removed and the comment explaining why was left in its place.

**3. The readiness check raced the thing it was checking.** Under
`Type=simple`, systemd runs `ExecStartPost` the instant `ExecStart` forks —
seconds before the gateway binds. Every start failed its own smoke test, and
`Restart=always` turned that into a crash loop. `smoke --wait 60` now polls for
the socket. A readiness check that races its subject is worse than none.

Two smaller ones from the same session:

*   The daemon's working directory is now **pinned to the project root**,
    matching the unit. State queries are answered relative to it, so inheriting
    the launching shell's directory made "what branch am I on" depend on where
    you were standing — and starting from `$HOME` put her outside any repo.
*   `install` **stops a direct-mode daemon first**. Installing the unit changes
    who owns the lifecycle, so an already-running bare daemon would go
    invisible to every later `friday` command while still holding the
    microphone, and systemd would start a second one that could not open it.
*   `smoke` now **fails on an empty reply**. A turn that routes, reports no
    error and says nothing is the worst thing to call green: the user asked and
    FRIDAY silently ignored them. This is how bug 4 above was caught — smoke had
    cheerfully printed "all stages passed" with `reply=''`.
