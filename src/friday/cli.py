"""The `friday` command.

    friday start | stop | restart | status | logs
    friday ask "what branch am i on"
    friday say "the build is green"
    friday doctor
    friday install | uninstall
    friday token

Design notes:

*   Every subcommand exits non-zero on failure, with distinct codes where the
    distinction is useful. This is meant to be scriptable, not just readable.
*   `status` is the only command that is safe to run in a loop, so it is the
    only one that does no work beyond asking.
*   `doctor` exists because this project's real failures were never in the
    Python -- they were a muted device, a leaked PipeWire module renumbering
    everyone's device list, an unexported key, a port already bound. Those are
    invisible from a traceback and obvious from a checklist.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from friday import daemon

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def _colour(enabled: bool):
    if enabled:
        return GREEN, RED, YELLOW, DIM, RESET
    return "", "", "", "", ""


def _tty() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _mark(good: Optional[bool]) -> str:
    g, r, y, _, x = _colour(_tty())
    if good is None:
        return f"{y}·{x}"
    return f"{g}✓{x}" if good else f"{r}✗{x}"


# --------------------------------------------------------------- lifecycle

def cmd_start(args: argparse.Namespace) -> int:
    if args.foreground:
        # Run in-process: no detach, no pidfile, Ctrl-C stops it. This is the
        # form to use when debugging, because the log goes to the terminal
        # instead of into a file you then have to go and read.
        from friday.core.app import main as app_main
        return app_main([])

    existing = daemon.status()
    if existing.running:
        print(f"{_mark(True)} already running "
              f"(pid {existing.pid}, {existing.supervisor.value})")
        # Attach anyway. "already running" plus a silent prompt is a dead end;
        # what the user wanted was to watch her, and she is watchable.
        return _attach() if not getattr(args, "detach", False) else 0

    print(f"{DIM if _tty() else ''}starting…{RESET if _tty() else ''}", flush=True)
    result = daemon.start(timeout=args.timeout)
    if result.healthy:
        info = result.health or {}
        print(f"{_mark(True)} friday is up (pid {result.pid}, "
              f"{result.supervisor.value}) — state={info.get('state', '?')}")
        # Attach after the health check, not instead of it: following a log
        # that never appears because startup failed is the least useful thing
        # this command could do.
        return _attach() if not getattr(args, "detach", False) else 0
    if result.running:
        # Up but not answering. Distinguished from "failed to start" because
        # the remedy is different: this one wants the log, not a retry.
        print(f"{_mark(None)} started but not healthy: {result.detail}")
        print(f"  logs: friday logs -n 40")
        return 3
    print(f"{_mark(False)} failed to start: {result.detail}")
    print(f"  logs: friday logs -n 40")
    return 1


def cmd_stop(args: argparse.Namespace) -> int:
    ok, note = daemon.stop(timeout=args.timeout)
    print(f"{_mark(ok)} {note}")
    return 0 if ok else 1


def cmd_restart(args: argparse.Namespace) -> int:
    result = daemon.restart(timeout=args.timeout)
    if result.healthy:
        print(f"{_mark(True)} restarted (pid {result.pid})")
        return _attach() if not getattr(args, "detach", False) else 0
    print(f"{_mark(False)} restart failed: {result.detail}")
    return 1


def cmd_status(args: argparse.Namespace) -> int:
    st = daemon.status()
    if args.json:
        print(json.dumps({
            "running": st.running, "healthy": st.healthy, "pid": st.pid,
            "supervisor": st.supervisor.value, "detail": st.detail,
            "health": st.health,
        }, indent=2))
        return 0 if st.healthy else 1

    print(f"{_mark(st.healthy or None if st.running else False)} "
          f"friday: {st.detail}"
          + (f" (pid {st.pid})" if st.pid else ""))
    print(f"  supervisor  {st.supervisor.value}")
    info = st.health
    if not info:
        if st.running:
            print("  gateway     not answering")
        return 0 if st.healthy else 1

    print(f"  state       {info.get('state', '?')}")
    print(f"  uptime      {info.get('uptime_s', 0):.0f}s")
    print(f"  voice loop  {'up' if info.get('voice_loop') else 'DOWN'}")
    print(f"  turns       {info.get('turns', 0)}"
          + (f" ({info['invalidated']} preempted)" if info.get("invalidated") else ""))
    print(f"  clients     {info.get('clients', 0)}")

    # Mic ownership is the thing most likely to explain "why isn't she
    # answering", so it is worth the extra round trip.
    mic = _call("state")
    if mic:
        detail = mic.get("mic_detail") or ""
        print(f"  mic         {mic.get('mic', '?')}"
              + (f" — {detail}" if detail and detail != "microphone free" else ""))
    return 0 if st.healthy else 1


def _log_command(*, follow: bool, lines: int) -> Optional[list]:
    """The command that emits the active supervisor's log, or None if there
    is no log to read yet."""
    if daemon.supervisor() is daemon.Supervisor.SYSTEMD:
        cmd = ["journalctl", "--user", "-u", daemon.UNIT_NAME,
               "-n", str(lines), "-o", "cat"]
        return [*cmd, "-f"] if follow else cmd

    if not daemon.LOG_PATH.exists():
        return None
    cmd = ["tail", "-n", str(lines)]
    if follow:
        cmd.append("-f")
    return [*cmd, str(daemon.LOG_PATH)]


def _tail(*, follow: bool, lines: int) -> int:
    """Stream the log to this terminal, coloured.

    Piped through the colouriser rather than exec'd straight at the terminal
    so the stored form stays plain: the daemon writes into the journal, and
    ANSI escapes committed there would follow the line through every later
    `grep` and `journalctl --grep`.
    """
    from friday.core import logfmt

    cmd = _log_command(follow=follow, lines=lines)
    if cmd is None:
        print(f"no log yet at {daemon.LOG_PATH}", file=sys.stderr)
        return 1

    color = logfmt.enabled()
    if not color:
        # Not a terminal (piped into a file, less, grep). Hand the child our
        # stdout directly: no reformatting, and no Python process sitting in
        # the middle of a pipeline it adds nothing to.
        return subprocess.call(cmd)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True,
                            errors="replace", bufsize=1)
    try:
        logfmt.stream_lines(proc.stdout, color=True)
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 0
    finally:
        if proc.stdout is not None:
            proc.stdout.close()


def _attach(lines: int = 12) -> int:
    """Stream the live log until Ctrl-C, which detaches without stopping her.

    A few lines of history rather than none: by the time the health check has
    passed, the interesting part of startup (which wake models loaded, whether
    the gateway bound, whether the voice loop came up) has already scrolled
    past into the journal.
    """
    print(f"{DIM if _tty() else ''}— streaming; Ctrl-C detaches "
          f"(she keeps running){RESET if _tty() else ''}")
    try:
        return _tail(follow=True, lines=lines)
    except KeyboardInterrupt:
        return 0


def cmd_logs(args: argparse.Namespace) -> int:
    try:
        return _tail(follow=args.follow, lines=args.lines)
    except KeyboardInterrupt:
        return 0


def cmd_reap(args: argparse.Namespace) -> int:
    """Unload a leftover echo-cancel module. Used by the unit's ExecStopPost.

    Exists as a subcommand rather than an inline `python -c` in the unit
    because systemd's parser does not survive the nested quoting that would
    need, and a mis-parsed ExecStopPost fails silently -- leaving exactly the
    leak this is here to prevent.
    """
    freed = daemon.reap_echo_cancel()
    if freed:
        print(f"{_mark(True)} freed {freed} leaked echo-cancel module"
              f"{'s' if freed != 1 else ''}")
    elif not getattr(args, "quiet", False):
        print(f"{_mark(True)} nothing to reap")
    return 0


def cmd_hear(args: argparse.Namespace) -> int:
    """Live mic probe. Runs in this process, not the daemon's."""
    from friday.voice import wake

    # The probe prints its own report; the event log would interleave
    # wake-init and per-frame lines into the middle of the table.
    os.environ.setdefault("FRIDAY_LOG", "silent")
    threshold = args.threshold if args.threshold is not None else wake.DEFAULT_THRESHOLD
    try:
        models = tuple(args.model) if getattr(args, "model", None) else None
        return asyncio.run(wake.probe(seconds=args.seconds,
                                      threshold=threshold, models=models))
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"{_mark(False)} probe failed: {exc}", file=sys.stderr)
        return 1


# ------------------------------------------------------------- interaction

def _call(method: str, params: Optional[dict] = None,
          timeout: float = 120.0) -> Optional[dict]:
    """One gateway call. None on any failure -- callers report their own errors."""
    from friday.gateway.client import GatewayClient

    async def _run() -> Optional[dict]:
        client = GatewayClient()
        try:
            await asyncio.wait_for(client.open(), timeout=5.0)
            reply = await client.connect(name="friday-cli")
            if not reply.get("ok"):
                return None
            reply = await client.request(method, params or {}, timeout=timeout)
            return reply.get("result") if reply.get("ok") else \
                {"__error__": reply.get("error", {})}
        except Exception:  # noqa: BLE001
            return None
        finally:
            await client.close()

    try:
        return asyncio.run(_run())
    except Exception:  # noqa: BLE001
        return None


def _require_running() -> bool:
    if daemon.status().running:
        return True
    print(f"{_mark(False)} friday is not running — start her with: friday start",
          file=sys.stderr)
    return False


def cmd_ask(args: argparse.Namespace) -> int:
    if not _require_running():
        return 2
    text = " ".join(args.text)
    result = _call("ask", {"text": text, "speak": args.speak})
    if result is None:
        print(f"{_mark(False)} no answer from the gateway", file=sys.stderr)
        return 1
    if "__error__" in result:
        print(f"{_mark(False)} {result['__error__'].get('message')}", file=sys.stderr)
        return 1
    if result.get("error"):
        print(f"{_mark(False)} {result['error']}", file=sys.stderr)
        return 1
    tier = result.get("tier") or "?"
    print(f"{DIM if _tty() else ''}[{tier}]{RESET if _tty() else ''} "
          f"{result.get('reply', '')}")
    return 0


def cmd_say(args: argparse.Namespace) -> int:
    if not _require_running():
        return 2
    text = " ".join(args.text)
    result = _call("say", {"text": text})
    if result is None:
        print(f"{_mark(False)} no answer from the gateway", file=sys.stderr)
        return 1
    if "__error__" in result:
        # The common case here is a refusal because a call owns the mic, which
        # is correct behaviour and deserves a plain explanation, not a stack.
        print(f"{_mark(False)} {result['__error__'].get('message')}", file=sys.stderr)
        return 1
    print(f"{_mark(True)} said: {text}")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    from friday.gateway.client import smoke
    return asyncio.run(smoke(args.url, None, ask=args.ask, wait=args.wait))


# ------------------------------------------------------------------ doctor

def cmd_doctor(args: argparse.Namespace) -> int:
    """Check the things that actually break, in the order they break.

    Every item here corresponds to a failure this project has really had. The
    checks are ordered by dependency, so the first ✗ is the one to fix.
    """
    problems = 0
    # Needed before the audio checks: whether a loaded echo-cancel module is
    # correct or a leak depends entirely on whether FRIDAY is running.
    st = daemon.status()

    def check(label: str, good: Optional[bool], note: str = "") -> None:
        nonlocal problems
        if good is False:
            problems += 1
        print(f"  {_mark(good)} {label}" + (f"  {DIM if _tty() else ''}{note}"
                                            f"{RESET if _tty() else ''}" if note else ""))

    print("credentials")
    env_file = Path.home() / ".friday" / "env"
    check("~/.friday/env exists", env_file.exists(), str(env_file))
    if env_file.exists():
        mode = env_file.stat().st_mode & 0o777
        check("~/.friday/env is 0600", mode == 0o600, f"mode {mode:o}")
    for key in ("DEEPGRAM_API_KEY", "ANTHROPIC_API_KEY"):
        present = bool(os.environ.get(key))
        check(f"{key} in this shell", present or None,
              "" if present else "not exported here (the daemon reads the env file)")

    print("audio")
    default_source = _run_ok(["pactl", "get-default-source"])
    check("pipewire answers", default_source is not None)
    if default_source:
        check("default source is a real input",
              "monitor" not in default_source,
              default_source.split(".")[-1] if default_source else "")
        muted = _run_ok(["pactl", "get-source-mute", default_source])
        check("default source unmuted", muted == "Mute: no", muted or "")
    # Counted in Python, not with `grep -c`: grep exits 1 when the count is
    # zero, so _run_ok() returned None for the healthy case and the check
    # reported a leak precisely when there wasn't one.
    modules = _run_ok(["pactl", "list", "short", "modules"])
    loaded = modules is not None and "module-echo-cancel" in modules
    # A lingering echo-cancel module renumbers every application's device list,
    # which is what silently broke Chrome and VoiceWin for nine days. But
    # FRIDAY loads it on demand herself, so it is only a *leak* when she is not
    # running -- flagging her own working module would train the user to ignore
    # this line, which is worse than not checking at all.
    if st.running:
        check("echo-cancel module", None if loaded else True,
              "loaded on demand by friday (expected)" if loaded
              else "not currently needed")
    else:
        check("no leaked echo-cancel module", not loaded,
              "" if not loaded
              else "friday is stopped, so this is a leak — unload it: "
                   "pactl unload-module module-echo-cancel")

    print("gateway")
    token = Path(os.environ.get("FRIDAY_GATEWAY_TOKEN_FILE",
                                Path.home() / ".friday" / "gateway-token"))
    if token.exists():
        mode = token.stat().st_mode & 0o777
        check("token file is 0600", mode == 0o600, f"mode {mode:o}")
    else:
        check("token file", None, "not created yet (minted on first start)")
    check("daemon running", st.running or None, st.detail)
    if st.running:
        check("gateway healthy", st.healthy,
              "" if st.healthy else "up but not answering; see: friday logs")
        if st.health:
            check("voice loop up", bool(st.health.get("voice_loop")),
                  "" if st.health.get("voice_loop") else
                  "usually a missing credential; see: friday logs")

    print("supervisor")
    who = daemon.supervisor()
    check(f"managed by {who.value}", None,
          "install the unit with: friday install"
          if who is daemon.Supervisor.DIRECT else daemon.UNIT_NAME)

    print()
    if problems:
        print(f"{_mark(False)} {problems} problem(s) found")
        return 1
    print(f"{_mark(True)} no problems found")
    return 0


def _run_ok(cmd: list[str]) -> Optional[str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=5.0, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


# ----------------------------------------------------------------- install

def cmd_install(args: argparse.Namespace) -> int:
    source = Path(__file__).resolve().parent.parent.parent / "deploy" / "friday.service"
    if not source.exists():
        print(f"{_mark(False)} unit template not found at {source}", file=sys.stderr)
        return 1
    # Hand over cleanly. Installing the unit changes who owns the lifecycle, so
    # a daemon started in direct mode would become invisible to every later
    # `friday` command while still holding the microphone -- and systemd would
    # then start a second one that could not open it.
    before = daemon.status()
    if before.running and before.supervisor is daemon.Supervisor.DIRECT:
        ok, note = daemon.stop()
        print(f"{_mark(ok)} stopped the direct-mode daemon first ({note})")
        if not ok:
            print("  refusing to install while it is still running",
                  file=sys.stderr)
            return 1

    target_dir = Path.home() / ".config" / "systemd" / "user"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / daemon.UNIT_NAME
    target.write_text(source.read_text())
    print(f"{_mark(True)} installed {target}")

    if subprocess.call(["systemctl", "--user", "daemon-reload"]) != 0:
        return 1
    if args.now:
        # enable --now both starts it and makes it survive a reboot. Doing one
        # without the other is the classic "it worked until I rebooted".
        code = subprocess.call(["systemctl", "--user", "enable", "--now",
                                daemon.UNIT_NAME])
        if code != 0:
            print(f"{_mark(False)} enable failed; try: friday logs", file=sys.stderr)
            return code
        print(f"{_mark(True)} enabled and started")
    else:
        print(f"  start it with: systemctl --user enable --now {daemon.UNIT_NAME}")
    return 0


def cmd_uninstall(args: argparse.Namespace) -> int:
    subprocess.call(["systemctl", "--user", "disable", "--now", daemon.UNIT_NAME])
    target = Path.home() / ".config" / "systemd" / "user" / daemon.UNIT_NAME
    if target.exists():
        target.unlink()
        print(f"{_mark(True)} removed {target}")
    subprocess.call(["systemctl", "--user", "daemon-reload"])
    return 0


def cmd_token(args: argparse.Namespace) -> int:
    from friday.gateway import auth as gauth

    if args.rotate:
        # Rotation invalidates every connected client, so say so rather than
        # letting the user discover it when their phone stops working.
        path = gauth.TOKEN_PATH
        if path.exists():
            path.unlink()
        token = gauth.load_or_create_token()
        print(f"{_mark(True)} rotated; restart friday for it to take effect")
        print(token)
        return 0
    print(gauth.load_or_create_token())
    return 0


# -------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="friday",
        description="FRIDAY voice assistant.",
        epilog="Run `friday doctor` first when something is wrong.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("start", help="start the daemon")
    p.add_argument("--foreground", "-f", action="store_true",
                   help="run in this terminal instead of detaching")
    p.add_argument("--timeout", type=float, default=daemon.START_TIMEOUT_S)
    # Streaming is the default. Starting a voice assistant and being handed
    # back a silent prompt tells you nothing about whether she can hear you;
    # the log is the only feedback there is. Ctrl-C detaches, it does not stop.
    p.add_argument("--detach", "-d", action="store_true",
                   help="start and return to the prompt instead of streaming")
    p.set_defaults(func=cmd_start)

    p = sub.add_parser("stop", help="stop the daemon")
    p.add_argument("--timeout", type=float, default=daemon.STOP_TIMEOUT_S)
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("restart", help="stop then start")
    p.add_argument("--timeout", type=float, default=daemon.START_TIMEOUT_S)
    p.add_argument("--detach", "-d", action="store_true",
                   help="return to the prompt instead of streaming")
    p.set_defaults(func=cmd_restart)

    p = sub.add_parser("status", help="is she up, and is she well?")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("logs", help="tail the daemon log")
    p.add_argument("-n", "--lines", type=int, default=40)
    p.add_argument("-f", "--follow", action="store_true")
    p.set_defaults(func=cmd_logs)

    p = sub.add_parser("ask", help="run one turn from text")
    p.add_argument("text", nargs="+")
    p.add_argument("--speak", action="store_true", help="also say it out loud")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("say", help="speak text out loud")
    p.add_argument("text", nargs="+")
    p.set_defaults(func=cmd_say)

    p = sub.add_parser("smoke", help="staged health check")
    p.add_argument("--url", default=os.environ.get("FRIDAY_GATEWAY_URL",
                                                   "ws://127.0.0.1:8765"))
    p.add_argument("--ask", metavar="TEXT")
    p.add_argument("--wait", type=float, default=0.0, metavar="SECONDS",
                   help="poll for the gateway to come up first")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("doctor", help="check the things that actually break")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("install", help="install the systemd user unit")
    p.add_argument("--now", action="store_true", help="also enable and start it")
    p.set_defaults(func=cmd_install)

    p = sub.add_parser("uninstall", help="remove the systemd user unit")
    p.set_defaults(func=cmd_uninstall)

    p = sub.add_parser("reap", help="unload a leaked echo-cancel module")
    p.add_argument("--quiet", "-q", action="store_true")
    p.set_defaults(func=cmd_reap)

    p = sub.add_parser("hear", help="live mic probe: input level + wake score")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--model", action="append", metavar="NAME_OR_PATH",
                   help="wake model to test (repeatable); a path works too, so "
                        "a freshly trained model can be tried before install")
    p.set_defaults(func=cmd_hear)

    p = sub.add_parser("token", help="print the gateway token")
    p.add_argument("--rotate", action="store_true")
    p.set_defaults(func=cmd_token)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
