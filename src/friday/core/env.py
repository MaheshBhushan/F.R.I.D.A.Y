"""Credentials: the `~/.friday/env` file and the keys it provides."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from friday.core.paths import ENV_PATH


def load_env_file(path: Path | None = None) -> dict[str, str]:
    """Parse `~/.friday/env` into a dict. Missing file is not an error.

    Parsed in Python rather than sourced through a shell. Sourcing would handle
    exotic syntax, but it also executes whatever is in the file -- and this file
    exists specifically to hold API keys, so it is the last thing that should
    gain the ability to run commands as a side effect of starting the daemon.

    Understands the format the file actually uses: `export KEY=value`, optional
    surrounding quotes, `#` comments, blank lines. Anything it does not
    understand is skipped rather than guessed at.
    """
    target = path or ENV_PATH
    try:
        raw = target.read_text()
    except OSError:
        return {}

    values: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load_credentials() -> None:
    """Load ~/.friday/env into the environment if it is not already there.

    Done in the daemon's own startup, so every way of launching it behaves
    identically: `friday start`, `friday start --foreground`,
    `systemctl --user start friday`, and a bare `python -m friday`.

    This is not belt-and-braces with the unit's `EnvironmentFile=`. That
    directive does *not* understand `export KEY=value`, which is the format
    this file uses -- systemd parses the variable name as "export KEY" and
    silently drops it. Measured: a file containing `export FOO=bar` yields an
    empty $FOO. So under systemd the credentials never arrived at all, and the
    only symptom was `voice_loop=false` with no error anywhere.
    """
    for key, value in load_env_file().items():
        os.environ.setdefault(key, value)


def require_key(name: str, hint: str = "") -> str:
    """Return `$name`, or exit 1 with a message naming the missing variable.

    For the per-module `python -m friday.X` entry points, where a missing key
    is a usage error, not something to recover from.
    """
    key = os.environ.get(name)
    if not key:
        print(
            f"error: {name} is not set. {hint or f'Export {name}=<your key> and retry.'}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return key
