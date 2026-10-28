"""Where FRIDAY keeps its state on disk.

One definition, so every file under `~/.friday` moves together when
`FRIDAY_HOME` points somewhere else (a scratch home for a test run, a second
checkout that must not share the first one's pidfile or memory).
"""

from __future__ import annotations

import os
from pathlib import Path

FRIDAY_HOME = Path(os.environ.get("FRIDAY_HOME", Path.home() / ".friday"))
ENV_PATH = FRIDAY_HOME / "env"

# The checkout root: .../friday/src/friday/core/paths.py -> .../friday
PROJECT_ROOT = Path(__file__).resolve().parents[3]
