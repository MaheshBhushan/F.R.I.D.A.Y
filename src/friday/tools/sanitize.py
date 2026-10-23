"""Argument sanitisation for tool calls.

The LLM's tool arguments are untrusted input: the model may be steered by
whatever it just read out of a log file. Two defences live here, both
fail-closed by raising `PermissionDenied`:

  - `safe_path()`   -- resolves symlinks, confines reads to an allowlist of
                       roots, and refuses credential-shaped files outright.
  - `safe_argv()`   -- allowlists the command, refuses every shell
                       metacharacter, and returns an argv list that is only
                       ever executed with shell=False.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

from friday.permissions import PermissionDenied

# Reads are confined to these roots (after full symlink resolution). The home
# directory is NOT a root: only the user-content folders below are, so config
# and credential stores (~/.config, ~/.local, ~/.mozilla, ...) stay out of reach.
# FRIDAY_READ_ROOTS (colon-separated) adds project/workspace directories.
HOME_READ_DIRS = (
    "Documents", "Downloads", "Desktop", "notes", "Notes",
    "projects", "Projects", "code", "src", "workspace",
)


def _read_roots() -> tuple[Path, ...]:
    extra = [
        Path(os.path.expanduser(p)).resolve()
        for p in os.environ.get("FRIDAY_READ_ROOTS", "").split(":")
        if p.strip()
    ]
    return (
        *((Path.home() / d).resolve() for d in HOME_READ_DIRS),
        *extra,
        Path("/var/log"),
        Path("/proc"),
        Path("/sys"),
        Path("/tmp"),
    )


ALLOWED_READ_ROOTS: tuple[Path, ...] = _read_roots()

# Roots that are never readable even though they sit inside an allowed root.
DENIED_READ_ROOTS: tuple[Path, ...] = (
    Path.home() / ".ssh",
    Path.home() / ".gnupg",
    Path.home() / ".aws",
    Path.home() / ".config" / "gh",
    Path.home() / ".kube",
)

# Credential-shaped basenames/suffixes, matched on the resolved basename.
DENIED_NAME_RE = re.compile(
    r"""(?ix)
    ^\.env($|\.)                # .env, .env.local, .env.production
    | ^\.netrc$
    | ^\.pgpass$
    | ^\.git-credentials$
    | ^id_[a-z0-9]+(\.pub)?$    # id_rsa, id_ed25519
    | ^known_hosts$
    | ^(environ|cmdline)$        # /proc/<pid>/environ leaks exported API keys
    | \.(pem|key|p12|pfx|keystore|jks)$
    | (secret|password|passwd|credential|token|apikey|api_key)
    """
)

# Read-only commands, and the subcommands allowed for the ones that have
# write-capable modes. An empty frozenset means "any arguments allowed".
ALLOWED_COMMANDS: dict[str, frozenset[str]] = {
    "ls": frozenset(),
    "ps": frozenset(),
    "df": frozenset(),
    "free": frozenset(),
    "uptime": frozenset(),
    "uname": frozenset(),
    "whoami": frozenset(),
    "date": frozenset(),
    "pgrep": frozenset(),
    "ss": frozenset(),
    "git": frozenset({"status", "log", "diff", "branch", "show", "remote"}),
    "systemctl": frozenset({"status", "is-active", "is-enabled", "list-units"}),
    "journalctl": frozenset(),
    "tmux": frozenset({
        "has-session", "list-clients", "list-panes", "list-sessions",
        "list-windows", "show-options", "show-window-options",
    }),
}

# Any occurrence of these in a raw command string means we refuse: they are
# how a "read-only" command becomes a write (>, >>), a second command (;, &&,
# |, newline), or a substitution ($(), ``, <()).
SHELL_METACHARS = ";&|<>`$()\n\r{}*?!#\\'\""


def _reject(reason: str) -> None:
    raise PermissionDenied(reason)


def safe_path(raw: str, *, roots: tuple[Path, ...] = ALLOWED_READ_ROOTS) -> Path:
    """Resolve `raw` to a real path a read-only tool is allowed to open."""
    if not isinstance(raw, str) or not raw:
        _reject("path must be a non-empty string")
    if "\x00" in raw:
        _reject("path contains a NUL byte")

    expanded = Path(os.path.expanduser(raw))
    # resolve() collapses ../ and follows symlinks, so traversal and symlink
    # escapes are both decided on the real target, not the string.
    path = expanded.resolve()

    for denied in DENIED_READ_ROOTS:
        if path == denied or denied in path.parents:
            _reject(f"refused: {raw!r} is inside protected directory {denied}")

    if DENIED_NAME_RE.search(path.name):
        _reject(f"refused: {raw!r} looks like a credential file ({path.name})")

    root = next((r for r in roots if path == r or r in path.parents), None)
    if root is None:
        _reject(f"refused: {raw!r} resolves outside the readable roots ({path})")
    # Dot-directories and dotfiles inside a root (.git/config, .config,
    # .mozilla, .local/share/keyrings) hold config and credentials.
    if any(part.startswith(".") for part in path.relative_to(root).parts):
        _reject(f"refused: {raw!r} is a hidden file or inside a hidden directory")

    return path


# Mutating flags on otherwise read-only commands. The subcommand check below
# deliberately skips anything starting with "-", so flag-driven writes (which
# is how journalctl mutates) would otherwise pass unchecked.
DENIED_FLAGS: dict[str, frozenset[str]] = {
    "journalctl": frozenset({
        "--rotate", "--vacuum-size", "--vacuum-time", "--vacuum-files",
        "--flush", "--sync", "--relinquish-var", "--setup-keys",
        "--cursor-file", "--update-catalog",
    }),
    # `-H/--host` and `-M/--machine` make systemctl open an SSH or container
    # connection, which is not a local read.
    "systemctl": frozenset({"--host", "-H", "--machine", "-M"}),
}

# git global options are refused outright: `-c core.fsmonitor=<cmd>`,
# `--exec-path`, `--config-env`, `--git-dir` and friends turn `git status`
# into arbitrary command execution or point it at another repository.
GIT_GLOBAL_OPTIONS_ALLOWED = frozenset({"--no-pager", "--no-optional-locks"})

# Flags that make a read-only git subcommand write a file (`--output`) or
# read outside the repository (`--no-index` diffs two arbitrary paths, which
# would bypass `safe_path`), or run an external diff program.
GIT_DENIED_FLAGS = frozenset({"--output", "--no-index", "--ext-diff"})

# `git branch` and `git remote` mutate with positional arguments or these
# flags; only their listing forms are read-only.
GIT_BRANCH_MUTATING_SHORT = frozenset("dDmMcCfut")
GIT_BRANCH_MUTATING_LONG = frozenset({
    "--delete", "--move", "--copy", "--force", "--set-upstream-to",
    "--unset-upstream", "--edit-description", "--track", "--no-track",
    "--create-reflog", "--recurse-submodules",
})
GIT_BRANCH_LIST_FLAGS = frozenset({
    "-l", "--list", "--contains", "--no-contains", "--merged", "--no-merged",
    "--points-at",
})
GIT_REMOTE_READ_ONLY = frozenset({"show", "get-url"})


def _check_git(argv: list[str]) -> None:
    """Refuse git invocations that execute code or write despite a read-only subcommand."""
    sub_index = next((i for i, a in enumerate(argv[1:], 1) if not a.startswith("-")), None)
    if sub_index is None:
        return  # bare `git` / `git --version`: the subcommand check refuses it
    for arg in argv[1:sub_index]:
        if arg not in GIT_GLOBAL_OPTIONS_ALLOWED:
            _reject(f"refused: git global option {arg!r} is not allowed")
    sub, rest = argv[sub_index], argv[sub_index + 1:]
    for arg in rest:
        if arg.split("=", 1)[0] in GIT_DENIED_FLAGS:
            _reject(f"refused: git {sub} {arg!r} writes or reads outside the repository")
    if sub == "branch":
        for arg in rest:
            if arg.startswith("--") and arg.split("=", 1)[0] in GIT_BRANCH_MUTATING_LONG:
                _reject(f"refused: git branch {arg!r} modifies branches")
            if arg.startswith("-") and not arg.startswith("--") and (
                set(arg[1:]) & GIT_BRANCH_MUTATING_SHORT
            ):
                _reject(f"refused: git branch {arg!r} modifies branches")
        positional = [a for a in rest if not a.startswith("-")]
        if positional and not any(a.split("=", 1)[0] in GIT_BRANCH_LIST_FLAGS for a in rest):
            _reject("refused: git branch with a name creates a branch; use --list")
    elif sub == "remote":
        action = next((a for a in rest if not a.startswith("-")), None)
        if action is not None and action not in GIT_REMOTE_READ_ONLY:
            _reject(f"refused: git remote {action!r} is not read-only")


def safe_argv(raw: str) -> list[str]:
    """Turn a read-only shell command string into an argv list, or refuse."""
    if not isinstance(raw, str) or not raw.strip():
        _reject("command must be a non-empty string")
    if "\x00" in raw:
        _reject("command contains a NUL byte")

    bad = sorted({c for c in raw if c in SHELL_METACHARS})
    if bad:
        _reject(f"refused: command contains shell metacharacters {bad!r}: {raw!r}")

    try:
        argv = shlex.split(raw)
    except ValueError as exc:
        _reject(f"refused: unparseable command {raw!r} ({exc})")
    if not argv:
        _reject(f"refused: empty command {raw!r}")

    if "/" in argv[0]:
        _reject(f"refused: command must be a bare name, not a path: {argv[0]!r}")

    allowed_sub = ALLOWED_COMMANDS.get(argv[0])
    if allowed_sub is None:
        _reject(f"refused: {argv[0]!r} is not in the read-only command allowlist")
    denied_flags = DENIED_FLAGS.get(argv[0])
    if denied_flags:
        for arg in argv[1:]:
            if arg.split("=", 1)[0] in denied_flags:
                _reject(
                    f"refused: {argv[0]} {arg!r} is a mutating flag, not read-only"
                )

    if argv[0] == "git":
        _check_git(argv)

    if allowed_sub:
        sub = next((a for a in argv[1:] if not a.startswith("-")), None)
        if sub not in allowed_sub:
            _reject(
                f"refused: {argv[0]} {sub!r} is not a read-only subcommand "
                f"(allowed: {sorted(allowed_sub)})"
            )
    return argv


# web_search is the only tool that sends data to a third party (Exa), so a
# query that looks like it carries secrets or a local-data dump -- rather
# than a search phrase -- is refused outright, not silently truncated.
MAX_QUERY_CHARS = 300

# Known API-key/token prefixes issued by common providers.
KEY_PREFIX_RE = re.compile(
    r"(sk-ant-|sk-|ghp_|github_pat_|xoxb-|AKIA|AIza|glpat-)"
)

PEM_RE = re.compile(r"-----BEGIN [A-Z0-9 ]*-----")

# Absolute local paths: not a search phrase, a filesystem reference.
ABS_PATH_RE = re.compile(r"(?:^|\s)(/home/|/etc/|/proc/|/root/|/var/|/usr/|/sys/|/boot/)\S*")

# A run of 32+ non-whitespace chars mixing upper, lower and digits, with no
# whitespace of its own, is a token/secret, not a word a person would search.
HIGH_ENTROPY_RE = re.compile(r"(?=[A-Za-z0-9+/_=-]{32,}(?:\s|$))(?=\S*[a-z])(?=\S*[A-Z])(?=\S*\d)\S{32,}")


def safe_query(raw: str) -> str:
    """Turn a search-query string into one Exa is allowed to see, or refuse."""
    if not isinstance(raw, str) or not raw.strip():
        _reject("query must be a non-empty string")
    if "\x00" in raw:
        _reject("query contains a NUL byte")
    if len(raw) > MAX_QUERY_CHARS:
        _reject(
            f"refused: query is {len(raw)} chars (max {MAX_QUERY_CHARS}) -- "
            "looks like a data dump, not a search phrase"
        )
    if KEY_PREFIX_RE.search(raw):
        _reject("refused: query looks like it contains an API key or token")
    if PEM_RE.search(raw):
        _reject("refused: query contains a PEM-encoded key block")
    if ABS_PATH_RE.search(raw):
        _reject("refused: query contains an absolute local filesystem path")
    if HIGH_ENTROPY_RE.search(raw):
        _reject("refused: query contains a high-entropy token, not a search phrase")
    for token in re.split(r"[\s/]+", raw):
        name = token.rstrip(",;:!?()[]{}\"'").lstrip("([{\"'")
        if name and DENIED_NAME_RE.search(name):
            _reject(f"refused: query mentions a credential-shaped file ({name!r})")
    return raw.strip()
