"""State-query intent matching shared by the router and the Tier 2 answerer.

Two layers, tried in order:

1. **Exact regexes** -- the original Tier 2 shapes. Fast and precise.
2. **Fuzzy topic matching** -- a dependency-free "semantic" fallback for the
   paraphrases the regexes miss ("battery level?", "how much space is left",
   "is anything listening on 8080", "which branch"). It normalises synonyms
   (ram -> memory, storage -> disk, ...), strips filler, and scores the
   remaining tokens against a per-kind signature: every kind needs one of its
   *anchor* words present and no *veto* word (imperatives like install,
   delete, kill, commit) anywhere in the utterance. No ML, no model download,
   sub-millisecond, offline.

Both the router (does this go to Tier 2 at all?) and `state_query` (which
formatter answers it?) call `classify_kind()` here, so they can no longer
drift apart -- the earlier regex duplication routed utterances to Tier 2 that
the answerer then could not match.

A false-positive Tier 2 route is cheap: `state_query.answer()` escalates to
reasoning when the snapshot cannot honestly answer, so the fuzzy layer is
tuned for recall on short questions and precision only against imperatives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# --- normalisation ----------------------------------------------------------


def normalize(text: str) -> str:
    """Lower-case, collapse whitespace, drop trailing punctuation, and fold
    expanded contractions ("what is" -> "what's") so each shape needs one
    regex. Shared with the router; keep it the single copy."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip("?.!")
    text = re.sub(r"\bwhat is\b", "what's", text)
    text = re.sub(r"\bhow is\b", "how's", text)
    return text.strip()


# --- exact shapes -----------------------------------------------------------

PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("agent_doing", re.compile(r"^what(?:'s| is)\s+(\S+)\s+doing$")),
    ("whats_running", re.compile(r"^what'?s\s+running$")),
    ("whats_happening", re.compile(r"^what'?s\s+happening$")),
    ("in_progress", re.compile(r"^what'?s\s+in\s+progress$")),
    ("branch", re.compile(r"^what\s+branch\s+(?:am i on|is this)$")),
    ("is_running", re.compile(r"^is\s+(.+?)\s+running$")),
    ("failures", re.compile(r"^any\s+(?:failures|errors)$")),
    ("test_failures", re.compile(r"^any\s+tests?\s+fail(?:ing|ed)$")),
    ("test_failures_count", re.compile(r"^how many\s+tests?\s+fail(?:ed|ing)$")),
    (
        "resources",
        re.compile(
            r"^(?:how'?s|what'?s)\s+(?:the\s+)?(?:load|memory|ram|battery|disk)"
            r"(?:\s+(?:usage|looking|at))?$"
        ),
    ),
)

# Canonical phrasings, used ONLY for prefix testing on partial transcripts.
CANONICAL_EXAMPLES = (
    "what's codex doing",
    "what is codex doing",
    "what's running",
    "what's happening",
    "any failures",
    "any errors",
    "any tests failing",
    "what branch am i on",
    "what branch is this",
    "is the dev server running",
    "how many tests failed",
    "what's in progress",
)


# --- fuzzy layer ------------------------------------------------------------

# Synonym folding happens on whole tokens after normalisation.
_SYNONYMS = {
    "ram": "memory",
    "mem": "memory",
    "storage": "disk",
    "space": "disk",
    "drive": "disk",
    "ssd": "disk",
    "charge": "battery",
    "charged": "battery",
    "power": "battery",
    "cpu": "load",
    "processor": "load",
    "busy": "load",
    "alive": "running",
    "up": "running",
    "listening": "running",
    "active": "running",
    "live": "running",
    "processes": "running",
    "checkout": "branch",
    "errors": "failures",
    "error": "failures",
    "failing": "failures",
    "failed": "failures",
    "broken": "failures",
    "crashed": "failures",
}

_FILLER = frozenset(
    {
        "the", "a", "an", "my", "me", "this", "that", "of", "on", "in", "at",
        "to", "for", "please", "hey", "so", "um", "uh", "right", "now",
        "currently", "still", "there", "here", "any", "some", "much", "many",
        "how", "what", "what's", "how's", "which", "is", "are", "am", "do",
        "does", "i", "we", "you", "it", "its", "it's", "left", "like", "level",
        "usage", "status", "looking", "tell", "check", "show", "give", "know",
        "and", "with", "about", "got", "have", "has", "we've", "i've",
    }
)

# Any of these anywhere means "do something", never "tell me something".
_IMPERATIVE_VETO = frozenset(
    {
        "install", "remove", "uninstall", "delete", "kill", "stop", "start",
        "restart", "launch", "open", "close", "run", "commit", "push", "pull",
        "merge", "rebase", "deploy", "build", "fix", "write", "make", "create",
        "switch", "change", "set", "update", "upgrade", "download", "search",
        "find", "remind", "schedule", "send", "email", "call", "cancel",
        "wait", "summarize", "summarise", "explain", "read", "edit", "move",
        "copy", "rename", "clear", "clean", "free", "kill", "spawn", "deploy",
        "test", "mute", "play", "pause", "resume", "turn",
        # Requests ("can you check...") are tasks for the reasoning tier.
        "can", "could", "would", "will", "should",
    }
)

_MAX_FUZZY_WORDS = 9


@dataclass(frozen=True)
class _Signature:
    kind: str
    anchors: frozenset[str]
    # Extra tokens that must ALSO appear for the kind to fire (any one).
    requires: frozenset[str] = frozenset()


# Order matters: earlier signatures win ties, so the more specific kinds
# (test failures, agent_doing) are listed before the broad ones.
_SIGNATURES: tuple[_Signature, ...] = (
    _Signature("test_failures", frozenset({"tests", "test"}), frozenset({"failures", "passing", "pass", "green", "red"})),
    _Signature("branch", frozenset({"branch"})),
    _Signature("resources", frozenset({"battery", "disk", "memory", "load"})),
    _Signature("in_progress", frozenset({"progress", "uncommitted", "unstaged", "working"})),
    _Signature("whats_happening", frozenset({"happening", "going"})),
    _Signature("failures", frozenset({"failures", "wrong", "problems"})),
    _Signature("whats_running", frozenset({"running"})),
)

_AGENT_DOING_RE = re.compile(r"^(?:what(?:'s| is)\s+)?(\S+)\s+(?:doing|up to|working on|status)$")
_IS_RUNNING_RE = re.compile(
    r"^(?:is|are)?\s*(?:the\s+)?(.+?)\s+(?:still\s+)?(?:running|up|alive|listening|active|live)(?:\s+on\s+\S+)?$"
)


def _tokens(norm: str) -> list[str]:
    raw = re.findall(r"[a-z0-9']+", norm)
    return [_SYNONYMS.get(t, t) for t in raw]


def fuzzy_kind(norm: str) -> tuple[Optional[str], Optional[re.Match]]:
    """Best-effort kind for a *normalised* utterance the exact shapes missed.

    Returns (kind, match) like the regex layer. `match` carries the captured
    target for `agent_doing` / `is_running`, otherwise None."""
    if not norm:
        return None, None
    tokens = _tokens(norm)
    if not tokens or len(tokens) > _MAX_FUZZY_WORDS:
        return None, None
    if any(t in _IMPERATIVE_VETO for t in tokens):
        return None, None

    # Target-bearing shapes first: they need the original word order.
    m = _AGENT_DOING_RE.match(norm)
    if m and m.group(1) not in _FILLER:
        return "agent_doing", m
    m = _IS_RUNNING_RE.match(norm)
    if m and "running" in tokens:
        target_tokens = [
            t for t in re.findall(r"[a-z0-9']+", m.group(1)) if t not in _FILLER
        ]
        if target_tokens and not {"anything", "something", "what", "nothing"} & set(target_tokens):
            # Hand the formatter a clean target ("dev server", not "the dev
            # server still") through the same group(1) contract the exact
            # regex uses.
            return "is_running", re.match(r"(.+)", " ".join(target_tokens))

    content = [t for t in tokens if t not in _FILLER]
    if not content:
        return None, None
    present = set(content)
    for sig in _SIGNATURES:
        if not (present & sig.anchors):
            continue
        if sig.requires and not (present & sig.requires):
            continue
        # Reject when the utterance is mostly about something else: at most
        # two content words may fall outside the signature's vocabulary.
        known = sig.anchors | sig.requires | {"running", "failures", "tests", "test"}
        strangers = [t for t in content if t not in known]
        if len(strangers) > 2:
            continue
        return sig.kind, None
    return None, None


def classify_kind(norm: str) -> tuple[Optional[str], Optional[re.Match]]:
    """Exact shapes first, then the fuzzy layer. Input must be normalised."""
    for kind, pattern in PATTERNS:
        m = pattern.match(norm)
        if m:
            return kind, m
    return fuzzy_kind(norm)


def is_state_query(norm: str) -> Optional[str]:
    """Router entry point: the matched kind, or None. Marks fuzzy matches
    with a `fuzzy:` prefix so spans and tests can tell the layers apart."""
    for kind, pattern in PATTERNS:
        if pattern.match(norm):
            return pattern.pattern
    kind, _ = fuzzy_kind(norm)
    return f"fuzzy:{kind}" if kind else None


def is_prefix(norm: str) -> bool:
    if not norm:
        return False
    return any(example.startswith(norm) for example in CANONICAL_EXAMPLES)
