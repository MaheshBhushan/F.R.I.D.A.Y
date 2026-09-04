"""Three-tier reflex router (T4).

Classifies an utterance into one of three tiers and dispatches it, marking
`intent_classified`. The whole point is speed: Tier 1 and Tier 2 answer
with zero LLM calls and zero network I/O (hardcoded actions / local world
state respectively); Tier 3 is the only tier that ever leaves the machine,
and it fires an ack immediately rather than waiting on that round trip.

Deliberately just two lists of compiled regexes plus two literal phrase
lists used only for prefix testing on partial transcripts - no ML, no
plugin registry, no intent-classification framework (ladder rung 6/3).

The router must be callable on an interim (non-final) transcript, because
the ack for Tier 3 has to fire before STT finalizes. `classify()` therefore
takes an `is_final` flag and will decline to guess (returning tier=None)
when a partial prefix is genuinely ambiguous between tiers - most notably
bare reflex words ("stop", "wait", "cancel that", ...) which are Tier 1 by
themselves but the opening of a Tier 3 task when followed by more words
("stop the dev server").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from friday.core.spans import TurnSpan


class Tier(Enum):
    REFLEX = "reflex"
    STATE_QUERY = "state_query"
    REASONING = "reasoning"


@dataclass
class RouteDecision:
    """Result of classify(). `tier` is None when the prefix is ambiguous
    and the caller should wait for more transcript before acting."""

    tier: Optional[Tier]
    matched: Optional[str] = None  # which trigger/pattern matched, for debugging/tests


# --- Tier 1: hardcoded reflex phrases ---------------------------------------

# Bare phrases that are Tier 1 only when they are the *whole* utterance.
# Followed by more words ("stop the dev server") they become Tier 3 tasks,
# so these are also the phrases treated as ambiguous-on-partial below.
_TIER1_BARE_PHRASES = (
    "stop",
    "mute",
    "wait",
    "cancel that",
    "never mind",
    "nevermind",
    "shut up",
)

# Backchannels: always Tier 1, standalone acknowledgement noises.
_TIER1_BACKCHANNELS = (
    "okay",
    "ok",
    "yeah",
    "yep",
    "yup",
    "mhm",
    "mm-hmm",
    "uh huh",
    "uh-huh",
    "right",
    "sure",
    "got it",
    "cool",
    "alright",
)

_TIER1_ALL_PHRASES = _TIER1_BARE_PHRASES + _TIER1_BACKCHANNELS

_TIER1_PATTERNS = [re.compile(rf"^{re.escape(p)}$") for p in _TIER1_ALL_PHRASES]

# Hardcoded action names for dispatch - no LLM, no lookup table beyond this dict.
TIER1_ACTIONS = {
    "stop": "stop_playback",
    "mute": "mute_mic",
    "wait": "pause_turn",
    "cancel that": "cancel_last_action",
    "never mind": "cancel_last_action",
    "nevermind": "cancel_last_action",
    "shut up": "stop_playback",
    **{p: "noop_ack" for p in _TIER1_BACKCHANNELS},
}


# --- Tier 2: state queries answered from friday.state ------------------------
#
# The shapes themselves live in `friday.tiers.intent`, shared with the Tier 2
# answerer so the two cannot drift. Exact regexes first, then a fuzzy topic
# matcher for paraphrases ("battery level?", "which branch", "is anything
# listening on 8080"). See that module for the design.

from friday.tiers import intent as _intent

_normalize = _intent.normalize


def _matches_tier1(text: str) -> Optional[str]:
    for phrase, pattern in zip(_TIER1_ALL_PHRASES, _TIER1_PATTERNS):
        if pattern.match(text):
            return phrase
    return None


def _matches_tier2(text: str) -> Optional[str]:
    return _intent.is_state_query(text)


def _is_ambiguous_partial(text: str) -> bool:
    """True if `text` exactly equals a bare Tier 1 reflex phrase (with no
    trailing words yet) - it could still turn into a Tier 3 task, or it
    could be the complete utterance. Only the bare phrases are ambiguous
    this way; backchannels are treated as committed once said in full,
    since none of them are also the opening of a longer task phrase in
    this table.
    """
    return text in _TIER1_BARE_PHRASES


def _is_tier2_prefix(text: str) -> bool:
    return _intent.is_prefix(text)


def classify(text: str, is_final: bool = True) -> RouteDecision:
    """Classify `text` (partial or final transcript) into a tier.

    Returns tier=None when a partial prefix is ambiguous and the router
    should wait for more transcript rather than guess.
    """
    norm = _normalize(text)
    if not norm:
        return RouteDecision(tier=None)

    exact1 = _matches_tier1(norm)
    if exact1 is not None:
        if is_final or not _is_ambiguous_partial(norm):
            return RouteDecision(tier=Tier.REFLEX, matched=exact1)
        # Bare reflex word, transcript still growing: could extend into a
        # Tier 3 task ("stop" -> "stop the dev server"). Wait for more.
        return RouteDecision(tier=None, matched=exact1)

    exact2 = _matches_tier2(norm)
    if exact2 is not None:
        return RouteDecision(tier=Tier.STATE_QUERY, matched=exact2)

    if not is_final:
        # Partial that isn't yet a full match of anything. If it could still
        # complete into a bare Tier 1 phrase, stay ambiguous rather than
        # guess Tier 3 out from under a possible "stop"/"wait"/etc.
        if any(p.startswith(norm) for p in _TIER1_BARE_PHRASES):
            return RouteDecision(tier=None)
        if _is_tier2_prefix(norm):
            return RouteDecision(tier=Tier.STATE_QUERY, matched="partial:" + norm)
        # Doesn't head toward any Tier 1/2 shape: it can only resolve to
        # Tier 3 no matter what follows, so it's safe to route now - this
        # is what lets the ack fire before STT finalizes.
        return RouteDecision(tier=Tier.REASONING, matched="partial-default")

    return RouteDecision(tier=Tier.REASONING, matched=None)


def classify_and_mark(text: str, span: TurnSpan, is_final: bool = True) -> RouteDecision:
    """classify() plus marking the `intent_classified` span stage. Only
    marks when a tier was actually decided (not on an ambiguous decline),
    since "classified" means a decision was made."""
    decision = classify(text, is_final=is_final)
    if decision.tier is not None:
        span.mark("intent_classified")
    return decision


def dispatch_tier1(decision: RouteDecision) -> str:
    """Look up and return the hardcoded action name for a Tier 1 decision.
    No LLM call, no network I/O - a dict lookup and nothing else."""
    assert decision.tier is Tier.REFLEX
    return TIER1_ACTIONS[decision.matched]


def dispatch_tier2(decision: RouteDecision) -> str:
    """Answer a Tier 2 state query from friday.state, no LLM/network beyond
    the local snapshot() calls (subprocesses to git/tmux/ss, not network)."""
    from friday import state as state_mod

    assert decision.tier is Tier.STATE_QUERY
    snap = state_mod.snapshot()
    return state_mod.summarize(snap)
