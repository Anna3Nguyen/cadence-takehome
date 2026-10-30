"""Cadence pre-op scheduling policy (effective Jan 1, 2026), expressed as data.

Rule checks read every window, threshold, and alias from here, so a policy change
is an edit to this file rather than to rule logic.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from difflib import get_close_matches
import re


@dataclass(frozen=True)
class LabRequirement:
    test: str
    # Inclusive: a result exactly max_days before the procedure still counts.
    max_days: int


@dataclass(frozen=True)
class Policy:
    # Rule 1: required documentation
    hp_max_days: int

    # Rule 2: required testing, keyed by procedure risk
    required_labs: dict[str, tuple[LabRequirement, ...]]
    lab_code_aliases: dict[str, str]
    excluded_lab_statuses: frozenset[str]

    # Rule 3: anticoagulation management (any lowercase name -> generic name)
    anticoagulants: dict[str, str]
    # Minimum difflib similarity for a misspelled word to count as an anticoagulant.
    # Measured: typos such as "warfrin" or "lovonox" score 0.86-0.95, while the
    # closest unrelated drugs ("xanax"/lixiana, "enalapril"/enoxaparin) score <= 0.67.
    anticoagulant_match_cutoff: float

    # Rule 4: acute safety exclusions
    systolic_hold_at: float  # hold when systolic >= this
    diastolic_hold_at: float  # hold when diastolic >= this
    temp_f_hold_above: float  # hold when temperature (F) > this


# Generic name -> brand names. Antiplatelets (aspirin, clopidogrel, ...) are
# deliberately absent: they are not anticoagulants and need no Rule 3 plan.
_ANTICOAGULANT_BRANDS: dict[str, tuple[str, ...]] = {
    "warfarin": ("coumadin", "jantoven"),
    "apixaban": ("eliquis",),
    "rivaroxaban": ("xarelto",),
    "dabigatran": ("pradaxa",),
    "edoxaban": ("savaysa", "lixiana"),
    "enoxaparin": ("lovenox",),
    "dalteparin": ("fragmin",),
    "heparin": (),
    "fondaparinux": ("arixtra",),
}

POLICY = Policy(
    hp_max_days=30,
    required_labs={
        "LOW": (LabRequirement("CBC", 30),),
        "MODERATE": (LabRequirement("CBC", 30),),
        "HIGH": (LabRequirement("CBC", 14), LabRequirement("CMP", 14)),
    },
    # A BMP is not a CMP, so BMP codes are intentionally unmapped.
    lab_code_aliases={
        "CBC": "CBC",
        "LAB-CBC": "CBC",
        "CMP": "CMP",
        "LAB-CMP": "CMP",
    },
    excluded_lab_statuses=frozenset({"entered-in-error", "cancelled"}),
    anticoagulants={
        name: generic
        for generic, brands in _ANTICOAGULANT_BRANDS.items()
        for name in (generic, *brands)
    },
    anticoagulant_match_cutoff=0.8,
    systolic_hold_at=180,
    diastolic_hold_at=110,
    temp_f_hold_above=100.4,
)


# -------------------------
# Lookups
# -------------------------

_WORD = re.compile(r"[a-z]+")
# Shorter words ("mg", "bid", "po") are never fuzzy-matched.
_MIN_FUZZY_WORD_LEN = 5


def canonical_lab(code: str | None, policy: Policy = POLICY) -> str | None:
    """Map a lab code like "LAB-CBC" to its policy test name, or None if not required."""

    if not code:
        return None
    return policy.lab_code_aliases.get(code.strip().upper())


def _anticoagulant_matches(text: str, policy: Policy) -> Iterator[str]:
    """Yield generic names for anticoagulants in text: exact words first, then typos."""

    words = _WORD.findall(text.lower())
    for word in words:
        if word in policy.anticoagulants:
            yield policy.anticoagulants[word]
    for word in words:
        if word in policy.anticoagulants or len(word) < _MIN_FUZZY_WORD_LEN:
            continue
        close = get_close_matches(
            word, policy.anticoagulants, n=1, cutoff=policy.anticoagulant_match_cutoff
        )
        if close:
            yield policy.anticoagulants[close[0]]


def anticoagulant_generic(name: str | None, policy: Policy = POLICY) -> str | None:
    """Return the generic anticoagulant named in text like "Eliquis 5mg BID", or None."""

    if not name:
        return None
    return next(_anticoagulant_matches(name, policy), None)


def anticoagulants_named(text: str | None, policy: Policy = POLICY) -> set[str]:
    """Return every generic anticoagulant named in text, e.g. "warfarin to Eliquis"."""

    if not text:
        return set()
    return set(_anticoagulant_matches(text, policy))
