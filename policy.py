"""Cadence pre-op scheduling policy (effective Jan 1, 2026), expressed as data.

Rule checks read every window, threshold, and alias from here, so a policy change
is an edit to this file rather than to rule logic.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    systolic_hold_at=180,
    diastolic_hold_at=110,
    temp_f_hold_above=100.4,
)


# -------------------------
# Lookups
# -------------------------

_WORD = re.compile(r"[a-z]+")


def canonical_lab(code: str | None, policy: Policy = POLICY) -> str | None:
    """Map a lab code like "LAB-CBC" to its policy test name, or None if not required."""

    if not code:
        return None
    return policy.lab_code_aliases.get(code.strip().upper())


def anticoagulant_generic(name: str | None, policy: Policy = POLICY) -> str | None:
    """Return the generic anticoagulant named in free text like "Xarelto 20 mg PO daily"."""

    if not name:
        return None
    for word in _WORD.findall(name.lower()):
        generic = policy.anticoagulants.get(word)
        if generic is not None:
            return generic
    return None
