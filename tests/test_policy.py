from __future__ import annotations

from typing import get_args

import pytest

from core import ProcedureRisk
from policy import POLICY, anticoagulant_generic, anticoagulants_named, canonical_lab


def test_every_procedure_risk_has_lab_requirements() -> None:
    assert set(POLICY.required_labs) == set(get_args(ProcedureRisk))


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("CBC", "CBC"),
        ("LAB-CBC", "CBC"),
        (" lab-cbc ", "CBC"),
        ("CMP", "CMP"),
        ("LAB-CMP", "CMP"),
        ("BMP", None),
        ("LAB-BMP", None),
        ("TSH", None),
        ("", None),
        (None, None),
    ],
)
def test_canonical_lab(code: str | None, expected: str | None) -> None:
    assert canonical_lab(code) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Xarelto 20 mg PO daily", "rivaroxaban"),
        ("rivaroxaban", "rivaroxaban"),
        ("Coumadin 5 mg PO daily", "warfarin"),
        ("warfarin", "warfarin"),
        ("Eliquis 5 mg BID", "apixaban"),
        ("Pradaxa (dabigatran 150 mg BID)", "dabigatran"),
        ("edoxaban (Savaysa) 60 mg daily", "edoxaban"),
        ("Lovenox 40 mg SC", "enoxaparin"),
        # Doses, frequencies, and casing attached to the name
        ("Eliquis 5mg BID", "apixaban"),
        ("Eliquis5mg", "apixaban"),
        ("ELIQUIS 5 MG PO BID", "apixaban"),
        ("xArElTo 20mg q.d.", "rivaroxaban"),
        ("warfarin/Coumadin 5 mg (INR goal 2-3)", "warfarin"),
    ],
)
def test_anticoagulant_generic_recognizes_generic_and_brand_names(
    name: str, expected: str
) -> None:
    assert anticoagulant_generic(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("warfrin 5 mg daily", "warfarin"),
        ("Coumadine", "warfarin"),
        ("Elliquis 5mg BID", "apixaban"),
        ("apixiban", "apixaban"),
        ("Xarleto 20 mg", "rivaroxaban"),
        ("rivaroxiban", "rivaroxaban"),
        ("dabigatrin 150 mg BID", "dabigatran"),
        ("Pradexa", "dabigatran"),
        ("Lovonox 40 mg SC", "enoxaparin"),
        ("heprin drip", "heparin"),
    ],
)
def test_anticoagulant_generic_tolerates_misspellings(name: str, expected: str) -> None:
    assert anticoagulant_generic(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "aspirin 81 mg PO daily",
        "ASA 81 mg",
        "clopidogrel 75 mg daily",
        "tamsulosin 0.4 mg PO nightly",
        # Closest unrelated drugs by similarity; must stay below the cutoff
        "enalapril 10 mg daily",
        "metformin 500 mg BID",
        "Xanax 0.5 mg PRN",
        "hepatitis B vaccine",
        "",
        None,
    ],
)
def test_anticoagulant_generic_ignores_antiplatelets_and_other_drugs(
    name: str | None,
) -> None:
    assert anticoagulant_generic(name) is None


def test_anticoagulant_generic_known_miss_for_transposed_letters() -> None:
    # "xeralto" scores 0.71, below the 0.8 cutoff. Lowering the cutoff to catch it
    # would bring "xanax" (0.67) close to matching. Reviewers see unmatched names
    # in ignored_medications instead.
    assert anticoagulant_generic("Xeralto 20 mg") is None


def test_anticoagulants_named_finds_every_drug_in_text() -> None:
    text = "Switched from warfarin to Eliquis 5mg BID last month; continues aspirin."
    assert anticoagulants_named(text) == {"warfarin", "apixaban"}
    assert anticoagulants_named(None) == set()
