from __future__ import annotations

from typing import get_args

import pytest

from core import ProcedureRisk
from policy import POLICY, anticoagulant_generic, canonical_lab


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
    ],
)
def test_anticoagulant_generic_recognizes_generic_and_brand_names(
    name: str, expected: str
) -> None:
    assert anticoagulant_generic(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "aspirin 81 mg PO daily",
        "ASA 81 mg",
        "clopidogrel 75 mg daily",
        "tamsulosin 0.4 mg PO nightly",
        "",
        None,
    ],
)
def test_anticoagulant_generic_ignores_antiplatelets_and_other_drugs(
    name: str | None,
) -> None:
    assert anticoagulant_generic(name) is None
