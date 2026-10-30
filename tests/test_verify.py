from __future__ import annotations

from typing import Any

import pytest

from core import PatientSubmission
from facts import (
    AnticoagPlan,
    Consent,
    ExtractedFacts,
    HPDocument,
    MedicationMention,
    NoteVital,
    ProcedureDateMention,
)
from verify import verify_facts

NURSING_NOTE = (
    "PRE-PROCEDURE NURSING ASSESSMENT\n"  # 1
    "Arrival vitals: BP 189/112,  HR 88.  T 37.1 C.\n"  # 2
    "08:30 addendum: BP rechecked after 24 min in a quiet room, 137/78.\n"  # 3
    "Patient’s home meds reviewed; takes acetaminophen 500 mg PRN for pain.\n"  # 4
    "Reports taking Elliquis 5mg BID from outside cardiologist."  # 5
)
CARDIOLOGY_NOTE = (
    "CARDIOLOGY CONSULTATION - PERIOPERATIVE\n"  # 1
    "Patient maintained on Xarelto 20 mg PO daily for atrial fibrillation.\n"  # 2
    "RECOMMENDATIONS:\n"  # 3
    "  1. Hold Xarelto for the 2 days before surgery (last dose 02/26/2026).\n"  # 4
    "  2. Resume Xarelto 24-48 hours after surgery once hemostasis is secure.\n"  # 5
    "  3. Continue clopidogrel; hold 5 days pre-op."  # 6
)
HP_NOTE = (
    "HISTORY AND PHYSICAL: pre-op evaluation complete for planned procedure.\n"  # 1
    "\n"  # 2 (blank)
    "Vitals 10:05  BP: 127/80, HR 89, 98.1 °F temporal, SpO2 99% RA"  # 3
)


def _submission() -> PatientSubmission:
    return PatientSubmission.model_validate(
        {
            "documents": [
                {"doc_id": "nursing", "date": "2026-02-26", "text": NURSING_NOTE},
                {"doc_id": "cardiology", "date": "2026-02-20", "text": CARDIOLOGY_NOTE},
                {"doc_id": "hp", "date": "2026-02-21", "text": HP_NOTE},
            ]
        }
    )


def _facts(**lists: list[Any]) -> ExtractedFacts:
    empty: dict[str, list[Any]] = {name: [] for name in ExtractedFacts.model_fields}
    return ExtractedFacts(**{**empty, **lists})


def _hp(line: int, doc_id: str = "hp") -> HPDocument:
    return HPDocument(doc_id=doc_id, line=line, is_preop_hp=True)


def _vital(
    doc_id: str,
    line: int,
    systolic: float | None = None,
    diastolic: float | None = None,
    temperature: float | None = None,
) -> NoteVital:
    return NoteVital(
        doc_id=doc_id,
        line=line,
        kind="temperature" if temperature is not None else "blood_pressure",
        systolic=systolic,
        diastolic=diastolic,
        temperature=temperature,
        temperature_unit="F" if temperature is not None else None,
        measured_on=None,
        measured_time=None,
    )


def _plan(**overrides: Any) -> AnticoagPlan:
    fields = {
        "doc_id": "cardiology",
        "line": 2,
        "drug": "Xarelto",
        "pre_op_line": 4,
        "post_op_line": 5,
    }
    return AnticoagPlan(**{**fields, **overrides})


def _mention(name: str, line: int) -> MedicationMention:
    return MedicationMention(doc_id="nursing", line=line, name=name, status="taking")


# -------------------------
# Quotes come from the cited line
# -------------------------


def test_quote_is_the_cited_line_without_indentation() -> None:
    result = verify_facts(_facts(anticoag_plans=[_plan()]), _submission())

    [plan] = result.facts.anticoag_plans
    assert plan.quote == (
        "Patient maintained on Xarelto 20 mg PO daily for atrial fibrillation."
    )
    assert plan.pre_op_quote == (
        "1. Hold Xarelto for the 2 days before surgery (last dose 02/26/2026)."
    )
    assert plan.post_op_quote == (
        "2. Resume Xarelto 24-48 hours after surgery once hemostasis is secure."
    )


def test_special_characters_are_kept_exactly() -> None:
    # Regression: the model used to garble "°" when copying quotes (case_00003).
    result = verify_facts(
        _facts(note_vitals=[_vital("hp", 3, temperature=98.1)]), _submission()
    )

    [vital] = result.facts.note_vitals
    assert "98.1 °F temporal" in vital.quote
    assert result.issues == []


# -------------------------
# Bad citations become issues
# -------------------------


@pytest.mark.parametrize(
    ("fact", "expected_problem"),
    [
        (_hp(99), "line 99 does not exist"),
        (_hp(0), "line 0 does not exist"),
        (_hp(2), "line 2 is blank"),
        (_hp(1, doc_id="missing-doc"), "not in this submission"),
    ],
)
def test_bad_citation_discards_fact_and_raises_issue(
    fact: HPDocument, expected_problem: str
) -> None:
    result = verify_facts(_facts(hp_documents=[fact]), _submission())

    assert result.facts.hp_documents == []
    [issue] = result.issues
    assert issue.category == "MISSING_REQUIRED_DATA"
    assert issue.reason == "unverified_quote"
    assert issue.rule == "hp"
    assert expected_problem in issue.finding
    assert issue.sources[0].id == fact.doc_id
    assert issue.sources[0].quote == f"(line {fact.line} could not be read)"


def test_issue_source_carries_document_date() -> None:
    result = verify_facts(_facts(hp_documents=[_hp(99)]), _submission())

    assert result.issues[0].sources[0].date == "2026-02-21"


def test_vital_numbers_are_compared_as_numbers() -> None:
    reading = _vital("nursing", 3, systolic=137.0, diastolic=78)

    result = verify_facts(_facts(note_vitals=[reading]), _submission())

    assert [v.quote for v in result.facts.note_vitals] == [
        "08:30 addendum: BP rechecked after 24 min in a quiet room, 137/78."
    ]
    assert result.issues == []


def test_vital_with_number_not_on_the_line_is_discarded_and_shows_the_line() -> None:
    reading = _vital("nursing", 3, systolic=190, diastolic=78)

    result = verify_facts(_facts(note_vitals=[reading]), _submission())

    assert result.facts.note_vitals == []
    [issue] = result.issues
    assert issue.rule == "blood_pressure"
    assert "systolic 190 does not appear on line 3" in issue.finding
    assert issue.sources[0].quote.endswith("137/78.")


def test_temperature_is_checked_against_the_line() -> None:
    result = verify_facts(
        _facts(note_vitals=[_vital("nursing", 2, temperature=38.6)]), _submission()
    )

    assert result.facts.note_vitals == []
    assert result.issues[0].rule == "temperature"


def test_each_fact_type_maps_to_its_rule() -> None:
    facts = _facts(
        consents=[
            Consent(
                doc_id="hp",
                line=99,
                patient_signed=True,
                procedure_named=None,
                matches_scheduled_procedure=True,
            )
        ],
        procedure_date_mentions=[
            ProcedureDateMention(doc_id="hp", line=99, date="2026-03-01")
        ],
    )

    result = verify_facts(facts, _submission())

    assert sorted(issue.rule for issue in result.issues) == [
        "consent",
        "required_fields",
    ]


# -------------------------
# Medications and anticoagulation plans
# -------------------------


def test_misspelled_anticoagulant_mention_is_verified_and_kept() -> None:
    result = verify_facts(
        _facts(medication_mentions=[_mention("Elliquis 5mg BID", 5)]), _submission()
    )

    [mention] = result.facts.medication_mentions
    assert mention.quote == "Reports taking Elliquis 5mg BID from outside cardiologist."
    assert result.ignored_medications == []


def test_anticoagulant_mention_citing_a_line_about_another_drug_is_discarded() -> None:
    result = verify_facts(
        _facts(medication_mentions=[_mention("Xarelto", 5)]), _submission()
    )

    assert result.facts.medication_mentions == []
    assert result.issues[0].rule == "anticoagulation"
    assert "'Xarelto' does not appear on line 5" in result.issues[0].finding


def test_non_anticoagulant_mentions_are_recorded_not_verified_into_issues() -> None:
    good = _mention("acetaminophen 500 mg PRN", 4)
    bad = _mention("omeprazole 20 mg", 99)

    result = verify_facts(_facts(medication_mentions=[good, bad]), _submission())

    assert result.facts.medication_mentions == []
    assert result.issues == []
    assert [(m.name, m.quote_verified) for m in result.ignored_medications] == [
        ("acetaminophen 500 mg PRN", True),
        ("omeprazole 20 mg", False),
    ]
    first = result.ignored_medications[0]
    assert first.reason == "not an anticoagulant"
    assert (first.source.id, first.source.date) == ("nursing", "2026-02-26")
    assert "acetaminophen 500 mg PRN" in first.source.quote


def test_bad_post_op_line_clears_only_that_instruction() -> None:
    result = verify_facts(
        _facts(anticoag_plans=[_plan(post_op_line=42)]), _submission()
    )

    [kept] = result.facts.anticoag_plans
    assert kept.pre_op_quote is not None
    assert kept.post_op_quote is None
    [issue] = result.issues
    assert issue.rule == "anticoagulation"
    assert "after-surgery instruction" in issue.finding
    assert issue.sources[0].quote == "(line 42 could not be read)"


def test_plan_without_a_cited_instruction_has_no_quote_and_no_issue() -> None:
    result = verify_facts(
        _facts(anticoag_plans=[_plan(post_op_line=None)]), _submission()
    )

    assert result.facts.anticoag_plans[0].post_op_quote is None
    assert result.issues == []


def test_bad_main_line_discards_whole_plan() -> None:
    result = verify_facts(_facts(anticoag_plans=[_plan(line=3)]), _submission())

    assert result.facts.anticoag_plans == []
    assert "'Xarelto' does not appear on line 3" in result.issues[0].finding


def test_plan_for_non_anticoagulant_is_recorded_as_ignored() -> None:
    plan = _plan(line=6, drug="clopidogrel", pre_op_line=6, post_op_line=None)

    result = verify_facts(_facts(anticoag_plans=[plan]), _submission())

    assert result.facts.anticoag_plans == []
    assert result.issues == []
    assert result.ignored_medications[0].reason == "plan is for a non-anticoagulant"
    assert result.ignored_medications[0].quote_verified is True
