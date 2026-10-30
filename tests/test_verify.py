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
from verify import normalize, verify_facts

NURSING_NOTE = (
    "PRE-PROCEDURE NURSING ASSESSMENT\n"
    "Arrival vitals: BP 189/112,  HR 88.  T 37.1 C.\n"
    "08:30 addendum: BP rechecked after 24 min in a quiet room, 137/78.\n"
    "Patient’s home meds reviewed; takes acetaminophen 500 mg PRN for pain.\n"
    "Reports taking Elliquis 5mg BID from outside cardiologist."
)
CARDIOLOGY_NOTE = (
    "CARDIOLOGY CONSULTATION - PERIOPERATIVE\n"
    "Patient maintained on Xarelto 20 mg PO daily for atrial fibrillation.\n"
    "RECOMMENDATIONS:\n"
    "  1. Hold Xarelto for the 2 days before surgery (last dose 02/26/2026).\n"
    "  2. Resume Xarelto 24-48 hours after surgery once hemostasis is secure.\n"
    "  3. Continue clopidogrel; hold 5 days pre-op."
)
HP_NOTE = (
    "HISTORY AND PHYSICAL: pre-op evaluation complete for planned procedure.\n"
    "Procedure date target: 2026-03-01."
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


def _hp(quote: str, doc_id: str = "hp") -> HPDocument:
    return HPDocument(doc_id=doc_id, quote=quote, is_preop_hp=True)


def _bp(quote: str, systolic: float, diastolic: float) -> NoteVital:
    return NoteVital(
        doc_id="nursing",
        quote=quote,
        kind="blood_pressure",
        systolic=systolic,
        diastolic=diastolic,
        temperature=None,
        temperature_unit=None,
        measured_on=None,
        measured_time=None,
    )


def _plan(**overrides: Any) -> AnticoagPlan:
    fields = {
        "doc_id": "cardiology",
        "quote": "Patient maintained on Xarelto 20 mg PO daily",
        "drug": "Xarelto",
        "pre_op_quote": "Hold Xarelto for the 2 days before surgery",
        "post_op_quote": "Resume Xarelto 24-48 hours after surgery",
    }
    return AnticoagPlan(**{**fields, **overrides})


def _mention(name: str, quote: str) -> MedicationMention:
    return MedicationMention(doc_id="nursing", quote=quote, name=name, status="taking")


# -------------------------
# Normalization and quote matching
# -------------------------


def test_normalize_ignores_case_whitespace_and_quote_styles() -> None:
    assert normalize("Patient’s  BP\n189/112") == normalize("patient's bp 189/112")


@pytest.mark.parametrize(
    "quote",
    [
        "HISTORY AND PHYSICAL: pre-op evaluation complete for planned procedure.",
        "history and physical:   PRE-OP evaluation\ncomplete",
        "  pre-op evaluation complete for planned procedure.  ",
    ],
)
def test_quote_matches_despite_case_and_whitespace(quote: str) -> None:
    result = verify_facts(_facts(hp_documents=[_hp(quote)]), _submission())

    assert len(result.facts.hp_documents) == 1
    assert result.issues == []


def test_straight_apostrophe_matches_curly_apostrophe_in_document() -> None:
    # NURSING_NOTE writes "Patient’s" with a curly apostrophe.
    fact = _hp("Patient's home meds reviewed", doc_id="nursing")

    result = verify_facts(_facts(hp_documents=[fact]), _submission())

    assert result.issues == []
    assert result.facts.hp_documents == [fact]


# -------------------------
# Discarded facts become issues
# -------------------------


@pytest.mark.parametrize(
    ("fact", "expected_problem"),
    [
        (_hp("Pre-operative evaluation was completed."), "does not appear"),
        (_hp(HP_NOTE, doc_id="missing-doc"), "not in this submission"),
        (_hp(""), "quote is empty"),
        (_hp("   \n "), "quote is empty"),
    ],
)
def test_unverifiable_quote_discards_fact_and_raises_issue(
    fact: HPDocument, expected_problem: str
) -> None:
    result = verify_facts(_facts(hp_documents=[fact]), _submission())

    assert result.facts.hp_documents == []
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.category == "MISSING_REQUIRED_DATA"
    assert issue.reason == "unverified_quote"
    assert issue.rule == "hp"
    assert expected_problem in issue.finding
    assert issue.sources[0].id == fact.doc_id
    assert issue.sources[0].quote == fact.quote


def test_issue_source_carries_document_date() -> None:
    result = verify_facts(_facts(hp_documents=[_hp("not in the note")]), _submission())

    assert result.issues[0].sources[0].date == "2026-02-21"


def test_vital_numbers_are_compared_as_numbers() -> None:
    reading = _bp(
        "08:30 addendum: BP rechecked after 24 min in a quiet room, 137/78", 137.0, 78
    )

    result = verify_facts(_facts(note_vitals=[reading]), _submission())

    assert result.facts.note_vitals == [reading]
    assert result.issues == []


def test_vital_with_number_not_in_quote_is_discarded() -> None:
    reading = _bp(
        "08:30 addendum: BP rechecked after 24 min in a quiet room, 137/78", 190, 78
    )

    result = verify_facts(_facts(note_vitals=[reading]), _submission())

    assert result.facts.note_vitals == []
    assert result.issues[0].rule == "blood_pressure"
    assert "systolic 190" in result.issues[0].finding


def test_temperature_is_checked_against_quote() -> None:
    reading = NoteVital(
        doc_id="nursing",
        quote="T 37.1 C.",
        kind="temperature",
        systolic=None,
        diastolic=None,
        temperature=38.6,
        temperature_unit="C",
        measured_on=None,
        measured_time=None,
    )

    result = verify_facts(_facts(note_vitals=[reading]), _submission())

    assert result.facts.note_vitals == []
    assert result.issues[0].rule == "temperature"


def test_each_fact_type_maps_to_its_rule() -> None:
    bad = "this text is nowhere in the chart"
    facts = _facts(
        consents=[
            Consent(
                doc_id="hp",
                quote=bad,
                patient_signed=True,
                procedure_named=None,
                matches_scheduled_procedure=True,
            )
        ],
        procedure_date_mentions=[
            ProcedureDateMention(doc_id="hp", quote=bad, date="2026-03-01")
        ],
    )

    result = verify_facts(facts, _submission())

    assert sorted(issue.rule for issue in result.issues) == ["consent", "required_fields"]


# -------------------------
# Medications and anticoagulation plans
# -------------------------


def test_misspelled_anticoagulant_mention_is_verified_and_kept() -> None:
    mention = _mention("Elliquis 5mg BID", "Reports taking Elliquis 5mg BID")

    result = verify_facts(_facts(medication_mentions=[mention]), _submission())

    assert result.facts.medication_mentions == [mention]
    assert result.ignored_medications == []


def test_anticoagulant_mention_naming_a_different_drug_is_discarded() -> None:
    mention = _mention("Xarelto", "Reports taking Elliquis 5mg BID")

    result = verify_facts(_facts(medication_mentions=[mention]), _submission())

    assert result.facts.medication_mentions == []
    assert result.issues[0].rule == "anticoagulation"
    assert "'Xarelto' does not appear" in result.issues[0].finding


def test_non_anticoagulant_mentions_are_recorded_not_verified_into_issues() -> None:
    good = _mention("acetaminophen 500 mg PRN", "takes acetaminophen 500 mg PRN for pain")
    bad = _mention("omeprazole 20 mg", "takes omeprazole 20 mg daily")

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


def test_complete_plan_is_kept_unchanged() -> None:
    plan = _plan()

    result = verify_facts(_facts(anticoag_plans=[plan]), _submission())

    assert result.facts.anticoag_plans == [plan]
    assert result.issues == []


def test_bad_post_op_quote_clears_only_that_instruction() -> None:
    plan = _plan(post_op_quote="Restart Xarelto on post-op day 1.")

    result = verify_facts(_facts(anticoag_plans=[plan]), _submission())

    kept = result.facts.anticoag_plans[0]
    assert kept.pre_op_quote == plan.pre_op_quote
    assert kept.post_op_quote is None
    assert len(result.issues) == 1
    assert result.issues[0].rule == "anticoagulation"
    assert "after-surgery instruction" in result.issues[0].finding


def test_bad_main_quote_discards_whole_plan() -> None:
    plan = _plan(quote="Patient takes Xarelto twice daily.")

    result = verify_facts(_facts(anticoag_plans=[plan]), _submission())

    assert result.facts.anticoag_plans == []
    assert len(result.issues) == 1


def test_plan_for_non_anticoagulant_is_recorded_as_ignored() -> None:
    plan = _plan(
        quote="Continue clopidogrel; hold 5 days pre-op.",
        drug="clopidogrel",
        pre_op_quote="hold 5 days pre-op",
        post_op_quote=None,
    )

    result = verify_facts(_facts(anticoag_plans=[plan]), _submission())

    assert result.facts.anticoag_plans == []
    assert result.issues == []
    assert result.ignored_medications[0].reason == "plan is for a non-anticoagulant"
    assert result.ignored_medications[0].quote_verified is True
