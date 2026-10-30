from __future__ import annotations

from datetime import date, timedelta
from typing import Any, get_args

import pytest

from core import PatientSubmission, RuleCheck, RuleId, TriageIssue
from facts import (
    AnticoagPlan,
    Consent,
    ExtractedFacts,
    HPDocument,
    MedicationMention,
    NoteVital,
    ProcedureDateMention,
)
from policy import POLICY
from rules import (
    Chart,
    Finding,
    check_anticoagulation,
    check_blood_pressure,
    check_consent,
    check_hp,
    check_labs,
    check_required_fields,
    check_temperature,
    days_before,
    decide,
    evaluate_rules,
    ignored_structured_medications,
    parse_date,
)

PROCEDURE_DATE = date(2026, 3, 1)


def _on(days: int) -> str:
    """The date `days` days before the procedure, as YYYY-MM-DD."""

    return (PROCEDURE_DATE - timedelta(days=days)).isoformat()


def _doc(doc_id: str, days: int) -> dict[str, Any]:
    return {"doc_id": doc_id, "date": _on(days), "text": f"Text of {doc_id}."}


def _lab(
    code: str, days: int, status: str = "final", lab_id: str | None = None
) -> dict[str, Any]:
    return {
        "id": lab_id or f"{code}-{days}",
        "code": code,
        "display": code,
        "effective_at": f"{_on(days)}T08:00:00Z",
        "status": status,
    }


def _chart(
    *,
    procedure: dict[str, Any] | None = None,
    labs: list[dict[str, Any]] | None = None,
    documents: list[dict[str, Any]] | None = None,
    medications: list[dict[str, Any]] | None = None,
    vitals: list[dict[str, Any]] | None = None,
    **facts: list[Any],
) -> Chart:
    submission = PatientSubmission.model_validate(
        {
            "procedure": {
                "procedure_type": "Elective carpal tunnel release",
                "procedure_risk": "LOW",
                "procedure_date": PROCEDURE_DATE.isoformat(),
                **(procedure or {}),
            },
            "labs": labs or [],
            "documents": documents or [],
            "medications": medications or [],
            "vitals": vitals or [],
        }
    )
    empty: dict[str, list[Any]] = {name: [] for name in ExtractedFacts.model_fields}
    return Chart(submission=submission, facts=ExtractedFacts(**{**empty, **facts}))


def _hp(doc_id: str, is_preop: bool = True) -> HPDocument:
    return HPDocument(
        doc_id=doc_id, quote=f"H&P quote from {doc_id}", is_preop_hp=is_preop
    )


def _consent(doc_id: str, signed: bool = True, matches: bool = True) -> Consent:
    return Consent(
        doc_id=doc_id,
        quote=f"Consent quote from {doc_id}",
        patient_signed=signed,
        procedure_named="carpal tunnel release"
        if matches
        else "coronary artery bypass",
        matches_scheduled_procedure=matches,
    )


def _date_mention(doc_id: str, stated: str) -> ProcedureDateMention:
    return ProcedureDateMention(
        doc_id=doc_id, quote=f"Surgery on {stated}", date=stated
    )


def _single(findings: list[Finding]) -> Finding:
    assert len(findings) == 1, findings
    return findings[0]


# -------------------------
# Helpers
# -------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-02-11", date(2026, 2, 11)),
        ("2026-02-11T11:29:00Z", date(2026, 2, 11)),
        (" 2026-02-11 ", date(2026, 2, 11)),
        ("02/11/2026", None),
        ("pending OR availability", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_date(value: str | None, expected: date | None) -> None:
    assert parse_date(value) == expected


def test_days_before_crosses_month_boundary() -> None:
    # February 2026 has 28 days.
    assert days_before(date(2026, 3, 1), date(2026, 1, 30)) == 30
    assert days_before(date(2026, 3, 1), date(2026, 1, 29)) == 31


# -------------------------
# Required fields
# -------------------------


def test_required_fields_pass_lists_date_risk_and_agreeing_notes() -> None:
    chart = _chart(
        documents=[_doc("hp", 8)],
        procedure_date_mentions=[_date_mention("hp", PROCEDURE_DATE.isoformat())],
    )

    check = _single(check_required_fields(chart, POLICY))

    assert isinstance(check, RuleCheck)
    assert check.status == "passed"
    assert [s.id for s in check.sources] == [
        "procedure.procedure_date",
        "procedure.procedure_risk",
        "hp",
    ]


def test_missing_procedure_date_is_missing_data() -> None:
    issue = _single(
        check_required_fields(_chart(procedure={"procedure_date": None}), POLICY)
    )

    assert isinstance(issue, TriageIssue)
    assert (issue.category, issue.reason) == ("MISSING_REQUIRED_DATA", "missing")
    assert issue.sources[0].id == "procedure.procedure_date"
    assert issue.sources[0].quote == "null"


def test_missing_procedure_risk_is_missing_data() -> None:
    issue = _single(
        check_required_fields(_chart(procedure={"procedure_risk": None}), POLICY)
    )

    assert (issue.category, issue.reason) == ("MISSING_REQUIRED_DATA", "missing")
    assert "risk" in issue.finding


def test_note_with_different_procedure_date_is_a_conflict_with_both_sources() -> None:
    chart = _chart(
        documents=[_doc("hp", 8), _doc("nursing", 2)],
        procedure_date_mentions=[
            _date_mention("hp", PROCEDURE_DATE.isoformat()),
            _date_mention("nursing", "2026-03-08"),
        ],
    )

    issue = _single(check_required_fields(chart, POLICY))

    assert (issue.category, issue.reason) == ("MISSING_REQUIRED_DATA", "conflict")
    assert "2026-03-08" in issue.finding
    assert [s.id for s in issue.sources] == ["procedure.procedure_date", "nursing"]


# -------------------------
# Rule 1: H&P
# -------------------------


@pytest.mark.parametrize(
    ("days", "passes"), [(0, True), (30, True), (31, False), (-1, False)]
)
def test_hp_window_is_inclusive(days: int, passes: bool) -> None:
    chart = _chart(documents=[_doc("hp", days)], hp_documents=[_hp("hp")])

    finding = _single(check_hp(chart, POLICY))

    assert isinstance(finding, RuleCheck if passes else TriageIssue)
    assert finding.computed["days_before"] == days
    if not passes:
        assert finding.category == "REQUIRED_DOCUMENTATION"


def test_hp_uses_most_recent_preop_hp() -> None:
    chart = _chart(
        documents=[_doc("old", 45), _doc("new", 10)],
        hp_documents=[_hp("old"), _hp("new")],
    )

    check = _single(check_hp(chart, POLICY))

    assert check.status == "passed"
    assert check.sources[0].id == "new"


def test_wellness_visit_does_not_count_and_is_listed_as_evidence() -> None:
    chart = _chart(
        documents=[_doc("wellness", 5)], hp_documents=[_hp("wellness", is_preop=False)]
    )

    issue = _single(check_hp(chart, POLICY))

    assert issue.category == "REQUIRED_DOCUMENTATION"
    assert "none is a pre-op evaluation" in issue.finding
    assert issue.sources[0].id == "wellness"


def test_no_hp_at_all_is_required_documentation() -> None:
    issue = _single(check_hp(_chart(), POLICY))

    assert issue.category == "REQUIRED_DOCUMENTATION"
    assert issue.finding == "No pre-op H&P found."


def test_hp_window_not_evaluated_without_procedure_date() -> None:
    chart = _chart(
        procedure={"procedure_date": None},
        documents=[_doc("hp", 10)],
        hp_documents=[_hp("hp")],
    )

    check = _single(check_hp(chart, POLICY))

    assert check.status == "not_evaluated"


def test_hp_presence_still_checked_without_procedure_date() -> None:
    issue = _single(check_hp(_chart(procedure={"procedure_date": None}), POLICY))

    assert issue.category == "REQUIRED_DOCUMENTATION"


# -------------------------
# Rule 1: consent
# -------------------------


def test_newer_signed_consent_supersedes_pending_one() -> None:
    chart = _chart(
        documents=[_doc("pending", 12), _doc("signed", 2)],
        consents=[_consent("pending", signed=False), _consent("signed")],
    )

    check = _single(check_consent(chart, POLICY))

    assert check.status == "passed"
    assert "supersedes 1 earlier" in check.finding
    assert [s.id for s in check.sources] == ["signed", "pending"]


def test_newer_unsigned_consent_overrides_older_signed_one() -> None:
    chart = _chart(
        documents=[_doc("signed", 12), _doc("pending", 2)],
        consents=[_consent("signed"), _consent("pending", signed=False)],
    )

    issue = _single(check_consent(chart, POLICY))

    assert issue.category == "REQUIRED_DOCUMENTATION"
    assert "not signed by the patient" in issue.finding


def test_consent_for_a_different_procedure_fails() -> None:
    chart = _chart(
        documents=[_doc("cabg", 5)], consents=[_consent("cabg", matches=False)]
    )

    issue = _single(check_consent(chart, POLICY))

    assert "coronary artery bypass" in issue.finding
    assert "Elective carpal tunnel release" in issue.finding


def test_no_consent_is_required_documentation() -> None:
    issue = _single(check_consent(_chart(), POLICY))

    assert (issue.category, issue.finding) == (
        "REQUIRED_DOCUMENTATION",
        "No surgical consent found.",
    )


@pytest.mark.parametrize("valid_first", [True, False])
def test_same_day_consents_prefer_the_valid_one_regardless_of_order(
    valid_first: bool,
) -> None:
    valid, pending = _consent("a"), _consent("b", signed=False)
    chart = _chart(
        documents=[_doc("a", 3), _doc("b", 3)],
        consents=[valid, pending] if valid_first else [pending, valid],
    )

    assert _single(check_consent(chart, POLICY)).status == "passed"


# -------------------------
# Rule 2: labs
# -------------------------


@pytest.mark.parametrize(("days", "passes"), [(30, True), (31, False)])
def test_low_risk_cbc_window(days: int, passes: bool) -> None:
    finding = _single(check_labs(_chart(labs=[_lab("CBC", days)]), POLICY))

    assert isinstance(finding, RuleCheck if passes else TriageIssue)
    assert finding.computed["limit_days"] == 30


@pytest.mark.parametrize(("days", "passes"), [(14, True), (15, False)])
def test_high_risk_cbc_window(days: int, passes: bool) -> None:
    chart = _chart(
        procedure={"procedure_risk": "HIGH"}, labs=[_lab("CBC", days), _lab("CMP", 3)]
    )

    cbc, cmp = check_labs(chart, POLICY)

    assert isinstance(cbc, RuleCheck if passes else TriageIssue)
    assert cmp.status == "passed"


def test_high_risk_missing_cmp_fails_and_bmp_does_not_count() -> None:
    chart = _chart(
        procedure={"procedure_risk": "HIGH"},
        labs=[_lab("CBC", 3), _lab("BMP", 3), _lab("LAB-BMP", 2)],
    )

    cbc, cmp = check_labs(chart, POLICY)

    assert cbc.status == "passed"
    assert isinstance(cmp, TriageIssue)
    assert cmp.category == "REQUIRED_TESTING"
    assert cmp.finding.startswith("No usable CMP result")


def test_lab_alias_counts_as_cbc() -> None:
    check = _single(check_labs(_chart(labs=[_lab("LAB-CBC", 5)]), POLICY))

    assert check.status == "passed"


def test_entered_in_error_result_is_skipped_and_shown_as_evidence() -> None:
    chart = _chart(
        labs=[
            _lab("CBC", 24, lab_id="older"),
            _lab("CBC", 23, status="entered-in-error", lab_id="voided"),
        ]
    )

    check = _single(check_labs(chart, POLICY))

    assert check.status == "passed"
    assert check.computed["days_before"] == 24
    assert [s.id for s in check.sources] == ["older", "voided"]
    assert "Skipped 1 CBC result" in check.finding


def test_most_recent_result_decides_even_if_an_older_one_would_pass() -> None:
    # Only the most recent result counts; here it is the newer one, which passes.
    chart = _chart(labs=[_lab("CBC", 50, lab_id="old"), _lab("CBC", 10, lab_id="new")])

    check = _single(check_labs(chart, POLICY))

    assert check.sources[0].id == "new"


def test_missing_cbc_is_required_testing() -> None:
    issue = _single(check_labs(_chart(labs=[_lab("TSH", 3)]), POLICY))

    assert issue.category == "REQUIRED_TESTING"


def test_labs_not_evaluated_without_risk() -> None:
    check = _single(check_labs(_chart(procedure={"procedure_risk": None}), POLICY))

    assert check.status == "not_evaluated"


def test_lab_window_not_evaluated_without_procedure_date_but_presence_checked() -> None:
    present = _single(
        check_labs(
            _chart(procedure={"procedure_date": None}, labs=[_lab("CBC", 5)]), POLICY
        )
    )
    absent = _single(check_labs(_chart(procedure={"procedure_date": None}), POLICY))

    assert present.status == "not_evaluated"
    assert isinstance(absent, TriageIssue)


# -------------------------
# Rule 3: anticoagulation
# -------------------------


def _med(name: str, active: bool | None = True) -> dict[str, Any]:
    return {"name": name, "active": active}


def _mention(doc_id: str, name: str, status: str = "taking") -> MedicationMention:
    return MedicationMention(
        doc_id=doc_id, quote=f"{name} mentioned in {doc_id}", name=name, status=status
    )


def _plan(doc_id: str, drug: str, pre: bool = True, post: bool = True) -> AnticoagPlan:
    return AnticoagPlan(
        doc_id=doc_id,
        quote=f"Maintained on {drug}",
        drug=drug,
        pre_op_quote=f"Hold {drug} 2 days before surgery" if pre else None,
        post_op_quote=f"Resume {drug} 48 hours after surgery" if post else None,
    )


def test_anticoagulant_only_in_a_note_counts_as_taken() -> None:
    chart = _chart(
        documents=[_doc("nursing", 3)],
        medication_mentions=[_mention("nursing", "Xarelto 20 mg")],
    )

    issue = _single(check_anticoagulation(chart, POLICY))

    assert issue.category == "ANTICOAGULATION_MANAGEMENT"
    assert issue.finding == (
        "Patient takes rivaroxaban, but no perioperative plan is documented."
    )
    assert issue.sources[0].id == "nursing"


def test_inactive_warfarin_confirmed_stopped_is_not_applicable() -> None:
    chart = _chart(
        documents=[_doc("hp", 10)],
        medications=[_med("warfarin 5 mg PO daily", active=False)],
        medication_mentions=[_mention("hp", "warfarin", "stopped")],
    )

    check = _single(check_anticoagulation(chart, POLICY))

    assert (check.status, check.finding) == (
        "not_applicable",
        "warfarin is not currently taken.",
    )
    assert [s.id for s in check.sources] == ["medications[0]", "hp"]


def test_unknown_status_with_uncertain_note_is_missing_data() -> None:
    chart = _chart(
        documents=[_doc("nursing", 2)],
        medications=[_med("Coumadin 5 mg PO daily", active=None)],
        medication_mentions=[_mention("nursing", "Coumadin", "uncertain")],
    )

    issue = _single(check_anticoagulation(chart, POLICY))

    assert (issue.category, issue.reason) == ("MISSING_REQUIRED_DATA", "missing")


def test_unknown_status_with_no_notes_is_missing_data() -> None:
    chart = _chart(medications=[_med("apixaban 5 mg", active=None)])

    issue = _single(check_anticoagulation(chart, POLICY))

    assert (issue.category, issue.reason) == ("MISSING_REQUIRED_DATA", "missing")


def test_list_says_taking_but_note_says_stopped_is_a_conflict() -> None:
    chart = _chart(
        documents=[_doc("nursing", 2)],
        medications=[_med("Eliquis 5 mg BID")],
        medication_mentions=[_mention("nursing", "Eliquis", "stopped")],
    )

    issue = _single(check_anticoagulation(chart, POLICY))

    assert (issue.category, issue.reason) == ("MISSING_REQUIRED_DATA", "conflict")
    assert [s.id for s in issue.sources] == ["medications[0]", "nursing"]


def test_newer_note_overrides_older_note() -> None:
    chart = _chart(
        documents=[_doc("old", 30), _doc("new", 5)],
        medication_mentions=[
            _mention("old", "warfarin", "taking"),
            _mention("new", "warfarin", "stopped"),
        ],
    )

    assert _single(check_anticoagulation(chart, POLICY)).status == "not_applicable"


def test_brand_and_generic_names_are_one_drug() -> None:
    chart = _chart(
        documents=[_doc("cardiology", 9), _doc("nursing", 3)],
        medications=[_med("Coumadin 5 mg PO daily")],
        medication_mentions=[_mention("nursing", "warfarin")],
        anticoag_plans=[_plan("cardiology", "Coumadin")],
    )

    check = _single(check_anticoagulation(chart, POLICY))

    assert check.status == "passed"
    assert check.computed["drug"] == "warfarin"


def test_complete_plan_passes_and_leads_the_evidence() -> None:
    chart = _chart(
        documents=[_doc("cardiology", 9)],
        medications=[_med("Xarelto 20 mg")],
        anticoag_plans=[_plan("cardiology", "Xarelto")],
    )

    check = _single(check_anticoagulation(chart, POLICY))

    assert check.status == "passed"
    assert [s.quote for s in check.sources[:3]] == [
        "Maintained on Xarelto",
        "Hold Xarelto 2 days before surgery",
        "Resume Xarelto 48 hours after surgery",
    ]


def test_plan_without_resume_instruction_fails() -> None:
    chart = _chart(
        documents=[_doc("cardiology", 6)],
        medications=[_med("warfarin 5 mg")],
        anticoag_plans=[_plan("cardiology", "warfarin", post=False)],
    )

    issue = _single(check_anticoagulation(chart, POLICY))

    assert issue.category == "ANTICOAGULATION_MANAGEMENT"
    assert "no after-surgery instruction" in issue.finding


def test_plan_for_a_different_anticoagulant_does_not_count() -> None:
    chart = _chart(
        documents=[_doc("old-plan", 20)],
        medications=[_med("warfarin 5 mg", active=False), _med("apixaban 5 mg BID")],
        anticoag_plans=[_plan("old-plan", "warfarin")],
    )

    findings = {f.computed["drug"]: f for f in check_anticoagulation(chart, POLICY)}

    assert findings["warfarin"].status == "not_applicable"
    assert findings["apixaban"].category == "ANTICOAGULATION_MANAGEMENT"


def test_one_complete_plan_among_several_passes() -> None:
    chart = _chart(
        documents=[_doc("a", 9), _doc("b", 3)],
        medications=[_med("Eliquis 5 mg")],
        anticoag_plans=[_plan("b", "Eliquis", post=False), _plan("a", "Eliquis")],
    )

    check = _single(check_anticoagulation(chart, POLICY))

    assert check.status == "passed"
    assert check.sources[0].id == "a"


def test_instructions_split_across_plans_do_not_count_as_one_plan() -> None:
    chart = _chart(
        documents=[_doc("a", 9), _doc("b", 3)],
        medications=[_med("Eliquis 5 mg")],
        anticoag_plans=[
            _plan("a", "Eliquis", post=False),
            _plan("b", "Eliquis", pre=False),
        ],
    )

    issue = _single(check_anticoagulation(chart, POLICY))

    assert "no single plan covers both" in issue.finding


def test_antiplatelets_only_is_not_applicable() -> None:
    chart = _chart(medications=[_med("aspirin 81 mg"), _med("clopidogrel 75 mg")])

    check = _single(check_anticoagulation(chart, POLICY))

    assert check.status == "not_applicable"


def test_ignored_structured_medications_lists_non_anticoagulants() -> None:
    chart = _chart(
        medications=[_med("aspirin 81 mg"), _med("Xarelto 20 mg"), _med("tamsulosin")]
    )

    ignored = ignored_structured_medications(chart)

    assert [(m.name, m.source.id) for m in ignored] == [
        ("aspirin 81 mg", "medications[0]"),
        ("tamsulosin", "medications[2]"),
    ]
    assert all(m.quote_verified for m in ignored)


# -------------------------
# Rule 4: vitals
# -------------------------


def _bp_vital(
    systolic: float, diastolic: float, days: int, at: str = "10:00"
) -> dict[str, Any]:
    return {
        "type": "blood_pressure",
        "systolic": systolic,
        "diastolic": diastolic,
        "date": f"{_on(days)}T{at}:00Z",
        "source": "Pre-op clinic",
    }


def _temp_vital(value_f: float, days: int, at: str = "10:00") -> dict[str, Any]:
    return {
        "type": "temperature",
        "value_f": value_f,
        "date": f"{_on(days)}T{at}:00Z",
        "source": "Pre-op clinic",
    }


def _note_bp(
    doc_id: str,
    systolic: float,
    diastolic: float,
    at: str | None = None,
    measured_on: str | None = None,
) -> NoteVital:
    return NoteVital(
        doc_id=doc_id,
        quote=f"BP {systolic:g}/{diastolic:g}",
        kind="blood_pressure",
        systolic=systolic,
        diastolic=diastolic,
        temperature=None,
        temperature_unit=None,
        measured_on=measured_on,
        measured_time=at,
    )


def _note_temp(doc_id: str, value: float, unit: str | None) -> NoteVital:
    return NoteVital(
        doc_id=doc_id,
        quote=f"T {value:g} {unit}",
        kind="temperature",
        systolic=None,
        diastolic=None,
        temperature=value,
        temperature_unit=unit,
        measured_on=None,
        measured_time=None,
    )


@pytest.mark.parametrize(
    ("systolic", "diastolic", "held"),
    [(179, 109, False), (180, 80, True), (120, 110, True)],
)
def test_bp_thresholds_are_inclusive(
    systolic: float, diastolic: float, held: bool
) -> None:
    chart = _chart(vitals=[_bp_vital(systolic, diastolic, 2)])

    finding = _single(check_blood_pressure(chart, POLICY))

    assert isinstance(finding, TriageIssue) is held
    if held:
        assert finding.category == "ACUTE_SAFETY_EXCLUSION"


def test_later_timed_recheck_replaces_high_reading() -> None:
    chart = _chart(
        documents=[_doc("nursing", 1)],
        note_vitals=[
            _note_bp("nursing", 189, 112, at="08:06"),
            _note_bp("nursing", 137, 78, at="08:30"),
        ],
    )

    check = _single(check_blood_pressure(chart, POLICY))

    assert check.status == "passed"
    assert [s.quote for s in check.sources] == ["BP 137/78", "BP 189/112"]


def test_recheck_still_over_limit_is_held() -> None:
    chart = _chart(
        documents=[_doc("nursing", 1)],
        note_vitals=[
            _note_bp("nursing", 192, 106, at="08:00"),
            _note_bp("nursing", 183, 104, at="08:24"),
        ],
    )

    issue = _single(check_blood_pressure(chart, POLICY))

    assert issue.category == "ACUTE_SAFETY_EXCLUSION"
    assert issue.computed["reading"] == "183/104"


def test_same_day_untimed_readings_use_the_most_severe() -> None:
    chart = _chart(
        documents=[_doc("nursing", 1)],
        note_vitals=[_note_bp("nursing", 150, 90), _note_bp("nursing", 185, 95)],
    )

    issue = _single(check_blood_pressure(chart, POLICY))

    assert issue.computed["reading"] == "185/95"
    assert "could not be put in order" in issue.finding


def test_same_day_structured_and_note_times_are_not_compared() -> None:
    chart = _chart(
        documents=[_doc("nursing", 2)],
        vitals=[_bp_vital(120, 80, 2, at="10:00")],
        note_vitals=[_note_bp("nursing", 185, 100, at="08:30")],
    )

    issue = _single(check_blood_pressure(chart, POLICY))

    assert issue.computed["reading"] == "185/100"


def test_newer_day_wins_over_older_high_reading() -> None:
    chart = _chart(vitals=[_bp_vital(184, 112, 10), _bp_vital(123, 70, 3)])

    check = _single(check_blood_pressure(chart, POLICY))

    assert check.status == "passed"
    assert check.computed["reading"] == "123/70"


def test_note_reading_dated_by_its_own_text_not_its_document() -> None:
    # A note written 1 day out describes a reading from 10 days out; the structured
    # reading from 3 days out is newer.
    chart = _chart(
        documents=[_doc("summary", 1)],
        vitals=[_bp_vital(120, 80, 3)],
        note_vitals=[_note_bp("summary", 184, 112, measured_on=_on(10))],
    )

    assert _single(check_blood_pressure(chart, POLICY)).status == "passed"


def test_no_bp_anywhere_is_missing_data() -> None:
    issue = _single(check_blood_pressure(_chart(), POLICY))

    assert (issue.category, issue.reason) == ("MISSING_REQUIRED_DATA", "missing")


def test_different_values_at_the_same_time_is_a_conflict() -> None:
    chart = _chart(
        documents=[_doc("a", 1), _doc("b", 1)],
        note_vitals=[
            _note_bp("a", 120, 80, at="09:00"),
            _note_bp("b", 150, 95, at="09:00"),
        ],
    )

    conflict, result = check_blood_pressure(chart, POLICY)

    assert (conflict.category, conflict.reason) == ("MISSING_REQUIRED_DATA", "conflict")
    assert result.status == "passed"


def test_duplicate_reading_is_not_a_conflict() -> None:
    chart = _chart(
        documents=[_doc("a", 1), _doc("b", 1)],
        note_vitals=[
            _note_bp("a", 120, 80, at="09:00"),
            _note_bp("b", 120, 80, at="09:00"),
        ],
    )

    assert _single(check_blood_pressure(chart, POLICY)).status == "passed"


@pytest.mark.parametrize(("value_f", "held"), [(100.4, False), (100.5, True)])
def test_fever_threshold_is_exclusive(value_f: float, held: bool) -> None:
    finding = _single(
        check_temperature(_chart(vitals=[_temp_vital(value_f, 2)]), POLICY)
    )

    assert isinstance(finding, TriageIssue) is held


@pytest.mark.parametrize(
    ("celsius", "fahrenheit", "held"),
    [(38.6, "101.5 F", True), (37.9, "100.2 F", False), (38.0, "100.4 F", False)],
)
def test_celsius_note_readings_are_converted(
    celsius: float, fahrenheit: str, held: bool
) -> None:
    chart = _chart(
        documents=[_doc("nursing", 1)],
        note_vitals=[_note_temp("nursing", celsius, "C")],
    )

    finding = _single(check_temperature(chart, POLICY))

    assert finding.computed["reading"] == fahrenheit
    assert isinstance(finding, TriageIssue) is held


def test_newer_note_fever_overrides_older_normal_structured_temperature() -> None:
    chart = _chart(
        documents=[_doc("nursing", 1)],
        vitals=[_temp_vital(98.9, 3)],
        note_vitals=[_note_temp("nursing", 38.7, "C")],
    )

    issue = _single(check_temperature(chart, POLICY))

    assert issue.category == "ACUTE_SAFETY_EXCLUSION"
    assert [s.id for s in issue.sources] == ["nursing", "vitals[0]"]


def test_no_temperature_anywhere_is_missing_data() -> None:
    chart = _chart(vitals=[_bp_vital(120, 80, 2)])

    issue = _single(check_temperature(chart, POLICY))

    assert (issue.category, issue.reason) == ("MISSING_REQUIRED_DATA", "missing")


def test_temperature_without_unit_is_skipped_not_guessed() -> None:
    chart = _chart(
        documents=[_doc("nursing", 1)], note_vitals=[_note_temp("nursing", 37.1, None)]
    )

    issue = _single(check_temperature(chart, POLICY))

    assert issue.reason == "missing"
    assert issue.sources[0].id == "nursing"


# -------------------------
# Decision and engine
# -------------------------


def _issue_of(category: str) -> TriageIssue:
    return TriageIssue(
        category=category, reason="policy_failed", rule="hp", finding="x"
    )


@pytest.mark.parametrize(
    ("categories", "decision"),
    [
        ([], "READY"),
        (["REQUIRED_TESTING"], "NEEDS_FOLLOW_UP"),
        (["MISSING_REQUIRED_DATA"], "NEEDS_FOLLOW_UP"),
        (["REQUIRED_DOCUMENTATION", "ACUTE_SAFETY_EXCLUSION"], "NOT_CLEARED"),
    ],
)
def test_decide(categories: list[str], decision: str) -> None:
    assert decide([_issue_of(c) for c in categories]) == decision


def _ready_chart() -> Chart:
    return _chart(
        labs=[_lab("CBC", 5)],
        documents=[_doc("hp", 10), _doc("consent", 2)],
        vitals=[_bp_vital(120, 80, 2), _temp_vital(98.6, 2)],
        hp_documents=[_hp("hp")],
        consents=[_consent("consent")],
    )


def test_evaluate_rules_splits_issues_and_checks() -> None:
    issues, checks = evaluate_rules(_ready_chart())

    assert issues == []
    assert [check.rule for check in checks] == [
        "required_fields",
        "hp",
        "consent",
        "labs",
        "anticoagulation",
        "blood_pressure",
        "temperature",
    ]


@pytest.mark.parametrize(
    "chart",
    [
        _chart(),
        _chart(procedure={"procedure_date": None, "procedure_risk": None}),
        _ready_chart(),
        _chart(
            procedure={"procedure_risk": "HIGH"},
            medications=[_med("warfarin"), _med("Eliquis", active=None)],
            vitals=[_bp_vital(200, 120, 1)],
        ),
    ],
    ids=["empty", "no-date-or-risk", "ready", "high-risk-anticoagulated-hypertensive"],
)
def test_every_rule_reports_on_every_chart(chart: Chart) -> None:
    issues, checks = evaluate_rules(chart)

    reported = {finding.rule for finding in [*issues, *checks]}
    assert reported == set(get_args(RuleId)) - {"system"}
