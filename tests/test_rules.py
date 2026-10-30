from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest

from core import PatientSubmission, RuleCheck, TriageIssue
from facts import Consent, ExtractedFacts, HPDocument, ProcedureDateMention
from policy import POLICY
from rules import (
    Chart,
    Finding,
    check_consent,
    check_hp,
    check_labs,
    check_required_fields,
    days_before,
    evaluate_rules,
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
# Engine
# -------------------------


def test_evaluate_rules_splits_issues_and_checks() -> None:
    chart = _chart(
        labs=[_lab("CBC", 5)],
        documents=[_doc("hp", 40), _doc("consent", 2)],
        hp_documents=[_hp("hp")],
        consents=[_consent("consent")],
    )

    issues, checks = evaluate_rules(chart)

    assert [issue.rule for issue in issues] == ["hp"]
    assert [check.rule for check in checks] == ["required_fields", "consent", "labs"]
