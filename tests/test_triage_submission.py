from __future__ import annotations

from types import SimpleNamespace
from typing import Any, get_args
from unittest.mock import Mock

import pytest

import pipeline
from core import RuleId, TriageOutput, triage_submission
from facts import (
    Consent,
    ExtractedFacts,
    HPDocument,
    MedicationMention,
    NoteVital,
)
from pipeline import DOCUMENT_RULES, triage

ALL_RULES = set(get_args(RuleId)) - {"system"}


def _submission(**overrides: Any) -> dict[str, Any]:
    submission: dict[str, Any] = {
        "patient": {"id": "patient-1", "name": {"given": "Grace", "family": "Lopez"}},
        "procedure": {
            "procedure_type": "Elective carpal tunnel release",
            "procedure_risk": "LOW",
            "procedure_date": "2026-03-01",
        },
        "vitals": [
            {
                "type": "blood_pressure",
                "systolic": 130,
                "diastolic": 79,
                "date": "2026-02-17T09:05:00Z",
                "source": "Pre-op clinic",
            }
        ],
        "labs": [
            {
                "id": "lab-cbc",
                "code": "CBC",
                "display": "Complete Blood Count",
                "effective_at": "2026-02-11T11:29:00Z",
                "status": "final",
            }
        ],
        "medications": [{"name": "tamsulosin 0.4 mg PO nightly", "active": True}],
        "documents": [
            {
                "doc_id": "hp-1",
                "type": "History and Pyhsical",
                "date": "2026-02-21",
                "text": "HISTORY AND PHYSICAL\nPre-op evaluation complete for carpal tunnel release.",
            },
            {
                "doc_id": "consent-1",
                "type": "Surgery Consent (scanned)",
                "date": "2026-02-27",
                "text": "Procedure: carpal tunnel release\nPatient signature: /s/ Grace Lopez",
            },
            {
                "doc_id": "nursing-1",
                "type": "Pre-op Nursing Intake",
                "date": "2026-02-26",
                "text": "10:15 BP 119/73, T 97.9 F.\nTakes acetaminophen 500 mg PRN.",
            },
        ],
    }
    submission.update(overrides)
    return submission


def _facts(**overrides: Any) -> ExtractedFacts:
    facts: dict[str, Any] = {
        "hp_documents": [
            HPDocument(
                doc_id="hp-1",
                line=2,
                is_preop_hp=True,
            )
        ],
        "consents": [
            Consent(
                doc_id="consent-1",
                line=2,
                patient_signed=True,
                procedure_named="carpal tunnel release",
                matches_scheduled_procedure=True,
            )
        ],
        "medication_mentions": [
            MedicationMention(
                doc_id="nursing-1",
                line=2,
                name="acetaminophen 500 mg PRN",
                status="taking",
            )
        ],
        "anticoag_plans": [],
        "note_vitals": [
            NoteVital(
                doc_id="nursing-1",
                line=1,
                kind="blood_pressure",
                systolic=119,
                diastolic=73,
                temperature=None,
                temperature_unit=None,
                measured_on=None,
                measured_time="10:15",
            ),
            NoteVital(
                doc_id="nursing-1",
                line=1,
                kind="temperature",
                systolic=None,
                diastolic=None,
                temperature=97.9,
                temperature_unit="F",
                measured_on=None,
                measured_time="10:15",
            ),
        ],
        "procedure_date_mentions": [],
    }
    facts.update(overrides)
    return ExtractedFacts(**facts)


def _client(
    facts: ExtractedFacts | None = None, error: Exception | None = None
) -> Mock:
    client = Mock()
    if error is not None:
        client.responses.parse.side_effect = error
    else:
        client.responses.parse.return_value = SimpleNamespace(output_parsed=facts)
    return client


def _run(submission: Any, client: Mock) -> TriageOutput:
    return triage(submission, model="test-model", client=client, cache_dir=None)


def _reported_rules(output: TriageOutput) -> set[str]:
    return {finding.rule for finding in [*output.issues, *output.checks]}


# -------------------------
# End to end
# -------------------------


def test_clean_chart_is_ready_with_every_rule_checked() -> None:
    output = _run(_submission(), _client(_facts()))

    assert output.decision == "READY"
    assert output.issues == []
    assert {check.rule for check in output.checks} == ALL_RULES
    assert all(check.status != "not_evaluated" for check in output.checks)
    assert {m.name for m in output.ignored_medications} == {
        "acetaminophen 500 mg PRN",
        "tamsulosin 0.4 mg PO nightly",
    }
    assert output.explanation.startswith("READY: no blocking issues.")


def test_decision_and_explanation_come_from_rule_issues() -> None:
    unsigned = Consent(
        doc_id="consent-1",
        line=1,
        patient_signed=False,
        procedure_named="carpal tunnel release",
        matches_scheduled_procedure=True,
    )

    output = _run(_submission(), _client(_facts(consents=[unsigned])))

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert [issue.category for issue in output.issues] == ["REQUIRED_DOCUMENTATION"]
    assert output.issues[0].finding in output.explanation


def test_triage_submission_runs_the_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    sentinel = Mock(spec=TriageOutput)
    monkeypatch.setattr(
        pipeline, "triage", lambda submission, **kw: calls.append(kw) or sentinel
    )

    assert triage_submission(_submission(), model="m") is sentinel
    assert calls == [{"model": "m"}]


# -------------------------
# Fail-safe
# -------------------------


@pytest.mark.parametrize(
    "bad_input",
    [
        _submission(procedure={"procedure_risk": "EXTREME"}),
        _submission(documents="not a list"),
        "not a submission",
        None,
    ],
    ids=["unknown-risk", "documents-not-a-list", "string", "none"],
)
def test_invalid_submission_needs_follow_up_without_calling_the_model(
    bad_input: Any,
) -> None:
    client = _client(_facts())

    output = _run(bad_input, client)

    assert output.decision == "NEEDS_FOLLOW_UP"
    [issue] = output.issues
    assert (issue.rule, issue.reason) == ("system", "invalid_submission")
    client.responses.parse.assert_not_called()


@pytest.mark.parametrize(
    "client",
    [
        _client(error=TimeoutError("request timed out")),
        _client(facts=None),
    ],
    ids=["api-error", "refusal"],
)
def test_extraction_failure_needs_follow_up_and_skips_document_rules(
    client: Mock,
) -> None:
    output = _run(_submission(), client)

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert (output.issues[0].rule, output.issues[0].reason) == (
        "system",
        "extraction_failed",
    )
    not_evaluated = {c.rule for c in output.checks if c.status == "not_evaluated"}
    assert not_evaluated == DOCUMENT_RULES
    assert {c.rule for c in output.checks if c.status == "passed"} == {
        "required_fields",
        "labs",
    }
    assert _reported_rules(output) == ALL_RULES | {"system"}


def test_extraction_failure_keeps_a_structured_safety_exclusion() -> None:
    hypertensive = _submission(
        vitals=[
            {
                "type": "blood_pressure",
                "systolic": 200,
                "diastolic": 120,
                "date": "2026-02-27T09:05:00Z",
                "source": "Pre-op clinic",
            }
        ]
    )

    output = _run(hypertensive, _client(error=TimeoutError("request timed out")))

    assert output.decision == "NOT_CLEARED"
    assert {issue.category for issue in output.issues} == {
        "MISSING_REQUIRED_DATA",
        "ACUTE_SAFETY_EXCLUSION",
    }
    assert "blood_pressure" not in {c.rule for c in output.checks}


def test_citation_to_a_missing_line_needs_follow_up() -> None:
    invented = HPDocument(doc_id="hp-1", line=9, is_preop_hp=True)

    output = _run(_submission(), _client(_facts(hp_documents=[invented])))

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert "unverified_quote" in {issue.reason for issue in output.issues}


def test_quotes_in_the_output_are_copied_from_the_cited_lines() -> None:
    output = _run(_submission(), _client(_facts()))

    hp_check = next(check for check in output.checks if check.rule == "hp")
    assert hp_check.sources[0].quote == (
        "Pre-op evaluation complete for carpal tunnel release."
    )


def test_internal_error_needs_follow_up_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_rules(*args: Any, **kwargs: Any) -> None:
        raise KeyError("boom")

    monkeypatch.setattr(pipeline, "evaluate_rules", broken_rules)

    output = _run(_submission(), _client(_facts()))

    assert output.decision == "NEEDS_FOLLOW_UP"
    [issue] = output.issues
    assert (issue.rule, issue.reason) == ("system", "internal_error")
    assert "KeyError" in issue.finding
