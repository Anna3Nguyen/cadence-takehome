"""Policy rules for pre-op triage, applied in plain Python.

Rules read structured fields from the submission and verified facts from the
documents (see verify.py); they never call the LLM. Each rule returns issues for
what blocks scheduling and checks for what passed, did not apply, or could not be
evaluated, so every rule leaves evidence a reviewer can verify.

To add a rule, write a function with the Rule signature and list it in RULES.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

from core import (
    CheckStatus,
    ComputedValue,
    Document,
    IssueCategory,
    IssueReason,
    LabResult,
    PatientSubmission,
    ProcedureInfo,
    RuleCheck,
    RuleId,
    Source,
    TriageIssue,
)
from facts import ExtractedFacts
from policy import POLICY, LabRequirement, Policy, canonical_lab


@dataclass(frozen=True)
class Chart:
    """Everything a rule may look at: the submission plus verified extracted facts."""

    submission: PatientSubmission
    facts: ExtractedFacts

    @property
    def procedure(self) -> ProcedureInfo:
        return self.submission.procedure or ProcedureInfo()

    @property
    def procedure_date(self) -> date | None:
        return parse_date(self.procedure.procedure_date)

    @property
    def documents(self) -> dict[str, Document]:
        return {doc.doc_id: doc for doc in self.submission.documents if doc.doc_id}


Finding = TriageIssue | RuleCheck
Rule = Callable[[Chart, Policy], list[Finding]]


# -------------------------
# Helpers
# -------------------------


def parse_date(value: str | None) -> date | None:
    """Parse "2026-02-11" or "2026-02-11T11:29:00Z"; None if missing or unreadable."""

    if not value:
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def days_before(procedure_date: date, event_date: date) -> int:
    return (procedure_date - event_date).days


def _timing(days: int) -> str:
    if days < 0:
        return f"{-days} days after the procedure"
    return f"{days} days before the procedure"


def _issue(
    category: IssueCategory,
    rule: RuleId,
    finding: str,
    sources: Sequence[Source] = (),
    reason: IssueReason = "policy_failed",
    **computed: ComputedValue,
) -> TriageIssue:
    return TriageIssue(
        category=category,
        reason=reason,
        rule=rule,
        finding=finding,
        sources=list(sources),
        computed=computed,
    )


def _check(
    rule: RuleId,
    status: CheckStatus,
    finding: str,
    sources: Sequence[Source] = (),
    **computed: ComputedValue,
) -> RuleCheck:
    return RuleCheck(
        rule=rule,
        status=status,
        finding=finding,
        sources=list(sources),
        computed=computed,
    )


def _field_source(path: str, value: object) -> Source:
    return Source(kind="field", id=path, quote="null" if value is None else str(value))


def _doc_source(chart: Chart, doc_id: str, quote: str) -> Source:
    doc = chart.documents.get(doc_id)
    return Source(
        kind="document", id=doc_id, date=doc.date if doc else None, quote=quote
    )


def _doc_date(chart: Chart, doc_id: str) -> date | None:
    doc = chart.documents.get(doc_id)
    return parse_date(doc.date) if doc else None


def _lab_source(lab: LabResult, index: int) -> Source:
    return Source(
        kind="lab",
        id=lab.id or f"labs[{index}]",
        date=lab.effective_at,
        quote=f"{lab.code} - {lab.display} ({lab.status})",
    )


# -------------------------
# Rules
# -------------------------


def check_required_fields(chart: Chart, policy: Policy) -> list[Finding]:
    """Procedure date and risk must be present, and notes must not contradict the date."""

    procedure = chart.procedure
    date_source = _field_source("procedure.procedure_date", procedure.procedure_date)
    risk_source = _field_source("procedure.procedure_risk", procedure.procedure_risk)
    mentions = chart.facts.procedure_date_mentions
    findings: list[Finding] = []

    if chart.procedure_date is None:
        findings.append(
            _issue(
                "MISSING_REQUIRED_DATA",
                "required_fields",
                "Procedure date is missing or unreadable, so date-window checks cannot run.",
                [
                    date_source,
                    *(_doc_source(chart, m.doc_id, m.quote) for m in mentions),
                ],
                reason="missing",
            )
        )
    else:
        conflicting = [
            m for m in mentions if parse_date(m.date) != chart.procedure_date
        ]
        if conflicting:
            stated = ", ".join(sorted({m.date for m in conflicting}))
            findings.append(
                _issue(
                    "MISSING_REQUIRED_DATA",
                    "required_fields",
                    f"Procedure date {chart.procedure_date} conflicts with notes stating "
                    f"{stated}.",
                    [
                        date_source,
                        *(_doc_source(chart, m.doc_id, m.quote) for m in conflicting),
                    ],
                    reason="conflict",
                )
            )

    if procedure.procedure_risk is None:
        findings.append(
            _issue(
                "MISSING_REQUIRED_DATA",
                "required_fields",
                "Procedure risk is missing, so required tests cannot be determined.",
                [risk_source],
                reason="missing",
            )
        )

    if not findings:
        findings.append(
            _check(
                "required_fields",
                "passed",
                f"Procedure date {chart.procedure_date} and risk "
                f"{procedure.procedure_risk} are recorded.",
                [
                    date_source,
                    risk_source,
                    *(_doc_source(chart, m.doc_id, m.quote) for m in mentions),
                ],
            )
        )
    return findings


def check_hp(chart: Chart, policy: Policy) -> list[Finding]:
    """Rule 1: a pre-op H&P must be completed within the window before the procedure."""

    hp_documents = chart.facts.hp_documents
    preop = [hp for hp in hp_documents if hp.is_preop_hp]
    if not preop:
        others = [_doc_source(chart, hp.doc_id, hp.quote) for hp in hp_documents]
        note = (
            f" {len(others)} other H&P(s) found, but none is a pre-op evaluation."
            if others
            else ""
        )
        return [
            _issue(
                "REQUIRED_DOCUMENTATION", "hp", f"No pre-op H&P found.{note}", others
            )
        ]

    dated = [(d, hp) for hp in preop if (d := _doc_date(chart, hp.doc_id)) is not None]
    if not dated:
        return [
            _issue(
                "MISSING_REQUIRED_DATA",
                "hp",
                "Pre-op H&P found, but its document date is missing.",
                [_doc_source(chart, hp.doc_id, hp.quote) for hp in preop],
                reason="missing",
            )
        ]

    hp_date, latest = max(dated, key=lambda pair: pair[0])
    source = _doc_source(chart, latest.doc_id, latest.quote)
    if chart.procedure_date is None:
        return [
            _check(
                "hp",
                "not_evaluated",
                f"Pre-op H&P dated {hp_date} is on file; its window cannot be checked "
                "without a procedure date.",
                [source],
            )
        ]

    days = days_before(chart.procedure_date, hp_date)
    limit = policy.hp_max_days
    computed: dict[str, ComputedValue] = {
        "hp_date": hp_date.isoformat(),
        "procedure_date": chart.procedure_date.isoformat(),
        "days_before": days,
        "limit_days": limit,
    }
    if 0 <= days <= limit:
        finding = f"Pre-op H&P is {_timing(days)} (limit {limit})."
        return [_check("hp", "passed", finding, [source], **computed)]
    finding = f"Most recent pre-op H&P is {_timing(days)} (limit {limit} days before)."
    return [_issue("REQUIRED_DOCUMENTATION", "hp", finding, [source], **computed)]


def check_consent(chart: Chart, policy: Policy) -> list[Finding]:
    """Rule 1: the most recent consent must be signed by the patient for this procedure."""

    consents = chart.facts.consents
    if not consents:
        return [
            _issue("REQUIRED_DOCUMENTATION", "consent", "No surgical consent found.")
        ]

    # Newest document wins; on the same date a valid consent wins, so the result
    # does not depend on the order the model listed them.
    ordered = sorted(
        consents,
        key=lambda c: (
            _doc_date(chart, c.doc_id) or date.min,
            c.patient_signed and c.matches_scheduled_procedure,
        ),
    )
    newest = ordered[-1]
    sources = [_doc_source(chart, c.doc_id, c.quote) for c in reversed(ordered)]
    consent_date = _doc_date(chart, newest.doc_id)
    label = f"Most recent consent ({consent_date or 'undated'})"

    if newest.patient_signed and newest.matches_scheduled_procedure:
        earlier = len(ordered) - 1
        note = f" It supersedes {earlier} earlier consent(s)." if earlier else ""
        finding = f"{label} is signed by the patient for the scheduled procedure.{note}"
        return [_check("consent", "passed", finding, sources)]

    problems = []
    if not newest.patient_signed:
        problems.append("is not signed by the patient")
    if not newest.matches_scheduled_procedure:
        problems.append(
            f"names {newest.procedure_named or 'no procedure'}, not the scheduled "
            f"procedure ({chart.procedure.procedure_type or 'unknown'})"
        )
    finding = f"{label} {' and '.join(problems)}."
    return [_issue("REQUIRED_DOCUMENTATION", "consent", finding, sources)]


def check_labs(chart: Chart, policy: Policy) -> list[Finding]:
    """Rule 2: the most recent result of each required test must be within its window."""

    risk = chart.procedure.procedure_risk
    if risk is None:
        return [
            _check(
                "labs",
                "not_evaluated",
                "Required tests depend on procedure risk, which is missing.",
            )
        ]
    return [
        _check_lab(chart, requirement, risk, policy)
        for requirement in policy.required_labs[risk]
    ]


def _check_lab(
    chart: Chart, requirement: LabRequirement, risk: str, policy: Policy
) -> Finding:
    test, limit = requirement.test, requirement.max_days
    usable: list[tuple[date, int, LabResult]] = []
    skipped: list[Source] = []
    for index, lab in enumerate(chart.submission.labs):
        if canonical_lab(lab.code, policy) != test:
            continue
        lab_date = parse_date(lab.effective_at)
        if (
            lab.status or ""
        ).lower() in policy.excluded_lab_statuses or lab_date is None:
            skipped.append(_lab_source(lab, index))
        else:
            usable.append((lab_date, index, lab))

    skipped_note = (
        f" Skipped {len(skipped)} {test} result(s) that were excluded or undated."
        if skipped
        else ""
    )
    requirement_text = f"{risk} risk requires {test} within {limit} days"
    if not usable:
        finding = f"No usable {test} result on file ({requirement_text}).{skipped_note}"
        return _issue("REQUIRED_TESTING", "labs", finding, skipped, test=test)

    lab_date, index, latest = max(usable, key=lambda item: (item[0], item[1]))
    sources = [_lab_source(latest, index), *skipped]
    if chart.procedure_date is None:
        finding = (
            f"{test} dated {lab_date} is on file; its window cannot be checked "
            f"without a procedure date.{skipped_note}"
        )
        return _check("labs", "not_evaluated", finding, sources, test=test)

    days = days_before(chart.procedure_date, lab_date)
    computed: dict[str, ComputedValue] = {
        "test": test,
        "result_date": lab_date.isoformat(),
        "procedure_date": chart.procedure_date.isoformat(),
        "days_before": days,
        "limit_days": limit,
    }
    if 0 <= days <= limit:
        finding = f"{test} is {_timing(days)} ({requirement_text}).{skipped_note}"
        return _check("labs", "passed", finding, sources, **computed)
    finding = (
        f"Most recent {test} is {_timing(days)} ({requirement_text}).{skipped_note}"
    )
    return _issue("REQUIRED_TESTING", "labs", finding, sources, **computed)


# -------------------------
# Engine
# -------------------------

RULES: tuple[Rule, ...] = (
    check_required_fields,
    check_hp,
    check_consent,
    check_labs,
)


def evaluate_rules(
    chart: Chart, policy: Policy = POLICY
) -> tuple[list[TriageIssue], list[RuleCheck]]:
    """Run every rule and split the results into blocking issues and checks."""

    issues: list[TriageIssue] = []
    checks: list[RuleCheck] = []
    for rule in RULES:
        for finding in rule(chart, policy):
            if isinstance(finding, TriageIssue):
                issues.append(finding)
            else:
                checks.append(finding)
    return issues, checks
