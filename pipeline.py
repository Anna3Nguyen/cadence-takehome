"""The triage pipeline: extract facts, verify them, apply the rules, decide.

It fails safe. Invalid input, an extraction failure, or a bug in this code yields a
"system" issue explaining what went wrong, so the decision is at least
NEEDS_FOLLOW_UP. No failure path returns READY or raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

from pydantic import ValidationError

from core import (
    Decision,
    IgnoredMedication,
    IssueReason,
    PatientSubmission,
    RuleCheck,
    RuleId,
    TriageIssue,
    TriageOutput,
)
from extraction import DEFAULT_CACHE_DIR, ExtractionError, extract_facts
from facts import ExtractedFacts
from rules import Chart, decide, evaluate_rules, ignored_structured_medications
from verify import verify_facts

if TYPE_CHECKING:
    from openai import OpenAI

# Rules that depend on document text. If extraction fails, their results from
# structured data alone would claim more than is known (a note may hold a newer
# reading or a plan), so they are reported as not evaluated. A structured reading
# that already meets a safety exclusion is kept: it is evidence of danger either way.
DOCUMENT_RULES: frozenset[RuleId] = frozenset(
    {"hp", "consent", "anticoagulation", "blood_pressure", "temperature"}
)


def triage(
    submission: dict[str, Any] | PatientSubmission,
    *,
    model: str,
    client: OpenAI | None = None,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
) -> TriageOutput:
    """Triage one submission package. Never raises and never fails open."""

    try:
        parsed = PatientSubmission.model_validate(submission)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(map(str, error['loc'])) or 'submission'}: {error['msg']}"
            for error in exc.errors()[:5]
        )
        finding = f"Submission does not match the expected format ({problems})."
        return _build_output([_system_issue("invalid_submission", finding)], [], [])

    try:
        return _triage_parsed(parsed, model=model, client=client, cache_dir=cache_dir)
    except Exception as exc:  # Last resort: a bug must never crash or clear a case.
        finding = f"Triage stopped on an internal error ({type(exc).__name__}: {exc})."
        return _build_output([_system_issue("internal_error", finding)], [], [])


def _triage_parsed(
    submission: PatientSubmission,
    *,
    model: str,
    client: OpenAI | None,
    cache_dir: Path | None,
) -> TriageOutput:
    try:
        raw = extract_facts(submission, model=model, client=client, cache_dir=cache_dir)
    except ExtractionError as exc:
        return _triage_without_documents(submission, exc)

    verified = verify_facts(raw, submission)
    chart = Chart(submission, verified.facts)
    issues, checks = evaluate_rules(chart)
    return _build_output(
        [*verified.issues, *issues],
        checks,
        [*verified.ignored_medications, *ignored_structured_medications(chart)],
    )


def _triage_without_documents(
    submission: PatientSubmission, error: ExtractionError
) -> TriageOutput:
    empty = ExtractedFacts(**{name: [] for name in ExtractedFacts.model_fields})
    chart = Chart(submission, empty)
    issues, checks = evaluate_rules(chart)

    issues = [
        issue
        for issue in issues
        if issue.rule not in DOCUMENT_RULES
        or issue.category == "ACUTE_SAFETY_EXCLUSION"
    ]
    reported = {issue.rule for issue in issues}
    checks = [check for check in checks if check.rule not in DOCUMENT_RULES]
    checks += [
        RuleCheck(
            rule=rule,
            status="not_evaluated",
            finding="Not evaluated: facts could not be extracted from the documents.",
        )
        for rule in get_args(RuleId)
        if rule in DOCUMENT_RULES and rule not in reported
    ]
    finding = (
        "Facts could not be extracted from the documents, so rules that depend on "
        f"document text were not evaluated. {error}"
    )
    return _build_output(
        [_system_issue("extraction_failed", finding), *issues],
        checks,
        ignored_structured_medications(chart),
    )


def _system_issue(reason: IssueReason, finding: str) -> TriageIssue:
    return TriageIssue(
        category="MISSING_REQUIRED_DATA", reason=reason, rule="system", finding=finding
    )


def _build_output(
    issues: list[TriageIssue],
    checks: list[RuleCheck],
    ignored_medications: list[IgnoredMedication],
) -> TriageOutput:
    decision = decide(issues)
    return TriageOutput(
        decision=decision,
        issues=issues,
        checks=checks,
        ignored_medications=ignored_medications,
        explanation=explain(decision, issues, checks),
    )


def explain(
    decision: Decision, issues: list[TriageIssue], checks: list[RuleCheck]
) -> str:
    """Summarize the issues and checks; generated from them, so it cannot contradict them."""

    if issues:
        parts = [f"{decision}: {len(issues)} blocking issue(s)."]
        parts += [f"{issue.category}: {issue.finding}" for issue in issues]
    else:
        parts = [f"{decision}: no blocking issues."]
    counts = {status: 0 for status in ("passed", "not_applicable", "not_evaluated")}
    for check in checks:
        counts[check.status] += 1
    parts.append(
        f"Checks: {counts['passed']} passed, {counts['not_applicable']} not applicable, "
        f"{counts['not_evaluated']} not evaluated."
    )
    return " ".join(parts)
