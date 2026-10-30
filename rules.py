"""Policy rules for pre-op triage, applied in plain Python.

Rules read structured fields from the submission and verified facts from the
documents (see verify.py); they never call the LLM. Each rule returns issues for
what blocks scheduling and checks for what passed, did not apply, or could not be
evaluated, so every rule leaves evidence a reviewer can verify.

To add a rule, write a function with the Rule signature and list it in RULES.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Literal

from core import (
    BloodPressureVital,
    CheckStatus,
    ComputedValue,
    Decision,
    Document,
    IgnoredMedication,
    IssueCategory,
    IssueReason,
    LabResult,
    Medication,
    PatientSubmission,
    ProcedureInfo,
    RuleCheck,
    RuleId,
    Source,
    TemperatureVital,
    TriageIssue,
)
from facts import AnticoagPlan, ExtractedFacts, MedicationMention, NoteVital
from policy import (
    POLICY,
    LabRequirement,
    Policy,
    anticoagulant_generic,
    canonical_lab,
)


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


MedStatus = Literal["taking", "stopped", "uncertain", "conflict"]


def check_anticoagulation(chart: Chart, policy: Policy) -> list[Finding]:
    """Rule 3: each anticoagulant the patient takes needs a complete perioperative plan."""

    listed: defaultdict[str, list[tuple[int, Medication]]] = defaultdict(list)
    for index, med in enumerate(chart.submission.medications):
        if generic := anticoagulant_generic(med.name, policy):
            listed[generic].append((index, med))
    mentioned: defaultdict[str, list[MedicationMention]] = defaultdict(list)
    for mention in chart.facts.medication_mentions:
        if generic := anticoagulant_generic(mention.name, policy):
            mentioned[generic].append(mention)
    plans: defaultdict[str, list[AnticoagPlan]] = defaultdict(list)
    for plan in chart.facts.anticoag_plans:
        if generic := anticoagulant_generic(plan.drug, policy):
            plans[generic].append(plan)

    drugs = sorted(set(listed) | set(mentioned))
    if not drugs:
        finding = "No anticoagulant on the medication list or in the notes."
        return [_check("anticoagulation", "not_applicable", finding)]
    return [
        _check_anticoagulant(chart, drug, listed[drug], mentioned[drug], plans[drug])
        for drug in drugs
    ]


def _check_anticoagulant(
    chart: Chart,
    drug: str,
    listed: list[tuple[int, Medication]],
    mentioned: list[MedicationMention],
    plans: list[AnticoagPlan],
) -> Finding:
    status = _anticoagulant_status(chart, listed, mentioned)
    sources = [
        *(_med_source(index, med) for index, med in listed),
        *(_doc_source(chart, m.doc_id, m.quote) for m in mentioned),
    ]
    if status == "stopped":
        finding = f"{drug} is not currently taken."
        return _check("anticoagulation", "not_applicable", finding, sources, drug=drug)
    if status == "uncertain":
        finding = f"Unclear whether the patient currently takes {drug}."
        return _issue(
            "MISSING_REQUIRED_DATA",
            "anticoagulation",
            finding,
            sources,
            "missing",
            drug=drug,
        )
    if status == "conflict":
        finding = f"Sources disagree on whether the patient currently takes {drug}."
        return _issue(
            "MISSING_REQUIRED_DATA",
            "anticoagulation",
            finding,
            sources,
            "conflict",
            drug=drug,
        )

    # Complete plans first, so the evidence leads with the plan that satisfies the rule.
    plans = sorted(plans, key=lambda p: not (p.pre_op_quote and p.post_op_quote))
    plan_sources = [
        _doc_source(chart, plan.doc_id, quote)
        for plan in plans
        for quote in (plan.quote, plan.pre_op_quote, plan.post_op_quote)
        if quote
    ]
    if plans and plans[0].pre_op_quote and plans[0].post_op_quote:
        finding = f"Patient takes {drug}; a plan covers both before and after surgery."
        return _check(
            "anticoagulation", "passed", finding, [*plan_sources, *sources], drug=drug
        )

    if not plans:
        finding = f"Patient takes {drug}, but no perioperative plan is documented."
    else:
        missing = [
            label
            for label, attr in (
                ("before-surgery", "pre_op_quote"),
                ("after-surgery", "post_op_quote"),
            )
            if not any(getattr(plan, attr) for plan in plans)
        ]
        gap = (
            f"no {' or '.join(missing)} instruction"
            if missing
            else "no single plan covers both before and after surgery"
        )
        finding = f"Patient takes {drug}, but the plan is incomplete: {gap}."
    return _issue(
        "ANTICOAGULATION_MANAGEMENT",
        "anticoagulation",
        finding,
        [*plan_sources, *sources],
        drug=drug,
    )


def _anticoagulant_status(
    chart: Chart,
    listed: list[tuple[int, Medication]],
    mentioned: list[MedicationMention],
) -> MedStatus:
    """Combine the medication list with the most recent note that mentions the drug."""

    listed_status = _merge_statuses(
        {
            "taking"
            if med.active
            else "stopped"
            if med.active is False
            else "uncertain"
            for _, med in listed
        }
    )
    note_status = None
    if mentioned:
        latest = max(_doc_date(chart, m.doc_id) or date.min for m in mentioned)
        note_status = _merge_statuses(
            {
                m.status
                for m in mentioned
                if (_doc_date(chart, m.doc_id) or date.min) == latest
            }
        )

    if note_status == "uncertain":
        return "uncertain"
    if "conflict" in (note_status, listed_status):
        return "conflict"
    known = ("taking", "stopped")
    if note_status in known and listed_status in known and note_status != listed_status:
        return "conflict"
    if note_status in known:
        return note_status
    if listed_status in known:
        return listed_status
    return "uncertain"


def _merge_statuses(statuses: set[str]) -> MedStatus | None:
    if not statuses:
        return None
    if {"taking", "stopped"} <= statuses:
        return "conflict"
    if "uncertain" in statuses:
        return "uncertain"
    return "taking" if "taking" in statuses else "stopped"


def _med_source(index: int, med: Medication) -> Source:
    return Source(
        kind="medication",
        id=f"medications[{index}]",
        quote=f"{med.name} (active: {med.active})",
    )


def ignored_structured_medications(
    chart: Chart, policy: Policy = POLICY
) -> list[IgnoredMedication]:
    """Medication-list entries no rule acts on, so a reviewer can see they were seen."""

    return [
        IgnoredMedication(
            name=med.name or "unnamed medication",
            reason="not an anticoagulant",
            quote_verified=True,
            source=_med_source(index, med),
        )
        for index, med in enumerate(chart.submission.medications)
        if anticoagulant_generic(med.name, policy) is None
    ]


@dataclass(frozen=True)
class _Reading:
    day: date
    at: time | None
    # Structured timestamps are UTC; note times are local and unzoned, so times are
    # only compared between readings on the same clock.
    clock: Literal["utc", "note"]
    values: tuple[float, ...]  # (systolic, diastolic) or (temperature_f,)
    source: Source


def check_blood_pressure(chart: Chart, policy: Policy) -> list[Finding]:
    """Rule 4: the most recent blood pressure must be below the hold thresholds."""

    readings: list[_Reading] = []
    skipped: list[Source] = []
    for index, vital in enumerate(chart.submission.vitals):
        if not isinstance(vital, BloodPressureVital):
            continue
        source = Source(
            kind="vital",
            id=f"vitals[{index}]",
            date=vital.date,
            quote=f"BP {vital.systolic}/{vital.diastolic} ({vital.source})",
        )
        day, at = _parse_timestamp(vital.date)
        if day is None or vital.systolic is None or vital.diastolic is None:
            skipped.append(source)
        else:
            values = (float(vital.systolic), float(vital.diastolic))
            readings.append(_Reading(day, at, "utc", values, source))
    for note in chart.facts.note_vitals:
        if note.kind != "blood_pressure":
            continue
        reading = _note_reading(chart, note, (note.systolic, note.diastolic))
        if reading is None:
            skipped.append(_doc_source(chart, note.doc_id, note.quote))
        else:
            readings.append(reading)

    def exceeds(reading: _Reading) -> bool:
        systolic, diastolic = reading.values
        return (
            systolic >= policy.systolic_hold_at or diastolic >= policy.diastolic_hold_at
        )

    def severity(reading: _Reading) -> float:
        systolic, diastolic = reading.values
        return max(
            systolic / policy.systolic_hold_at, diastolic / policy.diastolic_hold_at
        )

    thresholds = (
        f"hold at systolic >= {policy.systolic_hold_at:g} "
        f"or diastolic >= {policy.diastolic_hold_at:g}"
    )
    return _check_vital(
        "blood_pressure",
        "blood pressure",
        readings,
        skipped,
        exceeds,
        severity,
        lambda r: f"{r.values[0]:g}/{r.values[1]:g}",
        thresholds,
    )


def check_temperature(chart: Chart, policy: Policy) -> list[Finding]:
    """Rule 4: the most recent temperature must not exceed the fever threshold."""

    readings: list[_Reading] = []
    skipped: list[Source] = []
    for index, vital in enumerate(chart.submission.vitals):
        if not isinstance(vital, TemperatureVital):
            continue
        source = Source(
            kind="vital",
            id=f"vitals[{index}]",
            date=vital.date,
            quote=f"Temp {vital.value_f} F ({vital.source})",
        )
        day, at = _parse_timestamp(vital.date)
        if day is None or vital.value_f is None:
            skipped.append(source)
        else:
            readings.append(_Reading(day, at, "utc", (float(vital.value_f),), source))
    for note in chart.facts.note_vitals:
        if note.kind != "temperature":
            continue
        temperature_f = _to_fahrenheit(note.temperature, note.temperature_unit)
        reading = _note_reading(chart, note, (temperature_f,))
        if reading is None:
            skipped.append(_doc_source(chart, note.doc_id, note.quote))
        else:
            readings.append(reading)

    return _check_vital(
        "temperature",
        "temperature",
        readings,
        skipped,
        lambda r: r.values[0] > policy.temp_f_hold_above,
        lambda r: r.values[0],
        lambda r: f"{r.values[0]:g} F",
        f"hold above {policy.temp_f_hold_above:g} F",
    )


def _to_fahrenheit(value: float | None, unit: str | None) -> float | None:
    if value is None or unit is None:
        return None
    # Rounded so 38.0 C is exactly 100.4 F rather than a float just above or below it.
    return round(value * 9 / 5 + 32, 1) if unit == "C" else value


def _parse_timestamp(value: str | None) -> tuple[date | None, time | None]:
    """Split "2026-02-24T10:12:00Z" into its date and time; a bare date has no time."""

    day = parse_date(value)
    if day is None or value is None or "T" not in value:
        return day, None
    try:
        return day, datetime.fromisoformat(value.strip()).time()
    except ValueError:
        return day, None


def _note_reading(
    chart: Chart, note: NoteVital, values: tuple[float | None, ...]
) -> _Reading | None:
    """A note vital dated by its own text if stated, else by its document."""

    day = parse_date(note.measured_on) or _doc_date(chart, note.doc_id)
    complete = tuple(float(value) for value in values if value is not None)
    if day is None or len(complete) != len(values):
        return None
    try:
        at = time.fromisoformat(note.measured_time) if note.measured_time else None
    except ValueError:
        at = None
    return _Reading(
        day, at, "note", complete, _doc_source(chart, note.doc_id, note.quote)
    )


def _check_vital(
    rule: RuleId,
    label: str,
    readings: list[_Reading],
    skipped: list[Source],
    exceeds: Callable[[_Reading], bool],
    severity: Callable[[_Reading], float],
    describe: Callable[[_Reading], str],
    thresholds: str,
) -> list[Finding]:
    """Pick the most recent reading (most severe when same-day order is unknowable)."""

    if not readings:
        finding = f"No usable {label} reading in the structured vitals or notes."
        return [_issue("MISSING_REQUIRED_DATA", rule, finding, skipped, "missing")]

    latest_day = max(r.day for r in readings)
    same_day = [r for r in readings if r.day == latest_day]
    orderable = (
        all(r.at is not None for r in same_day)
        and len({r.clock for r in same_day}) == 1
    )
    if orderable:
        latest_at = max(r.at for r in same_day if r.at is not None)
        candidates = [r for r in same_day if r.at == latest_at]
    else:
        candidates = same_day
    chosen = max(candidates, key=severity)
    others = sorted(
        (r for r in readings if r is not chosen),
        key=lambda r: (r.day, r.at or time.min),
        reverse=True,
    )
    sources = [chosen.source, *(r.source for r in others), *skipped]

    findings: list[Finding] = []
    by_moment: defaultdict[tuple[time, str], set[tuple[float, ...]]] = defaultdict(set)
    for r in same_day:
        if r.at is not None:
            by_moment[(r.at, r.clock)].add(r.values)
    if any(len(values) > 1 for values in by_moment.values()):
        finding = (
            f"Different {label} values are recorded for the same time on {latest_day}."
        )
        findings.append(
            _issue(
                "MISSING_REQUIRED_DATA",
                rule,
                finding,
                [r.source for r in same_day],
                "conflict",
            )
        )

    when = f"{chosen.day}" + (f" {chosen.at:%H:%M}" if chosen.at else "")
    how = (
        ""
        if orderable or len(same_day) == 1
        else " Same-day readings could not be put in order, so the most severe was used."
    )
    computed: dict[str, ComputedValue] = {
        "reading": describe(chosen),
        "measured_on": chosen.day.isoformat(),
        "thresholds": thresholds,
    }
    reading = f"Most recent {label} {describe(chosen)} ({when})"
    if exceeds(chosen):
        finding = f"{reading} meets the {thresholds}.{how}"
        findings.append(
            _issue("ACUTE_SAFETY_EXCLUSION", rule, finding, sources, **computed)
        )
    else:
        finding = f"{reading} is within limits ({thresholds}).{how}"
        findings.append(_check(rule, "passed", finding, sources, **computed))
    return findings


def decide(issues: Sequence[TriageIssue]) -> Decision:
    """Derive the decision from issue categories alone."""

    if any(issue.category == "ACUTE_SAFETY_EXCLUSION" for issue in issues):
        return "NOT_CLEARED"
    return "NEEDS_FOLLOW_UP" if issues else "READY"


# -------------------------
# Engine
# -------------------------

RULES: tuple[Rule, ...] = (
    check_required_fields,
    check_hp,
    check_consent,
    check_labs,
    check_anticoagulation,
    check_blood_pressure,
    check_temperature,
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
