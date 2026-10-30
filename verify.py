"""Verify extracted facts against the document lines they cite.

The model cites a document and line number for each fact; it never writes quote
text. A fact reaches the rules only if the cited line exists and is not blank, and
the values the fact claims appear on that line. The line's text is copied into the
fact's quote, so every quote a reviewer sees is exact document text. A fact that
fails is discarded and reported as a MISSING_REQUIRED_DATA issue, so an unverifiable
fact can never help a case reach READY.

Medication facts about non-anticoagulants are not acted on by any rule. They are
listed as ignored medications instead of being verified into issues, so an
irrelevant drug cannot force follow-up and a reviewer can still see it.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TypeVar

from core import (
    Document,
    IgnoredMedication,
    PatientSubmission,
    RuleId,
    Source,
    TriageIssue,
)
from facts import (
    AnticoagPlan,
    ExtractedFacts,
    Fact,
    MedicationMention,
    NoteVital,
)
from policy import POLICY, Policy, anticoagulant_generic, anticoagulants_named

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
F = TypeVar("F", bound=Fact)


@dataclass
class VerificationResult:
    facts: ExtractedFacts
    issues: list[TriageIssue]
    ignored_medications: list[IgnoredMedication]


def verify_facts(
    facts: ExtractedFacts,
    submission: PatientSubmission,
    policy: Policy = POLICY,
) -> VerificationResult:
    """Keep only facts whose cited lines exist and state the claimed values."""

    docs = {doc.doc_id: doc for doc in submission.documents if doc.doc_id}
    issues: list[TriageIssue] = []
    ignored: list[IgnoredMedication] = []

    def verified(fact: F, rule: RuleId, what: str) -> F | None:
        """Return the fact with its quote filled in, or None after recording why not."""

        doc = docs.get(fact.doc_id)
        text, problem = _cited_line(doc, fact.line)
        if text is not None:
            fact = fact.model_copy(update={"quote": text})
            problem = _value_problem(fact, policy)
        if problem:
            issues.append(
                _unverified_issue(
                    rule, what, fact.doc_id, fact.line, doc, text, problem
                )
            )
            return None
        return fact

    def ignore(name: str, reason: str, fact: Fact) -> None:
        doc = docs.get(fact.doc_id)
        text, problem = _cited_line(doc, fact.line)
        ignored.append(
            IgnoredMedication(
                name=name,
                reason=reason,
                quote_verified=problem is None,
                source=_source(fact.doc_id, fact.line, doc, text),
            )
        )

    medication_mentions: list[MedicationMention] = []
    for mention in facts.medication_mentions:
        if anticoagulant_generic(mention.name, policy) is None:
            ignore(mention.name, "not an anticoagulant", mention)
        elif kept := verified(
            mention, "anticoagulation", f"medication mention {mention.name!r}"
        ):
            medication_mentions.append(kept)

    anticoag_plans: list[AnticoagPlan] = []
    for plan in facts.anticoag_plans:
        if anticoagulant_generic(plan.drug, policy) is None:
            ignore(plan.drug, "plan is for a non-anticoagulant", plan)
        elif kept := verified(
            plan, "anticoagulation", f"anticoagulation plan for {plan.drug!r}"
        ):
            anticoag_plans.append(
                _resolve_plan_instructions(kept, docs.get(plan.doc_id), issues)
            )

    return VerificationResult(
        facts=ExtractedFacts(
            hp_documents=[
                kept
                for f in facts.hp_documents
                if (kept := verified(f, "hp", "H&P document"))
            ],
            consents=[
                kept
                for f in facts.consents
                if (kept := verified(f, "consent", "surgical consent"))
            ],
            medication_mentions=medication_mentions,
            anticoag_plans=anticoag_plans,
            note_vitals=[
                kept
                for f in facts.note_vitals
                if (kept := verified(f, f.kind, f"{f.kind.replace('_', ' ')} reading"))
            ],
            procedure_date_mentions=[
                kept
                for f in facts.procedure_date_mentions
                if (kept := verified(f, "required_fields", "procedure date mention"))
            ],
        ),
        issues=issues,
        ignored_medications=ignored,
    )


def _resolve_plan_instructions(
    plan: AnticoagPlan, doc: Document | None, issues: list[TriageIssue]
) -> AnticoagPlan:
    """Fill each instruction quote from its cited line; a bad citation leaves it empty."""

    updates: dict[str, str | None] = {}
    for line_field, quote_field, label in (
        ("pre_op_line", "pre_op_quote", "before-surgery instruction"),
        ("post_op_line", "post_op_quote", "after-surgery instruction"),
    ):
        line = getattr(plan, line_field)
        if line is None:
            continue
        text, problem = _cited_line(doc, line)
        if problem:
            what = f"{label} in anticoagulation plan for {plan.drug!r}"
            issues.append(
                _unverified_issue(
                    "anticoagulation", what, plan.doc_id, line, doc, text, problem
                )
            )
        updates[quote_field] = text
    return plan.model_copy(update=updates)


def _cited_line(doc: Document | None, line: int) -> tuple[str | None, str | None]:
    """Return (line text, None) for a usable citation, else (None, problem)."""

    if doc is None:
        return None, "the cited document is not in this submission"
    lines = doc.lines()
    if not 1 <= line <= len(lines):
        return None, f"line {line} does not exist (the document has {len(lines)} lines)"
    text = lines[line - 1].strip()
    if not text:
        return None, f"line {line} is blank"
    return text, None


def _value_problem(fact: Fact, policy: Policy) -> str | None:
    """Check that the values a fact claims appear on its cited line."""

    if isinstance(fact, NoteVital):
        quoted = {float(n) for n in _NUMBER.findall(fact.quote)}
        for label, value in (
            ("systolic", fact.systolic),
            ("diastolic", fact.diastolic),
            ("temperature", fact.temperature),
        ):
            if value is not None and value not in quoted:
                return f"{label} {value:g} does not appear on line {fact.line}"
    elif isinstance(fact, MedicationMention | AnticoagPlan):
        name = fact.name if isinstance(fact, MedicationMention) else fact.drug
        named = anticoagulants_named(fact.quote, policy)
        if anticoagulant_generic(name, policy) not in named:
            return f"{name!r} does not appear on line {fact.line}"
    return None


def _source(doc_id: str, line: int, doc: Document | None, text: str | None) -> Source:
    return Source(
        kind="document",
        id=doc_id,
        date=doc.date if doc else None,
        quote=text if text is not None else f"(line {line} could not be read)",
    )


def _unverified_issue(
    rule: RuleId,
    what: str,
    doc_id: str,
    line: int,
    doc: Document | None,
    text: str | None,
    problem: str,
) -> TriageIssue:
    return TriageIssue(
        category="MISSING_REQUIRED_DATA",
        reason="unverified_quote",
        rule=rule,
        finding=f"Could not verify extracted {what}: {problem}. The fact was discarded.",
        sources=[_source(doc_id, line, doc, text)],
    )
