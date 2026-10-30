"""Verify extracted facts against the documents they cite.

A fact reaches the rules only if its quote appears in the cited document
(ignoring case, whitespace, and quote/dash styling) and the values it claims
appear in that quote. A fact that fails is discarded and reported as a
MISSING_REQUIRED_DATA issue, so an unverifiable fact can never help a case
reach READY.

Medication facts about non-anticoagulants are not acted on by any rule. They are
listed as ignored medications instead of being verified into issues, so an
irrelevant drug cannot force follow-up and a reviewer can still see it.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

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

_PUNCTUATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
    }
)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class VerificationResult:
    facts: ExtractedFacts
    issues: list[TriageIssue]
    ignored_medications: list[IgnoredMedication]


def normalize(text: str) -> str:
    """Casefold, unify quote and dash styles, and drop all whitespace."""

    text = unicodedata.normalize("NFKC", text).translate(_PUNCTUATION).casefold()
    return "".join(text.split())


def verify_facts(
    facts: ExtractedFacts,
    submission: PatientSubmission,
    policy: Policy = POLICY,
) -> VerificationResult:
    """Keep only facts whose quotes and values check out against the submission."""

    docs = {doc.doc_id: doc for doc in submission.documents if doc.doc_id}
    issues: list[TriageIssue] = []
    ignored: list[IgnoredMedication] = []

    def verified(fact: Fact, rule: RuleId, what: str) -> bool:
        doc = docs.get(fact.doc_id)
        problem = _quote_problem(doc, fact.quote) or _value_problem(fact, policy)
        if problem:
            issues.append(
                _unverified_issue(rule, what, fact.doc_id, doc, fact.quote, problem)
            )
        return problem is None

    def ignore(name: str, reason: str, fact: Fact) -> None:
        doc = docs.get(fact.doc_id)
        ignored.append(
            IgnoredMedication(
                name=name,
                reason=reason,
                quote_verified=_quote_problem(doc, fact.quote) is None,
                source=_source(fact.doc_id, doc, fact.quote),
            )
        )

    medication_mentions: list[MedicationMention] = []
    for mention in facts.medication_mentions:
        if anticoagulant_generic(mention.name, policy) is None:
            ignore(mention.name, "not an anticoagulant", mention)
        elif verified(mention, "anticoagulation", f"medication mention {mention.name!r}"):
            medication_mentions.append(mention)

    anticoag_plans: list[AnticoagPlan] = []
    for plan in facts.anticoag_plans:
        if anticoagulant_generic(plan.drug, policy) is None:
            ignore(plan.drug, "plan is for a non-anticoagulant", plan)
        elif verified(plan, "anticoagulation", f"anticoagulation plan for {plan.drug!r}"):
            doc = docs.get(plan.doc_id)
            anticoag_plans.append(_verify_plan_instructions(plan, doc, issues))

    return VerificationResult(
        facts=ExtractedFacts(
            hp_documents=[
                f for f in facts.hp_documents if verified(f, "hp", "H&P document")
            ],
            consents=[
                f for f in facts.consents if verified(f, "consent", "surgical consent")
            ],
            medication_mentions=medication_mentions,
            anticoag_plans=anticoag_plans,
            note_vitals=[
                f
                for f in facts.note_vitals
                if verified(f, f.kind, f"{f.kind.replace('_', ' ')} reading")
            ],
            procedure_date_mentions=[
                f
                for f in facts.procedure_date_mentions
                if verified(f, "required_fields", "procedure date mention")
            ],
        ),
        issues=issues,
        ignored_medications=ignored,
    )


def _verify_plan_instructions(
    plan: AnticoagPlan, doc: Document | None, issues: list[TriageIssue]
) -> AnticoagPlan:
    """Null out any plan instruction whose quote fails, leaving the plan incomplete."""

    updates: dict[str, None] = {}
    for field_name, label in (
        ("pre_op_quote", "before-surgery instruction"),
        ("post_op_quote", "after-surgery instruction"),
    ):
        quote = getattr(plan, field_name)
        if quote is None:
            continue
        problem = _quote_problem(doc, quote)
        if problem:
            what = f"{label} in anticoagulation plan for {plan.drug!r}"
            issues.append(
                _unverified_issue("anticoagulation", what, plan.doc_id, doc, quote, problem)
            )
            updates[field_name] = None
    return plan.model_copy(update=updates)


def _quote_problem(doc: Document | None, quote: str | None) -> str | None:
    if doc is None:
        return "the cited document is not in this submission"
    if not quote or not quote.strip():
        return "the quote is empty"
    if normalize(quote) not in normalize(doc.text or ""):
        return "the quote does not appear in the cited document"
    return None


def _value_problem(fact: Fact, policy: Policy) -> str | None:
    """Check that the values a fact claims appear in its own quote."""

    if isinstance(fact, NoteVital):
        quoted = {float(n) for n in _NUMBER.findall(fact.quote)}
        for label, value in (
            ("systolic", fact.systolic),
            ("diastolic", fact.diastolic),
            ("temperature", fact.temperature),
        ):
            if value is not None and value not in quoted:
                return f"{label} {value:g} does not appear in the quote"
    elif isinstance(fact, MedicationMention | AnticoagPlan):
        name = fact.name if isinstance(fact, MedicationMention) else fact.drug
        named = anticoagulants_named(fact.quote, policy)
        if anticoagulant_generic(name, policy) not in named:
            return f"{name!r} does not appear in the quote"
    return None


def _source(doc_id: str, doc: Document | None, quote: str | None) -> Source:
    return Source(
        kind="document",
        id=doc_id,
        date=doc.date if doc else None,
        quote=quote or "",
    )


def _unverified_issue(
    rule: RuleId,
    what: str,
    doc_id: str,
    doc: Document | None,
    quote: str | None,
    problem: str,
) -> TriageIssue:
    return TriageIssue(
        category="MISSING_REQUIRED_DATA",
        reason="unverified_quote",
        rule=rule,
        finding=f"Could not verify extracted {what}: {problem}. The fact was discarded.",
        sources=[_source(doc_id, doc, quote)],
    )
