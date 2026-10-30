"""Shared schemas and the triage_submission entry point for the pre-op triage scripts."""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, Field

Decision = Literal["READY", "NEEDS_FOLLOW_UP", "NOT_CLEARED"]
ProcedureRisk = Literal["LOW", "MODERATE", "HIGH"]
IssueCategory = Literal[
    "REQUIRED_DOCUMENTATION",
    "REQUIRED_TESTING",
    "ANTICOAGULATION_MANAGEMENT",
    "ACUTE_SAFETY_EXCLUSION",
    "MISSING_REQUIRED_DATA",
]

# -------------------------
# Schemas
# -------------------------

class PatientName(BaseModel):

    given: str | None = None
    family: str | None = None

class PatientInfo(BaseModel):

    id: str | None = None
    mrn: str | None = None
    name: PatientName | None = None
    dob: str | None = None
    sex: str | None = None

class ProcedureInfo(BaseModel):

    case_id: str | None = None
    procedure_type: str | None = None
    procedure_risk: ProcedureRisk | None = None
    procedure_date: str | None = None
    is_elective: bool | None = None
    location: str | None = None

class BloodPressureVital(BaseModel):

    type: str | None = None
    systolic: float | int | None = None
    diastolic: float | int | None = None
    date: str | None = None
    source: str | None = None

class TemperatureVital(BaseModel):

    type: str | None = None
    value_f: float | int | None = None
    date: str | None = None
    source: str | None = None

class GenericVital(BaseModel):

    type: str | None = None
    date: str | None = None
    source: str | None = None

Vital = BloodPressureVital | TemperatureVital | GenericVital

class LabResult(BaseModel):

    id: str | None = None
    code: str | None = None
    display: str | None = None
    effective_at: str | None = None
    status: str | None = None
    source: str | None = None

class Medication(BaseModel):

    name: str | None = None
    active: bool | None = None

class Condition(BaseModel):

    name: str | None = None
    active: bool | None = None

class Document(BaseModel):

    doc_id: str | None = None
    type: str | None = None
    date: str | None = None
    author: str | None = None
    text: str | None = None

    def lines(self) -> list[str]:
        """The text split into the lines the model cites; line n is lines()[n - 1]."""

        return (self.text or "").splitlines()

class SubmissionMetadata(BaseModel):

    submission_received_at: str | None = None
    source_system: str | None = None

class PatientSubmission(BaseModel):
    """Single submission package shape from the take-home prompt."""

    patient: PatientInfo | None = None
    procedure: ProcedureInfo | None = None
    vitals: list[Vital] = Field(default_factory=list)
    labs: list[LabResult] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    metadata: SubmissionMetadata | None = None

RuleId = Literal[
    "required_fields",
    "hp",
    "consent",
    "labs",
    "anticoagulation",
    "blood_pressure",
    "temperature",
    "system",
]
IssueReason = Literal[
    "policy_failed",
    "missing",
    "conflict",
    "unverified_quote",
    "extraction_failed",
    "invalid_submission",
    "internal_error",
]
CheckStatus = Literal["passed", "not_applicable", "not_evaluated"]
SourceKind = Literal["document", "lab", "vital", "medication", "field"]
ComputedValue = str | int | float | bool | None

class Source(BaseModel):
    """Where a piece of evidence came from, so a reviewer can find it in the chart."""

    kind: SourceKind
    # doc_id or lab id, else a path such as "vitals[1]" or "procedure.procedure_date"
    id: str
    date: str | None = None
    quote: str

class TriageIssue(BaseModel):
    """A blocking problem, with the evidence a reviewer needs to verify it."""

    category: IssueCategory
    reason: IssueReason
    rule: RuleId
    finding: str
    sources: list[Source] = Field(default_factory=list)
    computed: dict[str, ComputedValue] = Field(default_factory=dict)

class RuleCheck(BaseModel):
    """A rule that did not block, and why, so READY results can be spot-checked."""

    rule: RuleId
    status: CheckStatus
    finding: str
    sources: list[Source] = Field(default_factory=list)
    computed: dict[str, ComputedValue] = Field(default_factory=dict)

class IgnoredMedication(BaseModel):
    """A medication no rule acts on, listed so a reviewer can see it was not missed."""

    name: str
    reason: str
    quote_verified: bool
    source: Source

class TriageOutput(BaseModel):
    """Structured output contract for triage responses."""

    decision: Decision
    issues: list[TriageIssue] = Field(validation_alias=AliasChoices("issues"))
    checks: list[RuleCheck]
    ignored_medications: list[IgnoredMedication]
    explanation: str


def triage_submission(
    submission: dict[str, object] | PatientSubmission,
    *,
    model: str,
) -> TriageOutput:
    """Triage one submission: extract facts with the LLM, verify them, apply the policy.

    The decision comes from plain-Python rules, never from the model. Failures yield
    NEEDS_FOLLOW_UP with the reason; see pipeline.py.
    """

    # Imported here because pipeline.py (via rules.py) imports this module.
    from pipeline import triage

    return triage(submission, model=model)
