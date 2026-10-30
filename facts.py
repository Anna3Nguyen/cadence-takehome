"""Facts the LLM extracts from free-text documents.

Every fact cites the document (doc_id) and the numbered line it came from. The model
never writes quote text: verify.py copies the cited line from the document into
`quote`, so every quote is exact by construction, and then checks the fact's values
against it. Facts never contain decisions: the model reports what a document says,
and rules.py decides what it means.

Field descriptions are sent to the model as part of the JSON schema, so they double
as extraction instructions. Every model-facing field is required (optional values
are nullable) and extra fields are forbidden, as OpenAI strict structured outputs
require. Quote fields are hidden from the schema because code fills them in.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema


class Fact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(description="doc_id of the document this fact was read from.")
    line: int = Field(
        description="Number of the line in that document (shown as [n]) stating this fact."
    )
    # Filled from the cited line by verify.py; never written by the model.
    quote: SkipJsonSchema[str] = ""


class HPDocument(Fact):
    """A document that is a history and physical (H&P), whatever its title says."""

    is_preop_hp: bool = Field(
        description=(
            "True only if this H&P is the pre-operative evaluation for the scheduled "
            "procedure. False for wellness, annual, or establish-care visits."
        )
    )


class Consent(Fact):
    """A surgical consent form or consent record."""

    patient_signed: bool = Field(
        description=(
            "True only if the patient's signature is documented (wet, electronic, or "
            "scanned). False if it is pending, blank, sent but not completed, or only "
            "a clinician signed."
        )
    )
    procedure_named: str | None = Field(
        description="Procedure named on the consent, as written, or null if none."
    )
    matches_scheduled_procedure: bool = Field(
        description=(
            "True if procedure_named is the same procedure as the scheduled one "
            "(abbreviations and synonyms count as the same)."
        )
    )


class MedicationMention(Fact):
    """A statement about whether the patient takes a medication, including
    medications missing from the structured medication list. Exclude medications
    that are only proposed as part of a plan, such as a planned enoxaparin bridge."""

    name: str = Field(description="Medication name as written.")
    status: Literal["taking", "stopped", "uncertain"] = Field(
        description=(
            "taking: the patient currently takes it. stopped: discontinued or no "
            "longer taken. uncertain: the document says it is unclear or unconfirmed."
        )
    )


class AnticoagPlan(Fact):
    """Perioperative instructions for managing an anticoagulant around this
    procedure. line is the line naming the medication the plan manages."""

    drug: str = Field(description="Medication the plan manages, as written.")
    pre_op_line: int | None = Field(
        description=(
            "Line with the before-surgery instruction (when to hold or stop, or the "
            "last dose), or null if the document gives none."
        )
    )
    post_op_line: int | None = Field(
        description=(
            "Line with the after-surgery instruction (when or how to resume), or null "
            "if none. Deferrals such as 'will be addressed after surgery' are not "
            "instructions."
        )
    )
    # Filled from the cited lines by verify.py; never written by the model.
    pre_op_quote: SkipJsonSchema[str | None] = None
    post_op_quote: SkipJsonSchema[str | None] = None


class NoteVital(Fact):
    """A blood pressure or temperature reading written in a document's text.
    Extract every reading, including rechecks, as its own item. References such
    as 'see flowsheet' are not readings."""

    kind: Literal["blood_pressure", "temperature"]
    systolic: float | None = Field(description="mmHg; null for a temperature.")
    diastolic: float | None = Field(description="mmHg; null for a temperature.")
    temperature: float | None = Field(
        description="Value as written; null for a blood pressure."
    )
    temperature_unit: Literal["F", "C"] | None = Field(
        description="Unit as written; null for a blood pressure."
    )
    measured_on: str | None = Field(
        description="YYYY-MM-DD if the text states the reading's date, else null."
    )
    measured_time: str | None = Field(
        description="24-hour HH:MM if the text states the reading's time, else null."
    )


class ProcedureDateMention(Fact):
    """A statement of the scheduled date for this procedure."""

    date: str = Field(description="YYYY-MM-DD.")


class ExtractedFacts(BaseModel):
    """Everything extracted from one submission's documents."""

    model_config = ConfigDict(extra="forbid")

    hp_documents: list[HPDocument]
    consents: list[Consent]
    medication_mentions: list[MedicationMention]
    anticoag_plans: list[AnticoagPlan]
    note_vitals: list[NoteVital]
    procedure_date_mentions: list[ProcedureDateMention]
