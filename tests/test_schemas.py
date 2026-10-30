from __future__ import annotations

from collections.abc import Iterator
from typing import Any, get_args

import pytest
from pydantic import BaseModel, ValidationError

from core import TriageOutput
from facts import (
    AnticoagPlan,
    Consent,
    ExtractedFacts,
    HPDocument,
    MedicationMention,
    NoteVital,
    ProcedureDateMention,
)

FACT_TYPES: list[type[BaseModel]] = [
    HPDocument,
    Consent,
    MedicationMention,
    AnticoagPlan,
    NoteVital,
    ProcedureDateMention,
]


# -------------------------
# Extraction facts
# -------------------------


def test_extracted_facts_covers_every_fact_type() -> None:
    listed = {
        get_args(field.annotation)[0] for field in ExtractedFacts.model_fields.values()
    }
    assert listed == set(FACT_TYPES)


@pytest.mark.parametrize("fact_type", FACT_TYPES)
def test_every_fact_requires_doc_id_and_quote(fact_type: type[BaseModel]) -> None:
    for name in ("doc_id", "quote"):
        assert fact_type.model_fields[name].is_required()


def _iter_objects(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        if "properties" in node:
            yield node
        for value in node.values():
            yield from _iter_objects(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_objects(value)


def test_extracted_facts_schema_is_strict_compatible() -> None:
    objects = list(_iter_objects(ExtractedFacts.model_json_schema()))

    assert len(objects) == len(FACT_TYPES) + 1
    for obj in objects:
        assert set(obj["required"]) == set(obj["properties"])
        assert obj["additionalProperties"] is False
        assert not any("default" in prop for prop in obj["properties"].values())


# -------------------------
# Triage output
# -------------------------


def _output_payload() -> dict[str, Any]:
    return {
        "decision": "NEEDS_FOLLOW_UP",
        "issues": [
            {
                "category": "REQUIRED_DOCUMENTATION",
                "reason": "policy_failed",
                "rule": "hp",
                "finding": "Most recent pre-op H&P is 31 days before the procedure (limit 30).",
                "sources": [
                    {
                        "kind": "document",
                        "id": "doc-1",
                        "date": "2026-02-12",
                        "quote": "HISTORY AND PHYSICAL: pre-op evaluation complete.",
                    }
                ],
                "computed": {"days_before": 31, "limit_days": 30},
            }
        ],
        "checks": [
            {
                "rule": "labs",
                "status": "passed",
                "finding": "CBC is 8 days before the procedure (limit 30).",
                "sources": [
                    {
                        "kind": "lab",
                        "id": "lab-1",
                        "date": "2026-02-21",
                        "quote": "CBC - Complete Blood Count (final)",
                    }
                ],
                "computed": {"days_before": 8, "limit_days": 30},
            }
        ],
        "explanation": "H&P is outside the 30-day window.",
    }


def test_triage_output_round_trips_through_json() -> None:
    output = TriageOutput.model_validate(_output_payload())

    assert output.model_dump() == _output_payload()
    assert TriageOutput.model_validate_json(output.model_dump_json()) == output


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("issues", 0, "category"), "PAPERWORK"),
        (("issues", 0, "reason"), "unknown"),
        (("issues", 0, "rule"), "bmi"),
        (("issues", 0, "sources", 0, "kind"), "email"),
        (("checks", 0, "status"), "skipped"),
    ],
)
def test_triage_output_rejects_unknown_values(
    path: tuple[str | int, ...], bad_value: str
) -> None:
    payload = _output_payload()
    target: Any = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad_value

    with pytest.raises(ValidationError):
        TriageOutput.model_validate(payload)
