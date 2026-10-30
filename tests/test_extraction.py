from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core import PatientSubmission
from extraction import (
    EXTRACTION_PROMPT,
    ExtractionError,
    build_extraction_input,
    extract_facts,
)
from facts import ExtractedFacts, HPDocument

NOTE_TEXT = "PRE-OP H&P\nPre-op evaluation complete.\n  BP 120/80, T 98.6 F."

FACTS = ExtractedFacts(
    hp_documents=[HPDocument(doc_id="doc-1", line=2, is_preop_hp=True)],
    consents=[],
    medication_mentions=[],
    anticoag_plans=[],
    note_vitals=[],
    procedure_date_mentions=[],
)


@pytest.fixture
def submission() -> PatientSubmission:
    return PatientSubmission.model_validate(
        {
            "patient": {
                "id": "patient-1",
                "mrn": "MRN-3000000",
                "name": {"given": "Grace", "family": "Lopez"},
                "dob": "1941-01-01",
            },
            "procedure": {
                "procedure_type": "Elective carpal tunnel release",
                "procedure_risk": "LOW",
                "procedure_date": "2026-03-01",
            },
            "medications": [{"name": "tamsulosin 0.4 mg PO nightly", "active": True}],
            "documents": [
                {
                    "doc_id": "doc-1",
                    "type": "History and Physical",
                    "date": "2026-02-21",
                    "text": NOTE_TEXT,
                }
            ],
        }
    )


def _client(output: ExtractedFacts | None = FACTS) -> Mock:
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(output_parsed=output)
    return client


# -------------------------
# Request
# -------------------------


def test_request_uses_strict_schema_and_deterministic_settings(
    submission: PatientSubmission,
) -> None:
    client = _client()

    facts = extract_facts(submission, model="test-model", client=client, cache_dir=None)

    assert facts == FACTS
    call = client.responses.parse.call_args.kwargs
    assert call["model"] == "test-model"
    assert call["instructions"] == EXTRACTION_PROMPT
    assert call["text_format"] is ExtractedFacts
    assert call["temperature"] == 0
    assert call["store"] is False
    assert call["input"] == build_extraction_input(submission)


def test_input_numbers_each_document_line_verbatim(
    submission: PatientSubmission,
) -> None:
    user_input = build_extraction_input(submission)

    assert (
        "=== Document doc_id=doc-1 | type=History and Physical | date=2026-02-21 ===\n"
        "[1] PRE-OP H&P\n"
        "[2] Pre-op evaluation complete.\n"
        "[3]   BP 120/80, T 98.6 F."
    ) in user_input
    assert user_input.startswith("Scheduled procedure: Elective carpal tunnel release")


@pytest.mark.parametrize(
    "withheld",
    ["Grace", "Lopez", "MRN-3000000", "1941-01-01", "2026-03-01", "tamsulosin"],
)
def test_input_omits_identifiers_and_values_facts_are_checked_against(
    submission: PatientSubmission, withheld: str
) -> None:
    assert withheld not in build_extraction_input(submission)


# -------------------------
# Cache
# -------------------------


def test_cache_hit_skips_the_api(submission: PatientSubmission, tmp_path: Path) -> None:
    client = _client()

    first = extract_facts(submission, model="m", client=client, cache_dir=tmp_path)
    second = extract_facts(submission, model="m", client=client, cache_dir=tmp_path)

    assert first == second == FACTS
    assert client.responses.parse.call_count == 1


def test_cache_is_keyed_by_model_and_input(
    submission: PatientSubmission, tmp_path: Path
) -> None:
    client = _client()
    edited = submission.model_copy(deep=True)
    edited.documents[0].text = NOTE_TEXT + " Addendum."

    extract_facts(submission, model="m1", client=client, cache_dir=tmp_path)
    extract_facts(submission, model="m2", client=client, cache_dir=tmp_path)
    extract_facts(edited, model="m1", client=client, cache_dir=tmp_path)

    assert client.responses.parse.call_count == 3
    assert len(list(tmp_path.glob("*.json"))) == 3


def test_corrupt_cache_entry_is_refetched(
    submission: PatientSubmission, tmp_path: Path
) -> None:
    client = _client()
    extract_facts(submission, model="m", client=client, cache_dir=tmp_path)
    [entry] = tmp_path.glob("*.json")
    entry.write_text("{not json")

    facts = extract_facts(submission, model="m", client=client, cache_dir=tmp_path)

    assert facts == FACTS
    assert client.responses.parse.call_count == 2
    assert ExtractedFacts.model_validate_json(entry.read_text()) == FACTS


def test_no_cache_dir_always_calls_the_api(submission: PatientSubmission) -> None:
    client = _client()

    extract_facts(submission, model="m", client=client, cache_dir=None)
    extract_facts(submission, model="m", client=client, cache_dir=None)

    assert client.responses.parse.call_count == 2


# -------------------------
# Failures
# -------------------------


def test_api_error_raises_extraction_error_and_caches_nothing(
    submission: PatientSubmission, tmp_path: Path
) -> None:
    client = Mock()
    client.responses.parse.side_effect = TimeoutError("request timed out")

    with pytest.raises(ExtractionError, match="request timed out"):
        extract_facts(submission, model="m", client=client, cache_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_refusal_raises_extraction_error_and_caches_nothing(
    submission: PatientSubmission, tmp_path: Path
) -> None:
    with pytest.raises(ExtractionError, match="no facts"):
        extract_facts(
            submission, model="m", client=_client(output=None), cache_dir=tmp_path
        )

    assert list(tmp_path.iterdir()) == []
