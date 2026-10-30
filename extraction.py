"""Fact extraction: the only step that calls the LLM.

The model reads a submission's free-text documents and returns ExtractedFacts
(see facts.py). It never sees the policy and never makes a decision: verify.py
checks every quote and rules.py decides. Responses are cached on disk, so reruns
are free and return identical facts.

The model is sent only what extraction needs. Patient identifiers are omitted, and
so are the structured procedure date and medication list, so the model cannot echo
back the values its facts are later checked against.
"""

from __future__ import annotations

from functools import cache
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from core import PatientSubmission
from facts import ExtractedFacts

if TYPE_CHECKING:
    from openai import OpenAI

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "extractions"

EXTRACTION_PROMPT = """
You extract facts from the clinical documents in a pre-operative surgery package.
You do not decide whether the patient is ready for surgery and you do not apply any
policy; other software does that using the facts you return.

- Read every document. Return one item per fact in the lists the schema defines, and
  leave a list empty if no document states anything for it.
- doc_id is the id of the document the fact came from.
- Every quote is copied exactly from that document: one sentence or less, containing
  the value you extracted. Never paraphrase, summarize, correct typos, or join text
  from different places.
- Report only what the documents state. Do not infer or fill in values from medical
  knowledge.
- Record every blood pressure and temperature reading as its own item, including
  rechecks, with the time when the text gives one. Report temperatures in the unit
  written.
- Record every statement about whether the patient takes a medication, including
  medications the patient reports that may be missing from a medication list.
""".strip()


class ExtractionError(RuntimeError):
    """Extraction failed. Callers must never treat this as "no facts found"."""


def build_extraction_input(submission: PatientSubmission) -> str:
    """Render the scheduled procedure and documents as plain text for the model."""

    procedure = submission.procedure.procedure_type if submission.procedure else None
    blocks = [f"Scheduled procedure: {procedure or 'not stated'}"]
    for doc in submission.documents:
        header = (
            f"=== Document doc_id={doc.doc_id} | type={doc.type} | date={doc.date} ==="
        )
        blocks.append(f"{header}\n{doc.text or ''}")
    return "\n\n".join(blocks)


def extract_facts(
    submission: PatientSubmission,
    *,
    model: str,
    client: OpenAI | None = None,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
) -> ExtractedFacts:
    """Extract facts from the submission's documents, using the disk cache if present."""

    user_input = build_extraction_input(submission)
    cache_path = (
        cache_dir / f"{_cache_key(model, user_input)}.json" if cache_dir else None
    )
    if cache_path is not None and (cached := _read_cache(cache_path)) is not None:
        return cached

    facts = _call_model(client or _default_client(), model, user_input)
    if cache_path is not None:
        _write_cache(cache_path, facts)
    return facts


def _call_model(client: OpenAI, model: str, user_input: str) -> ExtractedFacts:
    try:
        response = client.responses.parse(
            model=model,
            instructions=EXTRACTION_PROMPT,
            input=user_input,
            text_format=ExtractedFacts,
            temperature=0,
            store=False,
        )
    except Exception as exc:  # Any failure must surface as ExtractionError.
        raise ExtractionError(f"Extraction call failed: {exc}") from exc

    facts = response.output_parsed
    if facts is None:
        raise ExtractionError("Model returned no facts (refusal or empty output).")
    return facts


@cache
def _default_client() -> OpenAI:
    # Imported lazily so the rest of the pipeline works without OpenAI installed.
    from openai import OpenAI

    return OpenAI(max_retries=3, timeout=60.0)


def _cache_key(model: str, user_input: str) -> str:
    material = {
        "model": model,
        "prompt": EXTRACTION_PROMPT,
        "schema": ExtractedFacts.model_json_schema(),
        "input": user_input,
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def _read_cache(path: Path) -> ExtractedFacts | None:
    try:
        return ExtractedFacts.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # Missing or corrupt entries are simply refetched.
        return None


def _write_cache(path: Path, facts: ExtractedFacts) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(facts.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)  # Atomic, so a crash never leaves a half-written entry.
    except OSError:
        pass  # A cache that cannot be written only costs a future API call.
