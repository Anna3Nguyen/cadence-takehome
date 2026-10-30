# Pre-Op Triage Take-Home

`triage_submission(...)` in `core.py` evaluates one pre-op submission package against the Cadence pre-operative scheduling policy and returns:

- `decision`: `READY | NEEDS_FOLLOW_UP | NOT_CLEARED`
- `issues[]`: each with a `category`, the rule, a plain-language `finding`, and `sources` (the exact document line, lab, vital, or field it came from)
- `checks[]`: every rule that passed, did not apply, or could not be evaluated, with the same evidence
- `ignored_medications[]`: medications no rule acted on, so a reviewer can see nothing was missed
- `explanation`

An LLM only extracts facts from the free-text documents, citing a document and line for each. Plain Python verifies every citation and applies the policy rules; the model never makes the decision. See [WRITEUP.md](WRITEUP.md) for the design and reasoning.

## How it works

```
submission ─► extraction.py ─► verify.py ─► rules.py ─► decision + evidence
              (LLM: facts,     (check each   (policy in
               cite doc+line)   cited line)   plain Python)
```

| File | Purpose |
|---|---|
| `core.py` | Input/output schemas and the `triage_submission` entry point |
| `pipeline.py` | Runs the steps above; fails safe to `NEEDS_FOLLOW_UP` on any error |
| `extraction.py` | The only LLM call: strict structured output, temperature 0, on-disk cache |
| `facts.py` | Schema of facts the LLM may return (each cites `doc_id` + `line`) |
| `verify.py` | Checks each cited line exists and states the claimed values; copies it in as the quote |
| `rules.py` | The four policy rules, missing-data/conflict checks, and the final decision |
| `policy.py` | Policy windows, thresholds, lab codes, and anticoagulant names as data |
| `run_baseline.py`, `run_evals.py`, `view_report.py` | Provided harness: run, score, and browse results |

## Setup

1. Confirm `uv` is installed:

```bash
uv --version
```

2. Provide your OpenAI API key. Either export it:

```bash
export OPENAI_API_KEY="<your_api_key>"
```

or put `OPENAI_API_KEY=<your_api_key>` in a `.env` file at the repo root (it is gitignored) and load it into your shell before running `make`:

```bash
set -a; . ./.env; set +a
```

Dependencies are declared inline in the scripts and installed by `uv` automatically.

## Run

```bash
make baseline   # triage all cases -> data/baseline_outputs.jsonl
make evals      # score against labels -> data/eval_report.json
make score      # print the score
make report     # browse results in a terminal UI (press f on a metric to filter failures, q to quit)
make all        # baseline + evals + score
```

Extractions are cached in `.cache/extractions/`, keyed by model, prompt, schema, and input, so re-running `make baseline` makes no API calls unless something changed. Delete `.cache/` to force fresh extractions.

`make determinism` is disabled: with the cache it would replay identical results and always report 100%. See [WRITEUP.md](WRITEUP.md) for the cache-off stability check.

## Tests

```bash
make test
```

The tests use a mocked OpenAI client and make no API calls.

## Outputs

- Triage outputs: `data/baseline_outputs.jsonl`
- Eval report: `data/eval_report.json`
- Extraction cache: `.cache/extractions/`

## Configurable Variables

- `MODEL` (default `gpt-4.1-mini`)
- `INPUT` (default `data/patients_sample_50.jsonl`)
- `OUTPUT` (default `data/baseline_outputs.jsonl`)
- `REPORT` (default `data/eval_report.json`)
