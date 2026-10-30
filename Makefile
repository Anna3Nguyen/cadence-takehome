INPUT ?= data/patients_sample_50.jsonl
OUTPUT ?= data/baseline_outputs.jsonl
REPORT ?= data/eval_report.json
DETERMINISM_REPORT ?= data/determinism_report.json
MODEL ?= gpt-4.1-mini

.PHONY: baseline evals score report test all clean

baseline:
	uv run run_baseline.py \
		--input $(INPUT) \
		--output $(OUTPUT) \
		--model $(MODEL)

evals:
	uv run run_evals.py \
		--input $(INPUT) \
		--outputs $(OUTPUT) \
		--report $(REPORT)

# Disabled: extractions are cached on disk (.cache/extractions), so repeated runs
# replay the cached result and this check would always report 100% without
# re-running the model. See WRITEUP.md for the cache-off stability check instead.
# determinism:
# 	uv run run_evals.py \
# 		--determinism \
# 		--input $(INPUT) \
# 		--model $(MODEL) \
# 		--report $(DETERMINISM_REPORT)

score:
	@python3 -c 'import json; r=json.load(open("$(REPORT)")); s=(r.get("primary_score",{}) or {}).get("value_pct"); print(s if s is not None else r.get("local_metrics_summary",{}).get("aggregate_local_score_pct", 0.0))'

report:
	uv run view_report.py --report $(REPORT)

test:
	uv run \
		--with 'openai>=2.0.0' \
		--with 'pydantic>=2.8.0' \
		--with 'pytest>=8.0.0' \
		python -m pytest tests

all: baseline evals score

clean:
	rm -f data/baseline_outputs.jsonl \
		data/eval_report.json \
		data/determinism_report.json
