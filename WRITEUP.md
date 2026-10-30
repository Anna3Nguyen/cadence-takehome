# Write-up

## Results

Score went from 74.67% (baseline) to 100% on the 50 sample cases.

## Decisions

**The LLM only extracts facts; plain Python makes the decision.** The baseline's mistakes were reasoning, not reading: it contradicted its own explanation, miscounted days, and made up rules (like treating aspirin as a blood thinner). Code gets dates and thresholds right every time and can be unit-tested.

**The LLM cites a line number, and code copies the quote.** At first the model copied quotes itself, but it messed up characters like `°F` and stitched pieces together, so good facts failed the check. Now it points to `doc_id` + line, code checks the values (BP, temp, drug) are actually on that line, and copies the exact text. This took the score from 87% to 100%.

**Unverifiable facts get thrown out and flagged.** If a cited line doesn't exist or doesn't contain the value, the fact is dropped and reported as `MISSING_REQUIRED_DATA`. A made-up fact can never push a case to READY.

**Errors never fail open.** Bad input, an API failure, or a bug returns `NEEDS_FOLLOW_UP` with the reason, never READY and never a crash. If the LLM call fails, a structured vital over the limit still returns `NOT_CLEARED`.

**Every result shows its evidence.** Issues and passed checks list the exact source (document line, lab, vital, or field) and the math, like "31 days before, limit 30". Medications no rule used are listed too, so a nurse can verify a READY case fast.

**Policy lives in one data file.** Windows, thresholds, lab codes, and the anticoagulant list are in `policy.py`, and each rule is a small function in a list. Changing the policy means editing data, and adding a rule means adding one function.

**Conflicts get resolved by the policy when possible, flagged when not.** A recheck replaces an older BP, and a newer signed consent replaces a pending one, with both shown as evidence. If sources truly disagree (med list says taking, note says stopped), it's flagged with both quotes.

**The model only gets what it needs.** No name, MRN, or DOB. It also doesn't see the procedure date or med list, so it can't just repeat values we later check its facts against.

**Runs are repeatable.** Temperature 0 plus an on-disk cache. `make determinism` is off because the cache would make it always say 100%.

## Policy calls I made

- Windows are inclusive (30 days passes, 31 fails).
- Latest reading wins; if same-day readings can't be ordered, the most severe one is used.
- 38.0 °C = 100.4 °F, which isn't a fever (policy says > 100.4).
- Aspirin and clopidogrel aren't anticoagulants. A blood thinner mentioned only in a note still counts.
- A plan has to cover both stopping and restarting the same drug; "will be addressed after surgery" doesn't count.
- Newest consent wins, and it must be signed by the patient for the scheduled procedure.
- Missing date, risk, temp, or BP is `MISSING_REQUIRED_DATA`. Entered-in-error labs are skipped, and a BMP isn't a CMP.

## Known limits

- Small, in-sample eval; a held-out set is the next step. Needs to be run on more data.
- Case 00048: the model sometimes counts a note that *mentions* an H&P as an H&P.
- 7 of 41 H&P checks cite a weak line ("Planned procedure: …") instead of the H&P heading. The documents are right, the evidence is just weaker.
- Badly misspelled drug names (e.g. "Xeralto") aren't matched, but they show up in `ignored_medications`.
