# SOL 6.1 — START HERE — VIRAC2 Infrared Discovery Engine

**Status at handoff: scientifically unvalidated research prototype, not a discovery machine.**
**Updated:** 2026-11-17. **Repository location:** `research/virac2-infrared-discovery-engine/` within `deepnilray/varwise`.

## Read in this exact order
1. `AGENTS.md`: operating constraints; what is forbidden.
2. `README.md`: installation, CLI, schemas, status meanings.
3. `BRUTAL_REVIEW.md`: verified deficiencies and patch history.
4. `VALIDATION_PROTOCOL.md`: mandatory scientific release gates.
5. `BACKLOG_AND_ACCEPTANCE.md`: precise ranked engineering tasks with pass/fail criteria.
6. `ARCHITECTURE_AND_DATA.md`: inputs, model, code map, provenance, limitations.
7. `REPRODUCE_AND_VERIFY.md`: exact commands and immutable checks.
8. `EVIDENCE_LEDGER.md`: facts demonstrated versus unverified hypotheses.

## User objective
Build a **fully independent, locally usable, scientifically credible astronomical discovery system** that finds real previously unreported astrophysical sources or events in public observational data. Do the implementation and verification autonomously. The user does not want a research proposal, an inflated metric, a toy benchmark, supervision as a prerequisite, or a mere list of suggestions. The existing engine is a frozen **microlensing-like infrared candidate ranker**, not a general new-star detector; explicitly distinguish discovering a new astronomical object, a new event in a known source, and a known event.

## Immediate next execution task
**P0: create and verify a truly independent, image-vetted artifact/negative benchmark, while guaranteeing no overlap with original v12 training, any old v7/v10 holdouts, or previous candidate fields.** Start by inventorying which historical sky tiles and source IDs are already opened in earlier experiments (they are not fully itemized in this standalone package). Do not silently assume missing tile lists are unseen. If no fresh archive access is available, implement and unit-test the data-acquisition and split infrastructure, record the hard blocker, and do not invent measurements.

Only after a genuinely fresh benchmark passes should P1/P2 engineering open a blind target search. Preserve a sealed holdout (test only once, then retire it for future tuning), source-level and sky-level leakage prevention, audit logs, positive/negative eligibility standards, reproducible method baseline and source/image verification.

## Repro commands (run from THIS directory)
```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python verify_bundle.py
python engine.py info
python engine.py score --input examples/known_training_lightcurves.csv --out /tmp/virac2_control_run_unique
python training/reproduce_frozen_model.py --out /tmp/virac2_model_reproduced.joblib
```
Use unique output paths; scorer intentionally refuses preexisting nonempty result directories. The demo stars are **known training controls**, not independent validation or discoveries.

## Verified facts and red lines
- Frozen Extra Trees model: 400 trees, 43 features, 366 **published labeled** sources, 138 development groups. Manifest SHA256 `655b77c78a56e557d884cceef88fa9c69db89b40a3b80081b5088291f2be1b3d` for the joblib.
- Grouped **development** confusion matrix [[269,3],[4,90]] (TN,FP,FN,TP); AP 0.993139817. Not independently validated against field artifacts/ordinary stars.
- 27 adversarial tests and 10 archive integrity assertions passed on 2026-11-17 locally. These are software tests, not scientific discovery validation.
- Real field scanning, independent event recovery, image-level verification, novelty check, online ESO ingestion and Windows GUI remain incomplete or untested.
- `release_validated=false`, `discovery_confirmed=false` are intentional. Never change them on the basis of CV scores, plausibility, or a high classifier score.

## How to report work to the user
For every changed commit: say what code or records actually changed, exact tests/data run, outcome (including failures), artifact paths, what remains blocked, and whether evidence is new/independent. No claims of a 'new object' without literature searches, original images, independently verified physical evidence, and scientific review. Work in a separate branch or clearly scoped subdirectory; do not damage older ZTF/VarWise work.
