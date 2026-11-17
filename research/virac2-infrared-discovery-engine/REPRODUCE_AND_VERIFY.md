# Reproduction, verification and environment

## Tested here on 2026-11-17
- `python -m unittest discover -s tests -q`: **27/27 passed**.
- `python verify_bundle.py`: **10 assertions passed**. Development confusion matrix [[269,3],[4,90]], grouped AP 0.993139817 and 366 labeled sources.
- `python engine.py info`: frozen joblib SHA256 `655b77c78a56e557d884cceef88fa9c69db89b40a3b80081b5088291f2be1b3d`, flags `release_validated=false`, `discovery_confirmed=false`.
- Local demo scorer regenerated CSV+JSON+HTML on eight known training curves; both exact per-control score parity and frozen-model binary reproduction were checked in the original audit.
- NOT tested: ESO live network, windows launcher, GUI window, FITS image inspection, independently sampled sky regions, novelty matching.

## Clean checkout instructions
```bash
cd research/virac2-infrared-discovery-engine
python3.12 -m venv .venv
# activate .venv as appropriate for the OS
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python verify_bundle.py
python engine.py info
python engine.py score --input examples/known_training_lightcurves.csv --out results/my_control_run_001
python training/reproduce_frozen_model.py --out /tmp/virac2_verification.joblib
```
`results/my_control_run_001` MUST NOT already be a nonempty directory. On Windows use a fresh directory of your choice rather than `/tmp`.

## Expected outputs
`candidate_ranking.csv`, `candidate_audit.json`, `candidate_report.html`, `review_checklist.csv`, `run_provenance.json`, `summary.json`. `run_provenance.json` must record hashes, dependency versions and run settings. Check that demo records are `known_training_control_not_discovery` (or other appropriately blocked status), not discoveries.

## Model reproduction
`training/reproduce_frozen_model.py` uses saved `training/training_rows.json`, the frozen training recipe and the known SHA. It must agree bit-for-bit in the tested Python/scikit-learn environment or loudly reject mismatches; binary hash differences on different library versions can occur and must be investigated, not silently accepted. Keep the original frozen joblib immutable.

## Quality gates
Always verify correct passband (`Ks`), distinct exposure `catid`, canonical string `sourceid`, finite `mag`/`emag`, valid `mjdobs`, consistent astrometry, sufficient valid nights, plausible peak coverage and physical fit stability. Insufficient evidence means abstain, **not** non-variable. Fail closed on unknown/ambiguous inputs. Never reuse already opened sky as an independent holdout.

## Integrity index
`SHA256SUMS.txt` contains cryptographic hashes of the shipped files. Verify with `sha256sum -c SHA256SUMS.txt` from this directory. The checksum file excludes itself. These hashes protect against accidental changes, **not** authenticated signatures.
