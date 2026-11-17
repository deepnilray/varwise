# Infrared Discovery Engine — audited, research-only

**Release status: NOT VALIDATED FOR DISCOVERY.** This is a working local candidate scorer using a genuine, frozen Extra Trees model trained on 366 published real-object examples. It does not verify telescope images, catalogue novelty, or scientific discovery. Do not use high scores as proof of new physical objects.

## What changed in the audited build

- **Fail-closed schema.** Explicit `filter` is mandatory; only Ks observations enter scoring. Source/exposure IDs must be canonical positive ASCII integers, preventing scientific-notation rounding and training-ID spoofing. Exposure IDs >2^53 are refused rather than rounded by the frozen feature extractor. Exact light-curve feature fingerprints are also compared against all 366 training curves to block relabelled training examples masquerading as new sources. This catches exact feature reuse, not arbitrary altered copies or catalog matches.
- **Quality gates.** Inconsistent astrometry, duplicated good exposure IDs, impossible observation eras/magnitudes or no valid observations trigger abstention. Missing coordinates block any high-score prioritization.
- **Physical caution.** The saved 43-feature ML model is **unchanged**. High-score sources are separately checked for fit convergence, a stable local refit under tiny perturbations, support on both sides of the putative event, >=2 event nights and improvement over a constant baseline. These are **new heuristic review rules, not independently validated sensitivity estimates.** Their cutoffs require new data.
- **No false verification claims.** All rows explicitly mark image inspection, independent confirmation and catalog novelty as incomplete. High scores remain **unvetted leads** or **blocked review cases**, never discoveries.
- **Audit trail.** Writes immutable run directories with input/model SHA-256, software versions, complete raw rankings, optional per-source evidence and a human-review checklist. Refuses to overwrite existing output directories.
- **Safer ESO fetch.** Validates requested and returned IDs, passband and row ceiling; rejects unexpected responses and doesn't overwrite prior downloads. The network endpoint could not be exercised in this offline environment.
- **Reproducibility and tests.** `python training/reproduce_frozen_model.py` re-fits the shipped real feature matrix and must match predictions exactly. `python verify_bundle.py` verifies archived development folds. `python -m unittest discover -s tests -v` runs additional adversarial integrity tests.

## Quick start

1. Install Python 3.12 (Windows launcher: `py`) and unpack the ZIP.
2. Run `RUN_WINDOWS.bat` (Windows) or `bash RUN_LINUX_MAC.sh` (Linux/macOS), or install `pip install -r requirements.txt` and run `python gui.py`.
3. Choose the real ESO VIRAC2 DR3 light-curve CSV and a **new, empty** output directory. The demo is an archive of eight *known training objects*, not a discovery test.
4. Examine `candidate_report.html`, `candidate_ranking.csv`, `candidate_audit.json`, `review_checklist.csv`, `run_provenance.json`, and `summary.json`.

### Command line

```bash
python engine.py info
python engine.py score --input examples/known_training_lightcurves.csv --out results/my_first_control_run
python verify_bundle.py
python -m unittest discover -s tests -v
python training/reproduce_frozen_model.py --out training/my_frozen_model_reproduction.joblib
# Internet required and official ESO endpoint behavior must be verified locally:
python engine.py fetch --ids 13726720002998 --out results/known_source_photometry.csv
python engine.py score --input results/known_source_photometry.csv --out results/known_source_review
```

**Required CSV fields:** `sourceid,catid,mjdobs,filter,mag,emag,phot_flag,ambiguous_match`. Each source needs at least 5 valid observing nights and 5 distinct exposure IDs. Optional coordinates: `ra`, `de` (ESO name), or `dec`; missing coordinates cause high-score leads to be blocked. The `filter` value must explicitly identify Ks; other bands are excluded from scoring, never silently treated as Ks. RA and Dec are in degrees, `mjdobs` is Modified Julian Date and `mag/emag` are magnitudes. A source must not contain duplicated usable `catid` exposures.

**Input-size limit:** <=250 MB CSV and <=800,000 photometry rows per batch; this is not a full-survey streaming system. Re-running into an existing nonempty output directory is deliberately rejected. For Windows, the launcher creates a Python environment and attempts to install packages; this cannot be Windows-tested here.

## Status meanings

- `known_training_control_not_discovery`: the source ID is in the frozen training set.
- `blocked_training_lightcurve_reuse`: photometry duplicates the features of a frozen training curve despite a different claimed source ID.
- `unvetted_high_score_needs_images_and_novelty_checks`: high score plus passing provisional photometric/fit guards; still NOT scientifically validated.
- `blocked_high_score_requires_data_or_fit_review`: ML score above cutoff but data/fit problems make it unsafe to prioritize automatically.
- `abstain_data_quality_failure`, `abstain_insufficient_valid_nights`: insufficient trustworthy inputs.
- `below_development_threshold_not_nonvariable` and `below_cutoff_possible_physical_anomaly`: model did not produce a high-score classification. Neither means the source is constant.

## Important scientific limitations

The original 90/94 microlensing recovery and 3/272 comparison-object flags were grouped *development* measurements, not independent performance in an unsearched field. They predate the new safeguards, so they **must not be described as validated performance of this audited pipeline**. Comparison objects were named classes of real variables, **not a representative set of nonvariable sources or image artifacts**. High feature importance or score is not proof of genuine lensing, dark objects, or unseen stars. The 2021 VVV survey already used decision-tree detection to recover 1,959 microlensing events from about 700 million light curves; our pipeline must beat a fair baseline, not merely exist. The rejected VIRAC2 tables are contaminated by duplicated/erroneous detections and require original FITS image evidence.

See `BRUTAL_REVIEW.md` for the audit, `training/development_statistics_with_limits.json` for exact counts and confidence intervals and `VALIDATION_PROTOCOL.md` for the release-blocking scientific work. See [ESO VIRAC2 description](https://www.eso.org/sci/publications/announcements/sciann17684.html), [ESO rejected-source catalog](https://www.eso.org/qi/catalog/show/425), and [the published 700-million-light-curve survey](https://academic.oup.com/mnras/article/506/2/2482/6313300).
