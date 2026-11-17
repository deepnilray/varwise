# Architecture, scientific assumptions and data lineage

## Current program

Input CSV → `core/input_guard.py` raw ID/passband/date/coordinate validation → `engine.py` source grouping and nightly quality filtering → `core/feature_pipeline.py`, `core/physical_features.py`, `core/predict.py` extract 43 fields and physical diagnostics → pinned Extra Trees joblib scores → extra post-fit gates → immutable CSV/JSON/HTML report + provenance.

Important: `model/virac2_microlensing_v12_retrained.joblib` is frozen and model inference uses a serialized Python object; load only from this trusted repository and verify `model/manifest.json` SHA256 first. Joblib is not a safe interchange format for untrusted external files.

## File map
- `engine.py`: CLI `info`, `score`, `fetch` (by supplied source IDs only); core score stage, ESO API calls, report generation. No full field survey, no FITS processing.
- `gui.py`, `RUN_WINDOWS.bat`, `RUN_LINUX_MAC.sh`: local interface and dependency setup, not observed in live Windows/macOS environments.
- `core/input_guard.py`: strict schema, canonical IDs, filter, duplicate, date and coordinate guards.
- `core/feature_pipeline.py`, `core/predict.py`, `core/physical_features.py`: featurization, physical fits and ML feature row.
- `training/training_rows.json`: 366 saved real feature/label rows; it is **not** original FITS or complete raw photometry for 366 sources.
- `training/fold_predictions.json`: 366 development-fold predictions, not independent new test data.
- `training/reproduce_frozen_model.py`: re-train exact frozen weights from saved feature matrix and verify binary/prediction parity.
- `model/training_source_ids.txt`: list of known training IDs. Program also detects some disguised reuse via feature fingerprint; not a substitute for sky/catalog match.
- `examples/known_training_lightcurves.csv`: 8 known training sources, *never* blind evaluation.
- `results/known_training_controls/`: a saved eight-source software regression demonstration, not discoveries.
- `verify_bundle.py`: archived development checks; `tests/test_adversarial.py`: integrity/adversarial tests.
- `BRUTAL_REVIEW.md`, `VALIDATION_PROTOCOL.md`, `BACKLOG_AND_ACCEPTANCE.md`: limitations and next work.

## Real dataset lineage
Astronomical source: ESO VVV/VVVX VIRAC2 DR3 public photometric catalogs and published literature source labels. Origin of current feature/label matrix: archived `VIRAC2_v12_Expanded_Training.zip` (2026-11-09). Model retrained 2026-11-17 using 366 archived feature vectors. 138 training/exposure sky groups are reported. No guarantee the underlying source lists cover all events or false artifacts. Full per-source original telescope images/raw light curves for all 366 are **not** included here. The eight-object demo does contain real photometry but is not novel.

## What model predicts
A classification score for **microlensing-like light curves vs curated published other variable categories**, not a calibrated posterior that gravitational lensing occurred, that a lens is a dark object/planet, or that the observation is novel. The extra conservative post-score gates are unvalidated heuristics. Artifact detection, novel object finding and confirmed physical parameter inference are separate tasks.

## Data policy
Do not infer source discovery from absence in three catalogs. Preserve URLs/access times/checksums for queries, FITS originals, image cutouts, photometric data and all labels. Propagate uncertainties and known selection biases. Catalog astrometry is epoch-dependent; matching must account for uncertainties/proper motions and confusion in crowded fields.

## Public documentation
- ESO VIRAC2 DR3: https://www.eso.org/sci/publications/announcements/sciann17684.html
- ESO rejected source table: https://www.eso.org/qi/catalog/show/425
- IRSA/ZTF docs for *separate optical* project: https://irsa.ipac.caltech.edu/docs/program_interface/ztf_lightcurve_api.html
- VVV 700-million-light-curve microlensing study: https://academic.oup.com/mnras/article/506/2/2482/6313300
Check the current catalog schema and licensing directly from official services before any new ingestion.
