# VarWise real-data training result — 27 October 2026

## Verdict: trained, but NOT validated for discovery

The model was actually fit on ZTF observations of Gaia-labeled sources.
The held-out performance did not beat a shuffled-label control. **Do not
deploy this model, run a discovery search with it, or claim that it found new
objects.** The VarWISE target catalog was not accessed in this experiment.

| Measurement | Actual result |
| --- | ---: |
| Gaia training metadata sources | 120 |
| Usable ZTF training curves | 110 |
| Training known-variable labels | 42 |
| Training *proxy* nonvariable labels | 68 |
| Sealed validation curves, queried **after** fit | 28 |
| Validation positive / proxy-negative labels | 16 / 12 |
| Actual held-out share | 20.29% |
| Real validation AUC | **0.567708** |
| Mean shuffled-label control AUC | **0.568750** |
| Discovery acceptance threshold | 0.85 |
| Precision at strict signal cuts | 1.0, on very few alerts |
| Recall at strict signal cuts | **0.125** |
| Discovery objects accessed | **0** |
| Independent automated regression tests | **35/35 passed** |

**Protocol:** Gaia training and ZTF training curves were fetched from six
training sky tiles. The classifier was fit, and ten shuffled-label controls
were fit to training data. Only then were Gaia labels and ZTF observations in
two previously untouched validation sky tiles opened. There is no overlap
between the reserved training and validation tiles, and their sky cells are
checked for intersections before any search. The validation AUC and null
control both failed the model gate. The 80/20 split was based on usable real
observations (110 training, 28 held out), not a synthetic split.

The negative labels mean **no Gaia DR3 variability solution found**; they do
not prove those sources are physically constant. Thus even a high score would
not by itself demonstrate discovery precision.

## What went wrong

The training and validation groups differed. A comparison of their medians:

| Class | Training ZTF mean magnitude | Validation ZTF mean magnitude | Training log false-alarm probability | Validation log false-alarm probability |
| --- | ---: | ---: | ---: | ---: |
| Known Gaia variables | 16.355 | 16.989 | -18.740 | -7.488 |
| Gaia proxy negatives | 17.427 | 16.171 | -1.148 | -1.373 |

This distribution change, combined with only 110 training examples, likely
contributed to poor generalization; this is a diagnostic interpretation, not
a proven causal attribution. Exploratory simplified-model checks using the
**already opened** validation data did not produce a scientifically credible
model. That old validation set can no longer count as a fresh independent
test for any newly tuned model. A future model must use *new, untouched* sky
tiles for its final evaluation.

## Real archive repair, tested end to end

The ordinary ZTF positional endpoint began returning HTTP 400 with
`UsageFault: Internal error (invalid TAP query)` even for a position that had
worked previously. A documented alternate output format and a previous
public release produced the same error. The repaired code uses the official
public ZTF DR24 HATS Objects Table on S3 to find ZTF object IDs, then queries
the ZTF ID endpoint directly (bypassing the broken positional TAP lookup).
The normal astrometric and quality checks still run. A real fallback probe
retrieved **940 quality-filtered r-band observations** of ZTF object
`656210200002249`, near RA 63.38383, Dec 35.04224, in about 13.5 seconds.

Official documentation:

- https://irsa.ipac.caltech.edu/docs/program_interface/ztf_lightcurve_api.html
- https://caltech-ipac.github.io/irsa-tutorials/ztf-lightcurves/

## Saved artifacts

- `upload/varwise_gaia_hunt.py`: complete repaired pipeline, source SHA-256
  `8813b22a8bf954a1635e4b08677659cfa215554bd265d2a47edab9fcd94b7f69`.
- `test_varwise_gaia_hunt.py`, `verification/test_independent_audit.py`:
  35 passing tests total, including train/validation isolation.
- `validation_report.json`: exact scores, class counts, sky tile locations,
  source-code hash, and `validated_for_discovery: false`.
- `training_features.csv`: 110 **training-only** labeled feature rows.
- `sealed_validation_features.csv`: 28 truly unseen-on-fit rows, now exposed
  for auditing this trial.
- `model_RESEARCH_ONLY_FAILED_VALIDATION.joblib`: explicitly quarantined fit,
  deterministically reproduced from the **training** CSV, with the old
  validation checked only afterward. It reproduces AUC 0.567708 and is
  deliberately rejected by `load_validated_model`.
- `varwise_real_training_cache_20261027.zip`: successful ZTF source downloads
  and HATS object-ID lookups, verified intact, for reproducibility.
- `verification/export_research_only_model.py`: reproduction/export script.

## GitHub disposition

The connected GitHub account is `deepnilray`. No clearly corresponding
VarWise repository exists among the repositories accessible to the connection.
The quarantined model is **not uploaded** into an unrelated public repository.
The user can provide an `owner/repository` target; any upload must retain the
prominent research-only/failed-validation designation. A discovery-valid model
should be published only after a new genuinely unseen evaluation passes.

## Next valid science step

Build a larger, better-defined training cohort, including confidently quiet
controls and brightness/survey-coverage balancing. Fit before opening **new**
reserved validation sky tiles, and require a strong held-out accuracy *and*
low false-positive rate. The current trial proves the lock and real-data
downloader work; it does **not** prove a discovery-capable classifier.
