# Brutal engineering and scientific audit — 17 November 2026

## Executive verdict
**Working local astronomical candidate scorer: YES. Discovery-qualified detector: NO.** The frozen 400-tree Extra Trees model reproduces its archived development-fold scores. It has never passed an appropriately blind, image-vetted, representative archive evaluation. No new object or microlensing event has been discovered or confirmed by this package.

## Source-material audit
- Examined the delivered ZIP's `engine.py`, `core/feature_pipeline.py`, `core/predict.py`, `core/physical_features.py`, the actual trained joblib weights, frozen training features and development fold outputs. Compared with the saved VIRAC2 v12 archive and its own readiness notes.
- The original frozen data contain **366 scorable labeled sources** in **138 sky/exposure groups**, 43 selected numerical features and mixed published classification strengths; there are not 366 image-confirmed astrophysical lenses.
- Development confusion matrix: **TN 269, FP 3, FN 4, TP 90**. Development AP: **0.993140**. None of these counts describes a blind full-survey run.
- Original development sensitivity 90/94 = **95.74%**; 95% Wilson interval **89.56%–98.33%**. Comparator false-alarm fraction 3/272 = **1.10%**; 95% Wilson interval **0.38%–3.19%**. These intervals do not incorporate selection bias, survey artifacts or nonexchangeable observations, and therefore underestimate real-world uncertainty.
- Cohort imbalance: the original test recovered **61/61** cases in the `microlensing` cohort but only **29/33** in the distinct `Microlens` cohort. That matters for distribution shift.
- No training-cohort source pairs were separated by <=2 arcsec in a geometry-only check; this is not proof that all group dependence or label leakage has been eliminated.

## Demonstrated numerical instability
For archived source `12764128009522`, saved training diagnostics gave t0=55925.6075 MJD, tE=37.7948 days, and no fit convergence. Re-extraction from the bundled real demonstration photometry yielded approximately t0=55940.6668 MJD, tE=392.8862 days and a converged fit. Photometry export precision and local optimization can change the chosen minimum. **Physical parameters and derived features cannot be promoted to discovery evidence without robust modeling and uncertainty analysis.**

## Findings and disposition

| Severity | Issue | Outcome in this revised bundle | Remaining obligation |
| --- | --- | --- | --- |
| CRITICAL | No independent image-artifact negatives or fresh blind region | Cannot fix with archived curated rows; model blocked from discovery claims | Collect real artifact/quiet cohorts, seal new sky split, evaluate ONCE |
| CRITICAL | Numerical physical-fit sensitivity and occasional failed convergence | Added per-lead tiny-perturbation stability check, convergence and baseline flags | Refit surviving candidates with robust multistart optimization and posterior uncertainty |
| CRITICAL | No original FITS image review | All results explicitly mark `original_image_verified=false`; human review checklist provided | Image download, reference image, image subtraction, neighbor comparison, repeated detections |
| CRITICAL | No full catalogue/literature novelty check | All results explicitly mark `novelty_catalog_checked=false` | Check OGLE, VVV/VIRAC2 prior work, SIMBAD, VizieR, VSX and publications |
| HIGH | Blind field-scale ingestion absent | Explicitly documented; input batches and by-ID fetch retained | Build tiled catalog ingestion with bounded streaming and a permanent source registry |
| HIGH | Small biased sample, published candidate label uncertainty | Cohort-by-cohort report and uncertainty intervals, no inflated precision claims | Enlarge ground-truth classes; adjudicate labels; stratify sky/crowding/magnitude/time-scale |
| HIGH | No source-population-aware false-alert estimate | Added exact caveats; no extrapolation from curated variable comparisons | Count false leads per 1e5 source-night opportunities in representative unsearched fields |
| HIGH | Unsafe source/exposure identifier handling | **Fixed:** canonical ASCII integer IDs, no scientific notation/leading zeros, reject precision-unsafe catid | Confirm full ESO catalog ID type domain on actual TAP server |
| HIGH | No passband requirement | **Fixed:** `filter` mandatory, Ks only | Verify ESO network schema live |
| HIGH | Duplicated exposure IDs and anomalous astrometry ignored | **Fixed:** fail-closed preflight; circular RA mean; 5-arcsec scatter flag | Calibrate post-hoc rejection thresholds against real unsearched images |
| HIGH | Short/one-night flashes can fool physical models | Added event-night and pre-/post-baseline flags; weak AIC triggers block | Quantify selection bias and short-event completeness on independent data |
| HIGH | Relabeling a known training curve to a different source ID bypassed training ID exclusion | **Fixed for exact feature reuse:** frozen 33-feature fingerprint registry, distinct block status and adversarial regression test | Similar curves/altered observations require sky-coordinate and source-level provenance checks |
| HIGH | Ranking output could be silently overwritten | **Fixed:** one-run/one-directory, checksums, staging and atomic directory commit | Immutable external storage/independent data provenance recommended |
| MEDIUM | ESO by-ID request lacked strict download validation | **Fixed:** reject unexpected IDs/bands, duplicate requests, exact row-ceiling and overwrite | Live TAP count/pagination test remains necessary; server may cap below requested MAXREC |
| MEDIUM | Large HTML table of potentially hundreds of thousands of sources | **Fixed:** HTML limited to first 500 rows; complete CSV preserved | Build scale-ready dashboard if blind ingestion is added |
| MEDIUM | Original replay script needed unavailable original archive paths | Added standalone **frozen-model reproduction** using saved 366 real feature rows; exact prediction and binary hash parity verified | Full raw-photometry→feature regeneration requires keeping the original full archive |
| MEDIUM | No automated safety regression tests | **Fixed:** 27 adversarial unit tests, 10 legacy fold/integrity assertions | Real-data acceptance/integration tests with ESO and FITS unavailable offline |
| MEDIUM | Model supply chain relied only on unsigned joblib+manifest | SHA-256 verifies internal consistency, but **NOT authenticity against malicious repackaging** | Sign releases, avoid loading untrusted joblib, record dependencies |

## Explicitly NOT fixed
1. **No new independent telescope data were downloaded** in this audit. Re-scoring old development or validation data cannot become a new blind result.
2. **No real negative sample of image-confirmed detector artifacts** is included, and the prior failed high-scoring alerts are not turned into new positives.
3. **No full-survey search, image-difference pipeline, astrophysical population completeness map, orbital/parallax lens modeling, event-specific posterior or novel-source announcement** has been completed.
4. New post-fit quality thresholds are conservative engineering choices, **not calibrated physical cuts**; the original development recall and false-positive statistics cannot be assumed after these changes.
5. The ESO network endpoint, Windows launcher and GUI window were not live-tested here. Local command-line inference and mocked network rejection tests passed.

## Priority order if the goal is actually to win
**P0** Obtain image-confirmed false-alert sample and diverse real ordinary/variable sources; freeze new sky-disjoint holdout, never tune on its results. **P1** Compare against the published VVV decision-tree baseline on identical data, adding nonvariable and transient-vs-artifact competitors. **P2** Build automatic tiled data ingestion, deduplication, night/source/neighbor evidence, FITS cutout retrieval and image-level artifact rules. **P3** Model real event detection efficiency versus brightness, timescale, crowding, Galactic coordinates and irregular observing cadence (carefully label injected signals as simulations; never as real discoveries). **P4** Multi-catalog match, independent follow-up, full physical uncertainty posterior and external review. **P5** Only then permit a discovery-quality claim.

## What a win would look like
A new event absent from established published surveys, showing repeatable and astrometrically consistent point-source amplification on independent original exposures, surviving crowded-field blends and all plausible contamination checks, with transparent catalogs, code, light curves, original images, quantitative completeness and a reproducible novelty audit. Discovery claims require independent scrutiny, not a score threshold.
