# Ranked backlog and explicit acceptance gates

The items below are **not all completed**. A code-only change cannot resolve a missing astronomical-evidence requirement. The operating strategy is to close each issue with an independently reproducible test and quantitative outcome.

## P0 — stop false discoveries (release blocker)

**P0.1 Inventory previous accesses and freeze untouched sky.** Build a machine-readable index of every sky tile, source ID, time range, published list and negative/positive sample used in v7/v10/v12/development, prior blind pilots, and old rejected-source alerts. The present standalone package has only 366 training IDs and 138 development groups, **not** a full authoritative registry of previously opened tiles. Acceptance: file manifest has coordinates, stable IDs, source files and hashes, provenance and overlap checks; assert test targets never intersect ANY known opened region/source/observation exposure. If data absent, mark registry `incomplete`, keep release gated and request archive provenance only if needed.

**P0.2 Build image-confirmed artifact and ordinary-source cohorts.** Include rejected detections, saturation ghosts, diffraction spikes, blends, bad pixels, duplicated detections, known quiet sources, ordinary non-lensing variables and published microlensing events, all from original telescope data. Acceptance: per-example selection rule, human-reviewed FITS images in ≥2 independent exposures, uncertain labels explicitly abstain, checksums and balanced/different sky populations. No synthetic negatives silently substituted.

**P0.3 Final blind evaluation.** Define geometry- and source-disjoint 80/20 split *before* opening final held-out data. Train only with training sky tiles, validate once on sealed untouched tiles, measure denominators for all classes, precision@10/30/100, sensitivity, false leads per 100k sources, abstention rate, grouped bootstrap/intervals and artifact strata. Acceptance: immutable holdout manifest+data hashes, raw predictions, exact confusion matrices, zero leakage, complete classification logs. Do not retune on same holdout.

**P0.4 Physical-model robustness.** Current single local PSPL fits can vary by order of magnitude in timescale. Add multistart fit, sensible bounds, profile likelihood/posterior, sampling cadence windows and alternative transients/variables; test numerical reproducibility across seeds/platforms. Acceptance: documented convergence diagnostics, uncertainties and fit failures; never promote unstable parameters to physical object claims.

## P1 — prove added value

**P1.1 Reproduce 2021 VVV decision-tree baseline**, plus simple non-ML threshold + published criteria, under identical new fields and ground truth. Acceptance: baseline code/parameters/data + paired performance and error bars, not a cross-paper raw accuracy comparison.

**P1.2 Research novelty question:** Do VIRAC2 DR3 reprocessed/main/rejected sources add real, recoverable astrophysical events absent from earlier VVV searches? Quantify missingness/selection efficiency and source-field overlap. Acceptance: predeclared selection, previously published candidate crossmatch, verified incremental source/event count with evidence. Zero incremental value is a valid negative result.

## P2 — scalable real search (engineering prerequisites)

**P2.1 ESO ingestion and streaming.** Verify TAP service live, table/column schema, row-count limits, pagination and restart logic, coordinate WCS, source matching, exposure uniqueness, Ks flag interpretation and rate limits. Acceptance: live e2e in two *non-holdout* fields, row-count reconciliation, deterministic chunk recovery, checksums, no dropped/duplicated data, bounded memory. Current `engine.py fetch` is only by known ID, not a blind survey.

**P2.2 Automatic candidate field ranking.** Build spatial tile scheduler and feature batching with reproducible source manifest. Acceptance: blinded search yields candidate ranking from unknown test targets with full provenance, not hand-picked examples; healthy scheduler under network failure; no overwrite; no injection of known training stars in discovery list.

**P2.3 FITS original-image verification.** Download good/reference cutouts for multiple seasons/epochs, perform positional and PSF consistency checks, neighbor-distance/crowding diagnostics, known detector masks and temporal co-detections. Acceptance: demonstrably rejects previously reviewed false alerts plus a new image-confirmed artifact set; save individual before/after evidence images and provenance; do not use old alerts to claim novel validation.

## P3 — astrophysical and novelty vetting

**P3.1 Literature crossmatch** against SIMBAD, VizieR, VSX, Gaia variability, OGLE, known VVV microlensing catalogues, survey/transient alerts. Match with positional uncertainty and epoch/proper-motion-aware logic when required. Acceptance: machine-readable query logs with dates, search radii, canonical IDs, nearby ambiguous sources and explicit prior-art verdict; absence from one catalog ≠ novelty.

**P3.2 Event evidence bundles** with light curves, original frames, baseline references, alternative model fits, uncertainties, source positions, field selection and reproducible code. Acceptance: independent astronomer can inspect each claim without private hidden assumptions.

**P3.3 New physical object branch:** if user's goal requires *new objects* rather than *new events*, design and train a separate image+astrometric anomaly pipeline. Existing microlensing scores cannot identify unique lens masses/identities or confirm a new star. Acceptance: distinct benchmark, real new-source validation, object identity and counterpart checks.

## P4 — continuous scientific QA

- GitHub CI covering unit suite, archived check, controlled score on demo, provenance/hashes; dependency pinning; signed release artifacts.
- Scientific error-budget report: saturation, crowding, seeing, filter contamination, varying baseline, cadence, masks and class prevalence.
- Compare different magnitude ranges, field positions and seasons; calibrate precision by strata.
- Event injection only as labelled simulations for efficiency, never as evidence of an actual new source.
- Archive complete negative findings, rejected candidates, false alarms and their original evidence.

## Stop rules

No `new discovery` claim if original images absent, if prior-art checks incomplete, if source/sky leakage uncertain, if full evidence cannot be reproduced, or if a purported physical fit is numerically unstable. The engine shall continue to output `unvetted` or `blocked` status until verified externally.
