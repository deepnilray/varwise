# AI-agent instructions for this directory

This document governs the VIRAC2 infrared engine subtree. These are scientific safety and reproducibility rules, not a claim about model capability.

- Act on the user's goal: independently discover **previously unreported real astronomical objects/events** using archival data, not just classify known objects. Maximize rigor, test failures and scientific usefulness. Proceed on feasible tasks without unnecessary questions.
- Do not claim any new object, event, outperforming baseline, or validated model until a **new** blind sky/source-disjoint evaluation plus relevant image/catalog evidence exists. The current model is a candidate ranker only.
- Training examples must be real, provenance-recorded observations with evidence-backed labels. Simulated/injected signals may be used as a clearly labeled recovery/completeness test, never called real discoveries or used to inflate real-data validation.
- Never reuse opened validation tiles for independent evaluation after algorithm changes. Do not mix overlapping sources/exposures/fields across splits. Freeze the split, algorithm, thresholds and checksums **before** obtaining blind labels/fluxes.
- Do not tune to the historical 366-source development set, the earlier v7/v10 fields, or old inspected failed alerts and then call the outcome fresh. Training can be expanded only with genuine additional samples with documented selection and separated untouched holdout.
- Refuse unsafe joblib loading from unknown/unverified origins; use the pinned checksums and trusted runtime, and preserve original files. Do not overwrite model, data, results or closed runs.
- For each real candidate: inspect original FITS frames, detect source/blend/edge issues, check astrometry and repeats, compare physical alternatives and uncertainty, crossmatch archival catalogs and published surveys with provenance, then seek independent review.
- Treat a known source with a new event differently from a newly discovered source, and a published/classified event differently from a novel event.
- Every modification requires tests, versioned manifest and an evidence ledger entry; include commands, elapsed observation window and scientific limitations.
- Keep old unrelated repository contents unmodified. See `SOL_6_1_START_HERE.md` and `BACKLOG_AND_ACCEPTANCE.md` before implementing changes.
