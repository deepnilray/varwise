# Expanded training completed — v12

Training grew from **259 to 366 usable real labeled sources** (+41.3%). All 107 additional clearly classified available examples passed feature checks. The combined set contains 70,783 distinct source/exposure/time/magnitude measurement records, 40,307 exposure IDs and 138 independent exposure/sky groups. Five other intended entries remain explicit abstentions. No synthetic observations or invented labels were added.

The new records add 33 published microlensing objects and 74 long-period/dipping variable comparisons. Generic YSO, sparse-source and undated nova labels were excluded rather than assigned unsupported labels. The data combine published classification tiers; not every positive is independently confirmed. This uses all clearly assignable unused labels in the currently downloaded source catalog, not all astronomical data in existence. Newly collected OGLE alert metadata are not mislabeled as downloaded confirmed training curves.

## Training results

86 configuration fits compared an ensemble, Extra Trees, physical-feature logistic regression and histogram boosting. Five outer grouped folds test models and cutoffs chosen using three inner grouped folds. These outer folds use approximately 80/20 groups, not exact row counts. Final model selection uses five grouped folds inside the complete development dataset, followed by fitting all 366 usable sources.

| Nested grouped development result | Count |
|---|---:|
| Known/published microlensing recovered | 90/94 (95.7%) |
| Missed microlensing | 4/94 |
| Other variable classes flagged | 3/272 (1.1%) |
| Other variable classes not flagged | 269/272 |

The three competing-class flags were 2 CV and 1 RR Lyrae. The four missed positives came from the newly included published Microlens class. Average precision was 0.9931. These are development results on a larger, harder cohort and must not be compared directly with the previous smaller cohort's percentage as an estimate of improvement or regression.

Final selected model: 400-tree Extra Trees classifier, minimum leaf size 2, maximum feature fraction 0.7, balanced class weights, 43 measured features. Locked threshold: 0.5228646028279441. The ensemble was evaluated but did not win the final training comparison.

## Verification

- No exposure IDs cross any of the 138 training groups.
- No training source or exposure overlaps the previously opened v7/v10 external checks.
- Existing test scores were not used to fit weights or select thresholds, and those tests were not rescored.
- Portable saved-model predictions agree exactly with pre-conversion predictions on all 366 feature inputs.
- CSV inference was run in a fresh process on a real training source and matched the cached-feature score; output denies discovery confirmation.

The final model is trained and saved. It is still a development candidate: no new independent validation, representative archive false-alert test, or original-image artifact verification has been completed for v12. Discovery readiness is not claimed, and the sealed discovery pilot remains untouched.

Run from project root:

    python v12/infer.py --input measurements.csv --output scores.json

Required columns: sourceid, catid, mjdobs, mag, emag, phot_flag, ambiguous_match; optional filter selects Ks. Dependencies: v7/requirements.txt. Frozen weights, code, input records, exposure audits and fold predictions are included.

Model SHA256: 06e23a8558396eb061eecf9ecc5eb6216e865ce8369d674396a748bff9bd7209
