# Evidence ledger — claims, evidence and unresolved scientific work

| Claim | Evidence in this repo | Status |
|---|---|---|
| Real supervised model weights exist | `model/*.joblib`, manifest, preserved feature rows | **Verified** |
| 366 training/development labeled real examples | `training/training_rows.json`, `verify_bundle.py` | **Verified within archived records** |
| 400-tree Extra Trees and 43 features | frozen joblib and feature code | **Verified** |
| Development recovery 90/94, other published variable flags 3/272 | `training/fold_predictions.json` | **Verified development; NOT blind discovery** |
| Local inference executes on example | `results/known_training_controls`, unit tests | **Verified, training controls only** |
| Source and input hardening rejects several malicious/malformed edge cases | `tests/test_adversarial.py` | **Verified by unit tests** |
| Features invariant across original FITS/raw archive and saved matrix | Some unstable PSPL fits found | **NOT established** |
| Real representative artifacts rejected at acceptable rate | No image-confirmed independent artifact benchmark | **NOT established** |
| Current model generalizes to fresh sky | No new independent holdout in this package | **NOT established** |
| ESO fetching works live | Only mock/error-handling tests | **NOT verified online** |
| Automated search covers previously unexplored fields | By-ID fetch only | **NOT implemented** |
| High-scoring alerts are real astrophysical objects | No original FITS image checks | **NOT established** |
| Any result is previously unreported | No complete literature/catalog audit | **NOT established** |
| Model beats published VVV baseline | No paired benchmark | **NOT established** |
| New event or object discovered | No surviving verified candidates | **NO** |

**Historic research warning:** Prior development examined several high-scoring VIRAC2 alerts, and seven failed image verification. Those fields and objects must be marked `previously_opened`, not recycled as fresh blind positives or held-out tests. The exact identifiers/fields need recovery from earlier archives before future benchmark design.

## Scientific distinction
- *Unclassified source* means catalog/classification absence, not a new astronomical object.
- *Previously unreported event* is new temporal behavior of a source that may already be catalogued.
- *New physical object* requires robust astrometry/photometry, exclusion of artifacts and association with physical counterpart analysis; a microlensing point-lens fit alone does not identify the lens.

Whenever new independent evidence arrives, append date, URLs, sky tiling and SHA256 records before updating this file. Never revise historic negative outcomes away.
