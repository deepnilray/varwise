# VarWise — Gaia × ZTF astronomy research

A reproducible pipeline for identifying potentially unreported *optical variability* among sources already listed in the VarWISE infrared-variable catalog.

> **Experimental. No confirmed discoveries. The real-world machine-learning model FAILED held-out validation and must NOT be used for discovery.**

## Verified real-data experiment · 27 October 2026

| Test | Actual result |
| --- | ---: |
| Real ZTF curves for model fitting | 110 |
| Never-seen-on-fit, sky-disjoint validation curves | 28 (20.3%) |
| Held-out ROC AUC | **0.568 — FAIL** |
| Mean shuffled-label baseline ROC AUC | **0.569** |
| Required validation AUC | **0.850** |
| Discovery target objects queried with this model | **0** |
| Regression/adversarial tests | **35 passed** |

The classifier is **fitted before any reserved validation sky tiles are queried**. Discovery runs fail closed when validation fails. Gaia sources without published variability solutions are only *proxy negatives*, not verified constant stars. The already-opened October 2026 holdout cannot be used again to claim a fresh independent test for a tuned model.

## Project layout

- `src/varwise_gaia_hunt.py` — complete Gaia/ZTF/VarWISE pipeline, independent signal checks and prior-art vetting.
- `tests/` — adversarial tests, including train-before-holdout.
- `reports/` — exact real training/validation records and scientific limitations.
- `models/` — quarantined **research-only** invalid model, if present.
- `requirements.txt` — Python dependencies.

## Run the code

```bash
python -m pip install -r requirements.txt
python train_first.py --min-train 400  # safety tests, then real training only
```

Only after a genuinely new independent validation passes can `run` or `auto` access VarWISE discovery targets. **Do not bypass the model-validation gate.**

## Archive access and novelty

The normal ZTF positional endpoint can return HTTP 400 `invalid TAP query`. The implementation can fall back to IRSA’s [public ZTF DR24 HATS Objects Table](https://caltech-ipac.github.io/irsa-tutorials/ztf-lightcurves/) and direct [ZTF object-ID light curves](https://irsa.ipac.caltech.edu/docs/program_interface/ztf_lightcurve_api.html), while retaining astrometric and epoch-quality checks.

A VarWISE catalog source is **already known** in infrared. Unreported optical *behavior*, if identified, needs independent multi-filter/temporal checks, catalog and literature vetting, imaging/blend analysis and expert confirmation before any discovery claim.

**Research status: failed validation; no verified new objects.**
