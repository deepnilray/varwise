# Deepnil Ray — real-label ZTF research training, Windows edition

## Start here: what has **actually** happened

**Completed 2026-10-27:** a model trained using actual ZTF DR24 r-band light
curves and exact published AAVSO VSX labels `RRAB`, `DCEP`, and `EW`.

| Dataset | Train | Validation | Separate RA-sector test | Validation macro-F1 | RA test macro-F1 |
|---|---:|---:|---:|---:|---:|
| 75-star functionality pilot (25/class) | 48 | 12 | 15 | 0.5556 | 0.8000 |
| 600-star larger completed run (200/class) | 384 | 96 | 120 | 0.83809 | 0.83093 |

Both completed models, original input manifests, confusion matrices, individual
holdout predictions, and audit results are stored in `results/ztf_supervised/runs/`.
The complete 600-source dataset was checked against its original ZTF query
response, SHA-256, archived VSX catalog row and exact published type: audit PASS.

We have **not** established detection of a novel astronomical object. This
classifier knows only three classes of already catalogued *variable* stars.
It has no real labeled nonvariable training class, no unknown-class rejection
that has been independently validated, and no independent-telescope check.
The first 75-star pilot is too small to infer scientific performance.

## Discovery screening — NEW, actually run on archived sky observations

Run **`windows_vet_original_field.bat`** after the 600-star reproduction to
recheck the original observed ZTF patch. This does not train a new model;
it uses recorded telescope brightness measurements and the actual saved
VSX/Gaia/ZTF catalog responses. The readout is `results/CANDIDATE_REVIEW.md`.

**Completed local execution on 2026-10-27:** of 750 ZTF lightcurve source
records in a single 45-arcsecond-radius field, 125 distinct quality-screened
positions were independently reviewed. Four have a close match in one of the
three checked published variable catalogs; 121 were **not ready for a discovery
claim**; zero passed every strict review gate. This is **not** a full-sky
discovery rate or proof that there are no new variables in the field.

The `ztf_candidate_vet.py` checks deliberately ask four plain questions:

1. Is the brightening or dimming larger than the telescope's quoted errors?
2. Does the pattern recur in the older AND newer observations?
3. Does another color filter show the same underlying pattern?
4. Could it be a nearby star, the telescope's daily observing rhythm, or an
   already known cataloged object?

Every accepted and rejected check is recorded for each real sky position.
No new objects are invented, no model probabilities are treated as proof,
and no source is labeled "discovered". The ZTF mirror's supplied data do not
include exposure-level `catflags`; independently recovering original flags,
inspection of real sky images, independent survey confirmation and expert
review are still mandatory before reporting any discovery. The current
thresholds have **not** been calibrated on confirmed real constant stars and
the system has **no verified false-discovery rate**.

We additionally ran `ztf_vetting_positive_benchmark.py` on **all 600**
original archived ZTF r-band stars with exact published VSX types (200 of
each). Passing all *single-band* screening gates: **119/200 RRAB**,
**133/200 DCEP**, **47/200 EW** (299/600 overall). This is a warning of
substantial missed real variables, especially contact binaries; it is NOT an
estimate of discovery sensitivity on an unselected field. The distinct
two-color gate was **not** part of that benchmark, so full-stage sensitivity
is still unknown. All 600 source hashes were reverified in this check. See
`results/ztf_positive_screen_benchmark.json` for exact original IDs and checks.

For a different field, double-click `windows_hunt_new_field.bat`: it now
downloads real observations, checks the three published variable catalogs,
then runs exactly the same stricter vetting, saving an individual report per
tag. A field may contain zero review-ready objects and will say so.

## Two Windows jobs (Python 3.12, 64-bit)

1. Extract this ZIP, open the `ogle_research` directory.
2. Install Python 3.12 from the official python.org Windows installer if
   `py -3.12 --version` does not work in Command Prompt.
3. Double-click **`windows_reproduce_600star_ztf.bat`**. It installs the pinned
   Python dependencies, trains the model **again** on the bundled 600 real
   source records (384 fit, 96 holdout, 120 RA test), independently audits all
   600, and performs a cross-validated learning curve on fit-only data.
   It needs internet for package installation, but does **not** need to invent
   or download any photometry for that completed dataset.
4. For a *larger* training run, double-click **`windows_train_ztf_real.bat`**.
   Start with `400` accepted stars **per class** (1,200 total) and `1000`
   candidate catalog rows per 30-degree RA sector. This performs authentic
   catalog and light-curve downloads, selects one quality-checked ZTF-r curve
   per distinct VSX source, trains, tests, audits, and writes a learning curve.
   It may take a considerable time depending on internet, accepted-source
   availability and CPU; the code prints progress and caches every successful
   raw download for resuming. If the data quota cannot be reached, it exits
   with an error instead of fabricating sources or scores.
5. To investigate new ZTF sky patches, run `windows_hunt_new_field.bat` after
   the training environment is installed. Its ranking and forced known-class
   suggestions do **not** assert discovery.

The larger run is a proposed **next experiment**. A 1,200-star result is not
claimed to exist until its process finishes and source audit passes. Every
distinct `--per-class` target has a separate result directory. An existing
run cannot silently change its catalog selection quota.

## Code inventory

| File | Actual behavior |
|---|---|
| `ztf_supervised.py` | Retrieve AAVSO VSX exact classes; select RA-stratified, position-deduplicated sources; fetch ZTF DR24 genuine r-band curves; use only quality-filtered real points; group-disjoint training/evaluation; save RF model + all individual holdout predictions |
| `ztf_supervised_audit.py` | Recheck each actual VSX catalog row and each telescope source checksum, position, filter, point count and disjoint split |
| `ztf_learning_curve.py` | Three-fold CV exclusively within original fit split at 25/50/75/100% of fit-star counts; no holdout used to choose models |
| `ztf_discovery.py` | Exploratory candidate ranking of real ZTF light curves; does not manufacture labels or claim new objects |
| `ztf_catalog_check.py` | Checks VSX, Gaia DR3 variable summary, and historical ZTF periodic-variable catalogs; absence in these three is not proof of novelty |
| `ztf_candidate_vet.py` | Recheck untouched ZTF and catalog SHA-256; test signal against photometry error, repeated observations in two timespans and two color filters, day/night alias, nearby stars; issue conservative human-review status |
| `ztf_vetting_positive_benchmark.py` | Audit all 600 known real variables again and report how often the individual single-band screening conditions recover known variables; cannot measure nonvariable false positives |
| `windows_vet_original_field.bat` | Reproduce the completed real-field vetting result without downloading additional sky measurements |
| `run.py` | Shared real-photometry feature extraction and older independently completed OGLE benchmark |
| `requirements.txt` | Exact versions used in the completed computations |

All labels are **published catalog metadata**, not labels inferred by our
algorithm. Selected sources use the exact strings `RRAB`, `DCEP` and `EW`,
not fuzzy classes. The published catalog periods are stored only for audit
and source eligibility (0.2–80 d), not used as model input features.

The model never trains on source IDs, sky coordinates, date baseline, or
observation count. The selected 13 measured-photometry features include
observed magnitude distribution statistics and a Lomb-Scargle periodogram
computed from ZTF timestamps/magnitudes/uncertainties.

The primary validation is 20% of **distinct catalog sources** from the main
RA sectors. A completely separate RA 300–360 degree sector supplies the
external test. No training or hyperparameter tuning uses either untouched
test set. This is a **geographic** external test within ZTF, not an
independent instrument test.

## Why training is not "done" yet

There is still far more real labeled data available. The completed 600-star
result has macro-F1 near 0.83 on both validation sets. The train-only learning
curve in `training_only_learning_curve.json` does not show a decisive
performance plateau. Increasing dataset size and expanding the taxonomy may
be valuable, but must be measured.

A reasonable *proposed* research gate before treating this as a reliable
three-class component is stable external macro-F1 around 0.90 or better,
adequate recall for every class, repeated nonleaking validation and a tested
unknown-source rejection protocol. Even that would **not** confirm any new
astronomical source; candidate catalog crossmatches, imaging, blend checks,
alias vetting, and expert review are separate necessities.

## Sources and acknowledgments

- [AAVSO Variable Star Index](https://vsx.aavso.org/vsx/) — published labels;
  via [VizieR B/vsx](https://cdsarc.cds.unistra.fr/viz-bin/cat/B/vsx).
- [ZTF public DR24 at NASA/IPAC IRSA](https://irsa.ipac.caltech.edu/Missions/ztf.html).
- [SNAD real ZTF DR24 photometry interface](https://db.ztf.snad.space/api/v3/help).
- [OGLE Collection of Variable Stars](https://ogle.astrouw.edu.pl/main/collections.html)
  for the older comparison baseline, not for the new ZTF fit data.

When writing a formal report, cite the original AAVSO/ZTF/OGLE science papers
as appropriate, not just archive landing pages.
