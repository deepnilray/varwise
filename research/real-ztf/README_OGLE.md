# OGLE + ZTF real-data astronomy research pipeline — Deepnil Ray

## Completed results (2026-10-27)

Everything below comes from original OGLE-IV I-band photometric observations
and OGLE's published variable-star catalog classes, not generated objects.

| Evaluation | Samples | Correct | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| LMC held-out, 20% by star ID | 300 | 298 | 0.9933 | 0.99335 |
| SMC untouched out-of-region evaluation | 300 | 291 | 0.9700 | 0.96996 |

The model trained on 1,200 LMC stars (400 RRab, 400 fundamental-mode
Cepheids, 400 eclipsing binaries). The LMC validation set contained 100 per
class. SMC external evaluation used another 100 per class, with **no retraining**.

These are real completed runs. The exact metrics, class-by-class confusion
matrices, individual holdout predictions and audit files are in `results/`.

**Not a discovery claim.** This classifies known sources. It has not searched
previously unclassified ZTF objects, confirmed a new star, or demonstrated
original mathematics. The two regions share OGLE instrumentation, so the SMC
result is an out-of-region check, not an entirely independent telescope test.

## Data provenance

Original project: [OGLE Collection of Variable Stars](https://ogle.astrouw.edu.pl/main/collections.html).

- RRab: `https://ftp.astrouw.edu.pl/ogle/ogle4/OCVS/lmc/rrlyr/RRab.dat`
- Cepheid F: `https://ftp.astrouw.edu.pl/ogle/ogle4/OCVS/lmc/cep/cepF.dat`
- Eclipsing binary: `https://ftp.astrouw.edu.pl/ogle/ogle4/OCVS/lmc/ecl/ecl.dat`
- SMC variants use the same official paths with `/lmc/` changed to `/smc/`.
- Every photometry file comes from the corresponding `phot/I/<OGLE-ID>.dat` URL.

The source manifest records the **original URL, exact source ID, OGLE label,
number of valid observations and SHA-256 of the original file**. Original files
remain unaltered; invalid/missing points are excluded in memory according to
recorded uncertainty. No human has manually overridden labels.

Catalog classes are not infallible ground truth. Individual objects require
expert verification for any discovery report. Cite the relevant OGLE papers
identified on the official archive page when using this dataset.

## ZTF DR24 real-object pilot (2026-10-27)

We actually queried a 45-arcsecond radius field centered at
**RA 298.0025 deg, Dec +29.87147 deg** through the public SNAD ZTF DR24 API,
retaining its unmodified raw JSON response in
`data/ztf_dr24_field_298_29_45arcsec.json`.

- 750 actual ZTF filter/field light curves, containing 175,419 measurement rows.
- 338 curves passed a predeclared finite/uncertainty/minimum-80-point screen.
- Approximate positional grouping within 1 arcsecond yielded 125 sky positions.
- 4 of those positions matched at least one of three queried published catalogs:
  VSX (`B/vsx/vsx`), Gaia DR3 variable summary (`I/358/varisum`), or the
  published 2020 ZTF periodic-variable catalog (`J/ApJS/249/18`).
- 121 positions had no match in **those three limited searches**. This DOES NOT
  mean 121 discoveries; we have **zero confirmed new objects** so far.

The #1 rank, ZTF OID `686203400115400`, is **already cataloged** as
`ZTFJ195202.69+295209.3`. Our coarse periodogram peaks around `0.49925` d,
while the published ZTF catalog r-band period is `0.99895` d. The harmonic
and observing cadence must be examined; no novel-star claim follows.
See `results/ztf_known_source_period_alias.png`.

An authenticity test compared all 724 quality-flag-zero observations for
ZTF OID `686103400034440` in the mirror with an *independent direct IRSA*
download (838 original epochs, including flagged ones). Observation times,
magnitudes and errors agree within the archived numeric rounding. See
`results/ztf_integrity.json`. This verifies **one** sampled light curve,
not every one of the 750 independently.

Source docs:

- [Official IRSA ZTF DR24 release](https://irsa.ipac.caltech.edu/Missions/ztf.html)
- [SNAD public ZTF DR24 data service](https://db.ztf.snad.space/api/v3/help)
- [VizieR ZTF periodic variables](https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/249/18)

Do not interpret the transferred OGLE model's classes or class probabilities
on ZTF as validated classifications. This search uses only measured ZTF
brightnesses; unknown source identities are **not** turned into training labels.

## Method

1. Deterministically sample 500 real objects per class from the OGLE LMC catalogs.
2. Require at least 80 finite photometric observations with positive uncertainty.
3. Extract features solely from measured time, magnitude and uncertainty;
   compute a Lomb-Scargle period from the actual photometry.
4. Stratified 80/20 split by **unique source ID**, fixed seed `20261027`.
5. Train one predeclared Random Forest: 300 trees, balanced class weights,
   minimum leaf size 2; do not tune against the untouched validation set.
6. Evaluate using confusion matrix, class precision, recall, macro F1 and accuracy.
7. Independently download 300 OGLE SMC stars, run **inference only**.

The model does not read catalog-provided periods, catalog magnitudes, IDs or
labels as input features. Catalog labels train the supervised model and are
used only to evaluate predictions. Catalog periods are retained in an audit
file but not passed to the classifier.

No synthetic light curves, injection/recovery, artificial anomalies, invented
negatives, data augmentation, pseudo-labels or fabricated test rows were used.

## Reproduction on a personal PC

Python 3.12 recommended (and required by the bundled Windows launcher). In a terminal from this directory:

```bash
python -m pip install -r requirements.txt
python audit.py
```

To independently rerun training (will re-use cached original OGLE light curves):

```bash
python run.py --n-per-class 500 --workers 8
python external_smc.py
python audit.py
```

The original downloaded light curves and trained `.joblib` model are bundled,
so rerunning the audit requires no network connection. Re-training may give
identical results when dependency versions and numerical behavior match;
the published metric files are the original completed run, not an estimate.

### Windows: what Deepnil should run

1. Install **Python 3.12 for Windows** from python.org, with Python launcher.
2. Extract the ZIP and open its `ogle_research` directory.
3. Double-click `windows_run.bat`. The first run installs all pinned packages,
   audits the 1,800 OGLE stars and 750 ZTF light curves, reruns the candidate
   screen and catalog checks, and generates the diagnostic plot. The bundled
   evidence makes this first analysis possible without downloading survey
   light curves again (package installation does require internet).
4. Optionally double-click `windows_retrain_ogle.bat` to **retrain** the
   supervised OGLE model using real catalog labels and telescope observations.
   That may need internet to replace missing original source files.
5. To hunt a new patch of ZTF sky, double-click
   `windows_hunt_new_field.bat`. Enter a distinct tag and sky coordinates in
   decimal degrees. Each run is stored separately and the original pilot
   cannot be overwritten by a reused tag. It downloads real ZTF observations,
   ranks signal candidates and queries three published variable catalogs.

Send back **the folder `results/fields/<tag>`** from any new field and we can
work together to vet the top candidates with additional independent catalogs,
image cutouts and time-series reliability tests. Use an additional ZIP if
needed to preserve the original downloaded observations in `data/fields/<tag>.json`.

Crucially, the real-only supervised training is **already complete on OGLE**;
the blind ZTF field remains unlabelled. We do not train on speculative labels
or report a ZTF accuracy without a new, independent labeled test set.

## Honest limitations and next research phase

- The class set is intentionally narrow: RRab / classical F Cepheid / eclipse.
- Object classes can correlate with light-curve cadence, precision and time span.
  Their differences may inflate scores without proving physical generalization.
- Daily observation aliases affect derived Lomb-Scargle periods.
- The independent test is in a separate sky region but within the OGLE survey.
- No ZTF candidate can be described as a discovery without independent catalog
  and image checks, repeat detections and expert scientific vetting.

Next, evaluate cross-instrument performance with real publicly classified ZTF
observations, then screen unclassified ZTF sources **without** turning model
predictions into supposed ground-truth labels.

