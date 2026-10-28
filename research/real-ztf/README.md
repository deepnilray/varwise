# Real ZTF variable-star research — independent experiment

> **Research prototype, not a confirmed discovery.** This is a separate
> experiment from the VarWise/Gaia model in the parent repository. Do not
> compare their scores or infer that the parent project's failed model passes.

This project tests whether a machine can recognize three *already cataloged*
types of variable stars from real Zwicky Transient Facility (ZTF) brightness
measurements. It then screens a small public telescope field for candidates
that might deserve human attention. **It has not discovered a new star.**

## Real completed results (27–28 October 2026)

| Check | Actual result |
|---|---:|
| Published AAVSO VSX-labeled ZTF-r source records | 600: 200 RRAB, 200 DCEP, 200 EW |
| Train fit | 384 separate stars |
| Normal held-out validation | 96 stars; macro F1 **0.83809**; 81/96 correct |
| Held-out sky region, RA 300–360 degrees | 120 stars; macro F1 **0.83093**; 100/120 correct |
| Independent original-data/label SHA checks | **PASS, 600/600** |
| Original field, raw ZTF light curves | 750 source IDs, 175,419 real photometry rows |
| Screened usable positions in that field | 125 |
| Positions matching one of the three variable catalogs checked | 4 |
| Unconfirmed/failed stricter review | 121 |
| Proven new object | **0** |

The research screen was *also* applied to all 600 authentic known positive
variables (one filter per star): 119/200 RRAB, 133/200 DCEP, and 47/200 EW
passed its four one-band checks. These 299/600 successes expose substantial
missed detections, particularly of eclipsing binaries. **This is not a
discovery F1 score.** We have no verified non-variable/artifact negative class
and therefore **cannot** calculate the discovery false-positive rate or F1.

On 28 October we extracted the complete archived dataset into a fresh folder,
retrained the model from the original genuine observations, reproduced both F1
scores, and reran the audit, field screen, and known-positive checks. All
packaged input/code SHA-256 values were separately verified.

## Why the GitHub folder does not contain 180 MB of raw inputs

The repository contains **executable code and compact results**, not the
original bulk telescope responses, 1,800 OGLE observations, or trained binary
model files. Those original data, catalogs, models, split manifests, and
checksums are all in the separately archived, CRC-checked ZIP:

**[Full Windows research ZIP (43 MB)](https://chatgpt.com/library?file_id=libfile_0db17b382e808191a000ecd05a3a15bc&account_id=personal)**

ChatGPT Library access might require signing in to the relevant account. A
person without ZIP access can still use the scripts to download new original
public data from [ZTF](https://irsa.ipac.caltech.edu/Missions/ztf.html) and
[VSX](https://vsx.aavso.org/vsx/), but new downloads are **not guaranteed** to
reproduce this archived fixed sample because catalogs and releases can change.
All archived identifiers and public query URLs are included in the full ZIP.

## Windows use

For **exactly reproducible archived experiments**, extract the ZIP, then:

1. `windows_reproduce_600star_ztf.bat` — run the actual supervised training,
   holdouts, real-data hashes, and training-only learning curve.
2. `windows_vet_original_field.bat` — reproduce screening of the real 125 sky
   positions.
3. `windows_hunt_new_field.bat` — download a new ZTF sky field and run the
   original-catalog crosscheck plus strict physical-signal screening.

To use **only this GitHub code**, start in this directory, install Python 3.12,
then create an environment and retrieve **fresh real** source data:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe ztf_supervised.py --per-class 200 --catalog-per-sector 200 --workers 3 --max-attempts 2500
```

Fresh catalog selection may fail if fewer usable real records are available;
the script stops with an error rather than inventing missing stars or scores.
The offline reproduction and packaging scripts require the complete ZIP's
archived sources/manifests; they are not standalone from the code-only folder.

## Files and proof

- `ztf_supervised.py`, `ztf_supervised_audit.py`: selection, fitting, honest
  unseen-star testing, and per-source raw-telescope/catalog checksum audit.
- `ztf_discovery.py`, `ztf_catalog_check.py`, `ztf_candidate_vet.py`: measured
  field ranking, checks against three original published catalogs, periodic
  repeatability, second-filter support, and close-neighbor screening.
- `ztf_vetting_positive_benchmark.py`: exactly 600 original published-variable
  examples used as a positive-control check, with no generated labels.
- `results/`: archived evidence in JSON, raw-record SHA hashes, individual
  held-out predictions, and a human-readable report. Bulk raw observations are
  intentionally kept in the full research ZIP.

## Essential limitations

- All classification training and holdout stars are from **ZTF**, not an
  independent telescope; geographic holdout is not independent instrumentation.
- The model can output only RRAB, DCEP, or EW; it does not recognize all other
  possible star types or trustworthy ordinary/nonvariable stars.
- The ZTF mirror input used in field screening lacks original `catflags`.
  Independently inspect original exposures/images and contamination.
- No catalog nonmatch proves novelty. Check recent literature, SIMBAD, Gaia,
  other variability catalogs, aliases and neighboring blended objects.
- **Neither the trained classifier nor the candidate screen is authorized to
  announce a discovery automatically.** Independent follow-up and expert
  verification remain required.

No fabricated training data, negative labels, or discovery claims.
