"""Train a *real-observations-only* OGLE stellar light-curve baseline.

No synthetic observations, data augmentation, synthetic negatives, pseudo-labels,
catalog features, or catalog-provided periods enter the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import requests
from astropy.timeseries import LombScargle
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "data" / "catalogs"
LIGHTCURVES = ROOT / "data" / "lightcurves"
RESULTS = ROOT / "results"
SEED = 20261027

SOURCES = {
    "RRab": ("rrab.dat", "rrlyr"),
    "Cepheid_F": ("cepheid_f.dat", "cep"),
    "Eclipsing": ("ecl.dat", "ecl"),
}
BASE = "https://ftp.astrouw.edu.pl/ogle/ogle4/OCVS/lmc"


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_catalog(class_name: str):
    filename, directory = SOURCES[class_name]
    lines = (CATALOG / filename).read_text().splitlines()
    objects = []
    seen = set()
    for line in lines:
        fields = line.split()
        if len(fields) < 4:
            continue
        obj_id = fields[0]
        if obj_id in seen or not obj_id.startswith("OGLE-LMC-"):
            continue
        seen.add(obj_id)
        try:
            catalog_period = float(fields[3])
        except ValueError:
            catalog_period = None
        url = f"{BASE}/{directory}/phot/I/{obj_id}.dat"
        objects.append({"source_id": obj_id, "class_label": class_name,
                        "catalog_period_days": catalog_period, "source_url": url,
                        "catalog_path": f"data/catalogs/{filename}"})
    return objects


def cleaned_observations(path: Path, minimum: int = 80):
    try:
        arr = np.loadtxt(path, usecols=(0, 1, 2), ndmin=2)
    except (ValueError, OSError, IndexError):
        return None
    if arr.shape[1] != 3:
        return None
    clean = arr[np.isfinite(arr).all(axis=1) & (arr[:, 2] > 0) & (arr[:, 2] < 1.0)]
    if len(clean) < minimum or np.ptp(clean[:, 0]) < 10:
        return None
    clean = clean[np.argsort(clean[:, 0])]
    if np.unique(clean[:, 0]).size < minimum:
        return None
    return clean


def get_one(rec, min_points: int):
    out = LIGHTCURVES / (rec["source_id"] + ".dat")
    if out.exists() and out.stat().st_size:
        arr = cleaned_observations(out, min_points)
        if arr is not None:
            return dict(rec, local_path=str(out.relative_to(ROOT)),
                        sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                        valid_points=len(arr), status="ok")
    try:
        # The official website contains the original measured photometry.
        resp = requests.get(rec["source_url"], timeout=25, headers={"User-Agent":"OGLE-research-audit/1.0"})
        if resp.status_code != 200 or len(resp.content) < 40:
            return dict(rec, status=f"http_{resp.status_code}")
        out.write_bytes(resp.content)
        arr = cleaned_observations(out, min_points)
        if arr is None:
            return dict(rec, status="insufficient_or_invalid_photometry")
        return dict(rec, local_path=str(out.relative_to(ROOT)),
                    sha256=hashlib.sha256(resp.content).hexdigest(),
                    valid_points=len(arr), status="ok")
    except requests.RequestException as exc:
        return dict(rec, status=f"network_{type(exc).__name__}")


def collect_records(n_per_class: int, min_points: int, workers: int):
    LIGHTCURVES.mkdir(parents=True, exist_ok=True)
    all_records = []
    rejection_counts = Counter()
    for name in SOURCES:
        candidates = read_catalog(name)
        rng = random.Random(SEED + len(name) * 307)
        rng.shuffle(candidates)
        selected = []
        tested = 0
        batch_size = min(128, max(16, workers * 8))
        while len(selected) < n_per_class and tested < len(candidates):
            batch = candidates[tested: tested+batch_size]
            tested += len(batch)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(get_one, item, min_points) for item in batch]
                for future in as_completed(futures):
                    rec = future.result()
                    if rec["status"] == "ok":
                        selected.append(rec)
                    else:
                        rejection_counts[rec["status"]] += 1
            # Sort by catalog ID for a deterministic record selection per batch.
            # The initial catalog shuffle supplies randomized choices.
            selected = sorted(selected, key=lambda r: r["source_id"])[:n_per_class]
            print(f"downloaded {name}: {len(selected)}/{n_per_class}, attempted {tested}", flush=True)
        if len(selected) != n_per_class:
            raise RuntimeError(f"Only {len(selected)} usable {name} objects, need {n_per_class}")
        all_records.extend(selected)
    assert len({r["source_id"] for r in all_records}) == len(all_records)
    write_json(RESULTS / "source_manifest.json", {
        "source": "Official OGLE-IV Collection of Variable Stars LMC classified catalogs",
        "label_policy": "Use OGLE's published class only; no generated labels",
        "selection_seed": SEED,
        "minimum_valid_points": min_points,
        "rejections": dict(rejection_counts),
        "records": all_records,
    })
    return all_records


def extract_features(rec):
    arr = cleaned_observations(ROOT / rec["local_path"], minimum=2)
    if arr is None:
        raise ValueError(f"Invalid photometry: {rec['source_id']}")
    return extract_features_from_array(arr)


def extract_features_from_array(arr):
    """Same frozen feature set for other surveys, using measured [time,mag,error]."""
    t, mag, error = arr.T
    t = t - min(t)
    median = np.median(mag)
    resid = mag - median
    variance = np.var(mag)
    diffs = np.diff(mag)
    # All features derive from real observations. Catalog class/period excluded.
    feats = {
        "log_n_obs": float(np.log1p(len(mag))),
        "log_timespan": float(np.log1p(np.ptp(t))),
        "std_mag": float(np.std(mag)),
        "mad_mag": float(1.4826 * np.median(np.abs(resid))),
        "q95_q05_mag": float(np.quantile(mag, 0.95)-np.quantile(mag, 0.05)),
        "iqr_mag": float(np.quantile(mag, .75)-np.quantile(mag, .25)),
        "skew_mag": float(skew(mag)),
        "kurtosis_mag": float(kurtosis(mag)),
        "von_neumann_ratio": float(np.mean(diffs**2)/(variance+1e-12)),
        "weighted_constant_chi2": float(np.mean((resid/error)**2)),
        "frac_bright_tail": float(np.mean(mag < median - np.std(mag))),
    }
    # Computing the period *from telescope observations* prevents label leakage
    # from the OGLE catalog's published orbital/pulsation period.
    frequencies = np.linspace(0.0125, 5.0, 4000)
    power = LombScargle(t, mag, error, center_data=True).power(frequencies, method="fast")
    top = int(np.argmax(power))
    period = 1.0 / frequencies[top]
    feats["ls_period_days"] = float(period)
    feats["ls_max_power"] = float(power[top])
    feats["ls_power_q95"] = float(np.quantile(power, .95))
    feats["ls_power_peak_ratio"] = float(power[top] / (np.quantile(power, .95)+1e-12))
    # The published period is never used as a feature.
    return feats, period


def train_model(records):
    feats = []
    pred_periods = {}
    for idx, rec in enumerate(records):
        feature, period = extract_features(rec)
        feats.append(feature)
        pred_periods[rec["source_id"]] = period
        if (idx+1) % 150 == 0:
            print(f"real photometry features: {idx+1}/{len(records)}", flush=True)
    feature_names = sorted(feats[0])
    x = np.array([[row[k] for k in feature_names] for row in feats], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite feature(s): inspect observed data; do not impute")
    y = np.array([r["class_label"] for r in records])
    ids = np.array([r["source_id"] for r in records])
    train_idx, val_idx = train_test_split(np.arange(len(records)), test_size=.20,
                                          random_state=SEED, stratify=y)
    if set(ids[train_idx]) & set(ids[val_idx]):
        raise AssertionError("source leakage detected")

    model = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                   random_state=SEED, class_weight="balanced",
                                   n_jobs=-1, min_samples_leaf=2)
    model.fit(x[train_idx], y[train_idx])
    predicted = model.predict(x[val_idx])
    labels = list(SOURCES)
    report = classification_report(y[val_idx], predicted, labels=labels,
                                   output_dict=True, zero_division=0)
    metrics = {
        "basis": "original OGLE-IV measured light curves / original class labels",
        "training_count": len(train_idx), "validation_count": len(val_idx),
        "split_seed": SEED, "class_counts_train": dict(Counter(y[train_idx])),
        "class_counts_validation": dict(Counter(y[val_idx])),
        "no_source_overlap": True,
        "validation_accuracy": float(accuracy_score(y[val_idx], predicted)),
        "validation_balanced_accuracy": float(balanced_accuracy_score(y[val_idx], predicted)),
        "validation_macro_f1": float(f1_score(y[val_idx], predicted, average="macro")),
        "confusion_labels": labels,
        "confusion_matrix": confusion_matrix(y[val_idx], predicted, labels=labels).tolist(),
        "classification_report": report,
        "feature_names": feature_names,
        "feature_importances": {k:float(v) for k,v in zip(feature_names, model.feature_importances_)},
        "limitations": [
            "In-survey validation, not an independent telescope/survey test.",
            "Published catalog classes are not error-free; this evaluates catalogue-label agreement.",
            "Classification among known variable-star types does not establish discovery of new stars.",
            "Finite period grid (0.2-80 days) and daily sampling can cause period aliases.",
        ],
    }
    split = {
        "policy": "stratified 80/20 by distinct OGLE source_id; no overlapping stars",
        "seed": SEED,
        "train": [str(s) for s in ids[train_idx]],
        "validation": [str(s) for s in ids[val_idx]],
    }
    write_json(RESULTS / "split.json", split)
    write_json(RESULTS / "metrics.json", metrics)
    write_json(RESULTS / "validation_predictions.json", [
        {"source_id": str(ids[i]), "true_catalog_label":str(y[i]),
         "predicted_label":str(predicted[j]),
         "predicted_period_from_photometry_days":float(pred_periods[str(ids[i])]),
         "catalog_period_for_audit_only_days": next(r["catalog_period_days"] for r in records if r["source_id"]==str(ids[i]))}
        for j,i in enumerate(val_idx)
    ])
    joblib.dump({"model": model, "feature_names":feature_names,
                 "labels":labels, "training_seed":SEED}, RESULTS/"model.joblib")
    print("VALIDATION METRICS " + json.dumps({k:metrics[k] for k in
          ("validation_accuracy","validation_balanced_accuracy","validation_macro_f1","confusion_matrix")}),flush=True)
    return metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--n-per-class",type=int,default=500)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--min-points",type=int,default=80)
    args=parser.parse_args()
    RESULTS.mkdir(parents=True,exist_ok=True)
    counts={name:len(read_catalog(name)) for name in SOURCES}
    print("OGLE catalog counts " + json.dumps(counts), flush=True)
    records=collect_records(args.n_per_class,args.min_points,args.workers)
    train_model(records)


if __name__=="__main__":
    main()
