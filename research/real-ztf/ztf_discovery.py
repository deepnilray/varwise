"""Honest, real-observation ZTF search and exploratory OGLE-model transfer.

This script does NOT infer a confirmed stellar type or novelty. It collects
published ZTF DR24 photometry through the documented SNAD ZTF data service,
preserves every input row, and ranks interesting measured variability.

Example Windows run, from this folder:
  python ztf_discovery.py --ra 298.0025 --dec 29.87147 --radius 45
  python ztf_discovery.py --input data/ztf_dr24_field_298_29_45arcsec.json
"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import requests

from run import ROOT, RESULTS, extract_features_from_array, write_json

BASE = "https://db.ztf.snad.space/api/v3/data/dr24/circle/full/json"
DEFAULT_RAW = ROOT / "data" / "ztf_dr24_field_298_29_45arcsec.json"


def angular_arcsec(a, b):
    # Small-angle approximation is more than sufficient within a 60-arcsec field.
    dra = (a[0] - b[0]) * math.cos(math.radians((a[1] + b[1]) / 2))
    return 3600 * math.hypot(dra, a[1] - b[1])


def observed_features(photometry):
    """Remove only flagged/invalid photometry, never invent replacement rows."""
    arr = []
    for point in photometry:
        try:
            value = (float(point["mjd"]), float(point["mag"]), float(point["magerr"]))
        except (KeyError, ValueError, TypeError):
            continue
        if not all(math.isfinite(x) for x in value):
            continue
        if not (0 < value[2] < .2 and 12 <= value[1] <= 22):
            continue
        arr.append(value)
    if len(arr) < 80:
        return None
    data = np.asarray(sorted(arr), dtype=float)
    if len(np.unique(data[:, 0])) < 80 or np.ptp(data[:, 0]) < 100:
        return None
    return data


def scan(records, model_path: Path):
    fitted = joblib.load(model_path)
    model = fitted["model"]
    features = fitted["feature_names"]
    model_origin=fitted.get("training_source", "OGLE labeled stars, not validated on ZTF")
    model.n_jobs = 1  # avoid launching a thread pool for every source
    counts = Counter()
    candidates = []
    feature_vectors = []
    for source_id, item in sorted(records.items()):
        counts["sources_seen"] += 1
        lightcurve = item.get("lc", [])
        counts["actual_measurement_rows"] += len(lightcurve)
        arr = observed_features(lightcurve)
        if arr is None:
            counts["sources_rejected_quality_or_sparse"] += 1
            continue
        try:
            values, period = extract_features_from_array(arr)
        except (ValueError, FloatingPointError):
            counts["sources_rejected_feature_error"] += 1
            continue
        if not all(np.isfinite(list(values.values()))):
            counts["sources_rejected_nonfinite_features"] += 1
            continue
        feature_vectors.append([values[name] for name in features])
        # Signal-based ranking is independent of the classifier's class output.
        # High power + robust scatter above point noise motivates manual review.
        noise_ratio = values["mad_mag"] / max(float(np.median(arr[:,2])),1e-6)
        score = float(values["ls_max_power"] * np.log1p(max(noise_ratio,0)))
        c = item["meta"]["coord"]
        candidates.append({
            "ztf_oid":source_id,
            "ra_deg":float(c["ra"]), "dec_deg":float(c["dec"]),
            "filter":item["meta"].get("filter"),
            "n_original_points":len(lightcurve), "n_quality_points":len(arr),
            "observed_span_days":float(np.ptp(arr[:,0])),
            "median_magnitude":float(np.median(arr[:,1])),
            "robust_amplitude_mag":float(values["q95_q05_mag"]),
            "robust_scatter_over_error":float(noise_ratio),
            "periodogram_best_period_days":float(period),
            "periodogram_peak_power":float(values["ls_max_power"]),
            "ranking_score":score,
            "annotation":"UNVETTED: forced known-class prediction; NOT a novelty detector or calibrated probability. Training source: "+model_origin,
        })
        counts["usable_source_lightcurves"] += 1
        if counts["sources_seen"] % 100 == 0:
            print(f"Screened {counts['sources_seen']}/{len(records)} ZTF light curves",flush=True)
    if candidates:
        x = np.asarray(feature_vectors)
        predicted = model.predict(x)
        probabilities = model.predict_proba(x)
        for row, label, probs in zip(candidates,predicted,probabilities):
            row["exploratory_ogle_type_suggestion"] = str(label)
            row["exploratory_ogle_max_probability"] = float(np.max(probs))
    candidates.sort(key=lambda x:(-x["ranking_score"], x["ztf_oid"]))
    # Group positional duplicates (different ZTF bands/fields) for *display*,
    # without merging photometry or inventing any object-level class label.
    groups = []
    for row in candidates:
        pos = (row["ra_deg"],row["dec_deg"])
        match = next((g for g in groups if angular_arcsec(pos,(g["ra_deg"],g["dec_deg"])) < 1),None)
        if match is None:
            groups.append({"ra_deg":row["ra_deg"],"dec_deg":row["dec_deg"],
                           "representative_oid":row["ztf_oid"],"top_score":row["ranking_score"],
                           "filters_and_oids":[{"filter":row["filter"],"oid":row["ztf_oid"]}],
                           "representative_period_days":row["periodogram_best_period_days"],
                           "n_quality_points":row["n_quality_points"]})
        else:
            match["filters_and_oids"].append({"filter":row["filter"],"oid":row["ztf_oid"]})
    counts["positional_groups_lt_1arcsec"] = len(groups)
    counts["periodogram_peaks_near_1_2_3_4_cycles_per_day"] = {
        str(cycles):sum(abs(1 / x["periodogram_best_period_days"]-cycles)<.05 for x in candidates)
        for cycles in (1,2,3,4)
    }
    return candidates, groups, dict(counts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ra",type=float,default=298.0025)
    p.add_argument("--dec",type=float,default=29.87147)
    p.add_argument("--radius",type=float,default=45,help="radius in arcsec (strictly between 0 and 60)")
    p.add_argument("--input",type=Path,help="an already downloaded, verbatim SNAD DR24 response; offline")
    p.add_argument("--output",type=Path)
    p.add_argument("--tag",help="safe distinct run name for new survey fields, e.g. field01")
    p.add_argument("--model",type=Path,help="optional saved model (otherwise latest trained ZTF model, then OGLE baseline)")
    args = p.parse_args()
    if args.tag and not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", args.tag):
        p.error("Tag must be 1-40 letters, digits, underscore or hyphen")
    output=args.output or (ROOT/"data"/"fields"/(args.tag+".json") if args.tag else DEFAULT_RAW)
    result_dir=RESULTS/"fields"/args.tag if args.tag else RESULTS
    if args.tag and (output.exists() or result_dir.exists()):
        p.error("Run tag already exists. Choose a new tag; old survey data will not be overwritten")
    if args.input:
        raw = args.input.read_bytes()
        url = "Previously downloaded SNAD DR24 public response; see archived query manifest"
    else:
        if not (0 < args.radius < 60 and -90 <= args.dec <= 90 and 0 <= args.ra < 360):
            p.error("Invalid position or radius (arcsec)")
        req = requests.get(BASE, params={"ra":args.ra,"dec":args.dec,"radius_arcsec":args.radius},timeout=90)
        req.raise_for_status()
        raw = req.content
        url = req.url
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_bytes(raw)
    j = json.loads(raw)
    if not isinstance(j,dict) or not j or not all("lc" in v and "meta" in v for v in j.values()):
        raise RuntimeError("Did not receive valid, labeled-as-unvetted ZTF observations")
    if args.model:
        model_path=args.model
    else:
        available=list((RESULTS/"ztf_supervised"/"runs").glob("n*/ztf_supervised_model.joblib"))
        available.sort(key=lambda p:int(p.parent.name[1:]) if p.parent.name[1:].isdigit() else -1)
        model_path=available[-1] if available else RESULTS/"model.joblib"
    rows, groups, counts = scan(j,model_path)
    result_dir.mkdir(parents=True,exist_ok=True)
    write_json(result_dir / "ztf_scan_results.json",{
        "data_release":"ZTF DR24", "observation_origin":"ZTF; public SNAD DR24 lightcurve service",
        "archive_sha256":hashlib.sha256(raw).hexdigest(),"request_url":url,
        "exploratory_model_used":str(model_path.relative_to(ROOT)) if model_path.is_relative_to(ROOT) else str(model_path),
        "field_query":{"ra":args.ra,"dec":args.dec,"radius_arcsec":args.radius},
        "warning":"Real measured rows, but unsupervised candidate selection; no novelty/catalog vetting claimed.",
        "counts":counts,"source_lightcurves_ranked":rows,"positional_groups_ranked":groups})
    print(json.dumps({"counts":counts,"top_five_unvetted_positional_groups":groups[:5]},indent=2))


if __name__=="__main__":
    main()
