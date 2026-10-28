"""Real-positive-only sensitivity check for the conservative discovery screen.

Exactly 600 original published variable-star examples, SHA-audited.  These are
NOT a nonvariable control set; false-positive rates cannot be measured here.
This does not train anything or generate new labels/photometry.
"""

import argparse
import hashlib
import json
from collections import Counter

from run import ROOT, RESULTS, write_json
from ztf_candidate_vet import band_diagnostics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-class", type=int, default=200)
    args = p.parse_args()
    if args.per_class != 200:
        p.error("Only the 600 audited original real-variable sample is currently benchmarked")
    base = RESULTS / "ztf_supervised" / "runs" / "n200"
    audit = json.loads((base / "training_integrity_audit.json").read_text())
    if audit.get("status") != "PASS" or audit.get("verified_real_source_count") != 600:
        raise RuntimeError("Source audit failed or is missing")
    records = json.loads((base / "accepted_training_sources.json").read_text())["records"]
    if len(records) != 600 or len({r["ztf_oid"] for r in records}) != 600:
        raise RuntimeError("Real-positive source count/identities changed")
    per_class = {label: Counter() for label in ("RRAB", "DCEP", "EW")}
    rows = []
    for i, r in enumerate(records):
        source_file = ROOT / r["source_file"]
        raw = source_file.read_bytes()
        if hashlib.sha256(raw).hexdigest() != r["source_sha256"]:
            raise RuntimeError(f"Real ZTF source integrity FAILED: {r['ztf_oid']}")
        payload = json.loads(raw)
        if r["ztf_oid"] not in payload:
            raise RuntimeError(f"Published variable-star ZTF source missing: {r['ztf_oid']}")
        band = band_diagnostics(r["ztf_oid"], payload[r["ztf_oid"]])
        if not band["usable"]:
            raise RuntimeError(f"Previously accepted real star now unreadable: {r['ztf_oid']}")
        check = {
            "strong_scatter_above_reported_noise": band["strong_scatter_above_reported_noise"],
            "strong_periodicity_analytical_only": band["strong_periodicity_analytical_only"],
            "repeated_signal_in_halves": band["repeated_signal_in_halves"],
            "without_strong_daily_alias": not band["strong_daily_alias_risk"],
        }
        passed = bool(all(check.values()))
        row = {"ztf_oid": r["ztf_oid"], "published_class": r["class_label"],
               "original_source_sha256": r["source_sha256"],
               "basic_one_band_gates": check, "all_one_band_gates_passed": passed}
        rows.append(row)
        counter = per_class[r["class_label"]]
        counter["actual_positive_stars"] += 1
        counter["all_one_band_gates_passed"] += int(passed)
        for key, success in check.items():
            counter[key] += int(success)
        if (i + 1) % 100 == 0:
            print(f"Tested {i + 1}/600 genuinely published variable stars", flush=True)
    report = {
        "purpose": "Positive-class recovery on original AAVSO-VSX-labeled ZTF data; no false-positive estimate",
        "source_count": len(records),
        "positive_counts_by_original_catalog_class": {k: dict(v) for k, v in per_class.items()},
        "individual_real_source_results": rows,
        "limitations": [
            "Only 3 published types of real variables; not representative of new or faint stars",
            "No verified real nonvariable or instrumental artifacts in this control set",
            "This is a single-band benchmark and cannot assess the separate two-filter gate",
            "Same archived ZTF sources as the supervised model, not independent survey data",
            "The screen thresholds were set before this benchmark run; not statistically calibrated",
        ]
    }
    out = RESULTS / "ztf_positive_screen_benchmark.json"
    write_json(out, report)
    print(json.dumps({"real_published_stars_checked": len(records),
                      "by_class": report["positive_counts_by_original_catalog_class"],
                      "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
