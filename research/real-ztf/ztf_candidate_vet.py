"""Conservative, reproducible pre-discovery checks on REAL archived ZTF data.

This is a candidate-review tool, not a detector with measured false-positive
rates, a novelty guarantee, or a claim of discovery. It refuses unverifiable
catalog archives. No generated photometry, labels or training examples.

Run after ztf_discovery.py and ztf_catalog_check.py::
    python ztf_candidate_vet.py
    python ztf_candidate_vet.py --tag my_field
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from astropy.timeseries import LombScargle

from run import ROOT, RESULTS, write_json
from ztf_discovery import DEFAULT_RAW, angular_arcsec, observed_features


FREQ_MIN = 0.0125
FREQ_MAX = 5.0
FREQ_GRID = np.linspace(FREQ_MIN, FREQ_MAX, 4000)
KNOWN_STATUS = "MATCHES_PUBLISHED_CATALOG"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_inside_root(rel: str) -> Path:
    p = (ROOT / rel).resolve()
    if not p.is_relative_to(ROOT.resolve()):
        raise ValueError("Unsafe path in archived catalog manifest")
    return p


def verify_inputs(tag: str | None):
    result_dir = RESULTS / "fields" / tag if tag else RESULTS
    source_path = ROOT / "data" / "fields" / (tag + ".json") if tag else DEFAULT_RAW
    scan = json.loads((result_dir / "ztf_scan_results.json").read_text())
    checked = json.loads((result_dir / "ztf_catalog_check.json").read_text())
    raw = source_path.read_bytes()
    if sha256(raw) != scan["archive_sha256"]:
        raise RuntimeError("ZTF raw-data SHA256 mismatch. Stop; cannot trust these scores")
    records = json.loads(raw)
    if not isinstance(records, dict) or not records:
        raise RuntimeError("Raw data has no ZTF records")
    for oid, record in records.items():
        if not isinstance(record, dict) or not isinstance(record.get("lc"), list):
            raise RuntimeError(f"Invalid original data for {oid}")
        if not isinstance(record.get("meta", {}).get("coord"), dict):
            raise RuntimeError(f"Missing real sky position for {oid}")

    if checked["count_candidate_positions"] != len(scan["positional_groups_ranked"]):
        raise RuntimeError("Sky grouping changed after the catalog check")
    left = [x["representative_oid"] for x in checked["ranked_positions"]]
    right = [x["representative_oid"] for x in scan["positional_groups_ranked"]]
    if left != right:
        raise RuntimeError("Catalog matches do not belong to this scan")

    # A malformed/HTML response is NOT evidence that a star is absent from a
    # real catalog. Both source and catalog hashes must be present and valid.
    provenance = checked["catalog_query_provenance"]
    if set(provenance) != {"vsx", "gaia_dr3_variable_summary", "ztf_2020_periodic"}:
        raise RuntimeError("One of the three mandatory real-catalog checks is missing")
    for name, entry in provenance.items():
        data = _safe_inside_root(entry["saved_query_response"]).read_bytes()
        if sha256(data) != entry["raw_sha256"]:
            raise RuntimeError(f"Catalog snapshot checksum mismatch: {name}")
        decoded = data.decode("utf-8", errors="replace")
        if "#INFO\tservice_protocol=ASU" not in decoded or "#Table" not in decoded:
            raise RuntimeError(f"Catalog {name} did not return a recognizable VizieR table")
        if "#INFO\tError" in decoded or "#INFO\tQUERY_STATUS=ERROR" in decoded:
            raise RuntimeError(f"Catalog {name} reports a query error")
        if entry["entries_in_queried_field"] >= 10000:
            raise RuntimeError(f"Catalog {name} may have been truncated; use smaller fields")
    center = scan.get("field_query") or {}
    cross = checked.get("search_field") or {}
    if center:
        if angular_arcsec((center["ra"], center["dec"]), (cross["ra"], cross["dec"])) > .5:
            raise RuntimeError("Catalog and telescope query centers disagree")
        if cross["catalog_radius_arcsec"] < center["radius_arcsec"] + 2:
            raise RuntimeError("The catalog search is smaller than the ZTF field")
    return result_dir, source_path, scan, checked, records


def _fit_period(t, mag, err):
    """Measured-time Lomb-Scargle, not any published period or ML prediction."""
    ls = LombScargle(t - np.min(t), mag, err, center_data=True)
    power = ls.power(FREQ_GRID, method="fast")
    idx = int(np.argmax(power))
    f = float(FREQ_GRID[idx])
    peak = float(power[idx])
    # Analytical white-noise FAP is a diagnostic, NOT an empirical survey FPR.
    fap = float(ls.false_alarm_probability(
        peak, method="baluev", minimum_frequency=FREQ_MIN,
        maximum_frequency=FREQ_MAX))
    return f, peak, max(0.0, min(1.0, fap)), ls


def _is_alias(freq: float) -> bool:
    # Sidereal/solar-day sampling can introduce misleading whole-day peaks.
    return any(abs(freq - n) < .04 for n in range(1, 6))


def _harmonic_match(f1: float, f2: float, tol: float = .02) -> bool:
    return any(abs(f1 - f2 * h) < tol for h in (.5, 1, 2))


def band_diagnostics(oid, record):
    arr = observed_features(record.get("lc", []))
    if arr is None:
        return {"ztf_oid": oid, "usable": False, "reason": "insufficient_clean_observations"}
    t, mag, err = arr.T
    mederr = float(np.median(err))
    medmag = float(np.median(mag))
    mad = float(1.4826 * np.median(np.abs(mag - medmag)))
    amp = float(np.quantile(mag, .95) - np.quantile(mag, .05))
    # Reduced weighted chi^2 around the best constant-brightness fit.
    w = err ** (-2)
    mu = float(np.sum(mag * w) / np.sum(w))
    chi2 = float(np.sum(((mag - mu) / err) ** 2) / (len(arr) - 1))
    freq, power, fap, ls = _fit_period(t, mag, err)
    by_time = np.argsort(t)
    a, b = np.array_split(by_time, 2)
    halves = []
    for part in (a, b):
        tt, mm, ee = t[part], mag[part], err[part]
        sub_freq, sub_power, sub_fap, sub_ls = _fit_period(tt, mm, ee)
        support = float(sub_ls.power(freq))
        halves.append({"n": len(part), "span_days": float(np.ptp(tt)),
                       "best_frequency_per_day": sub_freq, "peak_power": sub_power,
                       "power_at_full_frequency": support,
                       "analytic_fap": sub_fap,
                       "best_peak_agrees_or_harmonic": _harmonic_match(sub_freq, freq)})
    # Two separated halves must detect periodicity at the SAME full-data
    # frequency; their strongest alias does not have to match exactly.
    repeated = all(x["n"] >= 40 and x["span_days"] >= 60
                   and x["power_at_full_frequency"] >= .15
                   for x in halves)
    reliable_var = chi2 >= 5 and amp >= max(.15, 5 * mederr) and mad >= 2 * mederr
    return {
        "ztf_oid": oid, "usable": True,
        "filter": record["meta"].get("filter"),
        "n_good_measurements": int(len(arr)),
        "n_original_measurements": len(record.get("lc", [])),
        "span_days": float(np.ptp(t)),
        "median_magnitude": medmag, "median_reported_error_mag": mederr,
        "robust_amplitude_mag": amp,
        "reduced_chi2_constant_brightness": chi2,
        "frequency_per_day": freq, "period_days": 1 / freq,
        "peak_power": power,
        "analytic_white_noise_fap": fap,
        "strong_daily_alias_risk": _is_alias(freq),
        "independent_time_halves": halves,
        "repeated_signal_in_halves": bool(repeated),
        "strong_scatter_above_reported_noise": bool(reliable_var),
        "strong_periodicity_analytical_only": bool(fap < 1e-5 and power >= .3),
    }


def all_positions(records):
    """One representative per <1 arcsec source cluster; NO fake objects."""
    unique = []
    for rec in records.values():
        c = rec["meta"]["coord"]
        pos = float(c["ra"]), float(c["dec"])
        if not any(angular_arcsec(pos, old) < 1. for old in unique):
            unique.append(pos)
    return unique


def nearest_neighbor(group, positions):
    pos = group["ra_deg"], group["dec_deg"]
    neighbors = sorted(angular_arcsec(pos, other) for other in positions
                       if angular_arcsec(pos, other) >= 1.0)
    return float(neighbors[0]) if neighbors else None


def review_one(group, records, positions):
    oids = [str(x["oid"]) for x in group["filters_and_oids"]]
    bands = [band_diagnostics(oid, records[oid]) for oid in oids if oid in records]
    good = [b for b in bands if b["usable"]]
    support = [b for b in good if b["strong_scatter_above_reported_noise"]
               and b["strong_periodicity_analytical_only"]
               and b["repeated_signal_in_halves"]
               and not b["strong_daily_alias_risk"]]
    # Deliberately insist on color-independent corroboration before a star
    # goes to the top of a review queue. Missing a band means inconclusive.
    pair = None
    for i, x in enumerate(support):
        for y in support[i + 1:]:
            if x["filter"] != y["filter"] and _harmonic_match(
                x["frequency_per_day"], y["frequency_per_day"]):
                pair = [x["ztf_oid"], y["ztf_oid"]]
                break
        if pair:
            break
    nearest = nearest_neighbor(group, positions)
    matches = group["published_catalog_matches_within_2arcsec"]
    flags = []
    if matches:
        flags.append("already_in_at_least_one_published_catalog")
    if not good:
        flags.append("no_band_with_80_good_measurements")
    if not any(b["strong_scatter_above_reported_noise"] for b in good):
        flags.append("variability_not_clearly_above_photometric_errors")
    if not any(b["strong_periodicity_analytical_only"] for b in good):
        flags.append("no_very_strong_periodogram_peak")
    if not any(b["repeated_signal_in_halves"] for b in good):
        flags.append("signal_does_not_repeat_in_both_time_halves")
    if not any(not b["strong_daily_alias_risk"] for b in good):
        flags.append("possible_day_night_sampling_alias")
    if not pair:
        flags.append("not_independently_confirmed_in_two_filters")
    if nearest is not None and nearest < 3:
        flags.append("another_ztf_source_within_3_arcsec_blending_risk")
    # These are actionable review statuses, not machine-certified discoveries.
    if matches:
        status = "KNOWN_IN_CHECKED_CATALOG"
    elif not flags:
        status = "PRIORITY_FOR_HUMAN_REVIEW__NOT_A_DISCOVERY"
    else:
        status = "NOT_READY_FOR_DISCOVERY_CLAIM"
    return {
        "ztf_oid": group["representative_oid"],
        "ra_deg": float(group["ra_deg"]), "dec_deg": float(group["dec_deg"]),
        "status": status, "flags": flags,
        "matching_real_catalog_rows": matches,
        "nearest_distinct_ztf_source_arcsec": nearest,
        "two_band_corroborating_oids": pair,
        "bands": bands,
        "initial_exploratory_rank_score": group["top_score"],
        "aladin_sky_view": (
            "https://aladin.cds.unistra.fr/AladinLite/?target="
            f"{group['ra_deg']}%20{group['dec_deg']}&fov=0.03&survey=P%2FDSS2%2Fcolor"),
        "warning": "No novelty or discovery claim. Nearby blends, atmospheric effects, "
                   "additional published catalogs and independent telescope checks remain.",
    }


def write_report(path: Path, tag, outputs, counts, src, scan):
    lines = [
        "# ZTF candidate review — measured data only",
        "",
        "**No astronomical discovery has been established by this report.**",
        "",
        f"ZTF raw observations: `{src.relative_to(ROOT)}`",
        f"Raw SHA256: `{scan['archive_sha256']}`",
        f"Sky field tag: `{tag or 'original_45arcsec_field'}`",
        "",
        "## What the checks found",
        "",
    ]
    for k, v in sorted(counts.items()):
        lines.append(f"- {k}: **{v}**")
    lines += [
        "", "## What the automatic checks mean", "",
        "Each recorded ZTF measurement stays as observed. Periodic signals must "
        "exceed the reported measurement noise, appear in both time halves, "
        "and agree between independent filters. Close neighbors and peaks near "
        "the day/night observing rhythm block automatic promotion. Three "
        "original archived catalogs are checked, but catalog absence is "
        "**never** proof of discovery.",
        "", "### Strongest remaining review cases", "",
        "Only human-review priorities, never named or confirmed discoveries.",
        "",
    ]
    priorities = [x for x in outputs if x["status"].startswith("PRIORITY_")]
    for r in priorities[:15]:
        lines += [
            f"- **ZTF OID {r['ztf_oid']}**: RA {r['ra_deg']:.6f}°, Dec {r['dec_deg']:.6f}°. "
            f"[Inspect images]({r['aladin_sky_view']}).",
        ]
    if not priorities:
        lines.append("No object cleared every conservative screening gate in this field.")
    lines += ["", "### Reviewable leads that did NOT pass all checks", "",
              "These are **not** discoveries and may be uninteresting or spurious. "
              "The list is only a starting point for examining original sky images.", ""]
    unconfirmed = [r for r in outputs if r["status"] == "NOT_READY_FOR_DISCOVERY_CLAIM"]
    unconfirmed.sort(key=lambda r: (len(r["flags"]), -r["initial_exploratory_rank_score"]))
    for r in unconfirmed[:12]:
        lines.append(f"- **ZTF OID {r['ztf_oid']}**, RA {r['ra_deg']:.6f}°, "
                     f"Dec {r['dec_deg']:.6f}° — blocked by: "
                     f"{', '.join(r['flags'])}. "
                     f"[Inspect image]({r['aladin_sky_view']}).")
    lines += [
        "", "## Before reporting any discovery", "",
        "1. Inspect sky images and original ZTF observation flags (SNAD mirror "
        "   rows in this archive do **not** include ZTF `catflags`).",
        "2. Check SIMBAD, Gaia source astrometry, recent VSX updates, other "
        "   survey catalogs, and neighboring/blended objects individually.",
        "3. Confirm the period against aliases and independent survey images "
        "   or follow-up measurements; classify without relying on three-class ML guesses.",
        "4. Have an astronomer independently review the evidence before any "
        "   submission or public claim. The current pipeline has **no measured "
        "   real-world false-alarm rate**.",
        "", "## Technical caution", "",
        "The Lomb–Scargle false-alarm number assumes idealized white noise. "
        "It is **not** the measured probability of a false star discovery. "
        "The thresholds are deliberately strict exploratory screening values, "
        "not calibrated sensitivity/specificity on real control stars.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="same distinct field tag used in discovery and catalog scripts")
    args = parser.parse_args()
    if args.tag and not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", args.tag):
        parser.error("Invalid field tag")
    out_dir, source_path, scan, checked, records = verify_inputs(args.tag)
    positions = all_positions(records)
    report = []
    for i, group in enumerate(checked["ranked_positions"]):
        report.append(review_one(group, records, positions))
        if (i + 1) % 25 == 0:
            print(f"Vetted {i + 1}/{len(checked['ranked_positions'])} real positions", flush=True)
    totals = Counter(x["status"] for x in report)
    flag_totals = Counter(flag for x in report for flag in x["flags"])
    # Mirror the original ranking so every output row can be traced back.
    outcome = {
        "claim_level": "review_priorities_only__no_confirmed_new_objects",
        "source": "original_public_ZTF_DR24_and_three_archived_published_catalog_queries",
        "raw_sha256": scan["archive_sha256"],
        "catalog_sha256s": {key: val["raw_sha256"] for key, val
                            in checked["catalog_query_provenance"].items()},
        "number_of_real_telescope_positions": len(report),
        "status_counts": dict(totals),
        "risk_flag_counts": dict(flag_totals),
        "candidate_reviews": report,
        "unvalidated_assumptions": [
            "White-noise periodogram FAP is not an empirical artifact false-positive rate",
            "Mirror photometry lacks original exposure catflags",
            "Known catalogs are incomplete, including recent and non-variable catalogs",
            "Neighbor separation is not a full ZTF point-spread/blending analysis",
            "Additional telescope or new epoch confirmation not performed",
            "Threshold selection has not been calibrated on verified real non-variable stars",
        ],
    }
    write_json(out_dir / "ztf_candidate_vetting.json", outcome)
    write_report(out_dir / "CANDIDATE_REVIEW.md", args.tag, report, totals,
                 source_path, scan)
    print(json.dumps({"real_positions_checked": len(report), "status_counts": dict(totals),
                      "flags": dict(flag_totals), "report": str(out_dir / "CANDIDATE_REVIEW.md")},
                     indent=2))


if __name__ == "__main__":
    main()
