"""Network-free regression tests for no-label search and fail-closed prior art.

Run: python -m unittest -v test_varwise_gaia_hunt.py
"""
import argparse
import os
import tempfile
import unittest
import json
from unittest.mock import patch

import pandas as pd
import numpy as np

try:
    from upload import varwise_gaia_hunt as hunt  # attached-file workspace layout
except ModuleNotFoundError:
    import varwise_gaia_hunt as hunt  # downloaded files in one directory


class NoTargetLabelsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.args = argparse.Namespace(
            out=self.tmp.name, ra0=150.0, dec0=30.0, size=2.0,
            max_vw=20, n_train=20, max_prior=10, search_mode="signal",
            skip_gaia=False, include_anomalies=False,
            allow_untrained_diagnostics=True)
        self.vw = pd.DataFrame({"ra": [150.01], "dec": [30.02]})
        self.gaia = pd.DataFrame({
            "uid": [0], "source_id": [123456789], "g_ra": [150.01], "g_dec": [30.02],
            "sep": [0.5], "vari_id": [float("nan")], "n_gaia": [1],
            "phot_g_mean_mag": [15.0], "bp_rp": [1.0], "ruwe": [1.1]})
        self.ft = pd.DataFrame({
            "source_id": [123456789], "ra": [150.01], "dec": [30.02],
            "period": [3.7], "alias": [False], "log_fap": [-20.0],
            "amp_over_err": [15.0], "score": [float("nan")]})
        self.check = dict(confirmed=True, early_period=3.7, late_period=3.7,
                          reason="period independently seen in both halves")

    def _run_mocked(self, prior_hits):
        with patch.object(hunt, "get_varwise", return_value=self.vw), \
             patch.object(hunt, "gaia_match", return_value=self.gaia), \
             patch.object(hunt, "featurize", return_value=self.ft), \
             patch.object(hunt, "split_epoch_confirmation", return_value=self.check), \
             patch.object(hunt, "prior_art", return_value=prior_hits), \
             patch.object(hunt, "fold_plot"), \
             patch.object(hunt, "gaia_training_stars", side_effect=AssertionError(
                 "TARGET HAS NO TRAINING LABELS: should not be requested")):
            return hunt.run(self.args)

    def test_search_runs_without_any_target_training_labels(self):
        self.assertEqual(self._run_mocked([]), 0)
        c = pd.read_csv(os.path.join(self.tmp.name, "candidates.csv"))
        self.assertEqual(len(c), 1)
        self.assertEqual(c.iloc[0]["search_mode"], "untrained_signal")

    def test_failed_prior_art_cannot_be_claimed_as_discovery(self):
        self.assertEqual(self._run_mocked(["SIMBAD:ERROR(503)"]), 0)
        u = pd.read_csv(os.path.join(self.tmp.name, "unverified.csv"))
        self.assertEqual(len(u), 1)
        self.assertIn("UNVERIFIED", u.iloc[0]["status"])
        self.assertEqual(os.stat(os.path.join(self.tmp.name, "candidates.csv")).st_size, 1)

    def test_no_validated_model_never_falls_back_and_never_touches_targets(self):
        self.args.search_mode = "auto"
        with patch.object(hunt, "get_varwise", side_effect=AssertionError(
                "VarWISE access before held-out validation is forbidden")), \
             patch.object(hunt, "gaia_training_stars", side_effect=AssertionError(
                "run must use already-sealed model")):
            self.assertEqual(hunt.run(self.args), 2)
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "candidates.csv")))

    def test_target_without_a_gaia_counterpart_is_searched(self):
        self.args.skip_gaia = True
        target_features = self.ft.copy()
        target_features["source_id"] = -1
        with patch.object(hunt, "get_varwise", return_value=self.vw), \
             patch.object(hunt, "gaia_match", side_effect=AssertionError(
                 "Gaia matching must not be required for a new target")), \
             patch.object(hunt, "gaia_training_stars", side_effect=AssertionError(
                 "Training data must not be required")), \
             patch.object(hunt, "featurize", return_value=target_features), \
             patch.object(hunt, "split_epoch_confirmation", return_value=self.check), \
             patch.object(hunt, "prior_art", return_value=[]), \
             patch.object(hunt, "fold_plot"):
            self.assertEqual(hunt.run(self.args), 0)
        c = pd.read_csv(os.path.join(self.tmp.name, "candidates.csv"))
        self.assertEqual(c.iloc[0]["source_id"], -1)

    def test_anomaly_channel_recovers_untrained_event_and_rejects_flat_noise(self):
        rng = np.random.default_rng(5)
        t = np.arange(160, dtype=float) * 3.5
        e = np.full(len(t), 0.03)
        flat = 15 + rng.normal(0, e)
        event = flat.copy()
        event[[20, 21, 22, 23, 24, 25, 26]] += 0.4
        self.assertFalse(hunt.outlier_lead_features(t, flat, e)["flag"])
        self.assertTrue(hunt.outlier_lead_features(t, event, e)["flag"])

    def test_two_band_independent_filter_agreement(self):
        t = 58000 + np.repeat(np.arange(9) * 365 + 10, 6) + np.tile(np.arange(6), 9)
        trend = np.repeat([0, 0.1, 0.2, 0.7, 1, 0.8, 0.5, 0.3, 0.15], 6)
        r = pd.DataFrame({"mjd": t, "mag": 18.8 - trend})
        g = pd.DataFrame({"mjd": t, "mag": 19.5 - 1.1 * trend})
        r.attrs = dict(obj_ra=149.4, obj_dec=30.5)
        g.attrs = dict(obj_ra=149.40001, obj_dec=30.50001)
        with patch.object(hunt, "ztf_single_band", side_effect=[r, g]):
            result = hunt.ztf_two_band_check(149.4, 30.5, self.tmp.name)
        self.assertTrue(result["crossband_consistent"])
        self.assertGreater(result["crossband_rho"], 0.95)

    def test_spatial_holdout_is_approximately_80_20_without_leakage(self):
        d = pd.DataFrame({"cell": np.repeat([f"cell_{i}" for i in range(50)], 8),
                          "label": np.tile([0, 1], 200)})
        mask = hunt.spatial_test_mask(d)
        self.assertLess(abs(mask.mean() - 0.2), 0.05)
        self.assertFalse(set(d.loc[mask, "cell"]) & set(d.loc[~mask, "cell"]))

    def test_tiny_spatial_sample_is_rejected_instead_of_faking_accuracy(self):
        d = pd.DataFrame({"cell": np.repeat(["a", "b", "c"], 20),
                          "label": np.tile([0, 1], 30)})
        with self.assertRaisesRegex(RuntimeError, "insufficient spatial diversity"):
            hunt.spatial_test_mask(d)

    def test_known_agn_cannot_be_exported_as_new_discovery(self):
        self.assertEqual(self._run_mocked(["SIMBAD:KNOWN(AGN;AGN|Sy1)"]), 0)
        known = pd.read_csv(os.path.join(self.tmp.name, "known_objects.csv"))
        self.assertEqual(len(known), 1)
        self.assertIn("AGN", known.iloc[0]["prior_art"])
        self.assertEqual(os.stat(os.path.join(self.tmp.name, "candidates.csv")).st_size, 1)

    def test_known_agn_and_archive_outage_still_known_not_candidate(self):
        self.assertEqual(self._run_mocked(["SIMBAD:KNOWN(AGN;AGN)", "NED:ERROR(503)"]), 0)
        self.assertEqual(len(pd.read_csv(os.path.join(self.tmp.name, "known_objects.csv"))), 1)


class AdversarialPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_full_sky_tiles_cover_every_tested_coordinate_once(self):
        tiles = hunt.generate_boxes(9.0)
        self.assertTrue(any(d < -75 for _, d, _, _ in tiles))
        self.assertTrue(any(d > 75 for _, d, _, _ in tiles))
        # Every RA/Dec point belongs to exactly one geometrically bounded tile.
        for dec in (-89.9, -76, -30, -0.1, 0, 30, 76, 89.9):
            for ra in (0, 0.00001, 1, 90, 179.999, 270, 359.999):
                hits = sum((d - dh/2 <= dec < d + dh/2 or
                            (dec == 90 and d+dh/2 == 90)) and
                           ((r-rw/2 <= ra < r+rw/2) or (rw == 360))
                           for r, d, rw, dh in tiles)
                self.assertEqual(hits, 1, (ra, dec, hits))

    def test_tap_paging_recovers_dense_fields_without_duplicates(self):
        class Result:
            def __init__(self, frame):
                self.frame = frame
            def to_table(self):
                return self
            def to_pandas(self):
                return self.frame
        class Service:
            def __init__(self):
                self.queries = []
            def run_sync(self, q, maxrec=None):
                self.queries.append(q)
                # Distributed over one dimension: count simulated subtiles via WHERE dec.
                import re
                low = float(re.search(r"dec>=([-\d.]+)", q).group(1))
                high = float(re.search(r"dec<([-\d.]+)", q).group(1))
                ra_low = float(re.search(r"ra>=([-\d.]+)", q).group(1))
                ra_high = float(re.search(r"ra<([-\d.]+)", q).group(1))
                data = pd.DataFrame({"ra": [1.5]*15, "dec": np.arange(15)/15,
                                     "cluster_id": np.arange(15)})
                data = data[(data.dec >= low) & (data.dec < high) &
                            (data.ra >= ra_low) & (data.ra < ra_high)]
                if "COUNT(*)" in q:
                    return Result(pd.DataFrame({"n": [len(data)]}))
                return Result(data.reset_index(drop=True))
        service = Service()
        parts = hunt.fetch_complete_rectangle(service, "varwiseext", "ra", "dec",
                                               (0, 2, 0, 1), page_size=3)
        result = pd.concat(parts)
        self.assertEqual(len(result), 15)
        self.assertEqual(result.cluster_id.nunique(), 15)
        self.assertTrue(any("COUNT(*)" in q for q in service.queries))

    def test_tap_truncation_refuses_to_finish_tile(self):
        class Result:
            def __init__(self, df): self.df = df
            def to_table(self): return self
            def to_pandas(self): return self.df
        class Service:
            def run_sync(self, query, maxrec=None):
                return Result(pd.DataFrame({"n": [3]} if "COUNT(*)" in query else
                                       {"ra": [1.0], "dec": [1.0]}))
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            hunt.fetch_complete_rectangle(Service(), "varwiseext", "ra", "dec",
                                          (0, 2, 0, 2), page_size=4)

    def test_ztf_chooses_nearest_not_most_observed(self):
        far = pd.DataFrame({"oid": [10]*120, "ra": [150.00045]*120,
                            "dec": [30.0]*120})
        near = pd.DataFrame({"oid": [11]*45, "ra": [150.00002]*45,
                             "dec": [30.0]*45})
        raw = pd.concat([far, near], ignore_index=True)
        raw["mjd"] = np.arange(len(raw)) + 58000
        raw["mag"] = 16.0
        raw["magerr"] = 0.03
        raw["catflags"] = 0
        found = hunt.select_nearest_ztf_oid(raw, 150.0, 30.0)
        self.assertEqual(found.attrs["ztf_oid"], "11")
        self.assertEqual(len(found), 45)

    def test_ztf_rejects_ambiguous_neighbours(self):
        raw = pd.DataFrame({"oid": [10, 10, 11, 11],
                            "ra": [150.00008]*2 + [149.99987]*2,
                            "dec": [30.0]*4, "mjd": [1,2,3,4],
                            "mag": [16.0]*4, "magerr": [0.03]*4, "catflags": [0]*4})
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            hunt.select_nearest_ztf_oid(raw, 150.0, 30.0)

    def test_outage_and_no_match_are_not_cached(self):
        import requests
        with patch("requests.get", side_effect=requests.ConnectionError("offline")) as request:
            with patch.object(hunt.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, "unavailable"):
                    hunt.ztf_single_band(150, 30, "r", self.tmp.name)
        self.assertEqual(request.call_count, 3)
        self.assertEqual(os.listdir(self.tmp.name), [])

    def test_invalid_ztf_positional_tap_uses_documented_id_fallback(self):
        import requests
        # A genuine HTTP 400 must switch away from the broken positional TAP
        # lookup. No failure response may be cached or mislabeled as a star.
        bad = requests.Response()
        bad.status_code = 400
        bad._content = b"<VOTABLE>UsageFault: invalid TAP query</VOTABLE>"
        sample = pd.DataFrame({"mjd": [59000], "mag": [15.3], "magerr": [.01]})
        with patch("requests.get", return_value=bad) as direct, \
             patch.object(hunt, "ztf_single_band_via_hats", return_value=sample) as hats:
            result = hunt.ztf_single_band(63.38383, 35.04224, "r", self.tmp.name)
        self.assertIs(result, sample)
        direct.assert_called_once()
        hats.assert_called_once_with(63.38383, 35.04224, "r", self.tmp.name)
        self.assertEqual(os.listdir(self.tmp.name), [])

    def test_resume_is_transactional_idempotent_and_rejects_config_change(self):
        out = self.tmp.name
        token, cfg = "a"*64, {"mode": "signal"}
        hunt.initialize_survey(out, token, cfg)
        key = "180,20,1,1"
        record = dict(key=key, fingerprint=token, scanned=11,
                      candidates=[{"source_id": -123, "ra": 180.0, "dec": 20.0, "score": np.nan}],
                      unverified=[], anomalies=[], known=[])
        path = os.path.join(out, "committed_tiles", "tile1.json")
        hunt.atomic_json(path, record)
        a = hunt.survey_checkpoint(out, token)
        b = hunt.survey_checkpoint(out, token)
        self.assertEqual(a, b)
        self.assertEqual(len(pd.read_csv(os.path.join(out, "candidates.csv"))), 1)
        with open(path) as f:
            self.assertEqual(json.load(f)["candidates"][0]["score"], None)
        with self.assertRaisesRegex(RuntimeError, "DIFFERENT SURVEY"):
            hunt.initialize_survey(out, "b"*64, cfg)


class TrainBeforeValidationAndSearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_auto_does_not_even_query_varwise_if_training_fails(self):
        args = argparse.Namespace(out=self.tmp.name, search_mode="ml", retrain=True,
                                  box_size=6, bmin=0, dec_min=-90, dec_max=90,
                                  table="varwiseext", skip_gaia=True, hours=1, target=1)
        with patch.object(hunt, "train_and_validate", return_value=None), \
             patch.object(hunt, "get_varwise", side_effect=AssertionError(
                 "TEST TARGET LEAKED INTO TRAINING")) as target:
            self.assertEqual(hunt.auto(args), 2)
        target.assert_not_called()
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "candidates.csv")))

    def test_training_or_validation_region_is_forbidden_as_target(self):
        args = argparse.Namespace(out=self.tmp.name, search_mode="ml",
                                  ra0=150.0, dec0=30.0, size=2.0)
        fake_pack = {"model": object(), "report": {
            "training_tiles": [[150.0, 30.0, 6.0, 6.0]],
            "validation_tiles": [[180.0, 20.0, 6.0, 6.0]]}}
        with patch.object(hunt, "load_validated_model", return_value=fake_pack), \
             patch.object(hunt, "get_varwise", side_effect=AssertionError(
                 "OVERLAPPING TARGET ACCESS")) as target:
            self.assertEqual(hunt.run(args), 2)
        target.assert_not_called()

    def test_untrained_signal_mode_requires_explicit_override(self):
        args = argparse.Namespace(out=self.tmp.name, search_mode="signal",
                                  allow_untrained_diagnostics=False)
        with patch.object(hunt, "get_varwise", side_effect=AssertionError(
                 "UNTRAINED TARGET ACCESS")) as target:
            self.assertEqual(hunt.run(args), 2)
        target.assert_not_called()

    def test_holdout_lightcurves_remain_unread_until_after_model_fit(self):
        from sklearn import ensemble
        args = argparse.Namespace(out=self.tmp.name, box_size=6,
                                  train_per_box=20, min_train=160)
        boxes = [b for b in hunt.generate_boxes(6) if 0 <= b[1] <= 65]
        tracker = {"next_source": 0, "calls": 0, "main_fitted": False}
        sealed_ra_dec = {(x[0], x[1]) for x in boxes[16:24]}

        def fake_gaia(a, n):
            if (a.ra0, a.dec0) in sealed_ra_dec:
                self.assertTrue(tracker["main_fitted"],
                                "HOLDOUT GAIA LABELS WERE QUERIED BEFORE FIT!")
            rows = []
            for k in range(40):
                tracker["next_source"] += 1
                rows.append(dict(source_id=tracker["next_source"],
                                 ra=a.ra0 + (k % 5) - 2,
                                 dec=a.dec0 + (k // 5) * .55 - 2,
                                 label=k % 2))
            return pd.DataFrame(rows)

        def fake_featurize(stars, cache, label_col=None, workers=3, strict=True):
            tracker["calls"] += 1
            if tracker["calls"] == 1:
                self.assertFalse(tracker["main_fitted"], "training features before fitting")
            elif tracker["calls"] == 2:
                self.assertTrue(tracker["main_fitted"], "HOLDOUT WAS READ BEFORE FIT!")
            else:
                self.fail("unexpected extra feature call")
            out = stars.copy().rename(columns={"g_ra": "ra", "g_dec": "dec"})
            for name in hunt.FEATURES:
                out[name] = 0.0
            out["power"] = .05 + .9 * out["label"]
            out["log_fap"] = -10.0
            out["amp_over_err"] = 10.0
            out["alias"] = False
            return out

        class FakeModel:
            def __init__(self, max_iter=200, **kwargs):
                self.max_iter = max_iter
            def fit(self, x, y):
                if self.max_iter == 200:
                    tracker["main_fitted"] = True
                return self
            def predict_proba(self, x):
                score = x["power"].to_numpy() if self.max_iter == 200 else np.full(len(x), .5)
                return np.c_[1-score, score]

        with patch.object(hunt, "gaia_training_stars", side_effect=fake_gaia), \
             patch.object(hunt, "featurize", side_effect=fake_featurize), \
             patch.object(ensemble, "HistGradientBoostingClassifier", FakeModel), \
             patch("joblib.dump"), \
             patch.object(hunt, "get_varwise", side_effect=AssertionError("VarWISE touched")):
            pack = hunt.train_and_validate(args, boxes, self.tmp.name)
        self.assertIsNotNone(pack)
        self.assertEqual(tracker["calls"], 2)
        self.assertEqual(pack["report"]["validation_protocol"], hunt.VALIDATION_PROTOCOL)
        self.assertTrue(pack["validated_for_discovery"])
        self.assertLess(abs(pack["report"]["metadata_heldout"]/
                            (pack["report"]["metadata_heldout"] +
                             pack["report"]["metadata_train"]) - .2), .05)


if __name__ == "__main__":
    unittest.main()
