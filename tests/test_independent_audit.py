"""Independent, adversarial VarWISE pipeline checks (no real network required).

Run: python -m unittest -v verification/test_independent_audit.py
The failing tests are intentional defect demonstrations until production is fixed.
"""
import argparse
import json
import os
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from upload import varwise_gaia_hunt as h


class _Reply:
    def __init__(self, df):
        self.df = df

    def to_table(self):
        return self

    def to_pandas(self):
        return self.df


class _GeometricFakeTap:
    """Imitates TAP filtering and TOP output using generated WHERE clauses."""
    def __init__(self, coords):
        self.data = pd.DataFrame(coords, columns=["ra", "dec"])
        self.data["cluster_id"] = np.arange(len(self.data))
        self.queries = []
        self.truncate = False

    def run_sync(self, query, maxrec=None):
        self.queries.append(query)
        if "WHERE " not in query:
            raise AssertionError(f"unexpected query: {query}")
        where = query.split("WHERE ", 1)[1]
        d = self.data
        ral = re.findall(r"(?<![\w])ra>=(\d+(?:\.\d+)?)", where)
        rah = re.findall(r"(?<![\w])ra<(\d+(?:\.\d+)?)", where)
        if ral and rah:
            low, high = float(ral[0]), float(rah[0])
            rav = d["ra"]
            rmask = ((rav >= low) | (rav < high)) if " OR " in where else ((rav >= low) & (rav < high))
            d = d[rmask]
        decl = float(re.search(r"dec>=(\-?\d+(?:\.\d+)?)", where).group(1))
        decmatch = re.search(r"dec(<|<=)(\-?\d+(?:\.\d+)?)", where)
        dech = float(decmatch.group(2))
        d = d[(d["dec"] >= decl) & ((d["dec"] <= dech) if decmatch.group(1) == "<=" else (d["dec"] < dech))]
        if "COUNT(*)" in query:
            return _Reply(pd.DataFrame({"n": [len(d)]}))
        top = int(re.search(r"SELECT TOP (\d+)", query).group(1))
        if self.truncate and len(d):
            top = min(top, len(d) - 1)
        return _Reply(d.head(top).copy())


class PaginationTests(unittest.TestCase):
    def test_density_forces_recursive_split_without_data_loss(self):
        points = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2),
                  (2, 0), (2, 1), (2, 2), (3, 3), (4, 4)]
        tap = _GeometricFakeTap(points)
        chunks = h.fetch_complete_rectangle(tap, "varwiseext", "ra", "dec",
                                            (0, 6, 0, 6), page_size=3)
        d = pd.concat(chunks)
        self.assertEqual(sorted(d.cluster_id), list(range(len(points))))
        self.assertEqual(len(d), len(points))
        self.assertGreater(sum("COUNT(*)" in q for q in tap.queries), 1)

    def test_leaf_truncation_hard_fails(self):
        tap = _GeometricFakeTap([(1, 1), (2, 2), (3, 3)])
        tap.truncate = True
        with self.assertRaisesRegex(RuntimeError, "query incomplete"):
            h.fetch_complete_rectangle(tap, "varwiseext", "ra", "dec",
                                       (0, 5, 0, 5), page_size=4)

    def test_half_open_tile_edges_and_zero_ra(self):
        tap = _GeometricFakeTap([(359.9999, 0), (0, 0), (180, 1), (0, -90),
                                 (360 - 1e-6, 90), (180, 90)])
        tiles = [(0, 180, -90, 0), (180, 360, -90, 0),
                 (0, 180, 0, 90), (180, 360, 0, 90)]
        rows = []
        for tile in tiles:
            rows.extend(h.fetch_complete_rectangle(tap, "varwiseext", "ra", "dec",
                                                   tile, page_size=10))
        d = pd.concat(rows)
        self.assertEqual(len(d), len(tap.data))
        self.assertFalse(d.cluster_id.duplicated().any())


class SkyCoverageTests(unittest.TestCase):
    def test_whole_sky_random_points_have_exactly_one_tile(self):
        for size in (6, 17, 60):
            boxes = np.asarray(h.generate_boxes(size, bmin=0, seed=0))
            rng = np.random.default_rng(90210)
            coords = np.vstack((rng.uniform(0, 360, 500),
                                rng.uniform(-90, 90, 500))).T
            coords = np.vstack((coords, [[0, 0], [359.999999, -90], [180, 90],
                                         [0, -90], [0, 90]]))
            for ra, dec in coords:
                d0 = boxes[:, 1] - boxes[:, 3] / 2
                d1 = boxes[:, 1] + boxes[:, 3] / 2
                r0 = boxes[:, 0] - boxes[:, 2] / 2
                r1 = boxes[:, 0] + boxes[:, 2] / 2
                count = np.sum((ra >= r0) & (ra < r1) &
                               (dec >= d0) & ((dec < d1) | ((dec == 90) & (d1 == 90))))
                self.assertEqual(int(count), 1, f"size={size} ra={ra} dec={dec}")


class ZtfAssociationTests(unittest.TestCase):
    @staticmethod
    def frame(ra, dec, counts):
        rows = []
        for oid, off_arcsec, n in counts:
            rows.extend(dict(oid=oid, ra=ra + off_arcsec/3600,
                             dec=dec, mjd=58000 + j, mag=15 + j/100,
                             magerr=0.03, catflags=0) for j in range(n))
        return pd.DataFrame(rows)

    def test_closest_oid_wins_over_most_observed(self):
        d = self.frame(150, 0, [(1, 1.0, 70), (2, 0.1, 45)])
        selected = h.select_nearest_ztf_oid(d, 150, 0)
        self.assertEqual(selected.attrs["ztf_oid"], "2")
        self.assertEqual(len(selected), 45)

    def test_ambiguity_fail_closed(self):
        # Two astrometrically distinct candidates nearly tied in distance.
        d = self.frame(150, 0, [(1, 0.2, 70), (2, -0.4, 45)])
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            h.select_nearest_ztf_oid(d, 150, 0)

    def test_overlapping_field_oids_of_one_star_do_not_block_all_lc(self):
        """Live ZTF AGN has OIDs at 0.208 and 0.217 arcsec: same centroid."""
        d = self.frame(150, 0, [(6672, 0.208, 814), (16642, 0.217, 93)])
        # Both sets must agree photometrically; the generic fixture has a
        # deliberately artificial unbounded magnitude drift over 814 visits.
        d["mag"] = 15.4 + 0.05 * np.sin(d["mjd"].to_numpy())
        selected = h.select_nearest_ztf_oid(d, 150, 0)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.attrs["ztf_oid"], "6672")
        self.assertEqual(len(selected), 814)

    def test_oid_astrometry_crossing_ra_zero(self):
        """Circular RA arithmetic is mandatory; plain median gives 180 degrees."""
        d = self.frame(0, 0, [(1, 0.0, 60)])
        d.loc[::2, "ra"] = 359.9999
        d.loc[1::2, "ra"] = 0.0001
        selected = h.select_nearest_ztf_oid(d, 0, 0)
        self.assertIsNotNone(selected, "the matching OID is lost at RA=0")
        self.assertLess(selected.attrs["separation_arcsec"], 1.0)

    def test_transient_http_error_retries_and_caches_only_success(self):
        import requests

        with tempfile.TemporaryDirectory() as root:
            d = self.frame(150, 0, [(111, 0.1, 45)])

            class Resp:
                text = d.to_csv(index=False)

                @staticmethod
                def raise_for_status():
                    return None

            events = [requests.ConnectionError("temporary"),
                      requests.Timeout("temporary"), Resp()]

            def get(*_a, **_kw):
                obj = events.pop(0)
                if isinstance(obj, Exception):
                    raise obj
                return obj

            with patch("requests.get", side_effect=get) as req, patch.object(h.time, "sleep"):
                self.assertEqual(len(h.ztf_single_band(150, 0, "r", root)), 45)
                self.assertEqual(req.call_count, 3)
                self.assertEqual(len(h.ztf_single_band(150, 0, "r", root)), 45)
                self.assertEqual(req.call_count, 3)


class CheckpointTests(unittest.TestCase):
    def test_committed_tile_is_idempotent_and_state_recovered(self):
        with tempfile.TemporaryDirectory() as root:
            h.initialize_survey(root, "T1", {"table": "varwiseext"})
            folder = os.path.join(root, "committed_tiles")
            os.makedirs(folder)
            item = dict(key="0,0,5,5", fingerprint="T1", scanned=22,
                        candidates=[dict(source_id=7, ra=1, dec=2)],
                        unverified=[], anomalies=[], known=[])
            h.atomic_json(os.path.join(folder, "first.json"), item)
            s1 = h.survey_checkpoint(root, "T1")
            s2 = h.survey_checkpoint(root, "T1")
            self.assertEqual(s1, s2)
            self.assertEqual(s2["n_candidates"], 1)
            self.assertEqual(len(pd.read_csv(os.path.join(root, "candidates.csv"))), 1)
            h.atomic_json(os.path.join(root, "state.json"), {"garbled": True})
            s3 = h.survey_checkpoint(root, "T1")
            self.assertEqual(s3, s2)

    def test_different_survey_fingerprint_refused(self):
        with tempfile.TemporaryDirectory() as root:
            h.initialize_survey(root, "T1", {"table": "varwisepure"})
            with self.assertRaisesRegex(RuntimeError, "DIFFERENT SURVEY"):
                h.initialize_survey(root, "T2", {"table": "varwiseext"})


class EmptyTileTests(unittest.TestCase):
    def test_empty_varwise_tile_has_complete_five_part_result(self):
        args = argparse.Namespace(box_size=1.0, skip_gaia=True)
        with tempfile.TemporaryDirectory() as root, \
             patch.object(h, "get_varwise", return_value=pd.DataFrame()):
            good, unverified, anomalies, known, n = h.process_box(
                args, None, 10, 10, root, root, 1.0, 1.0)
        self.assertEqual((good, unverified, anomalies, known, n), ([], [], [], [], 0))

    def test_no_usable_ztf_tile_has_complete_five_part_result(self):
        args = argparse.Namespace(box_size=1.0, skip_gaia=True)
        vw = pd.DataFrame({"ra": [11.0], "dec": [11.0], "cluster_id": [42]})
        with tempfile.TemporaryDirectory() as root, \
             patch.object(h, "get_varwise", return_value=vw), \
             patch.object(h, "featurize", return_value=pd.DataFrame()):
            good, unverified, anomalies, known, n = h.process_box(
                args, None, 10, 10, root, root, 1.0, 1.0)
        self.assertEqual((good, unverified, anomalies, known, n), ([], [], [], [], 1))


if __name__ == "__main__":
    unittest.main()
