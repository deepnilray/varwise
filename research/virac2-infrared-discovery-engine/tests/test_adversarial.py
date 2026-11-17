"""Adversarial QA; no synthetic photometry used to claim model performance.
The generated fixtures here test parser failure modes ONLY, never scientific accuracy.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
sys.path.insert(0,str(R/'core'))
import engine
from input_guard import validate_input,source_preflight,audit_physical_evidence

class InputGuards(unittest.TestCase):
    def setUp(self):
        self.raw=pd.DataFrame({'sourceid':['123']*5,'catid':[str(i) for i in range(1,6)],
           'mjdobs':[55280.,55281.,55282.,55283.,55284.],
           'filter':['Ks']*5,'mag':[15.]*5,'emag':[.03]*5,
           'phot_flag':[0]*5,'ambiguous_match':[0]*5,
           'ra':[285.]*5,'de':[-30.]*5})
    def test_good(self):
        self.assertEqual(validate_input(self.raw)['sources'],1)
    def test_missing_required_fields(self):
        for c in ('sourceid','catid','mjdobs','filter','mag','emag','phot_flag','ambiguous_match'):
            with self.subTest(c=c),self.assertRaises(ValueError):validate_input(self.raw.drop(columns=c))
    def test_invalid_ids(self):
        for bad in ('=HYPERLINK("evil")','2e6','-1','1.0','',None,'NaN','٩','1'*25,'0012','0'):
            f=self.raw.copy();f.loc[0,'sourceid']=bad
            with self.subTest(bad=bad),self.assertRaises(ValueError):validate_input(f)
    def test_bad_exposure_id(self):
        for bad in ('1e3',str(2**53+1),'x',None,'01'):
            f=self.raw.copy();f.loc[0,'catid']=bad
            with self.subTest(bad=bad),self.assertRaises(ValueError):validate_input(f)
    def test_filter_missing(self):
        f=self.raw.copy();f.loc[0,'filter']=None
        with self.assertRaises(ValueError):validate_input(f)
    def test_filter_wrong(self):
        f=self.raw.copy();f['filter']='J'
        with self.assertRaises(ValueError):validate_input(f)
    def test_mixed_filters_explicit(self):
        f=self.raw.copy();f.loc[0,'filter']='J'
        self.assertEqual(set(validate_input(f)['observed_filters']),{'Ks','J'})
    def test_duplicate_exposure_fails_preflight(self):
        f=self.raw.copy();f.loc[1,'catid']='1'
        flags,_=source_preflight(f,f)
        self.assertIn('duplicate_exposure_ids',flags)
    def test_wrong_mjd_preflight(self):
        f=self.raw.copy();f.loc[0,'mjdobs']=1e40
        flags,_=source_preflight(f,f)
        self.assertIn('outside_virac2_observation_era',flags)
    def test_bad_magnitude_preflight(self):
        f=self.raw.copy();f.loc[0,'mag']=-900
        flags,_=source_preflight(f,f)
        self.assertIn('unphysical_magnitude',flags)
    def test_no_quality_rows(self):
        flags,coords=source_preflight(self.raw,self.raw.iloc[:0])
        self.assertIn('no_valid_photometry',flags)
        self.assertIsNone(coords)
    def test_astrometric_scatter(self):
        f=self.raw.copy();f.loc[0,'ra']+=.01
        flags,_=source_preflight(f,f)
        self.assertIn('astrometric_scatter_over_5_arcsec',flags)
    def test_coordinates_wrap_zero(self):
        f=self.raw.copy();f['ra']=[359.9999,0,0,359.9999,0]
        flags,coords=source_preflight(f,f)
        self.assertNotIn('astrometric_scatter_over_5_arcsec',flags)
        self.assertTrue(coords[0]>359.99 or coords[0]<.01)
    def test_missing_coords(self):
        f=self.raw.drop(columns=['ra','de'])
        flags,coords=source_preflight(f,f)
        self.assertEqual(flags,['no_astrometric_coordinates'])
        self.assertIsNone(coords)
    def test_coverage_and_convergence(self):
        flags,_=audit_physical_evidence(self.raw,{'t0_mjd':55282,'tE_days':.51,'fit_converged':False,'delta_aic':0})
        self.assertIn('optimizer_did_not_converge',flags)
        self.assertIn('weak_physical_model_improvement',flags)
        self.assertIn('inadequate_pre_or_post_event_baseline',flags)
    def test_nonfinite_physics(self):
        flags,_=audit_physical_evidence(self.raw,{'t0_mjd':float('nan'),'tE_days':0,'fit_converged':False})
        self.assertIn('unconstrained_microlensing_timescale',flags)
    def test_quality_no_out_of_range_conversion(self):
        f=self.raw.copy();f.loc[0,'mjdobs']=1e300
        self.assertEqual(len(engine._quality_nights(f)),4)
    def test_bad_weights_detected(self):
        m,w=engine._load()
        self.assertEqual(w['feature_count'],43)
        self.assertEqual(w['model'].n_features_in_,43)
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out';out.mkdir();(out/'existing.txt').write_text('protected')
            with self.assertRaises(FileExistsError):
                engine.score(R/'examples/known_training_lightcurves.csv',out,top=1)
            self.assertEqual((out/'existing.txt').read_text(),'protected')
    def test_scored_training_only(self):
        with tempfile.TemporaryDirectory() as d:
            input=R/'examples/known_training_lightcurves.csv';out=Path(d)/'new'
            result=engine.score(input,out,top=1)
            self.assertEqual(result['scored_sources'],8)
            self.assertEqual(result['known_training_sources'],8)
            self.assertEqual(result['review_leads'],0)
            self.assertFalse(result['independent_validation_passed'])
            data=json.loads((out/'candidate_audit.json').read_text())
            self.assertTrue(all(r['training_source'] for r in data['scores']))
            self.assertTrue(all(not r['independently_verified_discovery'] for r in data['scores']))
            self.assertTrue(all(r['original_image_verified'] is False for r in data['scores']))
            self.assertEqual(len(json.loads((out/'run_provenance.json').read_text())['output_files_sha256']),5)
    def test_relabelled_training_lightcurve_is_blocked(self):
        # A score on a known training star cannot become a scientific lead by changing its ID.
        all_rows=pd.read_csv(R/'examples/known_training_lightcurves.csv',dtype={'sourceid':str,'catid':str})
        f=all_rows.loc[all_rows.sourceid=='13726720002998'].copy()
        f['sourceid']='99999999999991'
        f['ra']=285.0;f['de']=-30.0
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'renamed.csv';o=Path(d)/'run';f.to_csv(p,index=False)
            summary=engine.score(p,o,top=1)
            scored=json.loads((o/'candidate_audit.json').read_text())['scores'][0]
            self.assertGreater(scored['score'],.9)
            self.assertEqual(scored['training_lightcurve_fingerprint_match'],'13726720002998')
            self.assertEqual(scored['status'],'blocked_training_lightcurve_reuse')
            self.assertEqual(summary['review_leads'],0)

    def test_filters_preserved(self):
        f=self.raw.copy();f.loc[2,'filter']='J'
        self.assertEqual(len(engine._quality_nights(f)),4)
    def test_no_file_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'foo.csv';self.raw.to_csv(p,index=False)
            with self.assertRaises(ValueError):engine.score(p,Path(d),top=1)
    def test_fetch_invalid_ids_before_network(self):
        with tempfile.TemporaryDirectory() as d:
            for ids in (['-2'],['1','1'],[str(i) for i in range(31)],[]):
                with self.subTest(ids=ids),self.assertRaises(ValueError):engine.fetch(ids,Path(d)/'x.csv')
    @patch('requests.post')
    def test_fetch_mismatched_ids(self,post):
        text='sourceid,catid,mjdobs,filter,mag,emag,phot_flag,ambiguous_match\n999,13,55281,Ks,14,0.03,0,0\n'
        post.return_value.text=text
        post.return_value.raise_for_status.return_value=None
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):engine.fetch(['123'],Path(d)/'out.csv')
            self.assertFalse((Path(d)/'out.csv').exists())
    @patch('requests.post')
    def test_fetch_mixed_filter_rejected(self,post):
        post.return_value.text='sourceid,catid,mjdobs,filter,mag,emag,phot_flag,ambiguous_match\n123,13,55281,J,14,0.03,0,0\n'
        post.return_value.raise_for_status.return_value=None
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):engine.fetch(['123'],Path(d)/'out.csv')
    @patch('requests.post')
    def test_fetch_valid_no_overwrite(self,post):
        post.return_value.text='sourceid,catid,mjdobs,filter,mag,emag,phot_flag,ambiguous_match\n123,13,55281,Ks,14,0.03,0,0\n'
        post.return_value.raise_for_status.return_value=None
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'out.csv'
            result=engine.fetch(['123'],p)
            self.assertEqual(result['matched_sources'],1)
            self.assertEqual(result['missing_source_ids'],[])
            self.assertTrue(p.is_file())
            with self.assertRaises(FileExistsError):engine.fetch(['123'],p)

if __name__=='__main__':unittest.main(verbosity=2)
