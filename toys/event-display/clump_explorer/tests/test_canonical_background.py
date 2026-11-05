"""Shared-law, campaign reproducibility and precision/metadata compatibility."""
from pathlib import Path
import sys
import tempfile
import unittest
import json
import math
import hashlib
import h5py
import numpy as np
from scipy.integrate import quad
from support import request, REPO
sys.path.insert(0,str(REPO/'toys/boltzmann'))
import canonical_background as bg
from generate_canonical_h5 import generate_dataset
from clump_explorer.generator_adapter import core
from clump_explorer.background import BackgroundProvider
from clump_explorer.scenario import generate_scenario
from clump_explorer.contracts import ScenarioSpec


class CanonicalBackgroundTests(unittest.TestCase):
    def test_exact_proton_sampler_parity_and_independent_default_streams(self):
        r=request();r['clump_config']['emitted_multiplicity']['value']=200
        r['clump_config']['boost_speed']['value']=0
        sample=bg.generate_background_event(200,.1,[0,0,0,0,1,0],stream_factory=core.StreamFactory(r['realization']))
        source=core.generate_clump(r['clump_config'],r['realization'])
        np.testing.assert_array_equal(sample.emitted_p4,source.rest_p4)
        np.testing.assert_array_equal(sample.accepted,source.accepted)
        default=bg.generate_background_event(200,.1,[0,0,0,0,1,0],root_seed=r['realization']['root_seed'])
        self.assertFalse(np.array_equal(default.emitted_p4,source.rest_p4))

    def test_all_species_share_the_correct_massive_radial_law(self):
        streams=bg.BackgroundStreams('123')
        for mass in [.13957039,.493677,core.PROTON_MASS]:
            p4=core.sample_rest_particles(6000,mass,.1,streams)
            momentum=np.linalg.norm(p4[:,1:],axis=1)
            np.testing.assert_allclose(p4[:,0]**2-momentum**2,mass**2,atol=3e-15)
            density=lambda p:p*p*math.exp(-(math.sqrt(mass*mass+p*p)-mass)/.1)
            normalization=quad(density,0,np.inf)[0]
            for edge in [.2,.4,.7]:
                expected=quad(density,0,edge)[0]/normalization
                self.assertLess(abs(float((momentum<edge).mean())-expected),.025)
            directions=p4[:,1:]/momentum[:,None]
            np.testing.assert_allclose(directions.mean(axis=0),0,atol=.025)

    def test_background_prefix_seed_replay_cuts_and_zero_count(self):
        probabilities=[.4,.4,.06,.06,.04,.04]
        a=bg.generate_background_event(30,.1,probabilities,root_seed='18446744073709551615',event_index=9)
        b=bg.generate_background_event(60,.1,probabilities,root_seed='18446744073709551615',event_index=9)
        np.testing.assert_array_equal(a.emitted_p4,b.emitted_p4[:30])
        np.testing.assert_array_equal(a.species_index,b.species_index[:30])
        strict=bg.generate_background_event(30,.1,probabilities,root_seed='18446744073709551615',event_index=9,
                    acceptance={'eta_max':.1,'pt_min_gev':100.,'pt_max_gev':None,'azimuth':'full'})
        np.testing.assert_array_equal(a.emitted_p4,strict.emitted_p4)
        self.assertEqual(int(strict.accepted.sum()),0)
        zero=bg.generate_background_event(0,.1,probabilities)
        self.assertEqual(zero.emitted_p4.shape,(0,4))

    def test_writer_replay_padding_metadata_no_refill_and_adapter(self):
        with tempfile.TemporaryDirectory() as first,tempfile.TemporaryDirectory() as second:
            a=generate_dataset('bes',12,.1,'18446744073709551615',first)
            b=generate_dataset('bes',12,.1,'18446744073709551615',second)
            with h5py.File(a) as fa,h5py.File(b) as fb:
                for name in ['event_record','point_cloud','observed/p4','truth/event/emitted_multiplicity']:
                    np.testing.assert_array_equal(fa[name][:],fb[name][:])
                self.assertEqual(fa.attrs['thermal_law'],core.THERMAL_LAW)
                self.assertEqual(fa.attrs['temperature_gev'],.1)
                self.assertEqual(fa.attrs['root_seed'],'18446744073709551615')
                self.assertEqual(fa['point_cloud'].dtype,np.float16)
                self.assertEqual(fa['observed/p4'].dtype,np.float32)
                self.assertEqual(fa['event_record'].dtype,np.float32)
                emitted=fa['truth/event/emitted_multiplicity'][:];accepted=fa['truth/event/accepted_multiplicity'][:]
                self.assertTrue(np.any(accepted<emitted))
                for i,m in enumerate(accepted):
                    self.assertEqual(fa['event_record'][i,1],m)
                    self.assertTrue(np.all(fa['point_cloud'][i,m:]==0))
                    self.assertTrue(np.all(fa['observed/p4'][i,m:]==0))
                self.assertEqual(json.loads(fa.attrs['acceptance_json'])['eta_max'],1.)
            provider=BackgroundProvider([a]);self.assertEqual(provider.warnings,[])
            d=provider.catalog()[0];self.assertEqual(d['physics']['temperature_gev'],.1)
            r=request();r['background']=dict(provider='hdf5',dataset_id=d['dataset_id'],fingerprint=d['fingerprint'],event_index=0)
            bundle=generate_scenario(r,provider);result=bundle.response(ScenarioSpec.parse(r))
            self.assertEqual(result['background']['metadata']['physics']['thermal_law'],core.THERMAL_LAW)
            self.assertEqual(result['background']['metadata']['physics']['temperature_gev'],.1)
            self.assertGreaterEqual(result['background']['metadata']['emitted_multiplicity'],len(result['background']['measured']['p4']))
            json.dumps(result,allow_nan=False)

    def test_existing_output_protected_and_empty_acceptance_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=generate_dataset('bes',2,output_dir=tmp,pt_min_gev=100.)
            before=hashlib.sha256(a.read_bytes()).hexdigest()
            with self.assertRaises(FileExistsError):generate_dataset('bes',2,output_dir=tmp)
            self.assertEqual(hashlib.sha256(a.read_bytes()).hexdigest(),before)
            with h5py.File(a) as f:
                self.assertTrue(np.all(f['event_record'][:,1]==0))
                self.assertTrue(np.all(f['point_cloud'][:]==0))
            provider=BackgroundProvider([a]);self.assertEqual(provider.warnings,[])
            d=provider.catalog()[0]
            r=request();r['background']=dict(provider='hdf5',dataset_id=d['dataset_id'],fingerprint=d['fingerprint'],event_index=0)
            b=generate_scenario(r,provider);self.assertEqual(b.data['background']['measured']['p4'],[])

    def test_invalid_mass_temperature_probability_and_counts(self):
        for mass in [0,-1,True,[True,False],['.1','.2'],[.1],np.nan]:
            with self.assertRaises(ValueError):core.sample_rest_particles(2,mass,.1,bg.BackgroundStreams('42'))
        for t in [0,np.nan,.6]:
            with self.assertRaises(ValueError):bg.generate_background_event(2,t,[1,0,0,0,0,0])
        for probs in [[1],[0]*6,[-1,1,1,1,1,1]]:
            with self.assertRaises(ValueError):bg.generate_background_event(2,.1,probs)
