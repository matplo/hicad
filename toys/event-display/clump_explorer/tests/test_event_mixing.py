from copy import deepcopy
import unittest
import numpy as np
from scipy.stats import ks_2samp
from support import request
from clump_explorer.generator_adapter import core
from clump_explorer.scenario import generate_scenario
from clump_explorer.contracts import ScenarioSpec
from clump_explorer.exports import recipe
from clump_explorer.diagnostics import for_bundle


class EventMixingTests(unittest.TestCase):
    def setUp(self):
        self.r=request()
        self.r['clump_config']['variants']=list(core.VARIANTS)
        self.r['clump_config']['emitted_multiplicity']['value']=30

    def test_independent_donors_mass_shell_full_boost_and_no_same_source_pairs(self):
        self.r['clump_config']['boost_direction']='isotropic_3d'
        bundle=generate_scenario(self.r);mixed=bundle.data['variants']['event_mixed']
        measured,truth=mixed['measured'],mixed['truth'];p=np.array(measured['p4'])
        np.testing.assert_allclose(p[:,0]**2-np.sum(p[:,1:]**2,axis=1),core.PROTON_MASS**2,atol=4e-15)
        self.assertEqual(len(set(truth['donor_event_id'])),len(p))
        self.assertTrue(all(c==-1 for c in truth['clump_id']))
        self.assertTrue(set(measured['particle_id']).isdisjoint(bundle.data['variants']['shared_boost']['measured']['particle_id']))
        for i in range(len(p)):
            expected=core.boost_four_vectors(np.array(truth['rest_p4'][i:i+1]),truth['beta_vectors'][i])
            np.testing.assert_array_equal(expected[0],p[i])
        d=for_bundle(bundle,'event_mixed','source_only','native')
        self.assertEqual(d['truth_pair_count'],0)
        self.assertFalse(d['truth_daughter_pairs']['available'])

    def test_mixed_prefix_rest_stream_and_acceptance_coupling(self):
        c=self.r['clump_config'];c['boost_direction']='isotropic_3d'
        a=core.generate_event_mixed_clump(c,self.r['realization'])
        c['emitted_multiplicity']['value']=40
        b=core.generate_event_mixed_clump(c,self.r['realization'])
        np.testing.assert_array_equal(a.lab_p4,b.lab_p4[:30]);self.assertEqual(a.particle_id,b.particle_id[:30])
        c['emitted_multiplicity']['value']=30;c['boost_speed']['value']=.8
        fast=core.generate_event_mixed_clump(c,self.r['realization'])
        np.testing.assert_array_equal(a.rest_p4,fast.rest_p4)
        self.r['realization']['direction_revision']+=1
        direction=core.generate_event_mixed_clump(c,self.r['realization'])
        np.testing.assert_array_equal(fast.rest_p4,direction.rest_p4)
        self.assertNotEqual(fast.donors['beta_vectors'],direction.donors['beta_vectors'])
        c['acceptance']['pt_min_gev']=100
        cut=core.generate_event_mixed_clump(c,self.r['realization'])
        np.testing.assert_array_equal(direction.lab_p4,cut.lab_p4);self.assertEqual(int(cut.accepted.sum()),0)

    def test_control_revision_preserves_shared_source_and_redraws_donors(self):
        first=generate_scenario(self.r)
        self.r['realization']['control_revision']+=1
        second=generate_scenario(self.r)
        self.assertEqual(first.data['variants']['shared_boost'],second.data['variants']['shared_boost'])
        self.assertNotEqual(first.data['variants']['event_mixed']['measured']['p4'],second.data['variants']['event_mixed']['measured']['p4'])
        self.assertNotEqual(first.data['variants']['azimuth_decohered']['measured']['p4'],second.data['variants']['azimuth_decohered']['measured']['p4'])

    def test_uniform_donor_parameters_independent_per_particle(self):
        c=self.r['clump_config']
        c['temperature_gev']={'distribution':'uniform','minimum':.06,'maximum':.16}
        c['boost_speed']={'distribution':'uniform','minimum':0.,'maximum':.6}
        a=core.generate_event_mixed_clump(c,self.r['realization'])
        self.assertGreater(len(set(a.donors['temperature_gev'])),1)
        self.assertGreater(len(set(a.donors['boost_speed'])),1)
        self.assertTrue(all(.06<=t<=.16 for t in a.donors['temperature_gev']))
        self.assertTrue(all(0<=b<=.6 for b in a.donors['boost_speed']))
        self.assertIsNone(a.realized['temperature_gev'])

    def test_old_variants_unchanged_and_recipe_replay_packed_orders(self):
        old=deepcopy(self.r);old['clump_config']['variants']=['shared_boost','azimuth_decohered']
        legacy=generate_scenario(old);bundle=generate_scenario(self.r)
        for v in old['clump_config']['variants']:self.assertEqual(legacy.data['variants'][v],bundle.data['variants'][v])
        saved=recipe(bundle);self.assertEqual(saved['provenance']['event_mixing_policy'],core.EVENT_MIXING_POLICY)
        replay=generate_scenario(saved);self.assertEqual(replay.data['variants'],bundle.data['variants'])
        self.r['observation_mode']='packed_float16';result=bundle.response(ScenarioSpec.parse(self.r))
        for composition in ['source_only','background_only','combined']:
            native=bundle.observation('event_mixed',composition)
            packed=result['observation_views']['event_mixed'][composition]
            self.assertEqual(native['particle_id'],packed['particle_id'])
        saved['provenance']['event_mixing_policy']='unsupported'
        with self.assertRaises(core.ParameterError):generate_scenario(saved)

    def test_empty_single_and_emitted_counts_match_but_acceptance_may_differ(self):
        for n in [0,1,30]:
            r=deepcopy(self.r);r['clump_config']['emitted_multiplicity']['value']=n
            r['clump_config']['acceptance']['eta_max']=.2
            b=generate_scenario(r)
            self.assertEqual(len(b.data['variants']['event_mixed']['measured']['p4']),n)
            d=for_bundle(b,'event_mixed','source_only','native');self.assertEqual(d['truth_pair_count'],0)
        r['clump_config']['enabled']=False;b=generate_scenario(r)
        self.assertEqual(b.data['variants']['event_mixed']['measured']['p4'],[])

    def test_ensemble_one_body_matching_and_longitudinal_correlation_removal(self):
        c=self.r['clump_config'];c['emitted_multiplicity']['value']=32;c['temperature_gev']['value']=.03
        c['boost_speed']['value']=.75;c['boost_direction']='isotropic_3d';c['acceptance']['eta_max']=100
        first={'shared':[],'mixed':[]};pair_products={'shared':[],'mixed':[]}
        for i in range(200):
            self.r['realization']['clump_revision']=i;self.r['realization']['direction_revision']=i
            for name,sample in [('shared',core.generate_clump(c,self.r['realization'])),
                                ('mixed',core.generate_event_mixed_clump(c,self.r['realization']))]:
                k=core.kinematics(sample.lab_p4);first[name].append([k['pt'][0],k['y'][0]])
                y=k['y'];pair_products[name].append((y.sum()**2-np.sum(y*y))/(len(y)*(len(y)-1)))
        # One daughter per event avoids treating correlated source daughters as
        # independent samples in the one-body two-sample check.
        for column in [0,1]:
            a=np.array(first['shared'])[:,column];b=np.array(first['mixed'])[:,column]
            self.assertLess(ks_2samp(a,b).statistic,.2)
        self.assertGreater(np.mean(pair_products['shared'])-np.mean(pair_products['mixed']),.1)
        self.assertLess(abs(np.mean(pair_products['mixed'])),.035)
