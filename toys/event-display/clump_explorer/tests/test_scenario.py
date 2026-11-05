from copy import deepcopy
import json
import unittest
import numpy as np
from support import request, DISPLAY
from clump_explorer.generator_adapter import core
from clump_explorer.scenario import generate_scenario
from clump_explorer.contracts import ScenarioSpec
from clump_explorer import diagnostics


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.r = request()

    def source(self, r, variant="shared_boost"):
        return generate_scenario(r).data["variants"][variant]

    def test_beta_direction_acceptance_and_count_coupling(self):
        a = self.source(self.r)
        for beta in [0, .8, .4]:
            r = deepcopy(self.r);r["clump_config"]["boost_speed"]["value"] = beta
            np.testing.assert_array_equal(self.source(r)["truth"]["rest_p4"], a["truth"]["rest_p4"])
        r = deepcopy(self.r);r["realization"]["direction_revision"] += 1
        b = self.source(r)
        self.assertEqual(a["truth"]["rest_p4"], b["truth"]["rest_p4"])
        self.assertNotEqual(a["measured"]["p4"], b["measured"]["p4"])
        r = deepcopy(self.r);r["clump_config"]["acceptance"]["eta_max"] = .01
        self.assertEqual(a["measured"]["p4"], self.source(r)["measured"]["p4"])
        r = deepcopy(self.r);r["clump_config"]["emitted_multiplicity"]["value"] = 20
        b = self.source(r)
        self.assertEqual(a["measured"]["p4"], b["measured"]["p4"][:8])
        self.assertEqual(a["measured"]["particle_id"], b["measured"]["particle_id"][:8])
        r["clump_config"]["clumps_per_injected_event"]["value"] = 3
        self.assertEqual(b["measured"]["p4"], self.source(r)["measured"]["p4"][:20])

    def test_default_daughter_leaves_and_reenters_eta_cut_without_resampling(self):
        samples=[]
        for beta in [.17,.18,.4,.68,.69]:
            r=deepcopy(self.r);r['clump_config']['boost_speed']['value']=beta
            samples.append(core.generate_clump(r['clump_config'],r['realization']))
        self.assertEqual([bool(s.accepted[4]) for s in samples],[True,False,False,False,True])
        self.assertEqual([s.particle_id[4] for s in samples],['c0:d4']*5)
        for s in samples:
            np.testing.assert_array_equal(s.rest_p4,samples[0].rest_p4)
            self.assertEqual(s.lab_p4[4,3],samples[0].lab_p4[4,3])
        self.assertGreater(samples[0].lab_p4[4,1],0)
        self.assertLess(samples[-1].lab_p4[4,1],0)

    def test_twins_and_acceptance_stable_control_angles(self):
        a = self.source(self.r);b = self.source(self.r, "azimuth_decohered")
        for key in ["particle_id", "accepted", "species", "charge"]:
            self.assertEqual(a["measured"][key], b["measured"][key])
        ka,kb = [core.kinematics(v["measured"]["p4"]) for v in (a,b)]
        for name in ["pt", "eta", "y"]:
            np.testing.assert_allclose(ka[name], kb[name], atol=2e-15)
        r = deepcopy(self.r);r["clump_config"]["acceptance"]["pt_min_gev"] = .5
        self.assertEqual(b["measured"]["p4"], self.source(r,"azimuth_decohered")["measured"]["p4"])

    def test_opt_in_comparison_controls_preserve_shared_source(self):
        r=request(include_controls=False);baseline=generate_scenario(r)
        self.assertEqual(set(baseline.data['variants']),{'shared_boost'})
        for extra in [['azimuth_decohered'],['event_mixed'],['azimuth_decohered','event_mixed']]:
            r['clump_config']['variants']=['shared_boost']+extra
            compared=generate_scenario(r)
            self.assertEqual(compared.data['variants']['shared_boost'],baseline.data['variants']['shared_boost'])

    def test_seed_replay_order_and_new_clump_direction(self):
        self.r["realization"]["root_seed"] = str(2**64-1)
        first = generate_scenario(self.r)
        other = deepcopy(self.r);other["realization"]["clump_revision"] = 8
        changed = generate_scenario(other)
        again = generate_scenario(self.r)
        self.assertEqual(first.data["variants"], again.data["variants"])
        self.assertEqual(first.data["source_truth"][0]["beta_vector"], changed.data["source_truth"][0]["beta_vector"])
        self.assertNotEqual(first.data["variants"], changed.data["variants"])
        self.assertEqual(first.data["scenario_id"], again.data["scenario_id"])

    def test_empty_disabled_zero_accepted_and_single(self):
        for kind in ["disabled", "zero", "zero_clumps", "rejected", "single"]:
            r=deepcopy(self.r)
            if kind=="disabled":r["clump_config"]["enabled"]=False
            if kind=="zero":r["clump_config"]["emitted_multiplicity"]["value"]=0
            if kind=="zero_clumps":r["clump_config"]["clumps_per_injected_event"]["value"]=0
            if kind=="rejected":r["clump_config"]["acceptance"]["pt_min_gev"]=1000
            if kind=="single":r["clump_config"]["emitted_multiplicity"]["value"]=1
            bundle=generate_scenario(r);result=bundle.response(ScenarioSpec.parse(r))
            json.dumps(result,allow_nan=False)
            s=result["summaries"]["shared_boost"]["combined"]
            self.assertLessEqual(s["count"],1)
            if not s["count"]:self.assertIsNone(s["mean_pt_gev"])
            d=diagnostics.for_bundle(bundle,"shared_boost","combined","native")
            self.assertEqual(d["pair_count"],0)
            self.assertEqual(d["message"],"No proton pairs")

    def test_variable_parameters_limits_and_validation(self):
        r=json.loads((DISPLAY/"clump_explorer/examples/variable_parameters.json").read_text())
        r["clump_config"]["clumps_per_injected_event"]["value"]=8
        a=generate_scenario(r);b=generate_scenario(r)
        self.assertEqual(a.data["source_truth"],b.data["source_truth"])
        for c in a.data["source_truth"]:
            self.assertTrue(2<=c["emitted"]<=20)
            self.assertTrue(.06<=c["temperature_gev"]<=.16)
        cases=[("temperature_gev",{"distribution":"fixed","value":0}),
               ("boost_speed",{"distribution":"fixed","value":.96}),
               ("emitted_multiplicity",{"distribution":"fixed","value":201}),
               ("source_frame_mode","exact_decay"),("variants",["independent_boost"]),
               ("signal_probability",.5),("enabled",1),("unknown",1)]
        for key,value in cases:
            r=deepcopy(self.r);r["clump_config"][key]=value
            with self.assertRaises(core.ParameterError,msg=key):generate_scenario(r)

    def test_exact_pair_diagnostics_and_chunking(self):
        self.r["clump_config"]["emitted_multiplicity"]["value"]=30
        b=generate_scenario(self.r);m=b.observation("shared_boost","source_only")
        ids=dict.fromkeys(m["particle_id"],0)
        a=diagnostics.calculate(m,ids,chunk=3);c=diagnostics.calculate(m,ids,chunk=100)
        self.assertEqual(a,c)
        n=len(m["p4"]);self.assertEqual(a["pair_count"],n*(n-1)//2)
        self.assertEqual(a["truth_pair_count"],a["pair_count"])
        for name,h in a["histograms"].items():
            self.assertEqual(sum(h["counts"])+h["underflow"]+h["overflow"],h["total_entries"])
        p=np.asarray(m["p4"]);i,j=np.triu_indices(n,1)
        q=diagnostics.q_inv(p[i],p[j]);edges=a["histograms"]["q_inv"]["edges"]
        self.assertEqual(np.histogram(q,edges)[0].tolist(),a["histograms"]["q_inv"]["counts"])

    def test_zero_boost_twins_statistically_isotropic_over_many_events(self):
        self.r['clump_config']['boost_speed']['value']=0
        self.r['clump_config']['emitted_multiplicity']['value']=100
        angles={v:[] for v in core.VARIANTS}
        for i in range(40):
            self.r['realization']['clump_revision']=i
            b=generate_scenario(self.r)
            for v in angles:
                p=b.observation(v,'source_only')['p4']
                angles[v].extend(core.kinematics(p)['phi'])
        for values in angles.values():
            for harmonic in (1,2,3):
                self.assertLess(abs(np.exp(1j*harmonic*np.array(values)).mean()),.045)
