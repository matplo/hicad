from copy import deepcopy
import json
import unittest
import numpy as np
from support import request
from clump_explorer.scenario import generate_scenario, summary
from clump_explorer.contracts import ScenarioSpec
from clump_explorer.exports import recipe,snapshot,model_observation
from clump_explorer.background import BackgroundProvider
from test_background import fixture
from pathlib import Path
import tempfile


class ExportTests(unittest.TestCase):
    def test_recipe_replay_and_truth_separation(self):
        r=request();r['realization']['root_seed']=str(2**64-1)
        bundle=generate_scenario(r)
        exported=json.loads(json.dumps(recipe(bundle),allow_nan=False))
        replay=generate_scenario(exported)
        self.assertEqual(bundle.data['scenario_id'],replay.data['scenario_id'])
        self.assertEqual(bundle.data['variants'],replay.data['variants'])
        snap=json.loads(json.dumps(snapshot(bundle),allow_nan=False))
        self.assertEqual(snap['source_truth'],bundle.data['source_truth'])
        self.assertIn('provenance',snap)
        for v in snap['variants'].values():
            self.assertNotIn('clump_id',v['measured']);self.assertIn('clump_id',v['truth'])
        model=model_observation(bundle)
        self.assertEqual(set(model),{'p4','species','charge'})

    def test_common_precision_per_composition_aligned_order_and_unchanged_canonical(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bg.h5';fixture(path);provider=BackgroundProvider([path]);ds=provider.catalog()[0]
            r=request();r['background']={k:ds[k] for k in ['dataset_id','fingerprint']};r['background'].update(provider='hdf5',event_index=0)
            r['observation_mode']='packed_float16';b=generate_scenario(r,provider);before=deepcopy(b.data)
            result=b.response(ScenarioSpec.parse(r))
            for variant in result['variants']:
                for composition,view in result['observation_views'][variant].items():
                    self.assertEqual(summary(view),result['summaries'][variant][composition])
                    other=result['observation_views']['shared_boost'][composition]
                    if variant!='event_mixed':
                        self.assertEqual(view['particle_id'],other['particle_id'])
                    else:
                        self.assertEqual(view['particle_id'],b.observation(variant,composition,'native')['particle_id'])
            self.assertEqual(before,b.data)
            source=result['observation_views']['shared_boost']['source_only'];combined=result['observation_views']['shared_boost']['combined']
            source_ids=set(source['particle_id']);positions={pid:p4 for pid,p4 in zip(combined['particle_id'],combined['p4'])}
            self.assertTrue(any(p4!=positions[pid] for pid,p4 in zip(source['particle_id'],source['p4'])))
            order=['source' if pid in source_ids else 'background' for pid in combined['particle_id']]
            self.assertNotEqual(order,sorted(order))

    def test_empty_packed_json(self):
        r=request();r['clump_config']['enabled']=False;r['observation_mode']='packed_float16'
        b=generate_scenario(r);s=snapshot(b)
        json.dumps(s,allow_nan=False)
        self.assertEqual(s['observation_views']['shared_boost']['combined']['p4'],[])
