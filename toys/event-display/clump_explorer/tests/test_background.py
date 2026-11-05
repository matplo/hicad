from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
from support import request
from clump_explorer.background import BackgroundProvider, decode_rows, MASSES
from clump_explorer.generator_adapter import core
from clump_explorer.scenario import generate_scenario


def fixture(path, observed=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    n=6;pt=np.array([.2,.35,.45,.6,.85,1.1]);eta=np.array([3.4,-.8,.1,.2,.3,.4]);phi=np.linspace(-2.8,2.8,n)
    mass=np.repeat(MASSES,2);charge=np.tile([1,-1],3)
    rows=np.column_stack((np.cos(phi),np.sin(phi),np.log(pt/pt.mean()),charge*np.log1p(mass),eta-eta.mean())).astype(np.float16)
    events=np.zeros((3,10),np.float32);events[:,0]=[0,.5,1];events[:,1]=n;events[:,7]=pt.mean();events[:,8]=pt.sum();events[:,9]=eta.mean()
    with h5py.File(path,'w') as f:
        f['mc_record']=np.array([9.2,.03],np.float32);f['event_record']=events
        padded=np.zeros((3,8,5),np.float16);padded[:,:n]=rows;f['point_cloud']=padded
        if observed:
            decoded,_,_=decode_rows(rows,events[0])
            arr=np.zeros((3,8,4),np.float32);arr[:,:n]=decoded
            f['observed/p4']=arr
    return rows,events


class BackgroundTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.path=self.root/'a'/'events.h5';self.rows,self.ev=fixture(self.path)
        fixture(self.root/'b'/'events.h5')
        self.provider=BackgroundProvider([self.path,self.root/'b'])
        self.dataset=self.provider.catalog()[0]

    def tearDown(self):
        self.tmp.cleanup()

    def select(self,**changes):
        body=dict(dataset_id=self.dataset['dataset_id'],mode='exact',event_index=0,
                  centrality_min=0,centrality_max=1,selection_seed='42');body.update(changes)
        selected=self.provider.select(body)
        return {k:selected[k] for k in ['provider','dataset_id','event_index','fingerprint']}

    def test_unique_ids_and_no_clipping_or_rounding(self):
        self.assertEqual(len(self.provider.catalog()),2)
        self.assertNotEqual(*[d['dataset_id'] for d in self.provider.catalog()])
        r=request();r['clump_config']['acceptance']['eta_max']=5;r['background']=self.select()
        result=generate_scenario(r,self.provider)
        m=result.data['background']['measured'];p=np.array(m['p4'])
        k=core.kinematics(p)
        np.testing.assert_allclose(k['eta'],self.rows[:,4].astype(float)+float(self.ev[0,9]),atol=1e-14)
        np.testing.assert_allclose(k['pt'],float(self.ev[0,7])*np.exp(self.rows[:,2].astype(float)),rtol=1e-14)
        self.assertGreater(k['eta'][0],3)
        self.assertEqual(m['species'],['pi+','pi-','K+','K-','p','pbar'])
        self.assertEqual(result.observation('shared_boost','background_only')['species'].count('p'),1)

    def test_selection_edges_empty_exact_and_seed(self):
        self.assertEqual(self.select(centrality_min=.5,centrality_max=1,event_index=2)['event_index'],2)
        self.assertEqual(self.select(mode='next',current_index=0)['event_index'],1)
        self.assertEqual(self.select(mode='previous',current_index=0)['event_index'],2)
        self.assertEqual(self.select(mode='random'),self.select(mode='random'))
        for changes in [dict(centrality_min=.1,centrality_max=.2),dict(event_index=99),dict(event_index=1,centrality_max=.5),dict(dataset_id='unknown')]:
            with self.assertRaises(core.ParameterError):self.select(**changes)

    def test_fingerprint_invalidates_even_cached_service(self):
        from clump_explorer.api import Service
        r=request();r['background']=self.select();service=Service(self.provider)
        service.scenario(r)
        with h5py.File(self.path,'r+') as f:f['event_record'][0,2]=9
        with self.assertRaises(core.ParameterError) as raised:service.scenario(r)
        self.assertEqual(raised.exception.status,409)
        self.path.unlink()
        with self.assertRaises(core.ParameterError) as raised:self.provider.read(r['background'],r['clump_config']['acceptance'])
        self.assertEqual(raised.exception.status,404)

    def test_background_changes_preserve_source_and_disabled_preserves_background(self):
        plain=generate_scenario(request())
        r=request();r['background']=self.select()
        injected=generate_scenario(r,self.provider)
        self.assertEqual(plain.data['variants'],injected.data['variants'])
        r['clump_config']['enabled']=False
        disabled=generate_scenario(r,self.provider)
        self.assertEqual(disabled.data['background'],injected.data['background'])
        self.assertEqual(disabled.observation('shared_boost','combined'),disabled.observation('shared_boost','background_only'))

    def test_signal_parameter_updates_keep_the_pinned_background(self):
        from clump_explorer.api import Service
        service=Service(self.provider);r=request();r['background']=self.select()
        first=service.scenario(r)
        for field,value in [('emitted_multiplicity',30),('temperature_gev',.2),('boost_speed',.8)]:
            changed=deepcopy(r);changed['clump_config'][field]={'distribution':'fixed','value':value}
            updated=service.scenario(changed)
            self.assertEqual(updated['resolved_request']['background'],first['resolved_request']['background'])
            self.assertEqual(updated['background'],first['background'])
            self.assertNotEqual(updated['scenario_id'],first['scenario_id'])

    def test_observed_p4_preferred_and_alignment_validated(self):
        p=self.root/'observed.h5';rows,ev=fixture(p,observed=True)
        provider=BackgroundProvider([p]);d=provider.catalog()[0]
        ref=dict(provider='hdf5',dataset_id=d['dataset_id'],fingerprint=d['fingerprint'],event_index=0)
        result=provider.read(ref,request()['clump_config']['acceptance'])
        with h5py.File(p,'r') as f:np.testing.assert_array_equal(result['measured']['p4'],f['observed/p4'][0,:6])
        decoded,_,_=decode_rows(rows,ev[0]);decoded[[0,1]]=decoded[[1,0]]
        with self.assertRaises(core.ParameterError):decode_rows(rows,ev[0],decoded)

    def test_source_only_without_files_and_bad_active_data(self):
        result=generate_scenario(request(),BackgroundProvider([self.root/'missing']))
        self.assertEqual(result.data['background']['measured']['p4'],[])
        rows=self.rows.copy();rows[0,2]=np.nan
        with self.assertRaises(core.ParameterError):decode_rows(rows,self.ev[0])
        rows=self.rows.copy();rows[0,3]=0
        with self.assertRaises(core.ParameterError):decode_rows(rows,self.ev[0])

    def test_full_particle_limit_has_no_truncation_and_rejects_overflow(self):
        p=self.root/'limit.h5'
        with h5py.File(p,'w') as f:
            ev=np.zeros((1,10),np.float32);ev[0,1]=20000;ev[0,7]=.5;ev[0,8]=10000
            f['event_record']=ev;f['mc_record']=np.array([9.2,.03],np.float32)
            rows=np.zeros((1,20000,5),np.float16);rows[:,:,0]=1;rows[:,:,3]=np.log1p(MASSES[0]);f['point_cloud']=rows
        provider=BackgroundProvider([p]);d=provider.catalog()[0];r=request()
        r['background']=dict(provider='hdf5',dataset_id=d['dataset_id'],event_index=0,fingerprint=d['fingerprint'])
        r['clump_config']['emitted_multiplicity']['value']=0
        b=generate_scenario(r,provider)
        self.assertEqual(len(b.observation('shared_boost','combined')['p4']),20000)
        r['clump_config']['emitted_multiplicity']['value']=1
        with self.assertRaises(core.ParameterError):generate_scenario(r,provider)
