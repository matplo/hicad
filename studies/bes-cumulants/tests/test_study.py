import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import kve
from scipy.stats import kstat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bes_cumulants import physics as p
from bes_cumulants.statistics import cumulants,metrics,Estimator
from bes_cumulants.study import analyze


class AcceptanceTests(unittest.TestCase):
    def test_normalization_and_direct_shared_sampler(self):
        t=.1;m=p.MASS
        z=4*np.pi*quad(lambda x:x*x*np.exp(-(np.sqrt(m*m+x*x)-m)/t),0,np.inf,epsabs=1e-11)[0]
        self.assertAlmostEqual(z,4*np.pi*m*m*t*kve(2,m/t),places=10)
        rest=p.clumps.sample_rest_protons(30000,t,p.clumps.StreamFactory(dict(root_seed='193',clump_revision=0,direction_revision=0,control_revision=0)))
        for beta,u in [(0,0),(.4,0),(.8,.8),(.8,1)]:
            lab=p.clumps.boost_four_vectors(rest,[beta*np.sqrt(1-u*u),0,beta*u]);k=p.clumps.kinematics(lab)
            measured=np.mean((abs(k['y'])<.5)&(k['pt']>.4)&(k['pt']<2))
            expected=p.acceptance(t,beta,u,.5)[0]
            self.assertLess(abs(measured-expected),.015)

    def test_quadrature_symmetry_convergence_and_nested_windows(self):
        a=p.acceptance(.06,.8,[.2,.7],.5,order=64)
        b=p.acceptance(.06,.8,[-.2,-.7],.5,order=128)
        np.testing.assert_allclose(a,b,atol=2e-7)
        table=p.acceptance_table(.1,.4,[.25,.5,1.])
        self.assertLess(table['quadrature_error'],2e-6)
        self.assertLess(table['interpolation_error'],2e-6)
        self.assertTrue(np.all(np.diff(table['values'],axis=0)>=0))
        point=.377
        for i,w in enumerate([.25,.5,1.]):
            interpolated=np.interp(point,table['grid'],table['values'][i])
            self.assertLess(abs(interpolated-p.acceptance(.1,.4,point,w,order=128)[0]),2e-6)

    def test_count_nesting_prefix_and_transverse_mixed_null(self):
        table=p.acceptance_table(.1,.4,[.25,.5,1.])
        a=p.source_counts(1000,[1,4,8],1,table,'transverse','42')
        b=p.source_counts(2000,[1,4,8,16],1,table,'transverse','42')
        np.testing.assert_array_equal(a['shared_boost'],a['event_mixed'])
        np.testing.assert_array_equal(a['shared_boost'],b['shared_boost'][:1000,:,:3])
        self.assertTrue(np.all(np.diff(a['shared_boost'].astype(int),axis=1)>=0))
        self.assertTrue(np.all(np.diff(a['shared_boost'].astype(int),axis=2)>=0))

    def test_isotropic_source_count_variance_retains_shared_direction(self):
        table=p.acceptance_table(.1,.8,[.5]);grid=table['grid'];a=table['values'][0]
        mean=np.trapezoid(a,grid);var=np.trapezoid(a*a,grid)-mean*mean
        samples=p.source_counts(50000,[16],1,table,'isotropic_3d','789')
        actual=np.var(samples['shared_boost'][:,0,0])-np.var(samples['event_mixed'][:,0,0])
        self.assertLess(abs(actual-16*15*var),.15)

    def test_geometry_independence_and_prefix(self):
        a=p.selected_geometry(1000,'42');b=p.selected_geometry(1500,'42')
        for name in a:np.testing.assert_array_equal(a[name],b[name][:1000])
        self.assertEqual(len(np.unique(a['event_id'])),1000)
        self.assertTrue(np.all((a['centrality']>=0)&(a['centrality']<.05)))
        protons,anti=p.background_counts(a['emitted'],[.2,.4,.6],'42')
        self.assertTrue(np.all(np.diff(protons,axis=1)>=0))
        self.assertTrue(np.all(protons+anti<=a['emitted'][:,None]))


class StatisticsTests(unittest.TestCase):
    def test_unbiased_kstatistics_match_scipy(self):
        x=np.random.default_rng(3).poisson(6,10000)-2
        moments=np.array([len(x),*[np.sum(x.astype(float)**r) for r in range(1,5)]])
        np.testing.assert_allclose(cumulants(moments),[kstat(x.astype(float),r) for r in range(1,5)],rtol=1e-10,atol=1e-9)

    def test_factorial_transform_and_poisson_reference(self):
        out=metrics(np.array([3.,3.,3.,3.]))
        np.testing.assert_array_equal(out[7:],0)
        self.assertEqual(out[6],1)

    def test_analytic_fixed_shift_binomial_and_rare_gate(self):
        table={'grid':np.array([0.,1.]),'values':np.array([[1.,1.]])}
        np.testing.assert_allclose(p.source_prediction(8,1,1,table,'transverse','shared_boost',0),[8,0,0,0])
        f=.1
        expected=[8*f,8**2*f*(1-f),8**3*f*(1-f)*(1-2*f),8**4*f*(1-f)*(1-6*f+6*f*f)]
        np.testing.assert_allclose(p.source_prediction(8,1,f,table,'transverse','shared_boost',0),expected)
        table['values'][:]=.6;a=.6;n=8
        expected=[n*a,n*a*(1-a),n*a*(1-a)*(1-2*a),n*a*(1-a)*(1-6*a*(1-a))]
        np.testing.assert_allclose(p.source_prediction(n,1,1,table,'transverse','shared_boost',0),expected,atol=1e-10)

    def test_paired_constant_shift_and_cbwc(self):
        r=np.random.default_rng(7);c=np.tile([.01,.04],5000);x=r.poisson(np.where(c<.025,3,9))
        e=Estimator(c,100,100,42,2)
        base=e.evaluate(x);shift=e.summarize(x+3,base)
        for mode in ['raw','cbwc']:
            self.assertAlmostEqual(shift[mode]['C1'][4],3)
            for name in ['C2','C3','C4']:
                self.assertLess(abs(shift[mode][name][4]),1e-7)
                self.assertLess(shift[mode][name][7],1e-7)
        self.assertGreater(base['raw'][0][1],base['cbwc'][0][1]+5)
        manual=(np.array([kstat(x[c<.025].astype(float),r) for r in range(1,5)])+np.array([kstat(x[c>.025].astype(float),r) for r in range(1,5)]))/2
        np.testing.assert_allclose(base['cbwc'][0][:4],manual,atol=1e-9)


class EndToEndTests(unittest.TestCase):
    def test_joint_grid_covers_crossed_parameters_and_restarts(self):
        c=json.loads((ROOT/'configs/pilot.json').read_text())
        c['selected_events']=1000;c['bootstrap_blocks']=20;c['bootstrap_replicates']=50
        c['rapidity_windows']=[.5];c.pop('shape');c.pop('clump_scan')
        c['strength']['n']=[8];c['strength']['f']=[0,.1]
        c['joint']={'temperature_gev':[.06,.1],'beta':[0,.8],
            'direction':['transverse','isotropic_3d'],'clumps':[1,2],
            'n':[2,8],'f':[0,.1]}
        with tempfile.TemporaryDirectory() as tmp:
            a=analyze(c,tmp,lambda *a,**k:None)
            b=analyze(c,tmp,lambda *a,**k:None)
            self.assertEqual(a['records'],b['records'])
            rows=[r for r in a['records'] if r['scan']=='joint']
            self.assertEqual(len(rows),2**6*3)
            keys=[tuple(r[k] for k in ['T','beta','direction','k','n','f','model','ymax']) for r in rows]
            self.assertEqual(len(keys),len(set(keys)))
            for row in rows:
                if row['f']==0:self.assertEqual(row['metrics']['net']['cbwc']['C4_C2'][4],0)

    def test_small_scan_resume_and_zero_injection(self):
        c=json.loads((ROOT/'configs/pilot.json').read_text());c['selected_events']=2000
        c['bootstrap_blocks']=20;c['bootstrap_replicates']=50;c['rapidity_windows']=[.25,.5]
        c['strength']['n']=[1,8];c['strength']['f']=[0,.1,1];c.pop('shape');c.pop('clump_scan')
        with tempfile.TemporaryDirectory() as tmp:
            a=analyze(c,tmp,lambda *a,**k:None);b=analyze(c,tmp,lambda *a,**k:None)
            self.assertEqual(a['records'],b['records']);self.assertEqual(len(a['records']),36)
            for row in a['records']:
                if row['f']==0:
                    for mode in ['raw','cbwc']:
                        self.assertEqual(row['metrics']['net'][mode]['C4_C2'][4],0)
                        self.assertEqual(row['metrics']['net'][mode]['C4_C2'][7],0)
            json.dumps(a,allow_nan=False)
