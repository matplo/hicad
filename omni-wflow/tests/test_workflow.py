import sys
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from omni_wflow.model import EventPET,collate,periodic_local
from omni_wflow.evaluate import paired_auc
from omni_wflow.data import features,generate,Clouds

class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(1)
        cls.model=EventPET(dict(periodic_embedding=True,features=4,dim=16,depth=1,head_depth=1,heads=2,neighbors=3)).eval()

    def test_logits_gradient_and_padding_permutation(self):
        a=np.random.default_rng(1).normal(size=(7,4)).astype('f4');a[0,2]=0
        x,mask,y=collate([(a,1)])
        result=self.model(x,mask);self.assertEqual(result.shape,(1,2))
        padded=torch.cat([x,torch.zeros(1,5,4)],1);pm=torch.cat([mask,torch.zeros(1,5,1,dtype=torch.bool)],1)
        torch.testing.assert_close(result,self.model(padded,pm),atol=2e-6,rtol=2e-6)
        order=torch.randperm(7);torch.testing.assert_close(result,self.model(x[:,order],mask),atol=2e-6,rtol=2e-6)
        result.sum().backward();self.assertTrue(any(p.grad is not None for p in self.model.pet.body.parameters()))

    def test_empty_and_small_events(self):
        for n in [0,1,2]:
            x,mask,_=collate([(np.zeros((n,4),dtype='f4'),0)])
            result=self.model(x,mask);self.assertTrue(torch.isfinite(result).all())
            padded=torch.cat([x,torch.zeros(1,8,4)],1);pm=torch.cat([mask,torch.zeros(1,8,1,dtype=torch.bool)],1)
            torch.testing.assert_close(result,self.model(padded,pm),atol=2e-6,rtol=2e-6)

    def test_periodic_local_and_reload(self):
        x=torch.randn(1,6,4);mask=torch.ones(1,6,1,dtype=torch.bool)
        original=periodic_local(self.model.pet.body.local_physics,x,mask)
        shifted=x.clone();shifted[:,0,1]+=2*np.pi
        torch.testing.assert_close(original,periodic_local(self.model.pet.body.local_physics,shifted,mask),atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(self.model(x,mask),self.model(shifted,mask),atol=2e-6,rtol=2e-6)
        copy=EventPET(self.model.spec).eval();copy.load_state_dict(self.model.state_dict())
        torch.testing.assert_close(self.model(x,mask),copy(x,mask))

    def test_auc_bootstrap_ties_and_mixture(self):
        b=np.array([.1,.5,.5,.9]);s=np.array([.5,.9,.2,.9]);w=np.ones((1,4))
        expected=roc_auc_score([0]*4+[1]*4,np.r_[b,s])
        self.assertAlmostEqual(paired_auc(b,s,w)[0],expected)
        w=np.array([[2,0,1,1]])
        self.assertAlmostEqual(paired_auc(b,s,w)[0],roc_auc_score([0]*4+[1]*4,np.r_[b,s],sample_weight=np.r_[w[0],w[0]]))
        for f in [0,.1,.5,1]:
            # Explicit positive mixture: another independent background copy and
            # injected score distribution; identical score values permit ties.
            y=np.r_[np.zeros(4),np.ones(8)];score=np.r_[b,b,s];weights=np.r_[np.ones(4),np.full(4,1-f),np.full(4,f)]
            self.assertAlmostEqual(roc_auc_score(y,score,sample_weight=weights),.5+f*(expected-.5))

    def test_family_splits_and_scramble_count_closure(self):
        config=json.loads((Path(__file__).resolve().parents[1]/'configs/run-smoke.json').read_text())['data'];config['parents']=40
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'clouds.h5';generate(config,p);c=Clouds(p)
            sets=[set(c.family_id[c.split==s]) for s in [0,1,2]]
            self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
            a=c.source(0,0);b=c.source(0,0,'azimuth_decohered')
            np.testing.assert_allclose(a[:,[0,3]],b[:,[0,3]])
            np.testing.assert_allclose(np.hypot(a[:,1],a[:,2]),np.hypot(b[:,1],b[:,2]),rtol=1e-6)
            x=features(*c.background(0));self.assertEqual(x.shape[1],4)
            self.assertTrue(np.isfinite(x).all())
            config['parents']=41
            with self.assertRaises(ValueError):generate(config,p)

    def test_training_evaluation_restart_integration(self):
        from omni_wflow.train import train
        from omni_wflow.evaluate import evaluate
        from omni_wflow.browser import build
        c=json.loads((Path(__file__).resolve().parents[1]/'configs/run-smoke.json').read_text())
        c['data']['parents']=40;c['training']['epochs']=1;c['training']['threads']=1
        c['evaluation']['bootstrap']=20;c['evaluation']['threads']=1
        with tempfile.TemporaryDirectory() as td:
            td=Path(td);generate(c['data'],td/'particles.h5');clouds=Clouds(td/'particles.h5')
            checkpoint=train(clouds,c['training'],td/'run',101)
            self.assertEqual(checkpoint,train(clouds,c['training'],td/'run',101))
            result=evaluate(clouds,checkpoint,td/'evaluation',c['evaluation'])
            self.assertFalse(result['validation']['family_overlap'])
            self.assertEqual(len(result['records']),32)
            resumed=evaluate(clouds,checkpoint,td/'evaluation',c['evaluation'])
            self.assertEqual(result['records'],resumed['records'])
            build(result,td/'index.html');self.assertIn('Whole-event', (td/'index.html').read_text())

if __name__=='__main__':unittest.main()
