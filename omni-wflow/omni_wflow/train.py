"""Device-independent supervised/denoising loops with validation-only selection."""
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from .data import rng,features,summaries,write_json
from .model import EventPET,collate,device


class Pairs(Dataset):
    def __init__(self,clouds,split,representation,seed,augment=False):
        self.rows=[];self.augment=augment;self.random=np.random.default_rng(seed)
        allowed=[j for j in range(len(clouds.settings)) if j not in clouds.heldout]
        for i in np.flatnonzero(clouds.split==split):
            j=allowed[int(rng(seed,11,int(clouds.family_id[i])).integers(len(allowed)))]
            for label,setting in [(0,None),(1,j)]:
                p4,s,_=clouds.event(i,setting)
                self.rows.append((features(p4,s,representation),label,summaries(p4,s),int(i),j))

    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        x,y,*_=self.rows[i];x=x.copy()
        if self.augment:
            angle=self.random.uniform(-np.pi,np.pi);x[:,1]=np.arctan2(np.sin(x[:,1]+angle),np.cos(x[:,1]+angle))
            x=x[self.random.permutation(len(x))]
        return x,y


def spec_for(config):
    if config['model_size']=='small':return dict(periodic_embedding=config.get('periodic_embedding',True),features=4 if config['representation']=='ideal' else 5,dim=128,depth=8,head_depth=2,heads=8,neighbors=8)
    return dict(periodic_embedding=config.get('periodic_embedding',True),features=4 if config['representation']=='ideal' else 5,dim=32,depth=2,head_depth=1,heads=4,neighbors=5)


def train(clouds,config,out,seed,init=None,freeze=False,pretrain=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);checkpoint=out/'best.pt'
    identity={'data_fingerprint':clouds.fingerprint,'training':config,'seed':seed,
              'init_sha256':None,'freeze':freeze,'objective':'denoising' if pretrain else 'classifier'}
    if init:
        import hashlib
        identity['init_sha256']=hashlib.sha256(Path(init).read_bytes()).hexdigest()
    if (out/'identity.json').exists():
        import json
        if json.loads((out/'identity.json').read_text())!=identity:raise ValueError('Run identity mismatch')
        if (out/'complete.json').exists():return checkpoint
    write_json(out/'identity.json',identity)
    torch.set_num_threads(config['threads']);torch.manual_seed(seed);np.random.seed(seed)
    dev=device(config['device']);model=EventPET(spec_for(config)).to(dev)
    if init:write_json(out/'transfer.json',model.initialize_encoder(init))
    if freeze:
        for p in model.pet.body.parameters():p.requires_grad=False
        if model.angle_embed is not None:
            for p in model.angle_embed.parameters():p.requires_grad=False
    if freeze and pretrain:raise ValueError('Denoising requires a trainable encoder')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],weight_decay=.01)
    print(f'Preparing train/validation clouds; device={dev}; parameters={sum(p.numel() for p in model.parameters()):,}',flush=True)
    sets=[Pairs(clouds,s,config['representation'],seed,augment=s==0) for s in [0,1]]
    loaders=[DataLoader(ds,batch_size=config['batch'],shuffle=s==0,collate_fn=collate,num_workers=0,
             generator=torch.Generator().manual_seed(seed)) for s,ds in enumerate(sets)]
    best=float('inf');bad=0;history=[];started=time.perf_counter()
    for epoch in range(config['epochs']):
        losses=[]
        for phase,loader in enumerate(loaders):
            model.train(phase==0);total=0.;count=0
            for x,mask,y in loader:
                x,mask,y=x.to(dev),mask.to(dev),y.to(dev)
                with torch.set_grad_enabled(phase==0):
                    if pretrain:
                        # Local toy-domain pretraining: denoise log-pT/log-E,
                        # training split only; validation corruption is reproducible.
                        gen=torch.Generator(device=dev).manual_seed(seed+epoch*10000+count if phase==0 else seed+count)
                        noisy=x.clone();noisy[:,:,2:4]+=.2*torch.randn(x[:,:,2:4].shape,device=dev,generator=gen)*mask
                        pred=model.reconstruct(noisy,mask)
                        loss=((pred-x[:,:,2:4]).square()*mask).sum()/(2*mask.sum().clamp_min(1))
                    else:loss=torch.nn.functional.cross_entropy(model(x,mask),y)
                    if phase==0:
                        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                if not torch.isfinite(loss):raise ArithmeticError('Non-finite training loss')
                total+=float(loss.detach())*len(x);count+=len(x)
            losses.append(total/count)
        history.append({'epoch':epoch+1,'train_loss':losses[0],'validation_loss':losses[1],'elapsed_seconds':time.perf_counter()-started})
        print(f'Seed {seed} epoch {epoch+1}: train {losses[0]:.5f}, validation {losses[1]:.5f}',flush=True)
        if losses[1]<best:
            best=losses[1];bad=0
            torch.save({'state':model.state_dict(),'spec':model.spec,'representation':config['representation'],
                'identity':identity,'epoch':epoch+1,'validation_loss':best},checkpoint)
        else:bad+=1
        write_json(out/'history.json',history)
        if bad>=config['patience']:break
    import joblib
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=config['threads'])
    from sklearn.ensemble import HistGradientBoostingClassifier
    # All baselines are fit exclusively on training families. Counts use ideal PID.
    a=sets[0].rows;summary=np.stack([r[2] for r in a]);labels=np.array([r[1] for r in a])
    baseline={}
    for name,cols in [('multiplicity',[0]),('proton_counts',[1,2,3]),('summaries',list(range(9)))]:
        baseline[name]=(HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=7,l2_regularization=1.,random_state=seed).fit(summary[:,cols],labels),cols)
    joblib.dump(baseline,out/'baselines.joblib')
    counts=summary[:,0].astype(int);bound=int(counts.max())+1
    p0=np.bincount(counts[labels==0],minlength=bound)+1.
    p1=np.bincount(counts[labels==1],minlength=bound)+1.
    p0/=p0.sum();p1/=p1.sum();target=np.minimum(p0,p1)
    write_json(out/'count_matching.json',{'background_weights':(target/p0).tolist(),'signal_weights':(target/p1).tolist(),
        'note':'Training-only smoothed pooled multiplicity matching; diagnostic point estimate, no claim of exact per-setting matching.'})
    write_json(out/'complete.json',{'best_validation_loss':best,'seconds':time.perf_counter()-started,'device':str(dev),'training_families':len(a)//2,'validation_families':len(sets[1])//2})
    return checkpoint
