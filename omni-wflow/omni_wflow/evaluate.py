"""Held-out, family-paired ROC/AUC evaluation; no tuning on test scores."""
from pathlib import Path
import json
import hashlib
import time
import joblib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,roc_curve,average_precision_score
from .data import rng,features,summaries,write_json
from .model import EventPET,collate,device


def paired_auc(background,signal,weights):
    """Vectorized family bootstrap with exact half-credit for score ties."""
    n=len(background);scores=np.r_[background,signal];label=np.r_[np.zeros(n),np.ones(n)]
    order=np.argsort(scores,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(scores[order]))+1]
    w=np.concatenate([weights,weights],axis=1)[:,order]
    pos=np.add.reduceat(w*label[order],starts,axis=1)
    neg=np.add.reduceat(w*(1-label[order]),starts,axis=1)
    return (pos*(np.cumsum(neg,axis=1)-.5*neg)).sum(1)/(pos.sum(1)*neg.sum(1))


def metrics(bg,sg,weights):
    y=np.r_[np.zeros(len(bg)),np.ones(len(sg))];score=np.r_[bg,sg]
    fpr,tpr,_=roc_curve(y,score);auc=float(roc_auc_score(y,score))
    boot=paired_auc(bg,sg,weights)
    # Conservative empirical operating points, not interpolated perfect rejection.
    result={'auc':auc,'low95':float(np.quantile(boot,.025)),'high95':float(np.quantile(boot,.975)),
        'auc_se':float(boot.std(ddof=1)),'average_precision':float(average_precision_score(y,score)),
        'eff_at_fpr_0.1':float(tpr[fpr<=.1].max()),'eff_at_fpr_0.01':float(tpr[fpr<=.01].max()),
        'n_background':len(bg),'n_signal':len(sg),'fpr_resolution':1/len(bg),'class_prior':.5,
        'roc':{'fpr':fpr.tolist(),'tpr':tpr.tolist()}}
    edges=np.linspace(0,1,31)
    result['score_hist']={'edges':edges.tolist(),'background':np.histogram(bg,edges)[0].tolist(),'signal':np.histogram(sg,edges)[0].tolist()}
    return result


def predict(model,events,representation,dev,batch_size,rotations):
    score=np.zeros(len(events))
    model.eval()
    with torch.no_grad():
        for angle in np.arange(rotations)*2*np.pi/rotations:
            for start in range(0,len(events),batch_size):
                batch=events[start:start+batch_size]
                x,mask,_=collate([(features(p,s,representation,angle),0) for p,s in batch])
                out=model(x.to(dev),mask.to(dev)).softmax(-1)[:,1].cpu().numpy()
                score[start:start+len(batch)]+=out/rotations
    return score


def evaluate(clouds,checkpoint,out,config):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved['identity']['objective']!='classifier':raise ValueError('Evaluate a classifier checkpoint, not a denoising pretraining head')
    identity={'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),'evaluation':config,'data_fingerprint':clouds.fingerprint}
    if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text())!=identity:raise ValueError('Evaluation checkpoint/config mismatch; choose a new output directory')
    write_json(out/'identity.json',identity)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=config['threads'])
    torch.set_num_threads(config['threads']);dev=device(config['device'])
    model=EventPET(saved['spec']).to(dev);model.load_state_dict(saved['state']);model.eval()
    if not config.get('external_validation',False) and saved['identity']['data_fingerprint']!=clouds.fingerprint:raise ValueError('Checkpoint/data mismatch')
    families=np.flatnonzero(clouds.split==2);n=len(families)
    weights=rng(clouds.config['data_seed'],40).multinomial(n,np.full(n,1/n),size=config['bootstrap'])
    backgrounds=[clouds.background(int(i)) for i in families]
    base_summary=np.stack([summaries(*e) for e in backgrounds])
    baselines=joblib.load(Path(checkpoint).parent/'baselines.joblib')
    bg={'PET2':predict(model,backgrounds,saved['representation'],dev,config['batch'],config['rotations'])}
    bg.update({name:m.predict_proba(base_summary[:,cols])[:,1] for name,(m,cols) in baselines.items()})
    np.savez_compressed(out/'background_scores.npz',family_id=clouds.family_id[families],**bg)
    matching_path=Path(checkpoint).parent/'count_matching.json'
    if not matching_path.exists():
        from .train import Pairs
        pairs=Pairs(clouds,0,saved['representation'],saved['identity']['seed'])
        counts=np.array([r[2][0] for r in pairs.rows],dtype=int);labels=np.array([r[1] for r in pairs.rows]);bound=int(counts.max())+1
        p0=np.bincount(counts[labels==0],minlength=bound)+1.;p1=np.bincount(counts[labels==1],minlength=bound)+1.
        p0/=p0.sum();p1/=p1.sum();target=np.minimum(p0,p1)
        write_json(matching_path,{'background_weights':(target/p0).tolist(),'signal_weights':(target/p1).tolist()})
    matching=json.loads(matching_path.read_text())
    def count_weight(counts,role):
        table=np.array(matching[role+'_weights']);idx=counts.astype(int)
        return np.where(idx<len(table),table[np.minimum(idx,len(table)-1)],0.)
    rows=[];started=time.perf_counter()
    for j,setting in enumerate(clouds.settings):
        models=['shared_boost']
        if j in config['control_settings']:models+=['event_mixed','azimuth_decohered','mean_matched_poisson']
        for control in models:
            destination=out/f'setting-{j:03d}-{control}.json'
            if destination.exists():rows.extend(json.loads(destination.read_text()));continue
            events=[];accepted=[]
            for i in families:
                p,s,k=clouds.event(int(i),j,control);events.append((p,s));accepted.append(k)
            summary=np.stack([summaries(*e) for e in events])
            predictions={'PET2':predict(model,events,saved['representation'],dev,config['batch'],config['rotations'])}
            predictions.update({name:m.predict_proba(summary[:,cols])[:,1] for name,(m,cols) in baselines.items()})
            np.savez_compressed(out/f'scores-{j:03d}-{control}.npz',family_id=clouds.family_id[families],accepted_source=accepted,**predictions)
            batch_rows=[]
            for name,score in predictions.items():
                m=metrics(bg[name],score,weights)
                has=np.asarray(accepted)>0
                m['auc_given_accepted_source']=float(roc_auc_score(np.r_[np.zeros(has.sum()),np.ones(has.sum())],np.r_[bg[name][has],score[has]])) if has.any() else None
                w0=count_weight(base_summary[:,0],'background');w1=count_weight(summary[:,0],'signal')
                m['multiplicity_reweighted_auc']=float(roc_auc_score(np.r_[np.zeros(n),np.ones(n)],np.r_[bg[name],score],sample_weight=np.r_[w0,w1])) if w0.sum()>0 and w1.sum()>0 else None
                m.update(setting=setting,setting_id=j,control=control,classifier=name,
                    heldout_parameter=j in clouds.heldout,zero_accepted_source_fraction=float(np.mean(np.array(accepted)==0)),
                    mean_accepted_source=float(np.mean(accepted)))
                m['mixture']=[{'f':f,'auc':.5+f*(m['auc']-.5),'low95':.5+f*(m['low95']-.5),'high95':.5+f*(m['high95']-.5)} for f in config['mixture_fractions']]
                batch_rows.append(m)
            write_json(destination,batch_rows);rows.extend(batch_rows)
        print(f'Test grid: {j+1}/{len(clouds.settings)} ({time.perf_counter()-started:.1f}s)',flush=True)
    # No-signal null uses disjoint halves of the held-out background pool.
    half=n//2;y=np.r_[np.zeros(half),np.ones(half)]
    null=float(roc_auc_score(y,np.r_[bg['PET2'][:half],bg['PET2'][-half:]]))
    all_ids=[set(clouds.family_id[clouds.split==s].tolist()) for s in [0,1,2]]
    result={'schema_version':'hicad-omni-results-1','data_fingerprint':clouds.fingerprint,
        'config':config,'data_config':clouds.config,'checkpoint':str(checkpoint),'checkpoint_epoch':saved['epoch'],'external_validation':config.get('external_validation',False),
        'software':{'torch':torch.__version__,'numpy':np.__version__,'omnilearned':__import__('omnilearned').__version__},
        'representation':saved['representation'],'training_seed':saved['identity']['seed'],'records':rows,
        'validation':{'split_families':[len(s) for s in all_ids],
            'family_overlap':any(all_ids[a]&all_ids[b] for a,b in [(0,1),(0,2),(1,2)]),
            'background_half_vs_half_auc':null,
            'permuted_test_label_auc':float(roc_auc_score(rng(clouds.config['data_seed'],41).permutation(np.r_[np.zeros(n),np.ones(n)]),np.r_[bg['PET2'],np.load(out/f'scores-{j:03d}-{control}.npz')['PET2']])),'null_approx_se':float(np.sqrt((2*half+1)/(12*half**2))),
            'bootstrap':'paired parent-family resampling, same weights across grid/model scores',
            'parameter_holdouts':sorted(clouds.heldout)},
        'limits':['Toy generator, no detector response or conservation.',
            'Ideal input includes true massive energy; proton-count and summary baselines include ideal species identification.',
            'Operating points are diagnostic curves on test data, not independently validated deployment thresholds.',
            'Mixture AUC is the exact population dilution prediction for the fixed score, not another training run.',
            'AUC grid points share test backgrounds and are correlated.']}
    write_json(out/'results.json',result)
    return result
