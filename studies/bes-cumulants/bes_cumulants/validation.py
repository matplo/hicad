"""Numerical acceptance, analytic closure and event-vs-block bootstrap checks."""
from pathlib import Path
import numpy as np
from . import physics as p
from .statistics import cumulants,metrics,Estimator


def validate_results(result,directory):
    directory=Path(directory);c=result['config'];checks={}
    sampler=[]
    realization=dict(root_seed='100193',clump_revision=0,direction_revision=0,control_revision=0)
    for t in [.06,.1,.18]:
        rest=p.clumps.sample_rest_protons(40000,t,p.clumps.StreamFactory(realization))
        for beta,u in [(0.,0.),(.8,0.),(.8,.8)]:
            lab=p.clumps.boost_four_vectors(rest,[beta*np.sqrt(1-u*u),0.,beta*u]);kin=p.clumps.kinematics(lab)
            measured=float(np.mean((abs(kin['y'])<.5)&(kin['pt']>.4)&(kin['pt']<2)))
            expected=float(p.acceptance(t,beta,u,.5,order=144)[0]);se=np.sqrt(expected*(1-expected)/len(rest))
            sampler.append(dict(T=t,beta=beta,cos_theta=u,n=len(rest),quadrature=expected,sampled=measured,z=(measured-expected)/se))
    checks['four_vector_acceptance']=sampler
    if max(abs(x['z']) for x in sampler)>5:raise ArithmeticError('Acceptance failed independent p4 validation')
    z=[];worst=[]
    for row in result['records']:
        for obs in ['proton','net']:
            for order in range(4):
                m=row['metrics'][obs]['raw'][f'C{order+1}'];expected=row['source_cumulants_expected'][order]
                if m[7] is not None and m[7]>1e-9:
                    score=(m[4]-expected)/m[7];z.append(score)
                    if abs(score)>3:worst.append(dict(id=row['id'],obs=obs,cumulant=order+1,z=score))
    checks['analytic_cumulant_addition']={'comparisons':len(z),'fraction_within_3se':float(np.mean(abs(np.array(z))<=3)),
        'rms_z':float(np.sqrt(np.mean(np.array(z)**2))),'max_abs_z':float(np.max(abs(np.array(z)))),'above_3se':worst,
        'interpretation':'Correlated scan checks, not independent significance tests; Monte Carlo fluctuations are expected.'}
    transverse={}
    for row in result['records']:
        if row['direction']=='transverse' and row['model'] in ['shared_boost','event_mixed']:
            key=tuple(row[k] for k in ['scan','T','beta','k','n','f','ymax'])
            transverse.setdefault(key,{})[row['model']]=row['metrics']
    checks['transverse_mixing_null_exact']=all(x.get('shared_boost')==x.get('event_mixed') for x in transverse.values())
    if not checks['transverse_mixing_null_exact']:raise ArithmeticError('Transverse count-control null failed')
    # Direct event bootstrap resamples a joint histogram: mathematically the same
    # as resampling paired individual events, without materializing N indices/B.
    archive=np.load(directory/'checkpoints/background.npz');source=np.load(directory/'checkpoints/source-000.npz')
    wi=min(range(len(c['rapidity_windows'])),key=lambda i:abs(c['rapidity_windows'][i]-.5))
    ni=min(range(len(c['strength']['n'])),key=lambda i:abs(c['strength']['n'][i]-8))
    f=min([f for f in c['strength']['f'] if f>0] or [0],key=lambda x:abs(x-.1))
    bg=archive['protons'][:,wi]-archive['antiprotons'][:,wi]
    signal=source['shared_boost'][:,wi,ni]*(p.rng(c['root_seed'],20).random(len(bg))<f)
    combined=bg+signal
    fine=np.minimum(c['centrality_fine_bins']-1,((archive['centrality']-c['centrality'][0])/(c['centrality'][1]-c['centrality'][0])*c['centrality_fine_bins']).astype(int))
    cells,freq=np.unique(np.column_stack([fine,bg,combined]),axis=0,return_counts=True)
    weights=p.rng(c['root_seed'],40).multinomial(len(bg),freq/len(bg),size=c['bootstrap_replicates']).astype(float)
    estimates=[]
    for column in [1,2]:
        feature=np.stack([np.column_stack([(cells[:,0]==j).astype(float)*cells[:,column].astype(float)**r for r in range(5)])
                          for j in range(c['centrality_fine_bins'])],axis=1)
        boot=(weights@feature.reshape(len(cells),-1)).reshape(-1,c['centrality_fine_bins'],5)
        raw=metrics(cumulants(boot.sum(axis=1)),False)
        cb=metrics(np.sum(cumulants(boot)*boot[:,:,0,None],axis=1)/boot[:,:,0].sum(axis=1)[:,None],False)
        estimates.append(dict(raw=raw,cbwc=cb))
    estimator=Estimator(archive['centrality'],c['bootstrap_blocks'],c['bootstrap_replicates'],int(p.rng(c['root_seed'],30).integers(2**32)),c['centrality_fine_bins'],*c['centrality'])
    base=estimator.evaluate(bg,False);values=estimator.evaluate(combined,False)
    comparison={}
    for mode in ['raw','cbwc']:
        direct_bg=np.std(estimates[0][mode][:,6],ddof=1)
        direct_delta=np.std(estimates[1][mode][:,6]-estimates[0][mode][:,6],ddof=1)
        block_bg=np.std(base[mode][1][:,6],ddof=1)
        block_delta=np.std(values[mode][1][:,6]-base[mode][1][:,6],ddof=1)
        comparison[mode]=dict(background_ratio_se_event=float(direct_bg),background_ratio_se_block=float(block_bg),
            delta_ratio_se_event=float(direct_delta),delta_ratio_se_block=float(block_delta),
            block_to_event_background=float(block_bg/direct_bg),block_to_event_delta=float(block_delta/direct_delta) if direct_delta else None)
    checks['paired_event_bootstrap_comparison']={'n':len(bg),'signal_n':c['strength']['n'][ni],'f':f,'ymax':c['rapidity_windows'][wi],'C4_C2':comparison}
    convergence=[]
    for n in [10000,50000,100000]:
        if n>len(bg):continue
        blocks=min(c['bootstrap_blocks'],n//50)
        while n%blocks:blocks-=1
        e=Estimator(archive['centrality'][:n],blocks,c['bootstrap_replicates'],int(p.rng(c['root_seed'],30).integers(2**32)),c['centrality_fine_bins'],*c['centrality'])
        b=e.evaluate(bg[:n],False)
        summary=e.summarize(combined[:n],b,False)
        convergence.append({'n':n,'baseline_C4_C2':float(b['cbwc'][0][6]),'signal_C4_C2':summary['cbwc']['C4_C2'][:4]})
    checks['prefix_convergence']=convergence
    return checks
