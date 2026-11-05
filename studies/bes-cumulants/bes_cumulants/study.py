"""Checkpointed common-background BES pilot scan; no HTTP or GPU dependency."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from . import VERSION
from . import physics as p
from .statistics import Estimator


def write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')
    tmp.replace(path)


def validate(c):
    if c['schema_version']!='hicad-bes-cumulants-1':raise ValueError('Unsupported schema')
    p.clumps.seed(c['root_seed'])
    n=c['selected_events'];blocks=c['bootstrap_blocks']
    if not isinstance(n,int) or not 1000<=n<=10000000:raise ValueError('Selected events outside [1000,10000000]')
    if not 10<=blocks<=1000 or n%blocks:raise ValueError('Invalid bootstrap blocks')
    if not 50<=c['bootstrap_replicates']<=2000:raise ValueError('Invalid bootstrap replicate count')
    if not 0<=c['centrality'][0]<c['centrality'][1]<=1:raise ValueError('Invalid centrality')
    if not 1<=c['centrality_fine_bins']<=20:raise ValueError('Invalid fine centrality bins')
    windows=c['rapidity_windows']
    if windows!=sorted(set(windows)) or any(not 0<x<=3 for x in windows):raise ValueError('Invalid windows')
    for task in tasks(c):
        if not .001<=task['T']<=.5 or not 0<=task['beta']<=.95:raise ValueError('Invalid source T/beta')
        if task['direction'] not in ['transverse','isotropic_3d']:raise ValueError('Invalid direction')
        if any(not isinstance(n,int) or not 1<=n<=200 for n in task['n']):raise ValueError('Invalid n')
        if any(not 0<=f<=1 for f in task['f']) or task['f']!=sorted(set(task['f'])):raise ValueError('Invalid f')
        if not 1<=task['k']<=8:raise ValueError('Invalid clumps')
    return c


def tasks(c):
    s=c['strength']
    result=[dict(scan='strength',T=s['temperature_gev'],beta=s['beta'],direction=s['direction'],k=s['clumps'],n=s['n'],f=s['f'])]
    s=c.get('shape')
    if s:
        result += [dict(scan='shape',T=t,beta=b,direction=d,k=s['clumps'],n=[s['n']],f=s['f'])
                   for t in s['temperature_gev'] for b in s['beta'] for d in s['direction']]
    s=c.get('clump_scan')
    if s:
        result += [dict(scan='clumps',T=s['temperature_gev'],beta=s['beta'],direction=d,k=k,n=[s['n']],f=s['f'])
                   for d in s['direction'] for k in s['clumps']]
    s=c.get('joint')
    if s:
        result += [dict(scan='joint',T=t,beta=b,direction=d,k=k,n=s['n'],f=s['f'])
                   for t in s['temperature_gev'] for b in s['beta']
                   for d in s['direction'] for k in s['clumps']]
    return result


def histogram(values):
    v,counts=np.unique(values,return_counts=True)
    return {'x':v.astype(int).tolist(),'counts':counts.tolist()}


def analyze(config,out_dir,log=print):
    started=time.perf_counter();c=validate(config);out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    cache=out/'checkpoints';cache.mkdir(exist_ok=True)
    identity={'config':c,'version':VERSION,'numpy':np.__version__,'scipy':scipy.__version__}
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    manifest=out/'run_identity.json'
    if manifest.exists() and json.loads(manifest.read_text())['sha256']!=fingerprint:
        raise ValueError('Output directory belongs to a different configuration/version; choose a new directory')
    write_json(manifest,{'sha256':fingerprint,**identity})
    n=c['selected_events'];windows=c['rapidity_windows'];seed=c['root_seed'];pt=c['pt_gev']
    estimator=None
    background_path=cache/'background.npz'
    if background_path.exists():
        background=dict(np.load(background_path))
    else:
        log(f'Generating {n:,} independent selected background events',flush=True)
        geometry=p.selected_geometry(n,seed,c['centrality'])
        a=np.array([p.acceptance(c['background_temperature_gev'],0,0,w,*pt,96)[0] for w in windows])
        protons,antiprotons=p.background_counts(geometry['emitted'],a,seed)
        background={**geometry,'protons':protons,'antiprotons':antiprotons,'acceptance':a}
        np.savez_compressed(background_path,**background)
    estimator=Estimator(background['centrality'],c['bootstrap_blocks'],c['bootstrap_replicates'],
                        int(p.rng(seed,30).integers(2**32)),c['centrality_fine_bins'],*c['centrality'])
    npbg=background['protons'];nbar=background['antiprotons'];baseline=[];baseline_eval=[]
    for w in range(len(windows)):
        evaluated={obs:estimator.evaluate(values,factorial=obs!='net') for obs,values in
                   [('proton',npbg[:,w]),('antiproton',nbar[:,w]),('net',npbg[:,w]-nbar[:,w])]}
        baseline_eval.append(evaluated)
        met={obs:estimator.summarize(values,evaluated[obs],obs!='net') for obs,values in
             [('proton',npbg[:,w]),('antiproton',nbar[:,w]),('net',npbg[:,w]-nbar[:,w])]}
        baseline.append({'ymax':windows[w],'metrics':met,'hist':{obs:histogram(values) for obs,values in
                        [('proton',npbg[:,w]),('antiproton',nbar[:,w]),('net',npbg[:,w]-nbar[:,w])]}})
    gates=p.rng(seed,20).random(n)
    table_cache={};records=[];integration=[]
    all_tasks=tasks(c)
    for task_index,task in enumerate(all_tasks):
        checkpoint=cache/f'task-{task_index:03d}.json'
        if checkpoint.exists():
            saved=json.loads(checkpoint.read_text());records.extend(saved['records']);integration.append(saved['integration'])
            log(f"[{task_index+1}/{len(all_tasks)}] cached {task['scan']} T={task['T']} beta={task['beta']} {task['direction']}",flush=True)
            continue
        log(f"[{task_index+1}/{len(all_tasks)}] {task['scan']} T={task['T']} beta={task['beta']} {task['direction']} k={task['k']}",flush=True)
        key=(task['T'],task['beta'])
        if key not in table_cache:
            table_cache[key]=p.acceptance_table(*key,windows,*pt,c['integration_tolerance'])
        table=table_cache[key]
        integration_info={**task,'quadrature_error':table['quadrature_error'],'interpolation_error':table['interpolation_error'],'direction_nodes':len(table['grid'])}
        integration.append(integration_info)
        source=p.source_counts(n,task['n'],task['k'],table,task['direction'],seed)
        np.savez_compressed(cache/f'source-{task_index:03d}.npz',**source)
        rows=[]
        mean_a=[table['values'][w,0] if task['direction']=='transverse' else float(np.trapezoid(table['values'][w],table['grid'])) for w in range(len(windows))]
        for ni,emitted in enumerate(task['n']):
            poisson=np.zeros((n,len(windows)),dtype=np.uint16)
            previous_f=0.
            for fi,f in enumerate(task['f']):
                gate=gates<f
                delta_f=f-previous_f;previous_f=f
                shells=np.diff(np.r_[0.,mean_a])*task['k']*emitted*delta_f
                additions=p.rng(seed,21,task_index,emitted,fi).poisson(shells,size=(n,len(windows)))
                poisson += np.cumsum(additions,axis=1).astype(np.uint16)
                for model in c['models']:
                    selected=poisson if model=='poisson' else source[model][:,:,ni]*gate[:,None]
                    for wi,w in enumerate(windows):
                        s=selected[:,wi];prediction=p.source_prediction(emitted,task['k'],f,table,task['direction'],model,wi)
                        values={'proton':npbg[:,wi]+s,'net':npbg[:,wi].astype(np.int32)+s-nbar[:,wi]}
                        metrics={obs:estimator.summarize(v,baseline_eval[wi][obs],obs!='net') for obs,v in values.items()}
                        rows.append({'id':len(records)+len(rows),'scan':task['scan'],'T':task['T'],'beta':task['beta'],
                            'direction':task['direction'],'k':task['k'],'n':emitted,'f':f,'model':model,'ymax':w,
                            'injected_events':int(gate.sum()) if model!='poisson' else None,
                            'source_mean':float(s.mean()),'source_mean_expected':float(prediction[0]),
                            'source_cumulants_expected':prediction.tolist(),'metrics':metrics,
                            'hist':{obs:histogram(v) for obs,v in values.items()},
                            'warnings':(['Fewer than 500 signal-bearing events'] if model!='poisson' and 0<int(gate.sum())<500 else [])})
        write_json(checkpoint,{'records':rows,'integration':integration_info});records.extend(rows)
    result={'schema_version':'hicad-bes-cumulant-results-1','config':c,'version':VERSION,'fingerprint':fingerprint,
            'provenance':{'numpy':np.__version__,'scipy':scipy.__version__,'python':platform.python_version(),
                'physics_core':p.clumps.CORE_VERSION,'thermal_law':p.clumps.THERMAL_LAW,'rng_policy':p.RNG_POLICY,
                'geometry':asdict(p.CONFIGS['bes']),'method':'Lorentz-transformed thermal acceptance quadrature + conditional count sampling',
                'background':'new independent selected events; no repeated input HDF5 pool',
                'bootstrap':'paired nonparametric bootstrap of independent event blocks; same block weights across scan points',
                'metric_columns':['value','low95','high95','se','delta_background','delta_low95','delta_high95','delta_se'],
                'source_gating':'same event uniforms for shared and mixed; Poisson yield control is ungated',
                'azimuth_scrambling':'identical count cumulants under full-azimuth analysis cuts',
                'limits':'Toy geometry and independent emission; no exact conservation or detector response; no QCD-critical-point inference'},
            'background':baseline,'records':records,'integration':integration,
            'timing_seconds':time.perf_counter()-started}
    write_json(out/'results.json',result)
    return result
