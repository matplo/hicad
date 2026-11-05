#!/usr/bin/env python3
"""Generate, train, evaluate and browse the whole-event OmniLearned study."""
import argparse
import json
from pathlib import Path
from omni_wflow.data import generate,Clouds,write_json,fingerprint

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['generate','train','pretrain','evaluate','browser','all'])
    p.add_argument('--config',type=Path,default=ROOT/'configs/run-pilot.json')
    p.add_argument('--output',type=Path,default=ROOT/'runs/pilot')
    p.add_argument('--external-data',type=Path,help='Evaluate a separately generated compatible parent pool')
    p.add_argument('--seed',type=int,help='Run one configured training seed')
    p.add_argument('--init',type=Path,help='Local encoder checkpoint; no automatic download')
    p.add_argument('--freeze-encoder',action='store_true')
    args=p.parse_args();c=json.loads(args.config.read_text());out=args.output;out.mkdir(parents=True,exist_ok=True)
    identity=out/'config.json'
    if identity.exists() and json.loads(identity.read_text())!=c:raise ValueError('Output directory config mismatch')
    write_json(identity,c);dataset=out/'particles.h5'
    if args.command in ['generate','all']:generate(c['data'],dataset)
    if args.command=='generate':return
    clouds=Clouds(args.external_data if args.external_data else dataset)
    seeds=[args.seed] if args.seed is not None else c['training_seeds']
    if any(s not in c['training_seeds'] for s in seeds):raise ValueError('Seed is not in the configured campaign')
    if args.external_data:
        if args.command!='evaluate':raise ValueError('--external-data only supports evaluate')
        for key in ['centrality','background_temperature_gev','acceptance','grid']:
            if clouds.config[key]!=c['data'][key]:raise ValueError('External physics mismatch: '+key)
        if clouds.config['data_seed']==c['data']['data_seed']:raise ValueError('Independent validation requires a different data seed')
    if not args.external_data and clouds.fingerprint!=fingerprint(c['data']):raise ValueError('Dataset fingerprint mismatch')
    if args.command in ['train','pretrain','all']:
        from omni_wflow.train import train
        for seed in seeds:
            train(clouds,c['training'],out/f'seed-{seed}',seed,args.init,args.freeze_encoder,args.command=='pretrain')
    if args.command in ['evaluate','all']:
        from omni_wflow.evaluate import evaluate
        from omni_wflow.browser import build
        for seed in seeds:
            directory=out/f'seed-{seed}'
            evaluation_dir=directory/(('evaluation-external-'+clouds.fingerprint[:10]) if args.external_data else 'evaluation')
            evaluation_config=dict(c['evaluation'])
            if args.external_data:evaluation_config['external_validation']=True
            result=evaluate(clouds,directory/'best.pt',evaluation_dir,evaluation_config)
            build(result,evaluation_dir/'index.html')
            from omni_wflow.plots import export
            export(result,evaluation_dir/'figures')
        from omni_wflow.browser import aggregate

        if not args.external_data and all((out/f'seed-{s}/evaluation/results.json').exists() for s in c['training_seeds']):aggregate(out,c['training_seeds'])
    if args.command=='browser':
        from omni_wflow.browser import build,aggregate
        for seed in seeds:
            directory=out/f'seed-{seed}/evaluation';build(json.loads((directory/'results.json').read_text()),directory/'index.html')
        aggregate(out,c['training_seeds'])
    print(f'Workflow {args.command} completed: {out}',flush=True)

if __name__=='__main__':main()
