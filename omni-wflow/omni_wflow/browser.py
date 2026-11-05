"""Offline scientific results explorer and cross-training-seed summary."""
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
from .data import write_json, CONFIGS, SPECIES, clumps


def build(result,path):
    template=(Path(__file__).with_name('browser.html')).read_text()
    cfg=CONFIGS['bes']
    payload={**result,'generation_info':{'ecm_gev':cfg.ecm,'geometry':asdict(cfg),
        'species':SPECIES,'species_probabilities':list(cfg.species_probs),
        'proton_mass_gev':clumps.PROTON_MASS}}
    encoded=json.dumps(payload,separators=(',',':'),allow_nan=False).replace('<',r'\u003c')
    Path(path).write_text(template.replace('__RESULTS__',encoded))


def aggregate(out,seeds):
    results=[json.loads((out/f'seed-{seed}/evaluation/results.json').read_text()) for seed in seeds]
    first=results[0];rows=[]
    for row in first['records']:
        matches=[next(x for x in d['records'] if x['setting_id']==row['setting_id'] and x['control']==row['control'] and x['classifier']==row['classifier']) for d in results]
        rows.append(dict(setting_id=row['setting_id'],setting=row['setting'],control=row['control'],classifier=row['classifier'],
            auc_mean=float(np.mean([x['auc'] for x in matches])),auc_seed_std=float(np.std([x['auc'] for x in matches],ddof=1)) if len(seeds)>1 else None,
            auc_by_seed={str(s):m['auc'] for s,m in zip(seeds,matches)}))
    write_json(out/'seed-summary.json',{'training_seeds':seeds,'records':rows,'note':'Same held-out families across training seeds. Seed spread is not a test-sample confidence interval.'})
    links=''.join(f'<li><a href="seed-{s}/evaluation/index.html">Training seed {s}</a></li>' for s in seeds)
    (out/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>OmniLearned study runs</title><h1>Whole-event OmniLearned study</h1><p>Held-out AUC maps, ROC curves and baselines. Same test families across model seeds.</p><ul>{links}</ul><p><a href="seed-summary.json">Cross-seed AUC summary (JSON)</a></p>')
