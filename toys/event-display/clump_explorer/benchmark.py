"""Reproducible local generation/serialization timings, no server required."""
from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
from .api import Service


def benchmark(provider, samples=30):
    default=json.loads((Path(__file__).parent/'examples/source_only.json').read_text())
    cases={"source_only":default}
    for entry in provider.catalog():
        name=entry['label'].lower()
        if 'bes' not in name and 'lhc' not in name:
            continue
        selected=provider.select(dict(dataset_id=entry['dataset_id'],mode='random',centrality_min=0,
                                      centrality_max=.05 if 'lhc' in name else 1,selection_seed='42'))
        r=deepcopy(default);r['background']={k:selected[k] for k in ['provider','dataset_id','event_index','fingerprint']}
        case_name='central_lhc' if 'lhc' in name else 'bes'
        if entry.get('physics',{}).get('thermal_law')=='massive_canonical_boltzmann_3d_v1':
            case_name+='_canonical'
        if case_name in cases:
            case_name+='_'+entry['dataset_id'][-8:]
        cases[case_name]=r
    results={"samples":samples,"numpy_version":np.__version__,"clock":"perf_counter", "cases":{}}
    for name,r in cases.items():
        service=Service(provider);times=[];sizes=[]
        for i in range(samples+2):
            r['realization']['clump_revision']=i
            start=time.perf_counter();response=service.scenario(r)
            serialized=json.dumps(response,allow_nan=False,separators=(',',':')).encode()
            elapsed=(time.perf_counter()-start)*1000
            if i>=2:times.append(elapsed);sizes.append(len(serialized))
        results['cases'][name]={"p50_ms":float(np.percentile(times,50)),"p95_ms":float(np.percentile(times,95)),
                "background_particles":len(response['background']['measured']['p4']),"json_bytes_p50":float(np.median(sizes)),
                "background_physics":response['background']['metadata'].get('physics'),
                "includes":"uncached generation, response preparation, cache accounting and JSON serialization; exact diagnostics deferred"}
    return results
