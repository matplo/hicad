#!/usr/bin/env python3
"""Score ragged p4 clouds with a trained event classifier (no truth required).

Input HDF5: p4 [total,4], offsets [events+1], species [total] for PID/track models.
For ideal-kinematics models species is optional and never used as a model input.
"""
import argparse
from pathlib import Path
import h5py
import numpy as np
import torch
from omni_wflow.model import EventPET,device
from omni_wflow.evaluate import predict


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='auto');p.add_argument('--batch',type=int,default=64);p.add_argument('--rotations',type=int,default=4)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a new output path')
    torch.set_num_threads(4);dev=device(a.device);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if saved['identity']['objective']!='classifier':raise ValueError('Fine-tune a classifier before inference')
    model=EventPET(saved['spec']).to(dev);model.load_state_dict(saved['state'])
    with h5py.File(a.input) as f:
        p4=f['p4'][:];offsets=f['offsets'][:]
        if saved['representation']!='ideal' and 'species' not in f:raise ValueError('This representation requires species/charge information')
        species=f['species'][:] if 'species' in f else np.zeros(len(p4),dtype='i1')
    if offsets[0]!=0 or offsets[-1]!=len(p4) or np.any(np.diff(offsets)<0):raise ValueError('Invalid ragged offsets')
    events=[(p4[s:e],species[s:e]) for s,e in zip(offsets[:-1],offsets[1:])]
    score=predict(model,events,saved['representation'],dev,a.batch,a.rotations)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,signal_score=score,event_index=np.arange(len(events)),checkpoint=str(a.checkpoint),rotations=a.rotations)
    print(f'Scored {len(events)} events: {a.output}')

if __name__=='__main__':main()
