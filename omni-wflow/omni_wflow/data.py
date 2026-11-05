"""Physical particle clouds, split by background ancestry before composition."""
import hashlib
import itertools
import json
from pathlib import Path
import sys
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'toys/boltzmann'))
import clumps
from canonical_background import generate_background_event, SPECIES, CHARGE
from generate_thermal_h5 import CONFIGS, sample_glauber_and_multiplicity


def rng(seed, role, *keys):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed),0x4f4d4e49,1,role,*map(int,keys)])))


def fingerprint(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,allow_nan=False,indent=2)+'\n');tmp.replace(path)


def grid(config):
    g=config['grid']
    return [dict(T=t,beta=b,n=n,k=k,direction=d) for t,b,n,k,d in itertools.product(
        g['temperature_gev'],g['beta'],g['protons_per_clump'],g['clumps_per_event'],g['direction'])]


def generate(config, path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    digest=fingerprint(config)
    if path.exists():
        with h5py.File(path) as f:
            if f.attrs['fingerprint']!=digest:raise ValueError('Dataset config mismatch; choose a different output path')
        return
    n=config['parents'];seed=config['data_seed'];cfg=CONFIGS['bes'];low,high=config['centrality']
    ids=[];geometry=[]
    chunk=0
    while len(ids)<n:
        c,b,np_,nc,m=sample_glauber_and_multiplicity(cfg,65536,rng(seed,1,chunk))
        for j in np.flatnonzero((c>=low)&(c<high)):
            ids.append(chunk*65536+int(j));geometry.append([c[j],b[j],np_[j],nc[j],m[j]])
            if len(ids)==n:break
        chunk+=1
    order=rng(seed,2).permutation(n);split=np.empty(n,dtype=np.int8)
    a=int(n*.7);b=int(n*.85);split[order[:a]]=0;split[order[a:b]]=1;split[order[b:]]=2
    rows=[];offsets=[0];species=[]
    for i,g in enumerate(geometry):
        sample=generate_background_event(int(g[4]),config['background_temperature_gev'],cfg.species_probs,
            root_seed=str(int(rng(seed,3,i).integers(1,2**63))),event_index=i,acceptance=config['acceptance'])
        rows.append(sample.accepted_p4.astype('f4'));species.append(sample.species_index[sample.accepted].astype('i1'))
        offsets.append(offsets[-1]+len(rows[-1]))
        if (i+1)%1000==0:print(f'Background particles: {i+1}/{n}',flush=True)
    tmp=path.with_suffix('.h5.tmp')
    with h5py.File(tmp,'w') as f:
        f.attrs['fingerprint']=digest;f.attrs['config_json']=json.dumps(config)
        f.attrs['columns']='E,px,py,pz';f.attrs['species_json']=json.dumps(SPECIES)
        for name,value in [('p4',np.concatenate(rows)),('species',np.concatenate(species)),('offsets',offsets),
                           ('family_id',ids),('geometry',geometry),('split',split)]:
            f.create_dataset(name,data=value,compression='lzf')
    tmp.replace(path)


class Clouds:
    def __init__(self,path):
        with h5py.File(path) as f:
            self.p4=f['p4'][:];self.species=f['species'][:];self.offsets=f['offsets'][:]
            self.family_id=f['family_id'][:];self.geometry=f['geometry'][:];self.split=f['split'][:]
            self.config=json.loads(f.attrs['config_json']);self.fingerprint=f.attrs['fingerprint']
        self.settings=grid(self.config)
        self.heldout=set(self.config.get('heldout_settings',[]))
        if not self.heldout<set(range(len(self.settings))):raise ValueError('Invalid held-out setting IDs')

    def background(self,i):
        a,b=self.offsets[i:i+2];return self.p4[a:b].copy(),self.species[a:b].copy()

    def source(self,i,setting,model='shared_boost'):
        s=self.settings[setting];seed=int(rng(self.config['data_seed'],4,int(self.family_id[i])).integers(1,2**63))
        streams=clumps.StreamFactory(dict(root_seed=str(seed),clump_revision=0,direction_revision=0,control_revision=0))
        pieces=[]
        # A Poisson number of emitted independent donors gives an accepted Poisson
        # excess with the same one-particle law and mean as the fixed k*n source.
        ks=s['k'];ns=[s['n']]*ks
        if model=='mean_matched_poisson':ns=[int(rng(seed,15).poisson(s['k']*s['n']))];ks=1
        for j in range(ks):
            rest=clumps.sample_rest_protons(ns[j],s['T'],streams,j)
            if model in ['event_mixed','mean_matched_poisson']:
                lab=np.array([clumps.boost_four_vectors(p[None,:],clumps.sample_boost_vector(s['beta'],s['direction'],rng(seed,16,j,l)))[0] for l,p in enumerate(rest)]).reshape(-1,4)
            else:
                lab=clumps.boost_four_vectors(rest,clumps.sample_boost_vector(s['beta'],s['direction'],streams.rng(6,j)))
            lab=lab[clumps.apply_acceptance(lab,self.config['acceptance'])]
            if model=='azimuth_decohered':
                phi=rng(seed,17,j).uniform(-np.pi,np.pi,len(lab));pt=np.hypot(lab[:,1],lab[:,2]);lab[:,1]=pt*np.cos(phi);lab[:,2]=pt*np.sin(phi)
            pieces.append(lab)
        return np.concatenate(pieces).astype('f4')

    def event(self,i,setting=None,model='shared_boost'):
        p4,species=self.background(i);n_source=0
        if setting is not None:
            source=self.source(i,setting,model);n_source=len(source)
            p4=np.concatenate([p4,source]);species=np.r_[species,np.full(len(source),4,dtype='i1')]
        return p4,species,n_source


def features(p4,species,representation='ideal',angle=0.):
    k=clumps.kinematics(p4);pt=np.maximum(k['pt'],1e-10)
    phi=np.arctan2(np.sin(k['phi']+angle),np.cos(k['phi']+angle))
    energy=p4[:,0] if representation=='ideal' else pt*np.cosh(k['eta'])
    x=np.column_stack([k['eta'],phi,np.log(pt),np.log(np.maximum(energy,1e-10))]).astype('f4')
    # No centering/scaling is fit: retain native physical log coordinates. Explicit
    # lengths define validity in our adapter; the native slot-2 sentinel is unused.
    if representation=='pid':x=np.column_stack([x,species/5.]).astype('f4')
    if representation=='track':x=np.column_stack([x,CHARGE[species]]).astype('f4')
    if not np.isfinite(x).all():raise ValueError('Non-finite particle features')
    return x


def summaries(p4,species):
    k=clumps.kinematics(p4);pt=k['pt'];phi=k['phi']
    n=len(pt);np_=int(np.count_nonzero(species==4));na=int(np.count_nonzero(species==5))
    # Rotationally invariant angular moments; no source IDs or geometry.
    return np.array([n,np_,na,np_-na,pt.sum(),pt.mean() if n else 0,
        *[abs(np.exp(1j*m*phi).mean()) if n else 0 for m in [1,2,3]]],dtype='f4')
