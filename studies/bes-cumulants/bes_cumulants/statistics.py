"""Finite-sample k-statistics, CBWC and paired independent-event-block bootstrap."""
import numpy as np

ORDINARY = ['C1','C2','C3','C4','C2_C1','C3_C2','C4_C2']
FACTORIAL = ['F2','F3','F4','F2_norm','F3_norm','F4_norm']


def cumulants(moments):
    """Last axis [N,sum x,sum x²,sum x³,sum x⁴]; N may include bootstrap weights."""
    n = moments[...,0]
    mean = np.divide(moments[...,1:],n[...,None],out=np.full_like(moments[...,1:],np.nan),where=n[...,None]>0)
    a,b,c,d = np.moveaxis(mean,-1,0)
    mu2 = b-a*a; mu3 = c-3*a*b+2*a**3; mu4 = d-4*a*c+6*a*a*b-3*a**4
    with np.errstate(divide='ignore',invalid='ignore'):
        result = np.stack([a,n/(n-1)*mu2,n*n/((n-1)*(n-2))*mu3,
            n*n*((n+1)*mu4-3*(n-1)*mu2*mu2)/((n-1)*(n-2)*(n-3))],axis=-1)
    return np.where((n>=4)[...,None],result,np.nan)


def metrics(c, factorial=True):
    a,b,c3,d = np.moveaxis(c,-1,0)
    with np.errstate(divide='ignore',invalid='ignore'):
        result = np.stack([a,b,c3,d,np.where(abs(a)>1e-12,b/a,np.nan),
                          np.where(b>1e-12,c3/b,np.nan),np.where(b>1e-12,d/b,np.nan)],axis=-1)
        if factorial:
            f2=b-a;f3=c3-3*b+2*a;f4=d-6*c3+11*b-6*a
            result=np.concatenate([result,np.stack([f2,f3,f4,
                np.where(a>1e-12,f2/a**2,np.nan),np.where(a>1e-12,f3/a**3,np.nan),
                np.where(a>1e-12,f4/a**4,np.nan)],axis=-1)],axis=-1)
    return result


def grouped_moments(values, group, n_groups):
    values=np.asarray(values,dtype=float)
    return np.stack([np.bincount(group,weights=None if r==0 else values**r,minlength=n_groups)
                     for r in range(5)],axis=-1).astype(float)


class Estimator:
    def __init__(self, centrality, n_blocks=200, n_bootstrap=400, seed=42, fine_bins=5, low=0., high=.05):
        self.n=len(centrality);self.n_blocks=n_blocks;self.fine_bins=fine_bins
        if self.n%n_blocks or self.n//n_blocks<20:
            raise ValueError('N must be divisible by blocks, with at least 20 events per block')
        block=np.arange(self.n)//(self.n//n_blocks)
        fine=np.minimum(fine_bins-1,((centrality-low)/(high-low)*fine_bins).astype(int))
        self.group=block*fine_bins+fine
        self.weights=np.random.Generator(np.random.PCG64(seed)).multinomial(n_blocks,np.full(n_blocks,1/n_blocks),size=n_bootstrap).astype(float)
        self.cache_background={}

    def evaluate(self, values, factorial=True):
        m=grouped_moments(values,self.group,self.n_blocks*self.fine_bins).reshape(self.n_blocks,self.fine_bins,5)
        all_m=m.sum(axis=0)
        boot=(self.weights@m.reshape(self.n_blocks,-1)).reshape(-1,self.fine_bins,5)
        result={}
        for mode in ['raw','cbwc']:
            if mode=='raw':
                point=cumulants(all_m.sum(axis=0));resamples=cumulants(boot.sum(axis=1))
            else:
                point=np.sum(cumulants(all_m)*all_m[:,0,None],axis=0)/all_m[:,0].sum()
                resamples=np.sum(cumulants(boot)*boot[:,:,0,None],axis=1)/boot[:,:,0].sum(axis=1)[:,None]
            result[mode]=(metrics(point,factorial),metrics(resamples,factorial))
        return result

    def summarize(self, values, baseline, factorial=True):
        evaluated=self.evaluate(values,factorial)
        names=ORDINARY+(FACTORIAL if factorial else [])
        output={}
        for mode,(point,boot) in evaluated.items():
            base_point,base_boot=baseline[mode]
            delta=point-base_point;delta_boot=boot-base_boot
            with np.errstate(invalid='ignore'):
                result=np.stack([point,np.nanpercentile(boot,2.5,axis=0),np.nanpercentile(boot,97.5,axis=0),
                    np.nanstd(boot,axis=0,ddof=1),delta,np.nanpercentile(delta_boot,2.5,axis=0),
                    np.nanpercentile(delta_boot,97.5,axis=0),np.nanstd(delta_boot,axis=0,ddof=1)],axis=-1)
            output[mode]={name:[None if not np.isfinite(x) else float(round(x,10)) for x in row] for name,row in zip(names,result)}
        return output
