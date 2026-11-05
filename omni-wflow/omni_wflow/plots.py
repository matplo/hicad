"""Standalone Matplotlib figures for the AUC pilot (not browser screenshots)."""
import os
from pathlib import Path
import numpy as np


def export(result,directory):
    directory=Path(directory);directory.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str((directory/'mpl-cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[r for r in result['records'] if r['classifier']=='PET2' and r['control']=='shared_boost' and r['setting']['k']==1 and r['setting']['direction']=='isotropic_3d']
    ns=sorted({r['setting']['n'] for r in rows});ts=sorted({r['setting']['T'] for r in rows});bs=sorted({r['setting']['beta'] for r in rows})
    fig,axes=plt.subplots(2,(len(ns)+1)//2,figsize=(10,7),squeeze=False,layout='constrained')
    for ax,n in zip(axes.flat,ns):
        matrix=np.array([[next(r['auc'] for r in rows if r['setting']['n']==n and r['setting']['T']==t and r['setting']['beta']==b) for b in bs] for t in ts])
        im=ax.imshow(matrix,origin='lower',vmin=.5,vmax=1,aspect='auto',cmap='viridis')
        ax.set_xticks(range(len(bs)),bs);ax.set_yticks(range(len(ts)),[round(t*1000) for t in ts]);ax.set_xlabel('Boost beta');ax.set_ylabel('T [MeV]');ax.set_title(f'n={n} protons / clump')
        for i in range(len(ts)):
            for j in range(len(bs)):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='white' if matrix[i,j]<.8 else 'black')
    for ax in list(axes.flat)[len(ns):]:ax.set_visible(False)
    fig.colorbar(im,ax=list(axes.flat),label='ROC AUC (injected events, f=1)',shrink=.8)
    fig.suptitle(f"PET2 whole-event classifier · seed {result['training_seed']}\nBES 9.2 GeV toy · |eta|<1 · isotropic boost · k=1 · ideal kinematics")
    for ext in ['png','svg']:fig.savefig(directory/f'auc_maps.{ext}',dpi=160)
    plt.close(fig)
