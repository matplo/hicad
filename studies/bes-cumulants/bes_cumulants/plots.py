"""Standalone scientific figures in PNG and editable SVG, generated with Matplotlib."""
import os
from pathlib import Path
import numpy as np


def export_figures(result,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    cache=directory/'_mpl_cache';cache.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache.resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.size':10,'figure.dpi':140,'savefig.dpi':180,'svg.fonttype':'none'})
    rows=result['records'];windows=result['config']['rapidity_windows']
    window=min(windows,key=lambda x:abs(x-.5));bg=next(b for b in result['background'] if b['ymax']==window)
    colors={'shared_boost':'#c62f35','event_mixed':'#3d70bb','poisson':'#b47c17'}
    names={'shared_boost':'Shared boost','event_mixed':'Event mixed','poisson':'Poisson yield control'}
    files=[]
    def finish(fig,name):
        fig.text(.01,.005,f"Synthetic Au+Au 9.2 GeV · {result['config']['selected_events']:,} central events · no detector/conservation corrections",fontsize=7)
        fig.tight_layout(rect=(0,.03,1,.96))
        for extension in ['png','svg']:
            path=directory/f'{name}.{extension}';fig.savefig(path);files.append(str(path))
        plt.close(fig)
    def draw(ax,x,m,color,label):
        m=np.array(m,dtype=float)
        ax.plot(x,m[:,0],'--o' if label=='Event mixed' else '-o',ms=5 if label=='Event mixed' else 3,markerfacecolor='none' if label=='Event mixed' else color,color=color,label=label)
        ax.vlines(x,m[:,1],m[:,2],color=color,alpha=.65,lw=1)
    primary=[r for r in rows if r['scan']=='strength' and r['ymax']==window]
    if primary:
        fig,axes=plt.subplots(1,2,figsize=(11,4.5));palette=plt.cm.viridis(np.linspace(.1,.9,len(result['config']['strength']['n'])))
        for n,color in zip(result['config']['strength']['n'],palette):
            values=sorted([r for r in primary if r['model']=='shared_boost' and r['n']==n],key=lambda r:r['f'])
            draw(axes[0],[r['f'] for r in values],[r['metrics']['net']['cbwc']['C4_C2'] for r in values],color,f'n={n}')
        baseline=bg['metrics']['net']['cbwc']['C4_C2'];axes[0].axhline(baseline[0],color='#298a53',ls='--',label='Background')
        axes[0].axhspan(baseline[1],baseline[2],color='#298a53',alpha=.08)
        for model in colors:
            values=sorted([r for r in primary if r['model']==model and r['n']==8],key=lambda r:r['f'])
            if not values:continue
            draw(axes[1],[r['f'] for r in values],[r['metrics']['net']['cbwc']['C4_C2'][4:8] for r in values],colors[model],names[model])
        axes[1].axhline(0,color='gray',ls='--')
        axes[0].set_ylabel('Net-proton C4 / C2');axes[1].set_ylabel('Delta (C4 / C2) from background')
        for ax in axes:ax.set_xlabel('Injection probability f');ax.legend(fontsize=8);ax.grid(alpha=.15)
        fig.suptitle(f'Strength response · CBWC · |y| < {window}, 0.4 < pT < 2 GeV · n=8 in right panel')
        finish(fig,'strength_net_c4_c2')
        fig,ax=plt.subplots(figsize=(7,4.5));values=[r for r in primary if r['n']==8 and r['f']==.1]
        distributions=[('Background',bg['hist']['net'],'#298a53')]+[(names[r['model']],r['hist']['net'],colors[r['model']]) for r in values]
        for label,h,color in distributions:
            low=min(h['x']);high=max(h['x']);map_=dict(zip(h['x'],h['counts']))
            counts=[map_.get(x,0)/result['config']['selected_events'] for x in range(low,high+1)]
            ax.stairs(counts,np.arange(low-.5,high+1.5),color=color,label=label,lw=1.5)
        ax.set_xlabel('Net protons');ax.set_ylabel('Event probability');ax.legend();ax.grid(alpha=.15)
        fig.suptitle('Multiplicity distributions · f=0.1, n=8 · raw selected ensemble')
        finish(fig,'net_proton_distributions')
    shape=[r for r in rows if r['scan']=='shape' and r['direction']=='isotropic_3d' and r['ymax']==window]
    if shape:
        ts=sorted(set(r['T'] for r in shape));bs=sorted(set(r['beta'] for r in shape));matrices=[]
        for model in ['shared_boost','event_mixed']:
            matrices.append(np.array([[next(r for r in shape if r['T']==t and r['beta']==b and r['model']==model)['metrics']['net']['cbwc']['C4_C2'][4] for b in bs] for t in ts]))
        bound=max(1e-6,*[np.max(abs(m)) for m in matrices]);norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound)
        fig,axes=plt.subplots(1,2,figsize=(10,4.5))
        for ax,matrix,title in zip(axes,matrices,['Shared boost','Event mixed']):
            image=ax.imshow(matrix,origin='lower',aspect='auto',cmap='coolwarm',norm=norm)
            ax.set_xticks(range(len(bs)),bs);ax.set_yticks(range(len(ts)),[round(t*1000) for t in ts]);ax.set_xlabel('Beta');ax.set_ylabel('T [MeV]');ax.set_title(title)
            for i in range(len(ts)):
                for j in range(len(bs)):ax.text(j,i,f'{matrix[i,j]:.2g}',ha='center',va='center',fontsize=8)
        color_axis=fig.add_axes([.88,.19,.018,.59])
        fig.colorbar(image,cax=color_axis,label='Delta net-proton C4/C2')
        fig.suptitle('Isotropic 3D boost · CBWC · f=0.1, n=8')
        # Use explicit spacing because a shared colorbar is not tight_layout-compatible.
        fig.subplots_adjust(bottom=.15,top=.82,wspace=.4,right=.82)
        fig.text(.01,.01,'Synthetic 9.2 GeV · colors show estimates, not discovery significance',fontsize=8)
        for ext in ['png','svg']:
            path=directory/f'shape_net_c4_c2.{ext}';fig.savefig(path);files.append(str(path))
        plt.close(fig)
    joint=[r for r in rows if r['scan']=='joint' and r['direction']=='isotropic_3d'
           and r['ymax']==window and r['T']==.1 and r['k']==1 and r['model']=='shared_boost']
    if joint:
        from matplotlib.colors import SymLogNorm
        ns=sorted({r['n'] for r in joint});bs=sorted({r['beta'] for r in joint});fs=sorted({r['f'] for r in joint})
        bound=max(abs(r['metrics']['net']['cbwc']['C4_C2'][4]) for r in joint)
        norm=SymLogNorm(linthresh=1,vmin=-bound,vmax=bound)
        fig,axes=plt.subplots(2,(len(ns)+1)//2,figsize=(12,7),squeeze=False)
        for ax,n in zip(axes.flat,ns):
            matrix=np.array([[next(r for r in joint if r['n']==n and r['beta']==b and r['f']==f)['metrics']['net']['cbwc']['C4_C2'][4] for f in fs] for b in bs])
            image=ax.imshow(matrix,origin='lower',aspect='auto',cmap='coolwarm',norm=norm)
            ax.set_xticks(range(len(fs)),fs);ax.set_yticks(range(len(bs)),bs)
            ax.set_xlabel('Injection fraction f');ax.set_ylabel('Boost beta');ax.set_title(f'n={n} protons / clump')
            for i in range(len(bs)):
                for j in range(len(fs)):ax.text(j,i,f'{matrix[i,j]:.2g}',ha='center',va='center',fontsize=8)
        fig.subplots_adjust(bottom=.12,top=.87,right=.84,hspace=.45,wspace=.2)
        fig.colorbar(image,cax=fig.add_axes([.88,.18,.015,.6]),label='Delta net-proton C4/C2 (symmetric log color)')
        fig.suptitle('Joint scan: f × beta × n · T=100 MeV · isotropic 3D · one clump\nCBWC · |y|<0.5, 0.4<pT<2 GeV')
        fig.text(.01,.02,'Synthetic Au+Au 9.2 GeV · 100,000 paired central events · color encodes estimates, not significance',fontsize=8)
        for ext in ['png','svg']:
            path=directory/f'joint_f_beta_n.{ext}';fig.savefig(path);files.append(str(path))
        plt.close(fig)
    return files
