import {$} from './controls.js';
import {fitCanvas,clumpProtonColor,speciesColors,variantsForComparison,variantLabels} from './renderer.js';
export class Diagnostics {
  constructor(post) {this.post=post;this.sequence=0;this.lastKey='';this.results=[];this.observer=new ResizeObserver(()=>this.paint());this.observer.observe($('histograms'));}
  update(scene,state) {
    if(!scene)return;
    const variants=variantsForComparison(scene,state.compare);
    const mode=scene.resolved_request.observation_mode;
    const configs=variants.map(variant=>({scenario_id:scene.scenario_id,observation_mode:mode,variant,composition:state.composition,binning_id:'proton_clump_diagnostics_v1'}));
    const key=JSON.stringify(configs);
    if(key===this.lastKey)return;
    this.lastKey=key;const sequence=++this.sequence;this.results=[];$('histograms').replaceChildren();
    if(!scene.resolved_request.diagnostics.enabled){$('diagnosticStatus').textContent='Disabled in recipe';return;}
    $('diagnosticStatus').textContent='Computing exact pairs…';
    const run=async()=>{
      try {
        const out=[];
        for(const config of configs) {
          if(sequence!==this.sequence)return;
          const r=await this.post('/api/v1/diagnostics',config);
          if(JSON.stringify(Object.fromEntries(Object.keys(config).map(k=>[k,r[k]])))!==JSON.stringify(config))return;
          out.push(r);
        }
        if(sequence!==this.sequence)return;
        this.results=out;this.paint();
        $('diagnosticStatus').textContent=out.map(r=>`${variantLabels[r.variant]||r.variant}: ${r.diagnostics.message}`).join(' · ');
      }catch(e){if(sequence!==this.sequence)return;if(e.status===503){$('diagnosticStatus').textContent='Pair worker busy · retrying';setTimeout(run,500);}else $('diagnosticStatus').textContent=`${e.message} · event remains available`;}
    };
    run();
  }
  paint() {
    if(!this.results.length)return;
    $('histograms').replaceChildren();
    for(const r of this.results) for(const [name,h] of Object.entries(r.diagnostics.histograms)) {
      const div=document.createElement('div');div.className='histogram';
      const title=document.createElement('strong');title.textContent=`${variantLabels[r.variant]||r.variant} · ${name} [${h.units}]`;title.style.fontSize='11px';
      const canvas=document.createElement('canvas'),note=document.createElement('p');
      const truth=r.diagnostics.truth_daughter_pairs.available===false?null:r.diagnostics.truth_daughter_pairs.histograms[name];
      note.textContent=`${h.total_entries} entries · bin width ${(h.edges[1]-h.edges[0]).toPrecision(3)} ${h.units} · under ${h.underflow}, over ${h.overflow}${truth?` · truth pairs ${truth.total_entries}`:''}`;
      const legend=document.createElement('p'),observedLabel=document.createElement('span');
      observedLabel.style.color=speciesColors.p;observedLabel.textContent=['pt','y'].includes(name)?'— All observed protons':'— All observed proton pairs';legend.append(observedLabel);
      if(truth){const sourceLabel=document.createElement('span');sourceLabel.style.color=clumpProtonColor;sourceLabel.textContent=' · ╌ Same-clump pairs (truth)';legend.append(sourceLabel);}
      if(r.variant==='event_mixed')note.textContent+=' · independent donors; no same-source pairs';
      legend.title='Green solid and red dashed histogram outlines, without fill. Each histogram is normalized to its own total entries, including overflow.';
      div.append(title,legend,canvas,note);$('histograms').append(div);
      const {ctx,w,h:height}=fitCanvas(canvas),pad=20;
      const maximum=Math.max(1e-12,...h.density,...(truth?.density||[]));
      ctx.strokeStyle='#294253';ctx.beginPath();ctx.moveTo(pad,10);ctx.lineTo(pad,height-pad);ctx.lineTo(w-5,height-pad);ctx.stroke();
      for(const [histogram,color,isTruth] of [[h,speciesColors.p,false],[truth,clumpProtonColor,true]]) {
        if(!histogram)continue;
        const edges=histogram.edges,span=edges.at(-1)-edges[0],baseline=height-pad;
        ctx.save();ctx.strokeStyle=color;ctx.lineWidth=isTruth?1.4:1;
        ctx.setLineDash(isTruth?[4,2]:[]);
        ctx.beginPath();ctx.moveTo(pad,baseline);
        histogram.density.forEach((value,i)=>{
          const left=pad+(edges[i]-edges[0])/span*(w-pad-5);
          const right=pad+(edges[i+1]-edges[0])/span*(w-pad-5);
          const barHeight=value/maximum*(height-pad-12),top=baseline-barHeight;
          ctx.lineTo(left,top);ctx.lineTo(right,top);
        });
        ctx.lineTo(w-5,baseline);ctx.stroke();
        ctx.restore();
      }
      ctx.fillStyle='#96adbd';ctx.font='9px system-ui';ctx.fillText(h.edges[0].toFixed(1),pad,height-5);ctx.fillText(h.edges.at(-1).toFixed(1),w-24,height-5);
      if(!h.total_entries)ctx.fillText(name==='pt'||name==='y'?'No positive protons':'No proton pairs',pad+10,height/2);
    }
  }
}
