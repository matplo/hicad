import {$,setupControls,syncControls,editControl} from './controls.js';
import {LatestRequestQueue} from './request_state.js';
import {Renderer,particlesFor,variantsForComparison,variantLabels} from './renderer.js';
import {Diagnostics} from './diagnostics.js';
import {recipeForScenario,saveJSON} from './downloads.js';

let draftSpec=null,committedScenario=null,initialSpec=null,selectionSequence=0;
const viewState={composition:'source_only',compare:'both',color:'truth',frame:'lab',axis:2,pzScale:1,ghost:false};
const renderer=new Renderer($('views'));
async function post(path,body) {
  const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await r.json();
  if(!r.ok){const e=new Error(`${data.error?.field||'request'}: ${data.error?.message||r.statusText}`);e.status=r.status;e.field=data.error?.field;throw e;}
  return data;
}
const diagnostics=new Diagnostics(post);
function error(e) {$('error').textContent=e.message;}
function status(text) {$('status').textContent=text;}
function display() {
  if(!committedScenario)return;
  renderer.update(committedScenario,viewState);diagnostics.update(committedScenario,viewState);
  for(const element of document.querySelectorAll('[data-particle-legend]'))element.hidden=viewState.color==='momentum';
  $('clumpLegend').hidden=viewState.color!=='truth';
  $('protonLegend').textContent=viewState.color==='truth'?'● background p':'● p · origin hidden';
  $('colorNote').textContent=viewState.color==='momentum'?'Color shows pT from cyan (0 GeV) to violet (3+ GeV).':'';
  const variants=variantsForComparison(committedScenario,viewState.compare),variant=variants[0]||Object.keys(committedScenario.variants)[0];
  const s=committedScenario.summaries[variant][viewState.composition];
  const sourceDetails=committedScenario.variants[variant].source_truth||committedScenario.source_truth;
  const generated=sourceDetails.reduce((n,c)=>n+c.emitted,0),accepted=sourceDetails.reduce((n,c)=>n+c.accepted,0);
  $('summary').replaceChildren();
  const sourceMean=committedScenario.summaries[variant].source_only.mean_pt_gev;
  const metrics=[[generated,'Source emitted'],[accepted,'Source accepted'],[sourceMean===null?'—':sourceMean.toFixed(3),'Source mean pT · GeV'],[s.count,'Observed total'],[`${s.protons} / ${s.antiprotons}`,'p / p̄'],[s.mean_pt_gev===null?'—':s.mean_pt_gev.toFixed(3),'Observed mean pT · GeV']];
  const backgroundMetadata=committedScenario.background.metadata;
  if(backgroundMetadata.event_record)metrics.push([backgroundMetadata.physics?.temperature_gev==null?'Unrecorded':(1000*backgroundMetadata.physics.temperature_gev).toFixed(1),'Background T · MeV']);
  for(const [value,label] of metrics){const d=document.createElement('div');d.className='metric';const b=document.createElement('b'),span=document.createElement('span');b.textContent=value;span.textContent=label;d.append(b,span);$('summary').append(d);}
  $('realized').textContent=`${variantLabels[variant]} readout · `+(sourceDetails.map(c=>c.independent_donors===undefined?`Clump ${c.clump_id}: ${c.emitted} emitted → ${c.accepted} accepted · T ${(1000*c.temperature_gev).toFixed(2)} MeV · β ${c.boost_speed.toFixed(4)}`:`${c.independent_donors} independent donors → ${c.accepted} accepted · T ${c.temperature_range_gev?c.temperature_range_gev.map(t=>(1000*t).toFixed(2)).join('–'):'—'} MeV · β ${c.boost_speed_range?c.boost_speed_range.map(b=>b.toFixed(4)).join('–'):'—'}`).join(' | ')||'No source daughters generated');
  if(variants.includes('event_mixed'))$('realized').textContent+=' · Event-mixed counts/spectra fluctuate independently; each panel reports its own source counts.';
  const bg=committedScenario.background.metadata;
  const originalCuts=typeof bg.source_acceptance==='object'&&bg.source_acceptance?`|eta| < ${bg.source_acceptance.eta_max}, pT >= ${bg.source_acceptance.pt_min_gev} GeV${bg.source_acceptance.pt_max_gev==null?'':`, pT < ${bg.source_acceptance.pt_max_gev} GeV`}`:bg.source_acceptance;
  const backgroundPhysics=bg.physics?.thermal_law==='massive_canonical_boltzmann_3d_v1'?`same massive 3D law as clumps · T ${bg.physics.temperature_gev==null?'unrecorded':`${(1000*bg.physics.temperature_gev).toFixed(1)} MeV`} · independent particles, beta ${bg.physics.boost_speed??'unrecorded'}`:'thermal model / input temperature unrecorded (legacy files retain their original physics)';
  $('eventMeta').textContent=bg.event_record?`Pinned ${bg.label}, event ${bg.event_index} · centrality ${(bg.centrality*100).toFixed(2)}% · ${backgroundPhysics} · ${bg.emitted_multiplicity==null?'emitted count unrecorded':`${bg.emitted_multiplicity} emitted → ${bg.event_record[1]} stored accepted`} · original acceptance ${originalCuts}. Looser cuts cannot restore absent tracks.`:'Source only: collision geometry and centrality are undefined.';
  $('precisionNote').textContent=committedScenario.resolved_request.observation_mode==='native'?'Native: source float64; HDF5 retains stored precision.':'Packed float16: each accepted composition re-encoded using float32 event means. Existing background quantization remains.';
  if(viewState.frame==='rest')$('precisionNote').textContent+=' Rest views are simulation truth; lab acceptance is highlighted, background omitted. Event-mixed points use separate donor rest frames, not one common frame.';
  const t=committedScenario.timing_ms;
  $('timings').textContent=`Generation ${t.generation.toFixed(1)} ms · preparation ${t.response_preparation?.toFixed(1)??'—'} ms · ${committedScenario.scenario_id} · pz scale ×${viewState.pzScale}`;
}
const queue=new LatestRequestQueue({send:spec=>post('/api/v1/scenario',spec),commit:scenario=>{
  const changed=!committedScenario||scenario.scenario_id!==committedScenario.scenario_id||scenario.resolved_request.observation_mode!==committedScenario.resolved_request.observation_mode;
  if(changed)renderer.stopReplay();
  committedScenario=scenario;$('error').textContent='';status(`Committed · ${scenario.scenario_id.slice(0,12)}`);
  for(const option of $('compare').options){option.disabled=option.value==='both'?false:option.value==='shared_mixed'?!(scenario.variants.shared_boost&&scenario.variants.event_mixed):!scenario.variants[option.value];}
  if($('compare').selectedOptions[0]?.disabled||!variantsForComparison(scenario,viewState.compare).length){viewState.compare='both';$('compare').value='both';}
  for(const id of ['recipeTop','recipe','snapshot','image'])$(id).disabled=false;
  display();
  if(changed&&$('replayOnChange').checked)renderer.replay();
},error,status});
function submit(immediate=false) {if(draftSpec)queue.draft(draftSpec,$('auto').checked,immediate);}
setupControls((id,immediate)=>{if(!draftSpec)return;editControl(draftSpec,id);submit(immediate);});
for(const button of document.querySelectorAll('[data-help]'))button.onclick=()=>{$('helpDialog').showModal();$('helpDialog').scrollTop=0;};
$('helpDialog').addEventListener('click',event=>{
  if(event.target!==$('helpDialog'))return;
  const rect=$('helpDialog').getBoundingClientRect();
  if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)$('helpDialog').close();
});
$('apply').onclick=()=>{if(draftSpec)queue.draft(draftSpec,true,true);};
$('auto').onchange=()=>{if(draftSpec)submit(true);};
$('newClump').onclick=()=>{draftSpec.realization.clump_revision++;submit(true);};
$('newDirection').onclick=()=>{draftSpec.realization.direction_revision++;submit(true);};
$('newControls').onclick=()=>{draftSpec.realization.control_revision++;submit(true);};
$('reset').onclick=()=>{selectionSequence++;queue.invalidate();draftSpec=structuredClone(initialSpec);syncControls(draftSpec);$('auto').checked=true;submit(true);};
$('precision').onchange=()=>{draftSpec.observation_mode=$('precision').value;submit(true);};
for(const id of ['composition','compare','color','frame','axis','pzScale','ghost']) $(id).addEventListener('input',()=>{
  viewState[id]=id==='ghost'?$(id).checked:['axis','pzScale'].includes(id)?Number($(id).value):$(id).value;
  if(!(viewState.axis>0)||!(viewState.pzScale>0))return;
  display();
});
$('fit').onclick=()=>{
  if(!committedScenario)return;
  // Explicit fit always uses the compared union AND both components, independent of visibility.
  let limit=.5;
  for(const v of Object.keys(committedScenario.variants))for(const p of particlesFor(committedScenario,v,{...viewState,composition:'combined'}))limit=Math.max(limit,Math.abs(p.p4[1])*1.1,Math.abs(p.p4[2])*1.1,Math.abs(p.p4[3]*viewState.pzScale)*1.1);
  viewState.axis=Math.ceil(limit*2)/2;$('axis').value=viewState.axis;display();renderer.gpu?.fit(viewState.axis);
};
$('replay').onclick=()=>{renderer.replay();status('Illustrative momentum reveal · no emission positions or lifetime');};
$('replayOnChange').onchange=()=>{if(!$('replayOnChange').checked){renderer.stopReplay();display();}};
$('applyAdvanced').onclick=()=>{try{draftSpec.clump_config=JSON.parse($('advanced').value);syncControls(draftSpec);submit(true);}catch(e){error(e);}};
$('variableExample').onclick=async()=>{try{const r=await fetch('/examples/variable_parameters.json');const example=await r.json();draftSpec.clump_config=example.clump_config;syncControls(draftSpec);submit(true);}catch(e){error(e);}};
$('loadRecipe').onchange=async e=>{try{const recipe=JSON.parse(await e.target.files[0].text());draftSpec=recipe;syncControls(draftSpec);queue.draft(draftSpec,true,true);}catch(e){error(e);}};

async function selectBackground(mode) {
  if(!draftSpec)return;
  const selection=++selectionSequence,dataset=$('dataset').value;
  if(dataset==='none'){draftSpec.background={provider:'none'};$('backgroundInfo').textContent='No HDF5 event selected.';submit(true);return;}
  try {
    $('backgroundInfo').textContent='Resolving and pinning background…';
    const body={dataset_id:dataset,mode,centrality_min:Number($('cmin').value),centrality_max:Number($('cmax').value),selection_seed:draftSpec.realization.root_seed};
    if(mode==='exact')body.event_index=Number($('eventIndex').value);
    if(mode==='previous'||mode==='next')body.current_index=Number($('eventIndex').value);
    const result=await post('/api/v1/background/select',body);
    if(selection!==selectionSequence)return;
    draftSpec.background=Object.fromEntries(['provider','dataset_id','event_index','fingerprint'].map(k=>[k,result[k]]));
    $('eventIndex').value=result.event_index;$('backgroundInfo').textContent=`Pinned event ${result.event_index} · ${result.fingerprint.slice(0,24)}…`;
    submit(true);
  }catch(e){if(selection===selectionSequence){error(e);$('backgroundInfo').textContent='Selection failed · previous pinned event retained.';}}
}
$('dataset').onchange=()=>selectBackground('exact');
for(const button of document.querySelectorAll('[data-nav]'))button.onclick=()=>selectBackground(button.dataset.nav);
function downloadURL(url) {const a=document.createElement('a');a.href=url;a.download='';document.body.append(a);a.click();a.remove();}
function saveRecipe() {
  if(!committedScenario)return;
  saveJSON(recipeForScenario(committedScenario),`clump-${committedScenario.scenario_id}-recipe.json`);
}
$('recipe').onclick=saveRecipe;$('recipeTop').onclick=saveRecipe;
$('snapshot').onclick=()=>{
  if(!committedScenario)return;
  const s=committedScenario,variant=s.variants[viewState.compare]?viewState.compare:Object.keys(s.variants)[0];
  downloadURL(`/api/v1/scenarios/${s.scenario_id}/snapshot?observation_mode=${s.resolved_request.observation_mode}&variant=${variant}`);
};
$('image').onclick=()=>renderer.image().toBlob(blob=>{const url=URL.createObjectURL(blob);downloadURL(url);setTimeout(()=>URL.revokeObjectURL(url),1000);});

async function start() {
  try {
    const [recipe,catalog,cap]=await Promise.all([fetch('/examples/source_only.json').then(r=>r.json()),fetch('/api/v1/backgrounds').then(r=>r.json()),fetch('/api/v1/capabilities').then(r=>r.json())]);
    initialSpec=recipe;draftSpec=structuredClone(recipe);
    for(const bg of catalog.backgrounds){const option=document.createElement('option');option.value=bg.dataset_id;option.textContent=`${bg.label} · ${bg.physics?.temperature_gev==null?'T unrecorded':`T ${(1000*bg.physics.temperature_gev).toFixed(0)} MeV`} · ${bg.event_count} events`;option.title=bg.dataset_id;$('dataset').append(option);}
    if(catalog.warnings.length)$('backgroundInfo').textContent=catalog.warnings.join(' · ');
    $('n').max=cap.limits.multiplicity_per_clump;$('beta').max=cap.limits.beta_max;
    syncControls(draftSpec);submit(true);
  }catch(e){error(e);status('Could not initialize · reload to retry');}
}
start();
// Optional GPU loading starts after controls, network queue and Canvas are ready.
import('./renderer3d.js').then(m=>m.create3D()).then(gpu=>renderer.setGPU(gpu)).catch(e=>renderer.unavailable(`3D unavailable: ${e.message}. Canvas views, controls and downloads remain active.`));
// Read-only inspection hook for reproducibility and browser acceptance checks.
window.hicadExplorer={get draftSpec(){return structuredClone(draftSpec);},get committedScenario(){return structuredClone(committedScenario);},get viewState(){return {...viewState};},get queueState(){return {busy:queue.busy,pending:!!queue.pending,sequence:queue.sequence};},get gpuStats(){return renderer.gpu?.stats()||null;}};
