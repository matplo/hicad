export const $=id=>document.getElementById(id);
export function variantsWithControls(azimuth=false,mixed=false) {
  return ['shared_boost',...(azimuth?['azimuth_decohered']:[]),...(mixed?['event_mixed']:[])];
}
export function setupControls(changed) {
  const definitions=[['n','Emitted protons n',0,40,1,8,200],['temperature','Temperature · MeV',20,250,1,100,500],['beta','Common boost β',0,.9,.01,.4,.95]];
  for(const [id,label,min,max,step,value,cap] of definitions) {
    const wrap=document.createElement('div');
    wrap.innerHTML=`<div class="sliderlabel"><label for="${id}">${label}</label>${id==='beta'?'<button type="button" class="helpShortcut" data-help aria-label="Help: common boost beta" aria-haspopup="dialog" aria-controls="helpDialog">?</button>':''}<input id="${id}" aria-label="${label}" type="number" min="${id==='temperature'?1:min}" max="${cap}" step="${step}" value="${value}"></div><input id="${id}Range" aria-label="${label} slider" type="range" min="${min}" max="${max}" step="${step}" value="${value}">`;
    $('sliders').append(wrap);
    $(id).addEventListener('input',()=>{$(id+'Range').value=$(id).value;changed(id,false);});
    $(id+'Range').addEventListener('input',()=>{$(id).value=$(id+'Range').value;changed(id,false);});
    for(const el of [$(id),$(id+'Range')]) {
      el.addEventListener('change',()=>changed(id,true));
      el.addEventListener('pointerup',()=>changed(id,true));
    }
  }
  for(const id of ['enabled','direction','seed','etaMax','ptMin','ptMax','includeAzimuth','includeMixed']) $(id).addEventListener('input',()=>changed(id,['includeAzimuth','includeMixed'].includes(id)));
}
export function syncControls(spec) {
  const c=spec.clump_config;
  $('enabled').checked=c.enabled; $('seed').value=spec.realization.root_seed;
  $('direction').value=c.boost_direction;
  for(const [id,key,factor] of [['n','emitted_multiplicity',1],['temperature','temperature_gev',1000],['beta','boost_speed',1]]) {
    const d=c[key];
    if(d.distribution==='fixed') {$(id).value=d.value*factor; $(id+'Range').value=d.value*factor;}
    $(id).title=d.distribution==='fixed'?'':`Advanced ${d.distribution} distribution; moving this control selects fixed.`;
  }
  $('etaMax').value=c.acceptance.eta_max; $('ptMin').value=c.acceptance.pt_min_gev;
  $('ptMax').value=c.acceptance.pt_max_gev??'';
  $('precision').value=spec.observation_mode;
  $('includeAzimuth').checked=c.variants.includes('azimuth_decohered');
  $('includeMixed').checked=c.variants.includes('event_mixed');
  $('newControls').disabled=!$('includeAzimuth').checked&&!$('includeMixed').checked;
  $('advanced').value=JSON.stringify(c,null,2);
  $('dataset').value=spec.background.provider==='none'?'none':spec.background.dataset_id;
  if(spec.background.provider==='hdf5') $('eventIndex').value=spec.background.event_index;
}
export function editControl(spec,id) {
  const c=spec.clump_config;
  if(id==='n') c.emitted_multiplicity={distribution:'fixed',value:Number($('n').value)};
  if(id==='temperature') c.temperature_gev={distribution:'fixed',value:Number($('temperature').value)/1000};
  if(id==='beta') c.boost_speed={distribution:'fixed',value:Number($('beta').value)};
  if(id==='enabled') c.enabled=$('enabled').checked;
  if(id==='direction') c.boost_direction=$('direction').value;
  if(id==='seed') spec.realization.root_seed=$('seed').value;
  if(['includeAzimuth','includeMixed'].includes(id)){
    c.variants=variantsWithControls($('includeAzimuth').checked,$('includeMixed').checked);
    $('newControls').disabled=c.variants.length===1;
  }
  if(['etaMax','ptMin','ptMax'].includes(id)) c.acceptance={eta_max:Number($('etaMax').value),pt_min_gev:Number($('ptMin').value),pt_max_gev:$('ptMax').value===''?null:Number($('ptMax').value),azimuth:'full'};
  $('advanced').value=JSON.stringify(c,null,2);
}
