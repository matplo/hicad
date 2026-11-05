// Pure JavaScript filter/render tests with a minimal DOM model (no UI automation).
// Usage: node studies/bes-cumulants/tests/check_browser.mjs path/to/index.html
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const html=fs.readFileSync(process.argv[2],'utf8');
class Element {
  constructor(id='',tag=''){this.id=id;this.tag=tag;this.options=[];this._value='';this.textContent='';this.innerHTML='';this.checked=false;this.style={};}
  set value(v){this._value=String(v);}
  get value(){return this._value;}
  get selectedOptions(){return this.options.filter(o=>o.value===this.value);}
  replaceChildren(){this.options=[];this.value='';}
  append(o){this.options.push(o);if(this.options.length===1)this.value=o.value;}
  addEventListener(){}
  querySelectorAll(){return [];}
  querySelector(selector){const value=selector.match(/value=["']?([^\]"']+)/)?.[1];return this.options.find(o=>o.value===value);}
}
const elements=new Map();
for(const m of html.matchAll(/<([a-z][a-z0-9]*)\b([^>]*\bid="([^"]+)"[^>]*)>/g)){
  const e=new Element(m[3],m[1]);e.checked=/\bchecked\b/.test(m[2]);elements.set(e.id,e);
}
for(const m of html.matchAll(/<select id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)){
  const e=elements.get(m[1]);for(const o of m[2].matchAll(/<option value="([^"]+)"[^>]*>([^<]*)/g)){const el=new Element();el.value=o[1];el.textContent=o[2];e.append(el);}
}
elements.get('resultsData').textContent=html.match(/<script id="resultsData" type="application\/json">([\s\S]*?)<\/script>/)[1];
const context=vm.createContext({document:{getElementById:id=>elements.get(id),createElement:tag=>new Element('',tag),querySelectorAll:selector=>[...elements.values()].filter(e=>selector==='select'?e.tag==='select':e.tag==='input'&&e.id!=='load')},console,Blob,URL,setTimeout,innerWidth:1400,innerHeight:1000});
const script=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];vm.runInContext(script,context);
const run=s=>vm.runInContext(s,context);
function check(){assert.match(elements.get('plot').innerHTML,/<svg/);assert.doesNotMatch(elements.get('plot').innerHTML,/NaN|Infinity/);assert.ok(run('selected.length')>0);}
if(run("DATA.records.some(r=>r.scan==='joint')")){
  check();assert.equal(run("$('view').value"),'facets');assert.equal(run('selected.length'),84);
  run("$('view').value='heatmap';render()");check();assert.equal(run('selected.length'),21);
  run("$('heatX').value='n';$('heatY').value='T';render()");check();assert.equal(run('selected.length'),12);
  run("$('heatX').value='n';$('heatY').value='n';render()");check();assert.notEqual(run("$('heatX').value"),run("$('heatY').value"));
  run("$('view').value='trend';$('xaxis').value='f';$('overlay').value='T';$('n').value='8';render()");check();assert.equal(run('selected.length'),21);
  run("$('overlay').value='none';$('scan').value='strength';scanOptions();$('view').value='trend';render()");
}
check();assert.equal(run('selected.length'),7);
run("$('n').value='all';render()");check();assert.equal(run('selected.length'),run("DATA.config.strength.n.length")*7);
run("$('n').value='8';$('event_mixed').checked=true;$('poisson').checked=true;render()");check();assert.equal(run('selected.length'),21);
for(const obs of ['net','proton','antiproton'])for(const method of ['raw','cbwc'])for(const presentation of ['absolute','delta']){
  run(`$('observable').value='${obs}';statsOptions();$('method').value='${method}';$('presentation').value='${presentation}';render()`);check();
  if(obs!=='net'){run("$('metric').value='F4_norm';render()");check();}
}
run("$('observable').value='net';statsOptions();$('view').value='distribution';render()");check();
if(run("DATA.records.some(r=>r.scan==='shape')")){run("$('scan').value='shape';scanOptions();$('direction').value='isotropic_3d';$('view').value='heatmap';render()");check();assert.equal(run('selected.length'),25);
for(const model of ['shared_boost','event_mixed','poisson']){run(`$('heatModel').value='${model}';render()`);check();}
}
if(run("DATA.records.some(r=>r.scan==='clumps')")){run("$('scan').value='clumps';scanOptions();$('view').value='trend';render()");check();}
run("$('shared_boost').checked=false;$('event_mixed').checked=false;$('poisson').checked=false;render()");assert.doesNotMatch(elements.get('plot').innerHTML,/NaN|Infinity/);
console.log('Browser initialization, model toggles, overlays, all observables, factorials, uncertainty/delta views, distributions, heatmaps, clump scan and empty selection: passed.');
