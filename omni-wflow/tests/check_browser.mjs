import fs from 'node:fs';import vm from 'node:vm';import assert from 'node:assert/strict';
const html=fs.readFileSync(process.argv[2],'utf8');
class Element{constructor(id,tag){this.id=id;this.tag=tag;this.value='';this.textContent='';this._html='';this.disabled=false;this.open=false;}showModal(){this.open=true;}close(){this.open=false;}set innerHTML(v){this._html=v;if(this.tag==='select'){this.options=[...v.matchAll(/value="([^"]+)"/g)].map(m=>m[1]);this.value=this.options[0]||'';}}get innerHTML(){return this._html;}}
const elements=new Map();for(const m of html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"[^>]*>/g))elements.set(m[2],new Element(m[2],m[1]));
elements.get('results').textContent=html.match(/<script id="results" type="application\/json">([\s\S]*?)<\/script>/)[1];
const context=vm.createContext({document:{getElementById:id=>elements.get(id),querySelectorAll:()=>[...elements.values()].filter(e=>e.tag==='select')}});const run=s=>vm.runInContext(s,context);
run(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
function check(){assert.match(elements.get('maps').innerHTML,/<svg/);assert.doesNotMatch(elements.get('maps').innerHTML,/NaN|Infinity/);assert.match(elements.get('roc').innerHTML,/<svg/);}
check();
// Parameter filters stay active even when a parameter is an axis or panel.
for(const k of ['T','beta','n','k']){
  assert.equal(elements.get(k).disabled,false);
  const v=run(`unique(D.records.map(r=>r.setting['${k}']))[0]`);
  run(`$('${k}').value='${v}';render()`);
  assert.ok(run('chosen.length')>0);
  assert.ok(run(`chosen.every(r=>r.setting['${k}']===${v})`));
  run(`$('${k}').value='all';render()`);check();
}
// Scanning the remaining fourth parameter must create extra panels rather than
// overlaying multiple settings in one heatmap cell.
assert.equal(run('chosen.length'),run("D.records.filter(r=>r.classifier===$('classifier').value&&r.control===$('control').value&&r.setting.direction===$('direction').value).length"));
run("$('info').onclick()");assert.equal(elements.get('infoDialog').open,true);
for(const term of ['Background and geometry','What is a clump','Mixture fraction','center-of-mass','Tbg=','Emission probability','gamma–Poisson'])assert.ok(elements.get('infoBody').innerHTML.includes(term));
run("$('infoClose').onclick()");assert.equal(elements.get('infoDialog').open,false);
run("$('info').onclick();$('infoDialog').onclick({target:$('infoDialog')})");assert.equal(elements.get('infoDialog').open,false);
run("$('reset').onclick()");check();
for(const model of run('unique(D.records.map(r=>r.classifier))')){run(`$('classifier').value='${model}';render()`);check();}
run("$('fraction').value='0';render()");assert.ok(run('chosen.every(r=>val(r)===.5)'));
run("$('x').value='n';$('y').value='n';render()");assert.notEqual(run("$('x').value"),run("$('y').value"));check();
run("$('control').value='event_mixed';render()");assert.doesNotMatch(elements.get('maps').innerHTML,/NaN|Infinity/);
console.log('Offline AUC browser: initialization, model comparisons, mixture null, axis collision, active parameter filters, INFO modal and controls passed.');
