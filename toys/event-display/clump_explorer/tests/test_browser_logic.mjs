import {test} from 'node:test';
import assert from 'node:assert/strict';
import {LatestRequestQueue} from '../static/request_state.js';
import {particlesFor,projectParticle,particleColor,clumpProtonColor,speciesColors,RevealTimeline,drawScatter,variantsForComparison,fitDistanceForSphere,clipCoordinatesVisible,sourceViewportCounts} from '../static/renderer.js';
import {recipeForScenario} from '../static/downloads.js';
import {variantsWithControls} from '../static/controls.js';
const tick=()=>new Promise(r=>setTimeout(r,5));
function deferred(){let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};}
function fixture() {
  const calls=[],commits=[],errors=[],statuses=[];
  const queue=new LatestRequestQueue({delay:1,retryDelay:1,send:body=>{const d=deferred();calls.push({body,...d});return d.promise;},commit:r=>commits.push(r),error:e=>errors.push(e),status:s=>statuses.push(s)});
  return {queue,calls,commits,errors,statuses};
}
test('rapid physical changes retain only the newest trailing draft; stale response cannot commit',async()=>{
  const f=fixture();f.queue.draft({n:8},true,true);await tick();
  for(const n of [9,20,40,12])f.queue.draft({n},true,true);
  await tick();assert.equal(f.calls.length,1);
  f.calls[0].resolve({request_id:f.calls[0].body.request_id,n:8});await tick();
  assert.equal(f.commits.length,0);assert.equal(f.calls.length,2);assert.equal(f.calls[1].body.n,12);
  f.calls[1].resolve({...f.calls[1].body});await tick();assert.equal(f.commits.length,1);assert.equal(f.commits[0].n,12);
});
test('reset during a request, auto-update off and explicit Apply',async()=>{
  const f=fixture();f.queue.draft({n:40},true,true);await tick();
  f.queue.invalidate();f.queue.draft({n:8},true,true);f.calls[0].resolve(f.calls[0].body);await tick();
  assert.equal(f.commits.length,0);assert.equal(f.calls[1].body.n,8);
  f.queue.draft({n:15},false);f.calls[1].resolve(f.calls[1].body);await tick();
  assert.equal(f.commits.length,0);assert.equal(f.calls.length,2);
  f.queue.draft({n:15},true,true);await tick();f.calls[2].resolve(f.calls[2].body);await tick();assert.equal(f.commits[0].n,15);
});
test('service errors preserve committed state and 503 retry sends only the latest draft',async()=>{
  const f=fixture();f.queue.draft({n:8},true,true);await tick();f.calls[0].resolve(f.calls[0].body);await tick();
  f.queue.draft({n:-1},true,true);await tick();f.calls[1].reject(Object.assign(new Error('invalid n'),{status:422}));await tick();
  assert.equal(f.errors.length,1);assert.equal(f.commits.length,1);
  f.queue.draft({n:9},true,true);await tick();f.calls[2].reject(Object.assign(new Error('busy'),{status:503}));
  f.queue.draft({n:21},true,true);await tick();
  assert.equal(f.calls[3].body.n,21);f.calls[3].resolve(f.calls[3].body);await tick();assert.equal(f.commits.at(-1).n,21);
});
test('source coordinates and observation arrays survive background toggles without mutation',()=>{
  const measured={particle_id:['c0:d0'],species:['p'],charge:[1],p4:[[2,1,.2,.4]],accepted:[true]};
  const scene={variants:{shared_boost:{measured,truth:{rest_p4:[[1,0,0,0]]}}},background:{measured:{particle_id:['b:0'],species:['pbar'],p4:[[2,-1,-.5,0]],accepted:[true]}},observation_order:{source_only:['c0:d0'],combined:['c0:d0','b:0']}};
  const before=structuredClone(scene),state={composition:'source_only',frame:'lab',ghost:false};
  const source=particlesFor(scene,'shared_boost',state),combined=particlesFor(scene,'shared_boost',{...state,composition:'combined'});
  assert.deepEqual(source[0].p4,combined.find(p=>p.id==='c0:d0').p4);assert.deepEqual(scene,before);
  const angles=[Math.PI-.01,-Math.PI+.01].map(phi=>projectParticle([2,Math.cos(phi),Math.sin(phi),.1],'yphi',Math.PI)[1]);
  assert.ok(Math.abs(angles[0]-angles[1])<.03,'cluster at original phi seam remains connected');
});
test('clump protons are red and background protons green in truth mode; neutral species hides origin',()=>{
  const source={accepted:true,source:true,species:'p',p4:[2,1,.2,.4]},background={...source,source:false};
  assert.equal(particleColor(source,{color:'truth'}),clumpProtonColor);
  assert.equal(particleColor(background,{color:'truth'}),speciesColors.p);
  assert.notEqual(clumpProtonColor,speciesColors.p);
  assert.equal(particleColor(source,{color:'species'}),particleColor(background,{color:'species'}));
  assert.notEqual(particleColor({...source,accepted:false},{color:'truth'}),clumpProtonColor);
});
test('saved recipe uses committed settings and exact uint64 seeds without view state or aliasing',()=>{
  const committed={resolved_request:{schema_version:'hicad-clump-explorer-1',request_id:17,
    realization:{root_seed:'18446744073709551615',clump_revision:2,direction_revision:3,control_revision:0},
    background:{provider:'hdf5',dataset_id:'pinned',event_index:9,fingerprint:'unchanged'},
    clump_config:{temperature_gev:{distribution:'fixed',value:.1}},observation_mode:'native'},
    provenance:{core_version:'canonical-proton-1.0.0',rng_policy:'seedsequence-pcg64-roles-v1',numpy_version:'2.5.4'},
    view_state:{color:'truth',replayOnChange:true}};
  const before=structuredClone(committed),recipe=JSON.parse(JSON.stringify(recipeForScenario(committed)));
  assert.equal(recipe.realization.root_seed,'18446744073709551615');
  assert.deepEqual(recipe.background,committed.resolved_request.background);
  assert.deepEqual(recipe.provenance,committed.provenance);assert.equal(recipe.view_state,undefined);
  recipe.clump_config.temperature_gev.value=.2;assert.deepEqual(committed,before);
});
test('Canvas replay reveals positions using a separate timeline and leaves authoritative momenta unchanged',()=>{
  const timeline=new RevealTimeline(700);assert.equal(timeline.fraction(0),1);
  timeline.start(100);assert.equal(timeline.fraction(100),0);assert.equal(timeline.fraction(450),.5);assert.equal(timeline.fraction(800),1);
  const particles=[{accepted:true,source:true,species:'p',p4:[2,1,.2,.4]}],before=structuredClone(particles),arcs=[];
  const context=Object.fromEntries(['setTransform','fillRect','beginPath','moveTo','lineTo','stroke','fillText','save','rect','clip','fill','restore'].map(k=>[k,()=>{}]));
  context.arc=(x,y)=>arcs.push([x,y]);
  const canvas={getBoundingClientRect:()=>({width:400,height:250}),getContext:()=>context};
  const original=globalThis.window;globalThis.window={devicePixelRatio:1};
  try {
    drawScatter(canvas,particles,{axis:2,color:'truth'},'xy',0,0);
    drawScatter(canvas,particles,{axis:2,color:'truth'},'xy',0,1);
    assert.notDeepEqual(arcs[0],arcs[1]);assert.deepEqual(particles,before);
  } finally {globalThis.window=original;}
  timeline.finish();assert.equal(timeline.active,false);assert.equal(timeline.fraction(200),1);
});
test('event-mixed comparison uses distinct donor IDs, its own accepted ordering and recorded policy',()=>{
  const shared={particle_id:['c0:d0'],species:['p'],p4:[[2,1,0,0]],accepted:[true]};
  const mixed={particle_id:['m:c0:d0','m:c0:d1'],species:['p','p'],p4:[[2,0,1,0],[2,0,-1,0]],accepted:[false,true]};
  const background={particle_id:['b:0'],species:['pbar'],p4:[[2,-1,0,0]],accepted:[true]};
  const scene={variants:{shared_boost:{measured:shared},azimuth_decohered:{measured:shared},event_mixed:{measured:mixed}},
    background:{measured:background},observation_order:{combined:['c0:d0','b:0']},
    observation_order_by_variant:{event_mixed:{combined:['b:0','m:c0:d1']}}};
  assert.deepEqual(variantsForComparison(scene,'shared_mixed'),['shared_boost','event_mixed']);
  assert.equal(variantsForComparison(scene,'both').length,3);
  const points=particlesFor(scene,'event_mixed',{composition:'combined',frame:'lab',ghost:false});
  assert.deepEqual(points.map(p=>p.id),['b:0','m:c0:d1']);assert.equal(points[1].source,true);
  const saved=recipeForScenario({resolved_request:{clump_config:{variants:['event_mixed']}},
    provenance:{core_version:'v1',rng_policy:'r1',numpy_version:'n1',event_mixing_policy:'independent-source-donors-v1'}});
  assert.equal(saved.provenance.event_mixing_policy,'independent-source-donors-v1');
});
test('optional controls start absent and can be enabled independently',()=>{
  assert.deepEqual(variantsWithControls(),['shared_boost']);
  assert.deepEqual(variantsWithControls(true,false),['shared_boost','azimuth_decohered']);
  assert.deepEqual(variantsWithControls(false,true),['shared_boost','event_mixed']);
  assert.deepEqual(variantsWithControls(true,true),['shared_boost','azimuth_decohered','event_mixed']);
});
test('viewport clipping counts camera exits separately from rejected truth ghosts',()=>{
  const particles=[{source:true,accepted:true,clip:{x:0,y:0,z:.5,w:1}},
    {source:true,accepted:true,clip:{x:2,y:0,z:.5,w:1}},
    {source:true,accepted:true,clip:{x:0,y:0,z:0,w:-1}},
    {source:true,accepted:false,clip:{x:0,y:0,z:.5,w:1}},
    {source:false,accepted:true,clip:{x:2,y:0,z:.5,w:1}}];
  assert.deepEqual(sourceViewportCounts(particles,p=>p.clip,true),{accepted:3,inView:1,offCamera:2,rejectedGhosts:1});
  assert.equal(clipCoordinatesVisible({x:0,y:0,z:-.1,w:1},true),false);
  assert.equal(clipCoordinatesVisible({x:0,y:0,z:-.1,w:1},false),true);
  for(const clip of [{x:0,y:2,z:0,w:1},{x:0,y:0,z:2,w:1},{x:0,y:0,z:-2,w:1},{x:NaN,y:0,z:0,w:1}])assert.equal(clipCoordinatesVisible(clip),false);
});
test('camera fit includes cube corners with perspective depth and narrow viewports',()=>{
  const axis=2,radius=Math.sqrt(3)*axis,fov=42;
  const normalize=v=>{const n=Math.hypot(...v);return v.map(x=>x/n);};
  const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const dot=(a,b)=>a.reduce((n,x,i)=>n+x*b[i],0);
  const direction=normalize([4,3,5]),right=normalize(cross([0,1,0],direction)),up=cross(direction,right);
  for(const aspect of [.4,1,2.5]){
    const distance=fitDistanceForSphere(radius,fov,aspect),slope=Math.tan(fov*Math.PI/360);
    for(const x of [-axis,axis])for(const y of [-axis,axis])for(const z of [-axis,axis]){
      const p=[x,y,z],depth=distance-dot(p,direction);
      assert.ok(depth>0);assert.ok(Math.abs(dot(p,right)/(depth*slope*aspect))<=1);assert.ok(Math.abs(dot(p,up)/(depth*slope))<=1);
    }
  }
  assert.ok(fitDistanceForSphere(radius,fov,.4)>fitDistanceForSphere(radius,fov,2.5));
});
