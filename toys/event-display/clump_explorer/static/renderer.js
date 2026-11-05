// Canvas views and particle selection initialize without Three.js or a GPU.
export const clumpProtonColor='#ff5b5b';
export const speciesColors={'p':'#55d58b','pbar':'#f28cb4','pi+':'#719ce9','pi-':'#719ce9','K+':'#f2be63','K-':'#f2be63'};
export const variantLabels={shared_boost:'Shared boost',azimuth_decohered:'Azimuth decohered',event_mixed:'Event mixed'};
export function variantsForComparison(scenario,comparison) {
  const requested=comparison==='both'?Object.keys(scenario.variants):comparison==='shared_mixed'?['shared_boost','event_mixed']:[comparison];
  return requested.filter(v=>scenario.variants[v]);
}
export function fitDistanceForSphere(radius,verticalFovDegrees,aspect) {
  if(![radius,verticalFovDegrees,aspect].every(Number.isFinite)||radius<0||verticalFovDegrees<=0||verticalFovDegrees>=180||aspect<=0)throw new Error('Invalid camera fit bounds');
  const vertical=verticalFovDegrees*Math.PI/360,horizontal=Math.atan(Math.tan(vertical)*aspect);
  return Math.max(.1,1.05*radius/Math.sin(Math.min(vertical,horizontal)));
}
export function clipCoordinatesVisible(p,zeroToOne=false) {
  if(![p.x,p.y,p.z,p.w].every(Number.isFinite)||p.w<=0)return false;
  return p.x>=-p.w&&p.x<=p.w&&p.y>=-p.w&&p.y<=p.w&&p.z>=(zeroToOne?0:-p.w)&&p.z<=p.w;
}
export function sourceViewportCounts(particles,project,zeroToOne=false) {
  let accepted=0,inView=0,rejectedGhosts=0;
  for(const particle of particles)if(particle.source){
    if(!particle.accepted){rejectedGhosts++;continue;}
    accepted++;if(clipCoordinatesVisible(project(particle),zeroToOne))inView++;
  }
  return {accepted,inView,offCamera:accepted-inView,rejectedGhosts};
}
export function particleColor(p,state) {
  if(!p.accepted) return '#53616e';
  if(state.color==='truth') return p.source?clumpProtonColor:(speciesColors[p.species]||'#ddd');
  if(state.color==='momentum') return `hsl(${190+Math.min(1,Math.hypot(p.p4[1],p.p4[2])/3)*120} 75% 65%)`;
  return speciesColors[p.species]||'#ddd';
}
// Presentation only: this timeline never mutates a particle or requests an event.
export class RevealTimeline {
  constructor(duration=700) {this.duration=duration;this.startedAt=null;}
  start(now) {this.startedAt=now;}
  finish() {this.startedAt=null;}
  get active() {return this.startedAt!==null;}
  fraction(now) {return this.active?Math.max(0,Math.min(1,(now-this.startedAt)/this.duration)):1;}
}
export function particlesFor(scenario,variant,state) {
  if(!scenario) return [];
  const source=scenario.variants[variant], map=new Map(source.measured.particle_id.map((id,i)=>[id,i]));
  let measured;
  if(state.frame==='rest') {
    measured={...source.measured,p4:source.truth.rest_p4};
  } else if(scenario.observation_views) {
    measured=scenario.observation_views[variant][state.composition];
  } else {
    const components=[];
    if(state.composition!=='source_only') components.push(scenario.background.measured);
    if(state.composition!=='background_only') components.push(source.measured);
    measured={particle_id:[],p4:[],species:[],accepted:[]};
    for(const m of components) for(let i=0;i<m.p4.length;i++) if(m.accepted[i]) {
      for(const key of Object.keys(measured)) measured[key].push(m[key][i]);
    }
    const orders=scenario.observation_order_by_variant?.[variant]||scenario.observation_order;
    const rank=new Map(orders[state.composition].map((id,i)=>[id,i]));
    const order=measured.p4.map((_,i)=>i).sort((a,b)=>rank.get(measured.particle_id[a])-rank.get(measured.particle_id[b]));
    measured=Object.fromEntries(Object.entries(measured).map(([k,v])=>[k,order.map(i=>v[i])]));
  }
  const out=[];
  for(let i=0;i<measured.p4.length;i++) if(measured.accepted[i] || (state.frame==='rest'&&state.ghost)) {
    out.push({id:measured.particle_id[i],p4:measured.p4[i],species:measured.species[i],accepted:measured.accepted[i],source:map.has(measured.particle_id[i])});
  }
  if(state.ghost && state.frame!=='rest' && state.composition!=='background_only') {
    for(let i=0;i<source.measured.p4.length;i++) if(!source.measured.accepted[i]) out.push({id:source.measured.particle_id[i],p4:source.measured.p4[i],species:'p',accepted:false,source:true});
  }
  return out;
}
export function fitCanvas(canvas) {
  const rect=canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);
  canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);
  return {ctx,w:rect.width,h:rect.height};
}
export function projectParticle(p4,projection,phiCenter=0) {
  const [e,px,py,pz]=p4,pt=Math.hypot(px,py),mass2=Math.max(0,e*e-px*px-py*py-pz*pz);
  if(pt===0)return null;
  return projection==='xy'?[px,py]:[Math.asinh(pz/Math.sqrt(mass2+pt*pt)),((Math.atan2(py,px)-phiCenter+3*Math.PI)%(2*Math.PI))-Math.PI];
}
export function drawScatter(canvas,particles,state,projection,phiCenter=0,reveal=1) {
  const {ctx,w,h}=fitCanvas(canvas),pad=34;
  ctx.fillStyle='#0b1621';ctx.fillRect(0,0,w,h);
  const xrange=projection==='xy'?state.axis:2.5,yrange=projection==='xy'?state.axis:Math.PI;
  const sx=x=>pad+(x+xrange)/(2*xrange)*(w-pad-12),sy=y=>h-pad-(y+yrange)/(2*yrange)*(h-pad-22);
  ctx.strokeStyle='#243c4e';ctx.lineWidth=.6;
  for(let i=-2;i<=2;i++) {const x=i*xrange/2,y=i*yrange/2;ctx.beginPath();ctx.moveTo(sx(x),22);ctx.lineTo(sx(x),h-pad);ctx.moveTo(pad,sy(y));ctx.lineTo(w-12,sy(y));ctx.stroke();}
  ctx.fillStyle='#91a9bb';ctx.font='10px system-ui';
  ctx.fillText(projection==='xy'?'px [GeV]':'rapidity y',w/2-20,h-7);
  ctx.fillText(projection==='xy'?'py [GeV]':'wrapped φ − φc [rad]',pad,13);
  ctx.fillText((-xrange).toFixed(1),pad-8,h-pad+13);ctx.fillText(xrange.toFixed(1),w-30,h-pad+13);
  ctx.fillText(yrange.toFixed(1),2,28);ctx.fillText((-yrange).toFixed(1),2,h-pad);
  ctx.save();ctx.beginPath();ctx.rect(pad,22,w-pad-12,h-pad-22);ctx.clip();
  let outside=0;
  for(const p of particles) {
    const projected=projectParticle(p.p4,projection,phiCenter);
    if(!projected)continue;
    const [x,y]=projected.map(v=>projection==='xy'?v*reveal:v);
    if(!Number.isFinite(x)||!Number.isFinite(y)) continue;
    if(Math.abs(x)>xrange||Math.abs(y)>yrange) outside++;
    ctx.globalAlpha=(p.accepted?.78:.35)*(projection==='xy'?1:reveal);ctx.fillStyle=particleColor(p,state);ctx.beginPath();ctx.arc(sx(x),sy(y),p.accepted?2.3:1.7,0,2*Math.PI);ctx.fill();
  }
  ctx.restore();ctx.globalAlpha=1;
  if(outside) {ctx.fillStyle='#f2be63';ctx.fillText(`${outside} outside fixed axes`,pad+4,34);}
  if(!particles.length) {ctx.fillStyle='#91a9bb';ctx.fillText('No particles in this view',pad+12,h/2);}
}

export class Renderer {
  constructor(container) {this.container=container;this.cards=new Map();this.scene=null;this.state=null;this.gpu=null;this.reveal=new RevealTimeline();this.animationFrame=null;this.gpuError='Loading optional 3D…';this.observer=new ResizeObserver(()=>this.draw());this.observer.observe(container);}
  setGPU(gpu) {this.gpu=gpu;this.draw();if(this.reveal.active)this.gpu.replay(this.reveal.startedAt);}
  unavailable(message) {this.gpuError=message;for(const c of this.cards.values()) c.fallback.textContent=message;}
  update(scenario,state) {this.scene=scenario;this.state=state;this.draw();}
  stopReplay() {if(this.animationFrame!==null)cancelAnimationFrame(this.animationFrame);this.animationFrame=null;this.reveal.finish();this.gpu?.stopReplay();}
  replay() {
    this.stopReplay();
    const start=performance.now();this.reveal.start(start);this.gpu?.replay(start);
    const tick=now=>{this.draw(true);if(this.reveal.fraction(now)<1)this.animationFrame=requestAnimationFrame(tick);else{this.reveal.finish();this.animationFrame=null;}};
    this.draw(true);this.animationFrame=requestAnimationFrame(tick);
  }
  draw(canvasOnly=false) {
    if(!this.scene||!this.state) return;
    const state=this.state,variants=variantsForComparison(this.scene,state.compare);
    this.container.classList.toggle('single',variants.length===1);
    for(const [v,card] of this.cards) card.root.hidden=!variants.includes(v);
    const beta=this.scene.source_truth[0]?.beta_vector||[0,0,0],phiCenter=Math.atan2(beta[1],beta[0]);
    for(const variant of variants) {
      if(!this.cards.has(variant)) {
        const root=document.createElement('article');root.className='viewcard';
        root.innerHTML=`<div class="viewtitle"><span>${variantLabels[variant]||variant}</span><small></small></div><p class="cutnote"></p><div class="gpu"><span class="axis3d"></span><span class="visibility3d" role="status"></span><span class="fallback"></span></div><div class="plots"><canvas class="scatter" aria-label="Transverse px py scatter"></canvas><canvas class="scatter" aria-label="Rapidity wrapped phi scatter"></canvas></div>`;
        this.container.append(root);
        const [xy,yp]=root.querySelectorAll('canvas');
        this.cards.set(variant,{root,xy,yp,gpu:root.querySelector('.gpu'),fallback:root.querySelector('.fallback'),count:root.querySelector('small')});
      }
      const card=this.cards.get(variant);card.root.hidden=false;
      const axisLabel=card.root.querySelector('.axis3d');axisLabel.textContent=`Momentum (px, py, pz × ${state.pzScale}) · GeV · box ±${state.axis}`;
      Object.assign(axisLabel.style,{position:'absolute',bottom:'6px',left:'10px',fontSize:'10px',color:'#96adbd',zIndex:'1',pointerEvents:'none'});
      const particles=canvasOnly&&card.particles?card.particles:particlesFor(this.scene,variant,state),accepted=particles.filter(p=>p.accepted).length;
      card.particles=particles;
      const source=this.scene.variants[variant].measured,sourceAccepted=source.accepted.filter(Boolean).length;
      const rejected=source.p4.length-sourceAccepted;
      const sourceLayerVisible=state.frame==='rest'||state.composition!=='background_only';
      card.root.querySelector('.cutnote').textContent=rejected?`${rejected} emitted source particles fail lab cuts.${!sourceLayerVisible?' Source layer hidden in this composition.':state.ghost?' Shown as gray truth ghosts.':' Hidden; tick Rejected daughters to follow them.'}`:'All emitted source particles pass lab cuts.';
      card.count.textContent=`Source ${source.p4.length} emitted → ${sourceAccepted} accepted · ${accepted} displayed · ${particles.length} rendered${state.frame==='rest'?(variant==='event_mixed'?' · DONOR REST FRAMES · TRUTH':' · REST TRUTH'):''}`;
      const reveal=this.reveal.fraction(performance.now());
      drawScatter(card.xy,particles,state,'xy',phiCenter,reveal);drawScatter(card.yp,particles,state,'yphi',phiCenter,reveal);
      card.fallback.textContent=this.gpu?(this.gpu.errors.get(variant)||''):this.gpuError;
      if(this.gpu&&!canvasOnly) this.gpu.update(variant,card.gpu,particles,state,variant==='event_mixed'?[0,0,0]:beta);
    }
  }
  image() {
    // Full-resolution 2D comparison export works even with no GPU/CDN.
    const visible=[...this.cards.values()].filter(c=>!c.root.hidden),width=1200,height=visible.length*410+70;
    const out=document.createElement('canvas');out.width=width;out.height=height;
    const ctx=out.getContext('2d');ctx.fillStyle='#080f18';ctx.fillRect(0,0,width,height);ctx.fillStyle='#e0ecf4';ctx.font='18px system-ui';
    ctx.fillText(`HICAD · ${this.scene.scenario_id} · ${this.state.composition} · ${this.state.frame} · ${this.scene.resolved_request.observation_mode}`,20,30);
    visible.forEach((c,i)=>{const top=60+i*410;ctx.fillStyle='#e0ecf4';ctx.fillText(c.root.querySelector('.viewtitle span').textContent,20,top+15);ctx.drawImage(c.xy,15,top+30,580,350);ctx.drawImage(c.yp,605,top+30,580,350);});
    return out;
  }
}
