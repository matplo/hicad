// Optional module. Any import/init failure leaves the Canvas explorer running.
import * as THREE from 'three/webgpu';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {particleColor,fitDistanceForSphere,sourceViewportCounts} from './renderer.js';

export async function create3D() {
  if(!navigator.gpu) throw new Error('3D unavailable: WebGPU is not available. Canvas views remain active.');
  const views=new Map();let linking=false,replayStart=0,disposedBuffers=0;
  async function make(key,host) {
    const renderer=new THREE.WebGPURenderer({antialias:true});
    renderer.setPixelRatio(Math.min(devicePixelRatio,2));
    await renderer.init();host.append(renderer.domElement);
    const scene=new THREE.Scene();scene.background=new THREE.Color('#0b1621');
    const camera=new THREE.PerspectiveCamera(42,1,.01,500);camera.position.set(4,3,5);
    const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
    const other=[...views.values()][0];
    if(other) {camera.position.copy(other.camera.position);camera.quaternion.copy(other.camera.quaternion);controls.target.copy(other.controls.target);}
    // Match the legacy WebGPU display's instanced octahedron geometry.
    const material=new THREE.MeshBasicNodeMaterial({color:0xffffff});
    const geometry=new THREE.OctahedronGeometry(1,0),points=new THREE.InstancedMesh(geometry,material,16);
    points.count=0;points.frustumCulled=false;points.instanceMatrix.setUsage(THREE.DynamicDrawUsage);scene.add(points);
    const v={renderer,scene,camera,controls,points,geometry,material,capacity:16,axis:0,particles:[],state:null,dirty:true,visibilityDirty:true,radius:0,viewportCounts:null};views.set(key,v);
    controls.addEventListener('change',()=>{v.visibilityDirty=true;if(linking)return;linking=true;for(const other of views.values())if(other!==v){other.camera.position.copy(camera.position);other.camera.quaternion.copy(camera.quaternion);other.controls.target.copy(controls.target);other.visibilityDirty=true;}linking=false;});
    return v;
  }
  const pending=new Map();
  const api={
    update(key,host,particles,state,beta) {
      if(api.failed.has(key))return;
      let v=views.get(key);
      if(!v) {
        if(!pending.has(key)) pending.set(key,make(key,host).then(()=>{pending.delete(key);const latest=api.latest.get(key);if(latest)api.update(key,...latest);}).catch(e=>{pending.delete(key);const message=`3D unavailable: ${e.message}. Canvas views remain active.`;host.querySelector('.fallback').textContent=message;api.failed.add(key);api.errors.set(key,message);}));
        api.latest.set(key,[host,particles,state,beta]);return;
      }
      api.latest.set(key,[host,particles,state,beta]);
      v.particles=particles;v.state=state;v.dirty=true;v.visibilityDirty=true;
      v.radius=0;for(const p of particles)v.radius=Math.max(v.radius,Math.hypot(p.p4[1],p.p4[2],p.p4[3]*state.pzScale));
      if(particles.length>v.capacity) {
        const capacity=2**Math.ceil(Math.log2(Math.max(16,particles.length)));
        v.scene.remove(v.points);v.points.dispose();disposedBuffers++;
        v.points=new THREE.InstancedMesh(v.geometry,v.material,capacity);v.points.frustumCulled=false;v.points.instanceMatrix.setUsage(THREE.DynamicDrawUsage);v.scene.add(v.points);v.capacity=capacity;
      }
      const color=new THREE.Color();
      particles.forEach((p,i)=>{color.set(particleColor(p,state));v.points.setColorAt(i,color);});
      if(v.points.instanceColor)v.points.instanceColor.needsUpdate=true;
      v.points.count=particles.length;
      if(v.axis!==state.axis || v.pzScale!==state.pzScale) {
        if(v.box){v.scene.remove(v.box);v.box.geometry.dispose();v.box.material.dispose();}
        const a=state.axis;v.box=new THREE.Box3Helper(new THREE.Box3(new THREE.Vector3(-a,-a,-a),new THREE.Vector3(a,a,a)),0x29485a);v.scene.add(v.box);
        v.axis=state.axis;v.pzScale=state.pzScale;
      }
      const showArrow=state.color==='truth' && state.frame==='lab' && Math.hypot(...beta)>0;
      if(showArrow) {
        const direction=new THREE.Vector3(beta[0],beta[1],beta[2]*state.pzScale).normalize();
        if(!v.arrow){v.arrow=new THREE.ArrowHelper(direction,new THREE.Vector3(),state.axis*.8,0xf2be63);v.scene.add(v.arrow);}
        else {v.arrow.setDirection(direction);v.arrow.setLength(state.axis*.8);}
      }
      if(v.arrow)v.arrow.visible=showArrow;
    },latest:new Map(),failed:new Set(),errors:new Map(),replay(start=performance.now()){replayStart=start;},
    stopReplay(){replayStart=0;for(const v of views.values())v.dirty=true;},
    fit(axis){
      const active=[...views.values()].filter(v=>!v.renderer.domElement.closest('[hidden]'));
      const radius=Math.max(Math.sqrt(3)*axis,...active.map(v=>v.radius));
      const distance=Math.max(.1,...active.map(v=>fitDistanceForSphere(radius,v.camera.fov,v.camera.aspect)));
      const first=active[0];if(!first)return;
      const direction=first.camera.position.clone().sub(first.controls.target);
      if(direction.lengthSq()===0)direction.set(4,3,5);direction.normalize().multiplyScalar(distance);
      linking=true;
      for(const v of views.values()){v.camera.position.copy(direction);v.controls.target.set(0,0,0);v.controls.update();v.visibilityDirty=true;}
      linking=false;
    },
    stats(){return {views:views.size,capacities:[...views.values()].map(v=>v.capacity),disposedBuffers,viewports:Object.fromEntries([...views].map(([key,v])=>[key,v.viewportCounts]))};}
  };
  const clip=new THREE.Vector4(),projection=new THREE.Matrix4();
  function animate(now) {
    const fraction=replayStart?Math.min(1,(now-replayStart)/700):1;
    if(fraction===1)replayStart=0;
    for(const v of views.values()) {
      const canvas=v.renderer.domElement,host=canvas.parentElement;
      if(!host||host.closest('[hidden]'))continue;
      const w=host.clientWidth,h=host.clientHeight;
      if(w<1||h<1)continue;
      if(canvas.width!==Math.round(w*v.renderer.getPixelRatio())||canvas.height!==Math.round(h*v.renderer.getPixelRatio())) {v.renderer.setSize(w,h,false);v.camera.aspect=w/h;v.camera.updateProjectionMatrix();v.visibilityDirty=true;}
      const positionsChanged=v.dirty||replayStart||v.wasReplaying;
      if(positionsChanged) {
        const matrix=new THREE.Matrix4().makeScale(.025,.025,.025);
        v.particles.forEach((p,i)=>{matrix.setPosition(p.p4[1]*fraction,p.p4[2]*fraction,p.p4[3]*v.state.pzScale*fraction);v.points.setMatrixAt(i,matrix);});
        v.points.instanceMatrix.needsUpdate=true;v.dirty=false;v.wasReplaying=!!replayStart;
      }
      v.controls.update();
      const far=Math.max(500,(v.camera.position.length()+Math.max(v.radius,Math.sqrt(3)*v.axis)+10)*1.1);
      if(v.camera.far!==far){v.camera.far=far;v.camera.updateProjectionMatrix();v.visibilityDirty=true;}
      v.renderer.render(v.scene,v.camera);
      if(v.visibilityDirty||positionsChanged){
        v.camera.updateMatrixWorld();projection.multiplyMatrices(v.camera.projectionMatrix,v.camera.matrixWorldInverse);
        v.viewportCounts=sourceViewportCounts(v.particles,p=>clip.set(p.p4[1]*fraction,p.p4[2]*fraction,p.p4[3]*v.state.pzScale*fraction,1).applyMatrix4(projection),v.camera.coordinateSystem===THREE.WebGPUCoordinateSystem);
        const counts=v.viewportCounts,note=host.querySelector('.visibility3d');
        note.textContent=counts.accepted?`3D source centers: ${counts.inView}/${counts.accepted} in view · ${counts.offCamera} off-camera${counts.offCamera?' · use Fit compared union':''}`:'No accepted source points in this composition';
        v.visibilityDirty=false;
      }
    }
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
  return api;
}
