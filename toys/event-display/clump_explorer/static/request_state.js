// One in-flight request + one replaceable pending draft. Independent of DOM/GPU.
export class LatestRequestQueue {
  constructor({send, commit, error, status, delay=200, retryDelay=400}) {
    Object.assign(this,{send,commit,error,status,delay,retryDelay});
    this.sequence=0; this.pending=null; this.busy=false; this.timer=null;
  }
  invalidate() { this.sequence++; this.pending=null; clearTimeout(this.timer); return this.sequence; }
  draft(spec, auto=true, immediate=false) {
    const seq=this.invalidate();
    this.status('Updating · draft differs from committed event');
    if(auto) {
      this.pending={spec:structuredClone(spec),seq};
      this.timer=setTimeout(()=>this.pump(),immediate?0:this.delay);
    }
    return seq;
  }
  flush() { clearTimeout(this.timer); return this.pump(); }
  async pump() {
    if(this.busy || !this.pending) return;
    clearTimeout(this.timer);
    const job=this.pending; this.pending=null; this.busy=true;
    try {
      const response=await this.send({...job.spec,request_id:job.seq});
      if(job.seq===this.sequence && response.request_id===job.seq) this.commit(response);
    } catch(e) {
      if(job.seq===this.sequence) {
        if(e.status===503) {
          this.pending=job;
          this.status('Service busy · retrying latest draft');
          this.timer=setTimeout(()=>this.pump(),this.retryDelay);
        } else { this.error(e); this.status('Update failed · previous event retained'); }
      }
    } finally {
      this.busy=false;
      // Retry honors its delay; a newer pending state runs immediately.
      if(this.pending && (this.pending.seq!==job.seq || !this.timer)) this.pump();
    }
  }
}
