import assert from 'node:assert/strict';
const cases=[];
const test=(name,run)=>cases.push({name,run});
let id=0;
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
const failure=name=>Object.assign(new Error(name),{name});
const track=()=>({stopped:0,events:{},stop(){this.stopped++;},addEventListener(name,fn){this.events[name]=fn;}});
const stream=t=>({getTracks:()=>[t]});
async function setup(media,play=async()=>{}) {
    const events={};
    Object.defineProperty(globalThis,'navigator',{configurable:true,value:{mediaDevices:{enumerateDevices:async()=>[],addEventListener(name,fn){events[name]=fn;},...media}}});
    globalThis.window={addEventListener(){}};
    globalThis.document={createElement:()=>({play,pause(){},readyState:2,videoWidth:640,videoHeight:480,currentTime:0})};
    const module=await import(`../web/camera.js?test=${id++}`);
    return {module,events};
}
test('discovery never activates capture and keeps default usable with concealed devices',async()=>{
    let starts=0;
    const {module:m}=await setup({getUserMedia(){starts++;throw new Error('unexpected capture');}});
    await m.cameraRefreshDevices();
    assert.equal(m.cameraDevices(),'[]');assert.equal(starts,0);assert.equal(m.cameraRunning(),false);
});
test('latest discovery wins and filters audio without exposing group IDs',async()=>{
    const a=deferred(),b=deferred();let count=0;
    const {module:m}=await setup({enumerateDevices:()=>count++?b.promise:a.promise});
    const first=m.cameraRefreshDevices(),second=m.cameraRefreshDevices();
    b.resolve([{kind:'audioinput',deviceId:'audio'},{kind:'videoinput',deviceId:'new',label:'USB webcam',groupId:'private'}]);await second;
    a.resolve([{kind:'videoinput',deviceId:'old',label:'Removed webcam'}]);await first;
    assert.deepEqual(JSON.parse(m.cameraDevices()),[{id:'new',label:'USB webcam'}]);
});
test('permission denial is actionable and is never retried',async()=>{
    let calls=0;const {module:m}=await setup({getUserMedia:async()=>{calls++;throw failure('NotAllowedError');}});
    await m.cameraStart('user');assert.equal(calls,1);assert.match(m.cameraError(),/access was denied/);assert.equal(m.cameraStarting(),false);
});
test('missing webcam is distinct from simulation recording',async()=>{
    const {module:m}=await setup({getUserMedia:async()=>{throw failure('NotFoundError');}});
    await m.cameraStart('user');assert.match(m.cameraError(),/No usable webcam/);assert.match(m.cameraError(),/not needed to record the world/);
});
test('default retries only an incompatible rate ceiling',async()=>{
    const t=track();const calls=[];
    const {module:m}=await setup({getUserMedia:async c=>{calls.push(c);if(calls.length===1)throw failure('OverconstrainedError');return stream(t);}});
    await m.cameraStart('environment');assert.equal(calls.length,2);assert.equal(calls[1].video.frameRate.max,undefined);assert.equal(m.cameraRunning(),true);
    m.cameraStop();assert.equal(t.stopped,1);
});
test('explicit device stays explicit and never silently falls back',async()=>{
    const calls=[];const {module:m}=await setup({getUserMedia:async c=>{calls.push(c);throw failure('OverconstrainedError');}});
    await m.cameraStart('user','chosen-camera');assert.equal(calls.length,1);assert.deepEqual(calls[0].video.deviceId,{exact:'chosen-camera'});assert.equal(calls[0].video.facingMode,undefined);
    assert.match(m.cameraError(),/selected webcam/);
});
test('a late acquired stream after stop is released without playback',async()=>{
    const d=deferred(),t=track();let plays=0;
    const {module:m}=await setup({getUserMedia:()=>d.promise},async()=>{plays++;});
    const start=m.cameraStart('user');m.cameraStop();d.resolve(stream(t));await start;
    assert.equal(t.stopped,1);assert.equal(plays,0);assert.equal(m.cameraRunning(),false);
});
test('late playback rejection cannot stop a newer session',async()=>{
    const wait=deferred(),a=track(),b=track();let starts=0,plays=0;
    const {module:m}=await setup({getUserMedia:async()=>stream(starts++?b:a)},()=>plays++?Promise.resolve():wait.promise);
    const first=m.cameraStart('user');await Promise.resolve();await Promise.resolve();
    await m.cameraStart('user');wait.reject(failure('AbortError'));await first;
    assert.equal(m.cameraRunning(),true);assert.equal(m.cameraError(),'');assert.equal(a.stopped,1);assert.equal(b.stopped,0);m.cameraStop();
});
test('ended track releases capture and refreshes occur on device changes',async()=>{
    const t=track();let lists=0;
    const {module:m,events}=await setup({getUserMedia:async()=>stream(t),enumerateDevices:async()=>{lists++;return [];}});
    await m.cameraStart('user');await Promise.resolve();events.devicechange();await Promise.resolve();
    assert.ok(lists>=2);t.events.ended();assert.equal(m.cameraRunning(),false);assert.equal(t.stopped,1);assert.match(m.cameraError(),/stream ended/);
});
for(const {name,run} of cases){await run();console.log(`PASS ${name}`);}
console.log(`${cases.length} webcam service checks passed; no physical camera opened.`);
