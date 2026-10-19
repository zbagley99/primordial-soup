import assert from 'node:assert/strict';
const cases=[];
const test=(name,run)=>cases.push({name,run});
const moduleUrl = new URL('../web/services.js',import.meta.url);
const fresh = (name) => import(`${moduleUrl}?case=${name}`);
test('clipboard rejection produces a PNG download and truthful result', async()=>{
    const downloads=[];
    Object.defineProperty(globalThis,'navigator',{configurable:true,value:{clipboard:{write:async()=>{throw new Error('permission denied');}}}});
    globalThis.ClipboardItem=class {constructor(value){this.value=value;}};
    globalThis.document={body:{append(){}},createElement(){return {click(){downloads.push(this.download);},remove(){}};}};
    const realTimer=globalThis.setTimeout;
    globalThis.setTimeout=(fn)=>{queueMicrotask(fn);return 0;};
    try {
        const {copyPng}=await fresh('clipboard');
        const message=await copyPng(new Uint8Array([137,80,78,71]),'fixture.png');
        assert.match(message,/Clipboard unavailable.*downloaded PNG instead/);
        assert.deepEqual(downloads,['fixture.png']);
    } finally {globalThis.setTimeout=realTimer;}
});
test('blocked project storage rejects instead of reporting a save', async()=>{
    globalThis.indexedDB={open(){const request={};queueMicrotask(()=>request.onblocked());return request;}};
    const {loadCore,saveCore}=await fresh('blocked');
    await assert.rejects(loadCore(),/blocked by another tab/);
    await assert.rejects(saveCore('{}'),/blocked by another tab/);
});
test('quota failure rejects the transaction and preserves separate archive storage', async()=>{
    const databases=[];
    globalThis.indexedDB={open(name){databases.push(name);const request={result:{createObjectStore(){},transaction(){const tx={error:new Error('QuotaExceededError'),objectStore(){return {put(){queueMicrotask(()=>tx.onabort());},add(){queueMicrotask(()=>tx.oncomplete());}};}};return tx;}}};queueMicrotask(()=>request.onsuccess());return request;}};
    const {saveCore,archiveAppend}=await fresh('quota');
    await assert.rejects(saveCore('{}'),/QuotaExceededError/);
    await archiveAppend('{"event":1}');
    assert.deepEqual(databases,['primordial-soup-projects','primordial-soup-archive']);
});
test('archive writes finish in order and export waits for pending writes', async()=>{
    const events=[];
    globalThis.indexedDB={open(){const request={result:{createObjectStore(){},transaction(){const tx={objectStore(){return {add(text){queueMicrotask(()=>{events.push(text);tx.oncomplete();});},getAll(){const r={};queueMicrotask(()=>{r.result=events.slice();r.onsuccess();});return r;}};}};return tx;}}};queueMicrotask(()=>request.onsuccess());return request;}};
    const {archiveAppend,archiveRead}=await fresh('serial');
    const first=archiveAppend('{"n":1}'); const second=archiveAppend('{"n":2}');
    assert.equal(await archiveRead(),'[{"n":1},{"n":2}]'); await Promise.all([first,second]);
});
test('video selection keeps hardware preference and survives its rejection', async()=>{
    const {chooseEncoder}=await import('../web/video.js');
    const quality={};
    const hardware=await chooseEncoder(async(codec,options)=>{
        assert.deepEqual(options,{width:128,height:128,framerate:60,quality,latencyMode:'quality',hardwareAcceleration:'prefer-hardware'});
        return codec==='avc';
    },128,128,quality);
    assert.deepEqual(hardware,{codec:'avc',acceleration:'prefer-hardware'});
    const fallback=await chooseEncoder(async(codec,options)=>codec==='vp9'&&options.hardwareAcceleration==='no-preference',128,128,{});
    assert.deepEqual(fallback,{codec:'vp9',acceleration:'no-preference'});
    await assert.rejects(chooseEncoder(async()=>false,128,128,{}),/no supported video encoder/);
});
for(const {name,run} of cases) { await run(); console.log(`PASS ${name}`); }
assert.equal(cases.length,5);
