import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createVideoService, createPayloadBudget, VIDEO_MAX_BYTES} from '../web/video.js';

const cases=[];
const test=(name,run)=>cases.push({name,run});
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
const pixels=()=>new Uint8Array(128*128*4);
globalThis.VideoEncoder=class {};
function fixture(options={}) {
    const sources=[],outputs=[],samples=[],downloads=[];
    class BufferTarget { buffer=new Uint8Array([1,2,3]).buffer;on(_,callback){this.onWrite=callback;} }
    class Source {
        constructor(config){this.config=config;this.closed=0;sources.push(this);}
        async add(sample){if(options.addGate)await options.addGate.promise;this.config.onEncodedPacket({data:new Uint8Array(options.packetBytes??4)});if(options.addError)throw options.addError;}
        close(){if(this.output.state==='pending')throw new Error('Cannot call close before output has been started.');this.closed++;this.graceful=true;}
    }
    class Output {
        constructor({target}){this.target=target;this.cancelled=0;this.state='pending';outputs.push(this);}
        addVideoTrack(source){this.source=source;source.output=this;}
        async start(){if(options.startError)throw options.startError;this.state='started';}
        async finalize(){this.state='finalizing';if(options.finishGate)await options.finishGate.promise;if(options.finishError)throw options.finishError;this.target.onWrite({end:options.containerBytes??3});}
        async cancel(){this.cancelled++;if(this.state==='finalizing')return;if(this.source.graceful&&options.addGate)await options.addGate.promise;this.state='canceled';}
    }
    class Sample {constructor(_,timing){this.timing=timing;this.closed=0;samples.push(this);}close(){this.closed++;}}
    const api={Output,BufferTarget,VideoSampleSource:Source,VideoSample:Sample,Quality:class {},Mp4OutputFormat:class {},WebMOutputFormat:class {},canEncodeVideo:async()=>true};
    const service=createVideoService({load:async()=>api,deliver:(bytes,name,type)=>{downloads.push({bytes,name,type});return name;},maxBytes:options.maxBytes??VIDEO_MAX_BYTES});
    return {...service,sources,outputs,samples,downloads};
}
test('video frames use fixed 60Hz timestamps, close samples and download MP4',async()=>{
    const f=fixture(),id=await f.startVideo(128,128,3);
    await f.addVideoFrame(id,pixels());await f.addVideoFrame(id,pixels());
    assert.deepEqual(f.samples.map(x=>x.timing.timestamp),[0,1/60]);
    assert.ok(f.samples.every(x=>x.closed===1&&x.timing.duration===1/60));
    assert.match(await f.finishVideo(id),/\.mp4$/);assert.equal(f.downloads[0].type,'video/mp4');
    await assert.rejects(f.addVideoFrame(id,pixels()),/no longer active/);
});
test('zero-frame finalization cancels without a download',async()=>{
    const f=fixture(),id=await f.startVideo(128,128,3);assert.equal(await f.finishVideo(id),'');
    assert.equal(f.downloads.length,0);assert.equal(f.outputs[0].cancelled,1);
});
test('discard during finalization suppresses download and permits an immediate independent restart',async()=>{
    const gate=deferred(),f=fixture({finishGate:gate}),id=await f.startVideo(128,128,3);
    await f.addVideoFrame(id,pixels());const finished=f.finishVideo(id);
    await f.cancelVideo(id);const newer=await f.startVideo(128,128,3);
    gate.resolve();await assert.rejects(finished,/cancelled/);assert.equal(f.downloads.length,0);
    await f.addVideoFrame(newer,pixels());await f.finishVideo(newer);assert.equal(f.downloads.length,1);
});
test('discard rejects a pending frame without closing its sample before add settles',async()=>{
    const gate=deferred(),f=fixture({addGate:gate}),id=await f.startVideo(128,128,3);
    const adding=f.addVideoFrame(id,pixels());await Promise.resolve();await Promise.resolve();
    assert.equal(f.samples.length,1);await f.cancelVideo(id);await assert.rejects(adding,/cancelled/);
    assert.equal(f.samples[0].closed,0);assert.equal(f.sources[0].closed,0);assert.equal(f.outputs[0].state,'canceled');gate.resolve();await Promise.resolve();await Promise.resolve();await Promise.resolve();
    assert.equal(f.samples[0].closed,1);assert.equal(f.downloads.length,0);
});
test('finalization failure cancels, removes the session and never reports a download',async()=>{
    const f=fixture({finishError:new Error('mux failed')}),id=await f.startVideo(128,128,3);
    await f.addVideoFrame(id,pixels());await assert.rejects(f.finishVideo(id),/mux failed/);
    assert.equal(f.outputs[0].cancelled,1);assert.equal(f.downloads.length,0);
    await assert.rejects(f.finishVideo(id),/no longer active/);
});
test('payload overflow is handled before finalization, including an asynchronous encoder callback',async()=>{
    const f=fixture({maxBytes:10}),id=await f.startVideo(128,128,3);
    await f.addVideoFrame(id,pixels());
    assert.doesNotThrow(()=>f.sources[0].config.onEncodedPacket({data:new Uint8Array(7)}));
    await assert.rejects(f.addVideoFrame(id,pixels()),/compressed-data limit/);
    await assert.rejects(f.finishVideo(id),/compressed-data limit/);
    assert.equal(f.outputs[0].cancelled,1);assert.equal(f.downloads.length,0);
});
test('payload failure releases an operation blocked by encoder backpressure',async()=>{
    const gate=deferred(),f=fixture({maxBytes:10,addGate:gate}),id=await f.startVideo(128,128,3);
    const adding=f.addVideoFrame(id,pixels());await Promise.resolve();await Promise.resolve();
    f.sources[0].config.onEncodedPacket({data:new Uint8Array(11)});
    await assert.rejects(adding,/compressed-data limit/);await f.cancelVideo(id);gate.resolve();
});
test('container overflow and failed startup clean up without downloading',async()=>{
    const f=fixture({maxBytes:10,containerBytes:11}),id=await f.startVideo(128,128,3);
    await f.addVideoFrame(id,pixels());await assert.rejects(f.finishVideo(id),/file-size limit/);
    assert.equal(f.downloads.length,0);assert.equal(f.outputs[0].cancelled,1);
    const failed=fixture({startError:new Error('start failed')});await assert.rejects(failed.startVideo(128,128,3),/start failed/);
    assert.equal(failed.outputs[0].cancelled,1);assert.equal(failed.sources[0].closed,0);
});
test('dimensions, hard budget and overlapping operations are rejected',async()=>{
    assert.throws(()=>createVideoService({maxBytes:VIDEO_MAX_BYTES+1}),/limit/);
    const gate=deferred(),f=fixture({addGate:gate});await assert.rejects(f.startVideo(129,128,3),/dimensions/);
    const id=await f.startVideo(128,128,3),adding=f.addVideoFrame(id,pixels());
    await assert.rejects(f.addVideoFrame(id,pixels()),/backpressure/);
    await assert.rejects(f.finishVideo(id),/pending/);gate.resolve();await adding;await f.cancelVideo(id);
});
test('packet budget includes alpha side data',async()=>{
    const budget=createPayloadBudget(10);assert.equal(budget({data:new Uint8Array(4),sideData:{alpha:new Uint8Array(4)}}),8);
    assert.throws(()=>budget({data:new Uint8Array(3)}),/compressed-data limit/);
});
test('the actual pinned muxer remuxes H264 and a small payload budget stops before the complete clip',async()=>{
    const {Input,ALL_FORMATS,BufferSource,EncodedPacketSink,EncodedVideoPacketSource,Output,BufferTarget,Mp4OutputFormat}=await import('../web/vendor/mediabunny.mjs');
    const input=new Input({source:new BufferSource(await readFile(new URL('./fixtures/recording-h264.mp4',import.meta.url))),formats:ALL_FORMATS});
    try {
        const track=await input.getPrimaryVideoTrack(),config=await track.getDecoderConfig();
        const packets=[];for await(const packet of new EncodedPacketSink(track).packets())packets.push(packet);
        assert.equal(packets.length,60);
        const mux=()=>{const source=new EncodedVideoPacketSource('avc'),target=new BufferTarget(),output=new Output({format:new Mp4OutputFormat({fastStart:'in-memory'}),target});output.addVideoTrack(source,{frameRate:60});return {source,target,output};};
        const full=mux();await full.output.start();const budget=createPayloadBudget();
        for(const packet of packets){budget(packet);await full.source.add(packet,{decoderConfig:config});}
        full.source.close();await full.output.finalize();assert.ok(full.target.buffer.byteLength>0);
        const limited=mux();await limited.output.start();const small=createPayloadBudget(packets[0].data.byteLength+packets[1].data.byteLength-1);
        let inserted=0;await assert.rejects(async()=>{for(const packet of packets){small(packet);await limited.source.add(packet,{decoderConfig:config});inserted++;}},/compressed-data limit/);
        assert.equal(inserted,1);await limited.output.cancel();
    } finally {input.dispose();}
});
for(const {name,run} of cases){await run();console.log(`PASS ${name}`);}
console.log(`${cases.length} video checks passed`);
