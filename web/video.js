// Local mediabunny 1.55.1 (MPL-2.0; original header retained in vendor/mediabunny.mjs).
// Offline timestamps follow simulation frames, independent of wall-clock time.
let mediabunny;
const loadMediabunny = () => mediabunny ??= import(new URL('./vendor/mediabunny.mjs', document.baseURI).href);
const qualities = ['very-low', 'low', 'medium', 'high', 'very-high'];
export const VIDEO_MAX_BYTES = 64 * 1024 * 1024;
const VIDEO_PAYLOAD_BYTES = 60 * 1024 * 1024;

export async function chooseEncoder(canEncodeVideo, width, height, quality) {
    for (const acceleration of ['prefer-hardware', 'no-preference']) {
        const candidates = acceleration === 'prefer-hardware' ? ['avc', 'vp9', 'av1'] : ['avc', 'vp9', 'vp8', 'av1'];
        for (const codec of candidates) {
            if (await canEncodeVideo(codec, { width, height, framerate:60, quality, latencyMode:'quality', hardwareAcceleration:acceleration })) return {codec, acceleration};
        }
    }
    throw new Error('This browser has no supported video encoder. Use native NVENC export or another WebCodecs browser.');
}

// Check before the muxer retains a packet. This bounds compressed payload, not total
// tab RAM: WebCodecs backlog, muxer buffers and the download Blob also need memory.
export function createPayloadBudget(limit = VIDEO_PAYLOAD_BYTES) {
    let bytes = 0;
    return packet => {
        bytes += packet.data.byteLength + (packet.sideData?.alpha?.byteLength ?? 0);
        if (bytes > limit) throw new Error('Video exceeds the compressed-data limit (60 MiB maximum). Use a shorter clip, lower resolution or lower quality.');
        return bytes;
    };
}

function download(bytes, name, type) {
    const url = URL.createObjectURL(new Blob([bytes], {type}));
    const a = document.createElement('a');
    a.href = url; a.download = name; document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
    return name;
}

// Dependency arguments allow deterministic lifecycle tests without camera access or
// a second app runtime. Production always uses local WebCodecs and the hard limits.
export function createVideoService({load = loadMediabunny, deliver = download, maxBytes = VIDEO_MAX_BYTES} = {}) {
    if (!(maxBytes > 0 && maxBytes <= VIDEO_MAX_BYTES)) throw new Error('Invalid video size limit.');
    const sessions = new Map();
    let nextId = 1;
    const closeSource = s => {
        if (!s.closed) { s.closed = true; s.source.close(); }
    };
    const cancelOutput = s => s.cancelTask ??= Promise.resolve().then(() => s.output.cancel()).catch(() => {});
    const fail = (s, error) => {
        s.error ??= error instanceof Error ? error : new Error(String(error));
        s.rejectFailure(s.error);
        // Never throw from WebCodecs' onEncodedPacket callback: it has no caller
        // awaiting it. Pending operations observe the stored terminal failure.
        void cancelOutput(s);
    };
    const check = s => {
        if (s.cancelled) throw new Error('Video recording cancelled.');
        if (s.error) throw s.error;
    };
    async function startVideo(width, height, quality) {
        if (![width, height].every(n => Number.isInteger(n) && n >= 128 && n <= 4096 && n % 2 === 0)) throw new Error('Video dimensions must be even and between 128 and 4096px.');
        const {Output, Mp4OutputFormat, WebMOutputFormat, BufferTarget, VideoSampleSource, Quality, canEncodeVideo} = await load();
        if (typeof VideoEncoder === 'undefined') throw new Error('WebCodecs video encoding is unavailable in this browser.');
        const selectedQuality = new Quality(qualities[quality] ?? 'high');
        const {codec, acceleration} = await chooseEncoder(canEncodeVideo, width, height, selectedQuality);
        const target = new BufferTarget();
        const output = new Output({format: codec === 'vp8' ? new WebMOutputFormat() : new Mp4OutputFormat({fastStart:'in-memory'}), target});
        const s = {width, height, target, output, frames:0, pending:false, finishing:false, cancelled:false, closed:false, error:null, cancelTask:null, webm:codec === 'vp8'};
        s.failure = new Promise((_, reject) => { s.rejectFailure = reject; });
        // A failure can arrive between add() calls; keep the promise handled then.
        s.failure.catch(() => {});
        const budget = createPayloadBudget(Math.min(VIDEO_PAYLOAD_BYTES, maxBytes));
        target.on('write', ({end}) => {
            if (end > maxBytes) throw new Error('Video exceeds the file-size limit (64 MiB maximum). Use a smaller recording.');
        });
        s.source = new VideoSampleSource({codec, quality:selectedQuality, latencyMode:'quality', hardwareAcceleration:acceleration,
            onEncodedPacket(packet) { try { budget(packet); } catch (error) { fail(s, error); } }});
        output.addVideoTrack(s.source, {frameRate:60});
        try { await output.start(); check(s); }
        catch (error) { await cancelOutput(s); throw error; }
        const id = nextId++;
        sessions.set(id, s);
        return id;
    }
    async function addVideoFrame(id, rgba) {
        const s = sessions.get(id);
        if (!s) throw new Error('Recording session is no longer active.');
        check(s);
        if (s.pending || s.finishing) throw new Error('Recording backpressure violation: a frame or finalization is still pending.');
        if (rgba.byteLength !== s.width * s.height * 4) throw new Error('Recording frame dimensions changed.');
        s.pending = true;
        let sample;
        let adding;
        try {
            const {VideoSample} = await load();
            check(s);
            sample = new VideoSample(rgba, {format:'RGBA', codedWidth:s.width, codedHeight:s.height, timestamp:s.frames / 60, duration:1 / 60});
            adding = Promise.resolve(s.source.add(sample)).finally(() => sample.close());
            await Promise.race([adding, s.failure]);
            check(s);
            s.frames++;
        } catch (error) {
            fail(s, error);
            throw s.error;
        } finally { if (!adding) sample?.close(); s.pending = false; }
    }
    async function finishVideo(id) {
        const s = sessions.get(id);
        if (!s) throw new Error('Recording session is no longer active.');
        if (s.pending || s.finishing) throw new Error('Cannot finalize while a frame or finalization is still pending.');
        s.finishing = true;
        try {
            check(s);
            if (!s.frames) { await cancelOutput(s); return ''; }
            closeSource(s);
            // Keep ownership until this settles. Mediabunny cannot interrupt an
            // in-progress finalize; discard marks it immediately and suppresses delivery.
            await s.output.finalize();
            check(s);
            return deliver(s.target.buffer, `primordial-soup-${Date.now()}.${s.webm ? 'webm' : 'mp4'}`, s.webm ? 'video/webm' : 'video/mp4');
        } catch (error) {
            fail(s, error);
            await cancelOutput(s);
            throw s.error;
        } finally {
            sessions.delete(id);
        }
    }
    async function cancelVideo(id) {
        const s = sessions.get(id);
        if (!s) return;
        s.cancelled = true;
        s.rejectFailure(new Error('Video recording cancelled.'));
        if (!s.finishing) sessions.delete(id);
        await cancelOutput(s);
    }
    return {startVideo, addVideoFrame, finishVideo, cancelVideo};
}

const service = createVideoService();
export const {startVideo, addVideoFrame, finishVideo, cancelVideo} = service;
