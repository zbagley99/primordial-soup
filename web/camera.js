// Camera stays off until the application Start button calls cameraStart.
let video = null, stream = null, generation = 0, frame = 0, error = '', running = false, starting = false;
let devices = [], devicesError = '', devicesRevision = 0, discovery = 0;
export function cameraFailure(failure, selected = false) {
  switch (failure?.name) {
    case 'NotFoundError': return 'No usable webcam is available. Connect a webcam, or start your virtual camera feed, then refresh the device list. A webcam is not needed to record the world.';
    case 'NotAllowedError': case 'PermissionDeniedError': return 'Camera access was denied. Allow camera access for this site in the browser, then press Start webcam again.';
    case 'NotReadableError': case 'TrackStartError': return 'The webcam could not be opened. Close other apps using it and check the camera driver or virtual camera feed.';
    case 'OverconstrainedError': return selected ? 'The selected webcam is unavailable or cannot provide a compatible stream. Refresh the device list and choose Default or another webcam.' : 'The webcam cannot provide a compatible stream. Try another device or check the virtual camera output.';
    case 'AbortError': return 'Webcam startup was interrupted. Try Start webcam again.';
    default: return `Webcam: ${failure?.message ?? failure}`;
  }
}
export async function cameraRefreshDevices() {
  const request = ++discovery;
  try {
    if (!navigator.mediaDevices?.enumerateDevices) throw new Error('Device discovery requires HTTPS or localhost and browser camera support.');
    const list = await navigator.mediaDevices.enumerateDevices();
    if (request !== discovery) return;
    devices = list.filter(d => d.kind === 'videoinput' && d.deviceId).map((d, i) => ({ id: d.deviceId, label: d.label || `Webcam ${i + 1}` }));
    devicesError = '';
  } catch (failure) { if (request === discovery) { devices = []; devicesError = cameraFailure(failure); } }
  if (request === discovery) devicesRevision++;
}
export function cameraDevices() { return JSON.stringify(devices); }
export function cameraDevicesError() { return devicesError; }
export function cameraDevicesRevision() { return devicesRevision; }
export async function cameraStart(facing, deviceId = '') {
  cameraStop();
  const request = generation;
  error = ''; starting = true;
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera capture requires HTTPS or localhost and browser camera support.');
    const constraints = { ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: { ideal: facing } }), width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15, max: 30 } };
    let acquired;
    try { acquired = await navigator.mediaDevices.getUserMedia({ audio: false, video: constraints }); }
    catch (failure) {
      // A device may reject the rate ceiling. Never retry permission denial,
      // or silently replace an explicitly selected webcam with another device.
      if (failure.name !== 'OverconstrainedError' || deviceId || request !== generation) throw failure;
      acquired = await navigator.mediaDevices.getUserMedia({ audio: false, video: { ...constraints, frameRate: { ideal: 15 } } });
    }
    if (request !== generation) { acquired.getTracks().forEach(track => track.stop()); return; }
    stream = acquired;
    video = document.createElement('video');
    video.muted = true; video.playsInline = true; video.srcObject = stream;
    await video.play();
    if (request !== generation) return;
    running = true; starting = false;
    void cameraRefreshDevices();
    for (const track of stream.getTracks()) track.addEventListener('ended', () => { if (request === generation) { cameraStop(); error = 'The camera stream ended. Start it again or choose another camera.'; } });
    if (video.requestVideoFrameCallback) {
      const tick = () => { if (request !== generation || !video) return; frame++; video.requestVideoFrameCallback(tick); };
      video.requestVideoFrameCallback(tick);
    }
  } catch (failure) {
    if (request !== generation) return;
    cameraStop();
    error = cameraFailure(failure, !!deviceId);
    void cameraRefreshDevices();
  }
}
export function cameraStop() {
  generation++; running = false; starting = false;
  stream?.getTracks().forEach(track => track.stop()); stream = null;
  if (video) { video.pause(); video.srcObject = null; video = null; }
}
export function cameraRunning() { return running; }
export function cameraStarting() { return starting; }
export function cameraError() { return error; }
export function cameraVideo() { return running && video?.readyState >= 2 ? video : null; }
export function cameraWidth() { return video?.videoWidth ?? 0; }
export function cameraHeight() { return video?.videoHeight ?? 0; }
export function cameraFrame() { return video?.requestVideoFrameCallback ? frame : (video?.currentTime ?? 0); }
window.addEventListener('pagehide', cameraStop);
navigator.mediaDevices?.addEventListener?.('devicechange', () => { void cameraRefreshDevices(); });

// The same GPU external-image operation as WGPU's wrapper, with transient frame exceptions caught.
export function cameraCopy(queue, texture, video, width, height) {
  try {
    queue.copyExternalImageToTexture({ source: video }, { texture }, { width, height });
    return true;
  } catch (failure) {
    console.debug('Camera frame skipped:', failure);
    return false;
  }
}
