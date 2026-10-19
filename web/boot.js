import init, { start } from './primordial_soup.js';
import './image-import.js';
const parameters = new URLSearchParams(location.search);
if (parameters.has('debug')) document.body.classList.add('debug');
document.querySelector('#reload').addEventListener('click', () => location.reload());
function fail(error) {
  console.error(error);
  document.querySelector('#loading').hidden = true;
  document.querySelector('#error').hidden = false;
  document.querySelector('#detail').textContent = String(error?.message ?? error);
}
try {
  if (!navigator.gpu) throw new Error('This browser has no WebGPU API. The simulation needs a hardware GPU and a secure HTTPS or localhost page.');
  const wasm = await init({ module_or_path: new URL('./primordial_soup_bg.wasm', import.meta.url) });
  await start(document.querySelector('#soup'));
  document.querySelector('#loading').hidden = true;
  if (parameters.has('debug')) {
    const output = document.querySelector('#memory');
    let previousRaf;
    const rafIntervals = [];
    const sampleRaf = (now) => {
      if (previousRaf !== undefined) {
        rafIntervals.push(now - previousRaf);
        if (rafIntervals.length > 60) rafIntervals.shift();
      }
      previousRaf = now;
      requestAnimationFrame(sampleRaf);
    };
    requestAnimationFrame(sampleRaf);
    let previousMemoryUpdate;
    const updateMemory = () => {
      const now = performance.now();
      if (previousMemoryUpdate !== undefined) output.dataset.timerIntervalMs = String(now - previousMemoryUpdate);
      previousMemoryUpdate = now;
      output.dataset.visibility = document.visibilityState;
      output.dataset.focused = String(document.hasFocus());
      if (rafIntervals.length) output.dataset.rafIntervalMs = String(rafIntervals.reduce((a,b)=>a+b,0) / rafIntervals.length);
      const wasmBytes = wasm.memory.buffer.byteLength;
      const jsBytes = performance.memory?.usedJSHeapSize;
      output.textContent = `WASM linear memory ${(wasmBytes / 1048576).toFixed(1)} MiB` +
        (Number.isFinite(jsBytes) ? ` · JS heap ${(jsBytes / 1048576).toFixed(1)} MiB` : '');
      output.dataset.wasmBytes = String(wasmBytes);
      if (Number.isFinite(jsBytes)) output.dataset.jsHeapBytes = String(jsBytes);
    };
    updateMemory();
    setInterval(updateMemory, 5000);
  }
} catch (error) { fail(error); }
window.addEventListener('unhandledrejection', event => fail(event.reason));
