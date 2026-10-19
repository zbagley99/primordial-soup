// Freeze the reference's actual fpsBand.ts outputs plus main.ts's two read windows.
// Usage: node tests/fixtures/performance.generate.mjs /path/to/Fluoddity-Web
import { pathToFileURL } from 'node:url';
import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
const { startBand, stepBand, fpsFrom } = await import(pathToFileURL(resolve(process.argv[2], 'src/perf/fpsBand.ts')));
let band = startBand(), now = 0;
const samples = [], frames = [];
function frame(ms, adjusting = false, active = true) {
  now += ms;
  if (ms > 0 && ms < 500) samples.push({at: now, ms});
  while (samples.length && now - samples[0].at > 250) samples.shift();
  const cutoff = (samples.at(-1)?.at ?? now) - (adjusting ? 80 : 250);
  const chosen = samples.filter(s => s.at >= cutoff);
  const mean = chosen.length ? chosen.reduce((sum,s) => sum+s.ms,0) / chosen.length : 0;
  if (active) band = stepBand(band, fpsFrom(mean), now, adjusting);
  frames.push({now, ms, adjusting, active, band: band.band, readout: Number(band.readout)});
}
for (const ms of [16.6667, 20, 20.1, 21.27, 21.28, 25, 28.5, 28.6, 31.24, 31.26, 40, 16, 8]) {
  for (let n=0;n<50;n++) frame(ms);
}
for (let n=0;n<60;n++) frame(n%2 ? 21.26 : 21.30);
for (let n=0;n<30;n++) frame(40, true);
for (let n=0;n<30;n++) frame(16.6667, true);
frame(500); frame(3000); frame(0);
for (let n=0;n<30;n++) frame(200, false, false);
for (let n=0;n<30;n++) frame(16.6667);
writeFileSync(new URL('./performance.fixture.json',import.meta.url),JSON.stringify({source:'aphid91/Fluoddity-Web@52787528700d49aff80a798b1e2490e4d537ea2b',frames},null,2)+'\n');
