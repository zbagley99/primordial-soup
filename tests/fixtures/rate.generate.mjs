// Regenerate from the verified MIT upstream checkout; no Rust implementation used.
// Usage: node tests/fixtures/rate.generate.mjs /path/to/Fluoddity-Web
import {writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
const {nextStep,MAX_PROBES,TARGET_MS}=await import(pathToFileURL(resolve(process.argv[2],'src/perf/rateSearch.ts')));
const cases=[];
const record=(history,lo,hi)=>{ const expected=nextStep(history,lo,hi); cases.push({history:history.map(p=>({...p})),lo,hi,expected}); return expected; };
for(const hi of [60,120]) {
    record([],1,hi);
    for(const cost of [.02,.05,.1,.25,.5,1,2,5]) for(const start of new Set([1,5,20,60,hi])) {
        const history=[]; let rate=start;
        for(let i=0;i<MAX_PROBES;i++) {
            history.push({rate,frameMs:Math.max(TARGET_MS,4+rate*cost)});
            const step=record(history,1,hi); if(step.kind==='done') break; rate=step.rate;
        }
    }
    record([{rate:hi,frameMs:16.7}],1,hi);
    record([{rate:hi,frameMs:400}],1,hi);
    record([{rate:1,frameMs:30}],1,hi);
    record([{rate:20,frameMs:16.7},{rate:34,frameMs:25}],1,hi);
    record(Array.from({length:8},(_,i)=>({rate:i+1,frameMs:i===7?30:16.7})),1,hi);
}
writeFileSync(new URL('./rate.fixture.json',import.meta.url),JSON.stringify({upstream:'52787528700d49aff80a798b1e2490e4d537ea2b',cases},null,2)+'\n');
console.log(`Frozen ${cases.length} independent upstream rate-search decisions.`);
