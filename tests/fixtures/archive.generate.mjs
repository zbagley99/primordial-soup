// Regenerate with: node archive.generate.mjs /path/to/Fluoddity-Web
import {readFileSync,writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
const root=resolve(process.argv[2]);
const {hashState}=await import(pathToFileURL(`${root}/src/archive/hash.ts`));
const {fromDocument}=await import(pathToFileURL(`${root}/src/config/persistence.ts`));
const fixtures=[];
for(const path of ['Core/Starcrossed.json','Core/Segments.json','Core/Diversity.json']) {
    let source;
    try { source=readFileSync(`${root}/public/configs/${path}`,'utf8'); }
    catch { continue; }
    const project=fromDocument(JSON.parse(source));
    fixtures.push({path,document:JSON.parse(source),hash:hashState(project)});
}
if (!fixtures.length) throw new Error('No presets found; check upstream public/configs path');
writeFileSync(new URL('./archive.fixture.json',import.meta.url),JSON.stringify({revision:'52787528700d49aff80a798b1e2490e4d537ea2b',fixtures},null,2)+'\n');
