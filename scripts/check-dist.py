#!/usr/bin/env python3
"""Check the actual Cloudflare deployment boundary, not the repository."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent.parent/'dist'
allowed={'.html','.js','.mjs','.wasm','.txt',''}
files=[]
for path in sorted(root.rglob('*')):
    if not path.is_file():continue
    size=path.stat().st_size
    if path.suffix not in allowed or path.name.startswith('.env'):
        raise SystemExit(f'Unexpected deployment file: {path.name}')
    if size >= 25*1024*1024:
        raise SystemExit(f'Cloudflare asset exceeds 25 MiB: {path.name}')
    files.append({'file':str(path.relative_to(root)),'bytes':size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
if len(files)>20000:raise SystemExit('Too many static assets for Cloudflare Free')
for required in ('index.html','boot.js','primordial_soup.js','primordial_soup_bg.wasm','_headers','Fluoddity-MIT.txt','Mediabunny-MPL-2.0.txt','Nayuki-MIT.txt','jsQR-Apache-2.0.txt','Primordial-Soup-MIT.txt'):
    if not (root/required).is_file():raise SystemExit(f'Missing deployment file: {required}')
print(json.dumps({'static_only':True,'files':files,'total_bytes':sum(f['bytes'] for f in files)},indent=2))
