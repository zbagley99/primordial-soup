#!/usr/bin/env python3
"""Give the generated module graph content versions without duplicating assets."""
import hashlib
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / 'dist'
rewritten = []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def version(path, references):
    text = path.read_text()
    for reference, asset in references:
        prefix = r'((?:\bfrom\s+|\bimport\s+|\bimport\s*\(\s*|\bnew\s+URL\s*\(\s*|\bsrc\s*=\s*))'
        pattern = prefix + r'([\'\"])' + re.escape(reference) + r'(?:\?v=[0-9a-f]{16})?\2'
        text, count = re.subn(pattern, lambda m: f'{m[1]}{m[2]}{reference}?v={digest(asset)}{m[2]}', text)
        if not count:
            raise SystemExit(f'Missing generated asset reference: {path.name}: {reference}')
    path.write_text(text)
    rewritten.append(path)


# Runtime URLs in these modules are relative to document.baseURI.
video = list(root.glob('snippets/*/web/video.js'))
if len(video) != 1:
    raise SystemExit('Expected exactly one generated video module')
version(video[0], [('./vendor/mediabunny.mjs', root / 'vendor/mediabunny.mjs')])
version(root / 'image-import.js', [('./vendor/jsqr.js', root / 'vendor/jsqr.js')])

# wasm-bindgen uses a fixed snippet directory across releases. Version every
# static import before hashing its parent, then version the WASM URL itself.
binding = root / 'primordial_soup.js'
snippets = sorted(set(re.findall(r'[\'\"](\./snippets/[^\'\"?]+\.js)', binding.read_text())))
version(binding, [(ref, root / ref) for ref in snippets] + [('primordial_soup_bg.wasm', root / 'primordial_soup_bg.wasm')])
version(root / 'boot.js', [(f'./{name}', root / name) for name in ['primordial_soup.js', 'image-import.js', 'primordial_soup_bg.wasm']])
version(root / 'index.html', [('./boot.js', root / 'boot.js')])

# Refuse a partial module graph if the binding generator or entrypoint changes.
asset_url = r'([\'\"])((?:\./|\.\./)?(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_-]+\.(?:js|mjs|wasm))(\?v=[0-9a-f]{16})?\1'
for path in rewritten:
    text = path.read_text()
    for match in re.finditer(asset_url, text):
        # These dictionary keys identify compiled WASM import modules, not URLs.
        # Changing them would break instantiation; only their JS imports get versions.
        if path == binding and match[2] in {'./primordial_soup_bg.js', *snippets} and re.match(r'\s*:\s*import\d+\b', text[match.end():]):
            continue
        if not match[3]:
            raise SystemExit(f'Unversioned application asset: {path.name}: {match[2]}')
