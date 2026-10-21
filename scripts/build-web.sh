#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v wasm-bindgen >/dev/null || { echo 'Install wasm-bindgen-cli matching Cargo.lock before building.' >&2; exit 1; }
rustup target list --installed | grep -qx wasm32-unknown-unknown || { echo 'Run: rustup target add wasm32-unknown-unknown' >&2; exit 1; }
if [[ "${LOW_PRIO:-0}" == 1 ]]; then
  python3 scripts/low-load.py cargo build --locked --profile web --target wasm32-unknown-unknown --lib
else
  cargo build --locked --profile web --target wasm32-unknown-unknown --lib
fi
web_stage=$(mktemp -d "$PWD/.dist-stage.XXXXXX")
web_backup=''
cleanup() {
  if [[ -n "$web_stage" && -d "$web_stage" ]]; then rm -rf -- "$web_stage"; fi
  if [[ -n "$web_backup" && -d "$web_backup" ]]; then
    if [[ ! -e dist ]]; then mv -- "$web_backup" dist; else rm -rf -- "$web_backup"; fi
  fi
}
trap cleanup EXIT
wasm-bindgen --target web --out-dir "$web_stage" --no-typescript --remove-name-section --remove-producers-section target/wasm32-unknown-unknown/web/primordial_soup.wasm
cp web/index.html web/boot.js web/_headers web/image-import.js "$web_stage/"
mkdir -p "$web_stage/vendor"
cp web/vendor/*.js web/vendor/*.mjs "$web_stage/vendor/"
cp LICENSES/Fluoddity-MIT.txt "$web_stage/Fluoddity-MIT.txt"
cp LICENSES/Mediabunny-MPL-2.0.txt "$web_stage/Mediabunny-MPL-2.0.txt"
cp LICENSES/Nayuki-MIT.txt "$web_stage/Nayuki-MIT.txt"
cp LICENSES/jsQR-Apache-2.0.txt "$web_stage/jsQR-Apache-2.0.txt"
cp LICENSE "$web_stage/Primordial-Soup-MIT.txt"
python3 scripts/version-web.py "$web_stage"
mkdir -p artifacts
python3 scripts/check-dist.py "$web_stage" > artifacts/static-dist.next.json
if [[ -e dist ]]; then
  web_backup=$(mktemp -d "$PWD/.dist-stage.XXXXXX")
  rmdir -- "$web_backup"
  mv -- dist "$web_backup"
fi
mv -- "$web_stage" dist
web_stage=''
mv artifacts/static-dist.next.json artifacts/static-dist.json
cat artifacts/static-dist.json
