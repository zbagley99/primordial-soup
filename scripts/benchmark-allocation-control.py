#!/usr/bin/env python3
"""Run the bounded allocation control with read-only NVIDIA hardware samples.

Invoke through low-load.py after building the native release binaries.
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
artifacts = root / "artifacts"
artifacts.mkdir(exist_ok=True)
binary = root / "target/release/gpu-check"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


command = [str(binary), "--allocation-control", "--output", str(artifacts / "allocation-control.json")]
query = [
    "nvidia-smi",
    "--query-gpu=timestamp,pstate,clocks.current.graphics,clocks.current.sm,clocks.current.memory,utilization.gpu,utilization.memory,power.draw,temperature.gpu",
    "--format=csv,noheader,nounits",
    "--loop-ms=100",
]
started = time.time()
with (artifacts / "allocation-control-hardware.csv").open("w") as samples:
    telemetry = subprocess.Popen(query, stdout=samples, stderr=subprocess.PIPE, text=True)
    try:
        with (artifacts / "allocation-control.log").open("w") as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=30)
    finally:
        telemetry.terminate()
        try:
            _, error = telemetry.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            telemetry.kill()
            _, error = telemetry.communicate()
        metadata = {
            "command": command,
            "telemetry_command": query,
            "started_unix_ms": round(started * 1000),
            "ended_unix_ms": round(time.time() * 1000),
            "binary_sha256": digest(binary),
            "source_sha256": {
                str(path.relative_to(root)): digest(path)
                for path in [root / "src/bin/gpu-check.rs", root / "src/render.rs", root / "src/gpu.rs", root / "src/config.rs", root / "Cargo.lock", *sorted((root / "shaders").glob("*.wgsl"))]
            },
            "telemetry_stderr": error,
            "note": "Read-only 100 ms hardware sampling shares the single low-priority CPU; no clocks or system settings are changed.",
        }
        (artifacts / "allocation-control-run.json").write_text(json.dumps(metadata, indent=2) + "\n")
if result.returncode:
    raise SystemExit(result.returncode)
print("Saved allocation-control.json, hardware samples and run metadata.")
