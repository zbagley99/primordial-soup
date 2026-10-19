.PHONY: run test web serve benchmark
LOW_PRIO ?= 0
RUN_PREFIX := $(if $(filter 1,$(LOW_PRIO)),python3 scripts/low-load.py)

run:
	$(RUN_PREFIX) cargo run --locked --release --bin primordial-soup
test:
	python3 scripts/low-load.py cargo test --locked -- --test-threads=1
web:
	bash scripts/build-web.sh
serve:
	python3 scripts/low-load.py python3 -m http.server 8080 --bind 127.0.0.1 --directory dist
benchmark:
	python3 scripts/low-load.py cargo run --locked --release --bin gpu-check -- --benchmark
