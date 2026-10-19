#!/usr/bin/env python3
"""Run builds/validation on one low-priority CPU, including linker children."""
import os,sys
if len(sys.argv)<2:raise SystemExit('usage: low-load.py COMMAND [ARGS...]')
os.nice(19)
if hasattr(os,'sched_getaffinity'):
    allowed=os.sched_getaffinity(0)
    os.sched_setaffinity(0,{max(allowed)})
env=os.environ.copy()
env['CARGO_BUILD_JOBS']='1'
os.execvpe(sys.argv[1],sys.argv[1:],env)
