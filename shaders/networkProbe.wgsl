// Bounded evaluator validation harness, also runnable with browser WebGPU.
#include "network.wgsl"
@group(0) @binding(0) var<storage,read> probe_inputs: array<vec4f>;
@group(0) @binding(1) var<storage,read_write> probe_outputs: array<vec4f>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let index = gid.x;
    if (index * 2u >= arrayLength(&probe_inputs)) { return; }
    let slot = i32(index % 3u);
    probe_outputs[index*2u]=brain_evaluate(slot,probe_inputs[index*2u],index,0u,vec3f(0.0));
    probe_outputs[index*2u+1u]=brain_evaluate(slot,probe_inputs[index*2u+1u],index,1u,vec3f(0.0));
}
