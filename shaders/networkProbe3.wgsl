#include "network.wgsl"
@group(0) @binding(0) var<storage,read> probe_inputs: array<vec4f>;
@group(0) @binding(1) var<storage,read_write> probe_outputs: array<vec4f>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid:vec3u){
 let index=gid.x;let slot=i32(index%3u);
 for(var stream=0u;stream<2u;stream++){
  let first=(index*2u+stream)*2u;
  probe_outputs[index*2u+stream]=brain_evaluate(slot,probe_inputs[first],index,stream,probe_inputs[first+1u].xyz);
 }
}
