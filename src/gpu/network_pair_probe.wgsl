@group(0) @binding(0) var<storage,read> pair_inputs:array<vec4f>;
@group(0) @binding(1) var<storage,read_write> pair_outputs:array<vec4f>;

@compute @workgroup_size(64)
fn reference(@builtin(global_invocation_id) gid:vec3u){
    let index=gid.x;
    if(index>=arrayLength(&pair_inputs)/3u){return;}
    let first=index*3u;let slot=i32(index%3u);let extra=pair_inputs[first+2u].xyz;
    pair_outputs[index*2u]=brain_evaluate(slot,pair_inputs[first],index,0u,extra);
    pair_outputs[index*2u+1u]=brain_evaluate(slot,pair_inputs[first+1u],index,1u,extra.yxz);
}

@compute @workgroup_size(64)
fn paired(@builtin(global_invocation_id) gid:vec3u){
    let index=gid.x;
    if(index>=arrayLength(&pair_inputs)/3u){return;}
    let first=index*3u;
    let result=brain_evaluate_pair(i32(index%3u),pair_inputs[first],pair_inputs[first+1u],index,pair_inputs[first+2u].xyz);
    pair_outputs[index*2u]=result.base;
    pair_outputs[index*2u+1u]=result.mirror;
}
