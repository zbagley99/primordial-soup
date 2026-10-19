#include "cellsCommon.wgsl"
@group(0) @binding(1) var<storage,read> particles:array<Entity>;
@group(0) @binding(2) var<storage,read_write> counts:array<atomic<u32>>;
@group(0) @binding(3) var source:texture_2d<f32>;
@group(0) @binding(4) var destination:texture_storage_2d<r32float,write>;
fn bounded_cell(p:vec2i,res:vec2i)->vec2i {
    if(c.camera.w==1.0){return (p%res+res)%res;}
    return clamp(p,vec2i(0),res-1);
}
fn count_at(p:vec2i)->f32 {
    let res=vec2i(c.grid.xy);
    if(c.camera.w!=1.0 && (any(p<vec2i(0)) || any(p>=res))){return 0.0;}
    let q=bounded_cell(p,res);
    return f32(atomicLoad(&counts[u32(q.y*res.x+q.x)]))*c.grid.x*c.grid.y/max(c.extra.w,1.0);
}
@compute @workgroup_size(256)
fn count(@builtin(global_invocation_id) id:vec3u){
    if(id.x>=arrayLength(&particles)){return;}
    let e=particles[id.x]; if(e_is_dead(e)){return;}
    var uv=cell_uv(e_pos(e));
    if(c.camera.w==1.0){uv=fract(uv);}else{if(any(uv<vec2f(-0.000001))||any(uv>vec2f(1.000001))){return;}uv=clamp(uv,vec2f(0),vec2f(0.9999999));}
    let p=vec2u(uv*c.grid.xy);
    atomicAdd(&counts[p.y*u32(c.grid.x)+p.x],1u);
}
@compute @workgroup_size(8,8)
fn horizontal(@builtin(global_invocation_id) id:vec3u){
    if(any(id.xy>=vec2u(c.grid.xy))){return;}
    let sigma=c.grid.x/(256.0*sqrt(c.canvas.x/c.canvas.y));var total=0.0;var weight=0.0;
    for(var i=-i32(ceil(3.0*sigma));i<=i32(ceil(3.0*sigma));i++){let w=exp(-0.5*f32(i*i)/(sigma*sigma));total+=w*count_at(vec2i(id.xy)+vec2i(i,0));weight+=w;}
    textureStore(destination,id.xy,vec4f(total/weight,0,0,0));
}
@compute @workgroup_size(8,8)
fn vertical(@builtin(global_invocation_id) id:vec3u){
    let res=vec2i(textureDimensions(source));if(any(id.xy>=vec2u(res))){return;}
    let sigma=c.grid.y*sqrt(c.canvas.x/c.canvas.y)/256.0;var total=0.0;var weight=0.0;
    for(var i=-i32(ceil(3.0*sigma));i<=i32(ceil(3.0*sigma));i++){
        let w=exp(-0.5*f32(i*i)/(sigma*sigma));let raw=vec2i(id.xy)+vec2i(0,i);
        if(c.camera.w==1.0 || (all(raw>=vec2i(0)) && all(raw<res))){total+=w*textureLoad(source,bounded_cell(raw,res),0).r;}weight+=w;
    }
    textureStore(destination,id.xy,vec4f(total/weight,0,0,0));
}
@compute @workgroup_size(8,8)
fn down(@builtin(global_invocation_id) id:vec3u){
    let dst=textureDimensions(destination);if(any(id.xy>=dst)){return;}
    let res=vec2i(textureDimensions(source));let p=vec2i(id.xy)*2;var v=0.0;
    for(var y=0;y<2;y++){for(var x=0;x<2;x++){v+=textureLoad(source,min(p+vec2i(x,y),res-1),0).r;}}
    textureStore(destination,id.xy,vec4f(v*0.25,0,0,0));
}
