#include "cellsBlur.wgsl"
const WASH_RANGE:f32=16.0;
// The first reduction already reads every full-resolution texel, so it also
// extracts mass for relief heights. Each workgroup writes the source texels of
// its own 16x16 block (the last row/column of blocks extends to the edge), so
// every texel is written exactly once.
@group(0) @binding(2) var mass_out:texture_storage_2d<r32float,write>;
fn owned(wg:u32,count:u32,res:i32)->vec2i {
 let lo=min(i32(wg)*16,res);return vec2i(lo,select(min(lo+16,res),res,wg+1u==count));
}
@compute @workgroup_size(8,8) fn first(@builtin(workgroup_id) wg:vec3u,@builtin(num_workgroups) n:vec3u,@builtin(local_invocation_id) lid:vec3u,@builtin(local_invocation_index) li:u32){
 gain=1.0/WASH_RANGE;
 let f=footprint(wg);let res=vec2i(textureDimensions(source));
 let x=owned(wg.x,n.x,res.x);let y=owned(wg.y,n.y,res.y);let w=x.y-x.x;
 if(f.fits){load_tile(f.origin,li);}
 for(var i=i32(li);i<w*(y.y-y.x);i+=64){
  let t=vec2i(x.x+i%w,y.x+i/w);let k=t-f.origin;var m:f32;
  if(f.fits && all(k>=vec2i(0)) && all(k<vec2i(SPAN))){m=tile[k.y*SPAN+k.x].a;}else{m=textureLoad(source,t,0).a;}
  textureStore(mass_out,t,vec4f(m,0,0,0));
 }
 if(f.fits){filter_tile(f,lid,li);}else{direct(f.first,lid);}
}
