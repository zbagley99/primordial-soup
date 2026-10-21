// Fast Cells compact-only tile gather. Every failure preserves the original draw.
#include "cellsBatch.wgsl"
struct TileConfig { tiles:vec2u, record_capacity:u32, index_capacity:u32, tile_count:u32, blocks:u32, max_occupancy:u32, average_slot:u32 }
struct Record { geometry:vec4f, pigment:vec4f }
struct TileState { overflow:atomic<u32>, references:u32, enabled:u32, max_occupancy:atomic<u32>, dispatch:vec4u }
struct PlainSites { first:u32, second:u32 }
@group(0) @binding(1) var<uniform> cfg:TileConfig;
@group(0) @binding(2) var<storage,read> particles:array<Entity>;
@group(0) @binding(3) var<storage,read> plan:BatchPlan;
@group(0) @binding(4) var<storage,read> sprites:array<vec2u>;
@group(0) @binding(5) var<storage,read> radii:array<f32>;
@group(0) @binding(6) var<storage,read_write> records:array<Record>;
@group(0) @binding(7) var<storage,read_write> counts:array<atomic<u32>>;
@group(0) @binding(8) var<storage,read_write> offsets:array<u32>;
@group(0) @binding(9) var<storage,read_write> indices:array<u32>;
@group(0) @binding(10) var<storage,read_write> state:TileState;
@group(0) @binding(11) var<storage,read_write> blocks:array<u32>;
@group(0) @binding(12) var<storage,read_write> indirect:array<vec4u>;
@group(0) @binding(15) var<storage,read_write> sites:array<PlainSites>;
@group(0) @binding(16) var old_color:texture_2d<f32>;
@group(0) @binding(17) var old_selection:texture_2d<f32>;
@group(0) @binding(18) var next_color:texture_storage_2d<rgba32float,write>;
@group(0) @binding(19) var next_selection:texture_storage_2d<r32float,write>;
@group(0) @binding(20) var<storage,read_write> dispatch_args:vec4u;
@compute @workgroup_size(1) fn setup(){
 let n=select(0u,min(plan.entries,cfg.record_capacity+1u),plan.mode==COMPACT);
 state.dispatch=vec4u((n+255u)/256u,1u,1u,0u);
 dispatch_args=state.dispatch;
 if(plan.mode==COMPACT && plan.entries>cfg.record_capacity){atomicStore(&state.overflow,1u);}
}
fn tile_half_world()->vec2f {return vec2f(sqrt(c.canvas.x/c.canvas.y),sqrt(c.canvas.y/c.canvas.x));}
fn tile_center(e:Entity,copy:u32)->vec2f {
 var position=e_pos(e);
 if(c.camera.w==1.0){let h=tile_half_world();let shift=vec2f(f32(copy%3u)-1.0,f32(copy/3u)-1.0);position=(fract(cell_uv(position))+shift)*h*2.0-h;}
 let ndc=cell_ndc(position)*c.canvas.zw/c.screen.xy;
 return vec2f(ndc.x*0.5+0.5,0.5-ndc.y*0.5)*c.screen.xy;
}
fn tile_bounds(geometry:vec4f)->vec4u {
 let lo=vec2u(clamp(floor((geometry.xy-geometry.z)/8.0),vec2f(0),vec2f(cfg.tiles-1u)));
 let hi=vec2u(clamp(floor((geometry.xy+geometry.z)/8.0),vec2f(0),vec2f(cfg.tiles-1u)));
 return vec4u(lo,hi);
}
@compute @workgroup_size(256) fn prepare(@builtin(global_invocation_id) id:vec3u){
 if(plan.mode!=COMPACT || id.x>=plan.entries){return;}
 if(id.x>=cfg.record_capacity){atomicStore(&state.overflow,1u);return;}
 let s=sprites[id.x];let index=s.x>>4u;let e=particles[index];let radius=radii[index];let center=tile_center(e,s.x&15u);
 let cp=e_col_params(e);let signal=select(cp.x,cp.y*0.75,c.color.z>0.5);let h=c.color.x*signal+c.color.y;
 var color=cell_hsv(h);if(c.look.x>0.5){let t=0.5+0.5*sin(h*6.2831853);color=mix(vec3f(0.28,0.43,0.04),vec3f(0.72,0.19,0.35),smoothstep(0.35,0.7,t));}
 // Conservative integer bounding box includes all potentially covered centers.
 let geometry=vec4f(center,radius,bitcast<f32>(s.y));let bounds=tile_bounds(geometry);let lo=bounds.xy;let hi=bounds.zw;
 records[id.x]=Record(geometry,vec4f(color,select(0.0,1.0,c.color.w<0.0 || floor(cp.y)==c.color.w)));
 for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){atomicAdd(&counts[y*cfg.tiles.x+x],1u);}}
}
var<workgroup> scan_values:array<u32,256>;
@compute @workgroup_size(256) fn scan(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) li:u32,@builtin(workgroup_id) wg:vec3u){
 var n=0u;if(id.x<cfg.tile_count){n=atomicLoad(&counts[id.x]);atomicMax(&state.max_occupancy,n);if(n>cfg.max_occupancy){atomicStore(&state.overflow,1u);}}
 scan_values[li]=n;workgroupBarrier();
 for(var step=1u;step<256u;step*=2u){var v=0u;if(li>=step){v=scan_values[li-step];}workgroupBarrier();scan_values[li]+=v;workgroupBarrier();}
 if(id.x<cfg.tile_count){offsets[id.x]=scan_values[li]-n;atomicStore(&counts[id.x],0u);}
 if(li==255u){blocks[wg.x]=scan_values[li];}
}
@compute @workgroup_size(1) fn scan_blocks(){
 var total=0u;for(var i=0u;i<cfg.blocks;i++){let n=blocks[i];blocks[i]=total;if(n>cfg.index_capacity || total>cfg.index_capacity-min(n,cfg.index_capacity)){atomicStore(&state.overflow,1u);total=cfg.index_capacity+1u;}else{total+=n;}}
 state.references=total;
 if(total>cfg.index_capacity){atomicStore(&state.overflow,1u);}
}
@compute @workgroup_size(256) fn add_offsets(@builtin(global_invocation_id) id:vec3u){
 if(id.x<cfg.tile_count){offsets[id.x]+=blocks[id.x/256u];}
}
@compute @workgroup_size(256) fn fill(@builtin(global_invocation_id) id:vec3u){
 if(plan.mode!=COMPACT || atomicLoad(&state.overflow)!=0u || id.x>=min(plan.entries,cfg.record_capacity)){return;}
 let b=tile_bounds(records[id.x].geometry);
 for(var y=b.y;y<=b.w;y++){for(var x=b.x;x<=b.z;x++){
  let tile=y*cfg.tiles.x+x;let slot=offsets[tile]+atomicAdd(&counts[tile],1u);
  if(slot<cfg.index_capacity){indices[slot]=id.x;}else{atomicStore(&state.overflow,1u);}
 }}
}
@compute @workgroup_size(1) fn finalize(){
 state.enabled=select(0u,1u,plan.mode==COMPACT && plan.entries<=cfg.record_capacity && atomicLoad(&state.overflow)==0u);
 if(state.enabled!=0u){indirect[0].y=0u;indirect[cfg.average_slot].x=0u;}
}
var<workgroup> cached:array<Record,64>;
var<workgroup> tile_enabled:u32;
var<workgroup> tile_length:u32;
@compute @workgroup_size(8,8) fn raster(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) li:u32,@builtin(workgroup_id) wg:vec3u){
 if(li==0u){tile_enabled=state.enabled;tile_length=atomicLoad(&counts[wg.y*cfg.tiles.x+wg.x]);}
 workgroupBarrier();
 let valid=all(id.xy<vec2u(c.screen.xy));let pixel=id.y*u32(c.screen.x)+id.x;
 if(workgroupUniformLoad(&tile_enabled)==0u){
  if(valid && c.structure.y>0.0){sites[pixel]=PlainSites(0u,0u);}
  return;
 }
 let p=vec2f(id.xy)+0.5;
 let inside=valid && (c.camera.w==1.0 || cell_inside(p));
 let tile=wg.y*cfg.tiles.x+wg.x;let n=workgroupUniformLoad(&tile_length);let first=offsets[tile];
 var red=0u;var green=0u;var blue=0u;var mass=0u;var chosen=0u;var nearest=0u;var second=0u;
 for(var start=0u;start<n;start+=64u){
  let len=min(64u,n-start);if(li<len){cached[li]=records[indices[first+start+li]];}workgroupBarrier();
  if(inside){for(var j=0u;j<len;j++){
   let r=cached[j];let uv=(p-r.geometry.xy)/r.geometry.z;let rr=dot(uv,uv);if(rr>1.0){continue;}
   let w=min(0.62,r.geometry.w*exp(-rr/(2.0*0.45*0.45)));let value=u32(round(w*c.render.w));
   mass+=value;red+=min(value,u32(round(w*c.render.w*r.pigment.r)));green+=min(value,u32(round(w*c.render.w*r.pigment.g)));blue+=min(value,u32(round(w*c.render.w*r.pigment.b)));
   if(c.color.w>=0.0 && r.pigment.w>0.5){chosen+=value;}
   if(c.structure.y>0.0){let key=cell_distance_key(rr);second=max(second,min(nearest,key));nearest=max(nearest,key);}
  }}workgroupBarrier();
 }
 if(valid){
  let value=vec4f(f32(red),f32(green),f32(blue),f32(mass))*c.render.x*c.render.y;
  let fresh=c.render.y==1.0;
  var color=value;if(!fresh){color=textureLoad(old_color,vec2i(id.xy),0)+value;}
  textureStore(next_color,id.xy,color);
  let selected_mass=select(chosen,mass,c.color.w<0.0);
  var selection=f32(selected_mass)*c.render.x*c.render.y;
  if(!fresh){selection=textureLoad(old_selection,vec2i(id.xy),0).r+selection;}
  textureStore(next_selection,id.xy,vec4f(selection,0,0,0));
  if(c.structure.y>0.0){sites[pixel]=PlainSites(nearest,second);}
 }
}
