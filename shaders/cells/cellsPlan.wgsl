// Cells sprite planning, once per temporal sample before the integer sprites.
//
// 1. box_rows/box_columns bound how many sprites can cover any one pixel. A
//    sprite never reaches further than sprite_radius_limit(), so every sprite
//    touching a pixel comes from a grid cell within k cells of that pixel's
//    own cell (projecting an outside pixel onto the world keeps that true).
//    The window sum of the occupancy grid over those cells is therefore an
//    upper bound; wrapping only adds repeated cells, which over-counts.
// 2. prepare lists the copies of each particle that can reach the framebuffer,
//    with their radius and amplitude, so the sprite pass does no culled work.
// 3. plan_batches keeps every listed copy in one pass when the bound is no
//    larger than the reference batch, so the reference quantum still cannot
//    overflow. Otherwise it restores particle batches no larger than the
//    reference. Either way the quantum, and so the image, is unchanged.
#include "cellsSpriteShape.wgsl"
struct PlanCounters { bound:atomic<u32>, entries:atomic<u32>, mode:u32, _pad:u32 }
@group(0) @binding(1) var<storage,read> particles:array<Entity>;
@group(0) @binding(2) var<storage,read> counts:array<u32>;
@group(0) @binding(3) var<storage,read_write> rows:array<u32>;
@group(0) @binding(4) var<storage,read_write> plan:PlanCounters;
@group(0) @binding(5) var<uniform> batch:BatchSettings;
@group(0) @binding(6) var density:texture_2d<f32>;
@group(0) @binding(7) var<storage,read_write> sprites:array<vec2u>;
@group(0) @binding(8) var<storage,read_write> radii:array<f32>;
@group(0) @binding(9) var<storage,read_write> indirect:array<vec4u>;
// Beyond this window the bound falls back to the whole population.
const MAX_REACH:i32=64;
fn reach()->vec2i {
 let cell=cell_period()/c.grid.xy;
 let cells=floor(sprite_radius_limit()/max(cell,vec2f(1e-6)))+2.0;
 return vec2i(min(cells,vec2f(f32(MAX_REACH+1))));
}
fn row_cell(x:i32,y:i32,res:vec2i)->i32 {
 if(c.camera.w==1.0){let q=(vec2i(x,y)%res+res)%res;return q.y*res.x+q.x;}
 if(x<0 || y<0 || x>=res.x || y>=res.y){return -1;}
 return y*res.x+x;
}
@compute @workgroup_size(8,8) fn box_rows(@builtin(global_invocation_id) id:vec3u){
 let res=vec2i(c.grid.xy);if(any(vec2i(id.xy)>=res)){return;}
 let k=reach().x;if(k>MAX_REACH){return;}
 var total=0u;
 for(var d=-k;d<=k;d++){let i=row_cell(i32(id.x)+d,i32(id.y),res);if(i>=0){total+=counts[i];}}
 rows[u32(i32(id.y)*res.x)+id.x]=total;
}
var<workgroup> peak:atomic<u32>;
@compute @workgroup_size(8,8) fn box_columns(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) li:u32){
 let res=vec2i(c.grid.xy);let k=reach();
 if(all(vec2i(id.xy)<res)){
  var total=0u;
  if(any(k>vec2i(MAX_REACH))){total=batch.count;}else{
   for(var d=-k.y;d<=k.y;d++){let i=row_cell(i32(id.x),i32(id.y)+d,res);if(i>=0){total+=rows[i];}}
  }
  atomicMax(&peak,total);
 }
 workgroupBarrier();
 if(li==0u){atomicMax(&plan.bound,atomicLoad(&peak));}
}
var<workgroup> staged:atomic<u32>;
var<workgroup> base:u32;
@compute @workgroup_size(256) fn prepare(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) li:u32){
 var listed:array<u32,9>;var weights:array<f32,9>;var found=0u;
 if(id.x<arrayLength(&particles)){
  let e=particles[id.x];
  if(!e_is_dead(e)){
   let uv=cell_uv(e_pos(e));
   var radius=sprite_radius(uv);
   if(c.camera.w==1.0){let span=world_span();radius=min(radius,0.49*min(span.x,span.y));}
   radii[id.x]=radius;
   for(var copy=0u;copy<select(1u,9u,c.camera.w==1.0);copy++){
    var center:vec2f;
    if(!sprite_copy(uv,e_pos(e),radius,copy,&center)){continue;}
    // A quad wholly outside the framebuffer rasterizes nothing.
    let center_px=vec2f(center.x*0.5+0.5,0.5-center.y*0.5)*c.screen.xy;
    if(any(center_px+radius<vec2f(-1.0)) || any(center_px-radius>c.screen.xy+1.0)){continue;}
    listed[found]=(id.x<<4u)|copy;weights[found]=sprite_amplitude(center,radius);found++;
   }
  }
 }
 let offset=atomicAdd(&staged,found);
 workgroupBarrier();
 if(li==0u){base=atomicAdd(&plan.entries,atomicLoad(&staged));}
 let first=workgroupUniformLoad(&base)+offset;
 for(var j=0u;j<found;j++){
  if(first+j<batch.capacity){sprites[first+j]=vec2u(listed[j],bitcast<u32>(weights[j]));}
 }
}
@compute @workgroup_size(1) fn plan_batches(){
 let bound=atomicLoad(&plan.bound);
 let entries=atomicLoad(&plan.entries);
 let compact=bound<=batch.limit && entries<=batch.capacity;
 let listed=!compact && entries<=batch.capacity && entries<(batch.count*batch.copies)/max(batch.batches,1u);
 plan.mode=select(select(1u,LISTED_BATCHES,listed),COMPACT,compact);
 let groups=(vec2u(c.screen.xy)+7u)/8u;
 for(var k=0u;k<batch.batches;k++){
  var instances=0u;
  // Every encoded average runs in fallback mode; in compact mode only the first.
  var work=vec4u(groups,1u,0u);
  if(compact){
   if(k==0u){instances=entries;}else{work=vec4u(0u,1u,1u,0u);}
  }else if(listed){
   instances=entries;
  }else{
   let first=k*batch.size;
   if(first<batch.count){instances=min(batch.size,batch.count-first)*batch.copies;}
  }
  indirect[k]=vec4u(4u,instances,0u,0u);
  indirect[batch.slots+k]=work;
 }
}
