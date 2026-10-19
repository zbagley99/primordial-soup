#include "cellsBatch.wgsl"
@group(0) @binding(1) var<storage,read_write> accum:array<CellAccum>;
@group(0) @binding(2) var<storage,read_write> selected:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read> plan:BatchPlan;
@group(0) @binding(6) var old_color:texture_2d<f32>;
@group(0) @binding(7) var old_selection:texture_2d<f32>;
@group(0) @binding(8) var next_color:texture_storage_2d<rgba32float,write>;
@group(0) @binding(9) var next_selection:texture_storage_2d<r32float,write>;
// Reading a pixel's sums also clears them for the next batch. Keep atomic
// exchanges: ordinary shader stores measured slower on the target GPU.
@compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id:vec3u){
 if(any(id.xy>=vec2u(c.screen.xy))){return;}let i=id.y*u32(c.screen.x)+id.x;
 let mass=atomicExchange(&accum[i].mass,0u);
 let red=atomicExchange(&accum[i].red,0u);
 let green=atomicExchange(&accum[i].green,0u);
 let blue=atomicExchange(&accum[i].blue,0u);
 let value=vec4f(f32(red),f32(green),f32(blue),f32(mass))*c.render.x*c.render.y;
 // A one-sample compact frame executes only batch zero, whose history was
 // cleared by Cells::clear. Keep history reads for all other batch schedules.
 let fresh=c.render.y==1.0 && plan.mode==COMPACT;
 var color=value;if(!fresh){color=textureLoad(old_color,vec2i(id.xy),0)+value;}
 textureStore(next_color,id.xy,color);
 // Without a highlighted cohort the sprites skip the selection sum, which would equal mass.
 var chosen=mass;if(c.color.w>=0.0){chosen=atomicExchange(&selected[i],0u);}
 var selection=f32(chosen)*c.render.x*c.render.y;
 if(!fresh){selection=textureLoad(old_selection,vec2i(id.xy),0).r+selection;}
 textureStore(next_selection,id.xy,vec4f(selection,0,0,0));
}
