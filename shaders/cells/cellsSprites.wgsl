#include "cellsSpriteShape.wgsl"
@group(0) @binding(1) var<storage,read> particles:array<Entity>;
@group(0) @binding(2) var density:texture_2d<f32>;
@group(0) @binding(3) var<storage,read_write> accum:array<CellAccum>;
@group(0) @binding(4) var<storage,read_write> selected:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read> plan:BatchPlan;
@group(0) @binding(6) var<storage,read> sprites:array<vec2u>;
@group(0) @binding(7) var<storage,read> radii:array<f32>;
@group(0) @binding(8) var<storage,read_write> sites:array<CellSites>;
@group(0) @binding(9) var<uniform> batch:BatchSettings;
// Index of this pass among the encoded particle batches.
@group(1) @binding(0) var<uniform> slot:vec4u;
struct Out {@builtin(position) pos:vec4f,@location(0) uv:vec2f,@location(1) @interpolate(flat) color:vec3f,@location(2) @interpolate(flat) radius:f32,@location(3) @interpolate(flat) chosen:f32,@location(4) @interpolate(flat) amplitude:f32,@location(5) @interpolate(flat) center_px:vec2f}
fn sprite(e:Entity,center:vec2f,radius:f32,amplitude:f32,vi:u32)->Out {
 var o:Out;
 let corner=strip_corner(vi)*2.0-1.0;
 o.pos=vec4f(center+corner*radius*2.0/c.screen.xy,0,1);o.uv=corner;o.radius=radius;o.amplitude=amplitude;
 o.center_px=vec2f(center.x*0.5+0.5,0.5-center.y*0.5)*c.screen.xy;
 let cp=e_col_params(e);let signal=select(cp.x,cp.y*0.75,c.color.z>0.5);let h=c.color.x*signal+c.color.y;
 o.color=cell_hsv(h);if(c.look.x>0.5){let t=0.5+0.5*sin(h*6.2831853);o.color=mix(vec3f(0.28,0.43,0.04),vec3f(0.72,0.19,0.35),smoothstep(0.35,0.7,t));}
 o.chosen=select(0.0,1.0,c.color.w<0.0 || floor(cp.y)==c.color.w);
 return o;
}
@vertex fn vertex(@builtin(vertex_index) vi:u32,@builtin(instance_index) i:u32)->Out {
 var o:Out;o.pos=vec4f(0);o.uv=vec2f(0);o.color=vec3f(0);o.radius=1.0;o.chosen=0;o.amplitude=0;o.center_px=vec2f(0);
 if(plan.mode==COMPACT){
  // cellsPlan.wgsl listed only reachable copies with their radius and amplitude.
  let s=sprites[i];let index=s.x>>4u;let e=particles[index];
  var center:vec2f;
  if(!sprite_copy(cell_uv(e_pos(e)),e_pos(e),radii[index],s.x&15u,&center)){return o;}
  return sprite(e,center,radii[index],bitcast<f32>(s.y),vi);
 }
 let copies=select(1u,9u,c.camera.w==1.0);
 var index:u32;var copy:u32;
 if(plan.mode==LISTED_BATCHES){
  index=sprites[i].x>>4u;
  let first=slot.x*batch.size;
  if(index<first || index>=min(first+batch.size,batch.count)){return o;}
  copy=sprites[i].x&15u;
 }else{
  let instance=slot.x*batch.size*copies+i;
  index=instance/copies;copy=instance%copies;
 }
 let e=particles[index];
 if(e_is_dead(e)){return o;}
 let uv=cell_uv(e_pos(e));
 var radius=sprite_radius(uv);
 // Disjoint periodic copies preserve the integer sum bound (one per particle/pixel).
 if(c.camera.w==1.0){let span=world_span();radius=min(radius,0.49*min(span.x,span.y));}
 var center:vec2f;
 if(!sprite_copy(uv,e_pos(e),radius,copy,&center)){return o;}
 return sprite(e,center,radius,sprite_amplitude(center,radius),vi);
}
@fragment fn fragment(i:Out)->@location(0) vec4f {
 // The experimental tile gather uses analytical pixel distances. Preserve
 // hardware interpolation for ordinary Cells rendering.
 var uv=i.uv;if(c.structure.z>0.5){uv=(i.pos.xy-i.center_px)/i.radius;}
 let rr=dot(uv,uv);if(rr>1.0 || (c.camera.w!=1.0 && !cell_inside(i.pos.xy))){discard;}
 // Integral in reference pixels is 0.4; radius includes physical output scale.
 let w=min(0.62,i.amplitude*exp(-rr/(2.0*0.45*0.45)));
 let value=u32(round(w*c.render.w));let p=vec2u(i.pos.xy);let index=p.y*u32(c.screen.x)+p.x;
 if(c.structure.y>0.0){
  let key=cell_distance_key(rr);
  let old=atomicMax(&sites[index].first,key);
  atomicMax(&sites[index].second,min(old,key));
 }
 atomicAdd(&accum[index].mass,value);atomicAdd(&accum[index].red,min(value,u32(round(w*c.render.w*i.color.r))));atomicAdd(&accum[index].green,min(value,u32(round(w*c.render.w*i.color.g))));atomicAdd(&accum[index].blue,min(value,u32(round(w*c.render.w*i.color.b))));
 // Without a highlighted cohort every sprite is chosen, so the average pass reuses mass.
 if(c.color.w>=0.0 && i.chosen>0.5){atomicAdd(&selected[index],value);}return vec4f(0);
}
