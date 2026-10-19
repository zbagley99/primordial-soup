#include "cellsCommon.wgsl"
@group(0) @binding(1) var<storage,read_write> sites:array<CellSites>;
@group(0) @binding(2) var<storage,read_write> history:array<vec2f>;
@group(0) @binding(3) var density:texture_2d<f32>;

fn occupancy(p:vec2f)->f32 {
 let res=vec2i(textureDimensions(density));let pos=cell_uv(cell_world(p))*vec2f(res)-0.5;
 let q=vec2i(floor(pos));let f=fract(pos);var d=0.0;
 for(var y=0;y<2;y++){for(var x=0;x<2;x++){
  var a=q+vec2i(x,y);
  if(c.camera.w==1.0){a=(a%res+res)%res;}else{a=clamp(a,vec2i(0),res-1);}
  d+=textureLoad(density,a,0).r*select(1.0-f.x,f.x,x==1)*select(1.0-f.y,f.y,y==1);
 }}return d;
}
fn coincident(p:vec2i)->bool {
 let q=clamp(p,vec2i(0),vec2i(c.screen.xy)-1);let index=u32(q.y)*u32(c.screen.x)+u32(q.x);
 return atomicLoad(&sites[index].first)==atomicLoad(&sites[index].second);
}
@compute @workgroup_size(8,8) fn accumulate(@builtin(global_invocation_id) id:vec3u){
 if(any(id.xy>=vec2u(c.screen.xy))){return;}
 let index=id.y*u32(c.screen.x)+id.x;
 let first=atomicLoad(&sites[index].first);if(first==0u){return;}
 let second=atomicLoad(&sites[index].second);
 let a=cell_key_distance(first);let b=cell_key_distance(second);
 let p=vec2f(id.xy)+0.5;
 let factor=clamp(pow(max(occupancy(p),0.1),-c.opts.y*0.5),0.5,2.0);
 var radius=clamp(1.5*c.opts.x*pow(c.camera.z,c.extra.z)*factor*c.render.z,max(0.75,c.render.z),24.0*c.render.z);
 if(c.camera.w==1.0){radius=min(radius,0.49*min(cell_period().x,cell_period().y));}
 let outer=(1.0-a)*radius;
 var edge=outer;
 // Equal distances on an actual bisector have unequal neighbors. A flat tie
 // from coincident particles is one shared exterior, never a filled-in rim.
 var duplicate=false;
 if(first==second){
  let q=vec2i(id.xy);
  duplicate=coincident(q+vec2i(1,0)) && coincident(q-vec2i(1,0)) && coincident(q+vec2i(0,1)) && coincident(q-vec2i(0,1));
 }
 if(second!=0u && !duplicate){edge=min(outer,0.5*max(b-a,0.0)*radius);}
 let width=max(0.8*c.render.z,0.65);
 let coverage=smoothstep(0.0,max(c.render.z,0.75),outer);
 let rim=(1.0-smoothstep(width*0.25,width*1.75,edge))*coverage;
 let interior=(0.12+0.08*(1.0-a*a))*coverage;
 // Shade before averaging. Averaging distances would invent false walls when
 // cells move between actual motion-blur samples.
 history[index]+=vec2f(interior,rim)*c.render.y;
}
