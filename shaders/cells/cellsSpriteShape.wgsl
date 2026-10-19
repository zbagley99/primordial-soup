// Sprite radius and amplitude, shared by the planning pass and the sprite
// vertex fallback. Including modules declare `density`.
#include "cellsBatch.wgsl"
fn density_at(uv:vec2f,l:i32)->f32 {
 let res=vec2i(textureDimensions(density,l));let p=uv*vec2f(res)-0.5;let q=vec2i(floor(p));let f=fract(p);var v=0.0;
 for(var y=0;y<2;y++){for(var x=0;x<2;x++){
 var a=q+vec2i(x,y);if(c.camera.w==1.0){a=(a%res+res)%res;}else{a=clamp(a,vec2i(0),res-1);}
 v+=textureLoad(density,a,l).r*select(1.0-f.x,f.x,x==1)*select(1.0-f.y,f.y,y==1);
 }}return v;
}
fn half_world()->vec2f {return vec2f(sqrt(c.canvas.x/c.canvas.y),sqrt(c.canvas.y/c.canvas.x));}
fn world_span()->vec2f {return abs(cell_ndc(half_world())-cell_ndc(-half_world()))*c.canvas.zw*0.5;}
// Radius in physical pixels before the wrapping limit.
fn sprite_radius(uv:vec2f)->f32 {
 let lod=clamp(log2(max(1.0,c.grid.y/max(cell_period().y,1.0))),0.0,f32(textureNumLevels(density)-1u));let lo=i32(floor(lod));let hi=min(lo+1,i32(textureNumLevels(density))-1);
 let d=mix(density_at(uv,lo),density_at(uv,hi),fract(lod));
 let factor=clamp(pow(max(d,0.1),-c.opts.y*0.5),0.5,2.0);
 return clamp(1.5*c.opts.x*pow(c.camera.z,c.extra.z)*factor*c.render.z,max(0.75,c.render.z),24.0*c.render.z);
}
// Largest radius any particle can take this frame: the density factor is at most 2.
fn sprite_radius_limit()->f32 {
 return clamp(1.5*c.opts.x*pow(c.camera.z,c.extra.z)*2.0*c.render.z,max(0.75,c.render.z),24.0*c.render.z);
}
// Centre of copy `copy` in framebuffer NDC, or false when the copy is culled.
fn sprite_copy(uv:vec2f,world:vec2f,radius:f32,copy:u32,center:ptr<function,vec2f>)->bool {
 var position=world;
 if(c.camera.w==1.0){
  let span=world_span();
  let offset=vec2f(f32(copy%3u)-1.0,f32(copy/3u)-1.0);
  position=(fract(uv)+offset)*half_world()*2.0-half_world();
  let margin=(radius+c.screen.zw)/span;
  let ghost_uv=cell_uv(position);
  if(any(ghost_uv < -margin) || any(ghost_uv > vec2f(1)+margin)){return false;}
 }else if(any(uv<vec2f(0)) || any(uv>vec2f(1))){return false;}
 *center=cell_ndc(position)*c.canvas.zw/c.screen.xy;
 return true;
}
// Peak weight of a sprite centred at `center` (framebuffer NDC).
fn sprite_amplitude(center:vec2f,radius:f32)->f32 {
 // Subpixel-thin wrapped worlds attenuate mass with their bounded footprint.
 let energy_scale=min(c.render.z,radius);
 var amplitude=0.4*energy_scale*energy_scale*0.85896/(radius*radius);
 // Correct the truncated discrete footprint where subpixel phase matters most.
 // At larger footprints the continuous normalization is within a few percent.
 if(radius<=3.0){
  let center_px=vec2f(center.x*0.5+0.5,0.5-center.y*0.5)*c.screen.xy;
  let origin=floor(center_px);var energy=0.0;
  for(var y=-3;y<=3;y++){for(var x=-3;x<=3;x++){
   let delta=(origin+vec2f(f32(x),f32(y))+0.5-center_px)/radius;
   let rr=dot(delta,delta);if(rr<=1.0){energy+=exp(-rr/(2.0*0.45*0.45));}
  }}
  if(energy>0.0){amplitude=0.4*energy_scale*energy_scale/energy;}
 }
 return amplitude;
}
