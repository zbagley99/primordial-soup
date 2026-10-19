#include "cellsCommon.wgsl"
@group(0) @binding(1) var image:texture_2d<f32>;
@group(0) @binding(2) var selection:texture_2d<f32>;
@group(0) @binding(3) var soft:texture_2d<f32>;
@group(0) @binding(4) var mapping:texture_2d<f32>;
@group(0) @binding(5) var map_out:texture_storage_2d<rgba32float,write>;
@group(0) @binding(6) var density:texture_2d<f32>;
@group(0) @binding(7) var<storage,read> membrane_history:array<vec2f>;
@group(0) @binding(9) var mass:texture_2d<f32>;
@group(0) @binding(10) var scene_out:texture_storage_2d<rgba16float,write>;
@group(0) @binding(11) var marks_out:texture_storage_2d<rgba16float,write>;
@group(0) @binding(12) var shaded_out:texture_storage_2d<rgba16float,write>;
@group(0) @binding(13) var shaded:texture_2d<f32>;
@group(0) @binding(14) var wash_sampler:sampler;
fn image_at(p:vec2f)->vec4f {
 let res=vec2i(textureDimensions(image));let q=vec2i(floor(p-0.5));let f=fract(p-0.5);var v=vec4f(0);
 for(var y=0;y<2;y++){for(var x=0;x<2;x++){let a=q+vec2i(x,y);if((all(a>=vec2i(0)) && all(a<res))){v+=textureLoad(image,a,0)*select(1.0-f.x,f.x,x==1)*select(1.0-f.y,f.y,y==1);}}}return v;
}
fn selected_at(p:vec2f)->f32 {
 let res=vec2i(textureDimensions(selection));let q=vec2i(floor(p-0.5));let f=fract(p-0.5);var v=0.0;
 for(var y=0;y<2;y++){for(var x=0;x<2;x++){let a=q+vec2i(x,y);if((all(a>=vec2i(0)) && all(a<res))){v+=textureLoad(selection,a,0).r*select(1.0-f.x,f.x,x==1)*select(1.0-f.y,f.y,y==1);}}}return v;
}
// Positive cubic B-spline reconstruction hides the coarse wash lattice without
// ringing, clipping weak pigment or dividing color by mass before filtering.
fn cubic(f:f32)->vec4f {
 let a=1.0-f;let ff=f*f;let fff=ff*f;
 return vec4f(a*a*a,3.0*fff-6.0*ff+4.0,-3.0*fff+3.0*ff+3.0*f+1.0,fff)/6.0;
}
fn cubic_derivative(f:f32)->vec4f {
 return vec4f(-0.5*(1.0-f)*(1.0-f),1.5*f*f-2.0*f,-1.5*f*f+f+0.5,0.5*f*f);
}
// Each pair of same-signed weights becomes one bilinear fetch placed between
// its two texels at the ratio of their weights: 4 fetches per value, 8 per
// gradient, instead of 16 loads each. Positions are texel-center coordinates.
struct Pairs { weight:vec2f, at:vec2f }
fn pairs(w:vec4f,q:f32)->Pairs {
 let s=vec2f(w.x+w.y,w.z+w.w);
 return Pairs(s,vec2f(q-1.0+w.y/s.x,q+1.0+w.w/s.y)+0.5);
}
fn wash_tap(x:f32,y:f32,dims:vec2f,l:i32)->vec4f {return textureSampleLevel(soft,wash_sampler,vec2f(x,y)/dims,f32(l));}
fn soft_value(p:vec2f,l:i32)->vec4f {
 let dims=vec2f(textureDimensions(soft,l));let pos=p/c.screen.xy*dims-0.5;let q=floor(pos);let f=pos-q;
 let x=pairs(cubic(f.x),q.x);let y=pairs(cubic(f.y),q.y);
 return (wash_tap(x.at.x,y.at.x,dims,l)*x.weight.x+wash_tap(x.at.y,y.at.x,dims,l)*x.weight.y)*y.weight.x
  +(wash_tap(x.at.x,y.at.y,dims,l)*x.weight.x+wash_tap(x.at.y,y.at.y,dims,l)*x.weight.y)*y.weight.y;
}
fn soft_gradient(p:vec2f,l:i32)->vec2f {
 let dims=vec2f(textureDimensions(soft,l));let pos=p/c.screen.xy*dims-0.5;let q=floor(pos);let f=pos-q;
 let x=pairs(cubic(f.x),q.x);let y=pairs(cubic(f.y),q.y);
 let dx=pairs(cubic_derivative(f.x),q.x);let dy=pairs(cubic_derivative(f.y),q.y);
 let gx=(wash_tap(dx.at.x,y.at.x,dims,l).a*dx.weight.x+wash_tap(dx.at.y,y.at.x,dims,l).a*dx.weight.y)*y.weight.x
  +(wash_tap(dx.at.x,y.at.y,dims,l).a*dx.weight.x+wash_tap(dx.at.y,y.at.y,dims,l).a*dx.weight.y)*y.weight.y;
 let gy=(wash_tap(x.at.x,dy.at.x,dims,l).a*x.weight.x+wash_tap(x.at.y,dy.at.x,dims,l).a*x.weight.y)*dy.weight.x
  +(wash_tap(x.at.x,dy.at.y,dims,l).a*x.weight.x+wash_tap(x.at.y,dy.at.y,dims,l).a*x.weight.y)*dy.weight.y;
 return vec2f(gx,gy)*dims/c.screen.xy;
}
// Half-float wash texels are stored divided by this (see cellsBlurFirst).
const WASH_RANGE:f32=16.0;
fn wash_value(p:vec2f,level:f32)->vec4f {
 let lo=i32(floor(level));let a=soft_value(p,lo);let f=fract(level);
 if(f==0.0){return a*WASH_RANGE;}
 return mix(a,soft_value(p,min(lo+1,i32(textureNumLevels(soft))-1)),f)*WASH_RANGE;
}
fn wash_gradient(p:vec2f,level:f32)->vec2f {
 let lo=i32(floor(level));let a=soft_gradient(p,lo);let f=fract(level);
 if(f==0.0){return a*WASH_RANGE;}
 return mix(a,soft_gradient(p,min(lo+1,i32(textureNumLevels(soft))-1)),f)*WASH_RANGE;
}
// Compensate optical coverage as normalized sprites grow. Keep accumulation,
// pigment ratios and selection mass unchanged; stop gain at the sprite cap.
fn density_at(uv:vec2f,l:i32)->f32 {
 let res=vec2i(textureDimensions(density,l));let p=uv*vec2f(res)-0.5;let q=vec2i(floor(p));let f=fract(p);var v=0.0;
 for(var y=0;y<2;y++){for(var x=0;x<2;x++){
 var a=q+vec2i(x,y);if(c.camera.w==1.0){a=(a%res+res)%res;}else{a=clamp(a,vec2i(0),res-1);}
 v+=textureLoad(density,a,l).r*select(1.0-f.x,f.x,x==1)*select(1.0-f.y,f.y,y==1);
 }}return v;
}
fn coverage_gain(p:vec2f)->f32 {
 let uv=cell_uv(cell_world(p));
 let lod=clamp(log2(max(1.0,c.grid.y/max(cell_period().y,1.0))),0.0,f32(textureNumLevels(density)-1u));
 let lo=i32(floor(lod));let hi=min(lo+1,i32(textureNumLevels(density))-1);
 let d=mix(density_at(uv,lo),density_at(uv,hi),fract(lod));
 let factor=clamp(pow(max(d,0.1),-c.opts.y*0.5),0.5,2.0);
 let base=clamp(1.5*c.opts.x*factor*c.render.z,max(0.75,c.render.z),24.0*c.render.z);
 let radius=clamp(1.5*c.opts.x*pow(c.camera.z,c.extra.z)*factor*c.render.z,max(0.75,c.render.z),24.0*c.render.z);
 // Small discrete kernels have sharper peaks than the continuous footprint.
 // Ease in a modest correction as cells become large enough to resolve.
 return max(1.0,(radius*radius)/(base*base))*mix(1.0,1.5,smoothstep(1.0,8.0,c.camera.z));
}
// Mass alone (written by the first wash reduction), so relief heights fetch
// one scalar per tap instead of four.
fn mass_at(p:vec2f)->f32 {
 let res=vec2i(textureDimensions(mass));let q=vec2i(floor(p-0.5));let f=fract(p-0.5);var v=0.0;
 for(var y=0;y<2;y++){for(var x=0;x<2;x++){let a=q+vec2i(x,y);if((all(a>=vec2i(0)) && all(a<res))){v+=textureLoad(mass,a,0).r*select(1.0-f.x,f.x,x==1)*select(1.0-f.y,f.y,y==1);}}}return v;
}
fn height_at(p:vec2f)->f32 {let m=mass_at(p);return log(1.0+64.0*min(m,10000.0));}
@compute @workgroup_size(8,8) fn build_map(@builtin(global_invocation_id) id:vec3u){
 let res=textureDimensions(map_out);if(any(id.xy>=res)){return;}
 let p=vec2f(id.xy)/vec2f(res-1u)*c.screen.xy;
 let step=max(c.render.z,0.5);let g=vec2f(height_at(p+vec2f(step,0))-height_at(p-vec2f(step,0)),height_at(p+vec2f(0,step))-height_at(p-vec2f(0,step)))*0.5;
 let period=cell_period();
 let relief=select(c.opts.z,min(c.opts.z,0.1*min(period.x,period.y)/c.render.z),c.camera.w==1.0);
 let d=clamp(relief*g/(1.0+length(g)),vec2f(-0.9),vec2f(0.9));
 textureStore(map_out,id.xy,vec4f(d,0,0));
}
fn displacement(p:vec2f)->vec2f {
 let res=vec2i(textureDimensions(mapping));let pos=clamp(p/c.screen.xy,vec2f(0),vec2f(1))*vec2f(res-1);let q=vec2i(floor(pos));let f=fract(pos);var v=vec2f(0);
 for(var y=0;y<2;y++){for(var x=0;x<2;x++){v+=textureLoad(mapping,min(q+vec2i(x,y),res-1),0).xy*select(1.0-f.x,f.x,x==1)*select(1.0-f.y,f.y,y==1);}}return v*c.render.z;
}
fn pigment(v:vec4f)->vec3f {if(v.a<=0.0000001){return vec3f(0);}return clamp(v.rgb/v.a,vec3f(0),vec3f(1));}
fn ink(m:f32)->f32 {let value=max(m*c.extra.x,0.0);let compressed=log(1.0+value*c.extra.y)/max(c.extra.y,0.1);return 1.0-exp(-compressed*2.5);}
// World-space occupancy is independent of hue, exposure and sprite size.
// Zero bypasses suppression exactly. A soft shoulder avoids hard contour cuts.
fn structure_visibility(p:vec2f)->f32 {
 let threshold=c.structure.x;if(threshold<=0.0){return 1.0;}
 let d=density_at(cell_uv(cell_world(p)),0);
 return smoothstep(threshold,threshold*1.5,d);
}
fn membrane_at(p:vec2f)->f32 {
 if(c.structure.y<=0.0){return 1.0;}
 let res=vec2i(c.screen.xy);let q=vec2i(floor(p-0.5));let f=fract(p-0.5);var v=vec2f(0);
 for(var y=0;y<2;y++){for(var x=0;x<2;x++){
  let a=q+vec2i(x,y);
  if(all(a>=vec2i(0)) && all(a<res)){
   v+=membrane_history[u32(a.y)*u32(res.x)+u32(a.x)]*select(1.0-f.x,f.x,x==1)*select(1.0-f.y,f.y,y==1);
  }
 }}
 return mix(1.0,clamp(v.x+0.9*v.y,0.0,1.0),c.structure.y);
}
fn fine_ink(m:f32,p:vec2f)->f32 {return ink(m*coverage_gain(p)*mix(4.0,1.0,c.background.w))*structure_visibility(p)*membrane_at(p);}
// One evaluation supplies the display color and the bloom marks, which share
// the displaced mass sample and its fine ink.
struct Shaded { color:vec4f, marks:vec3f }
fn shade(p:vec2f)->Shaded {
 let background=c.background.rgb;
 if(!cell_inside(p)){return Shaded(vec4f(background,1),vec3f(0));}
 let q=p+displacement(p);let a=image_at(q);let period=cell_period();let shortest=min(period.x,period.y);
 let limit=select(f32(textureNumLevels(soft)-1u),max(0.0,floor(log2(max(1.0,shortest/32.0)))),c.camera.w==1.0);
 // Wider Gaussian prefilter at a finer mip keeps the halo round while reducing
 // reconstruction texel spacing, rather than merely enlarging a coarse tile.
 let level=clamp(log2(max(1.0,24.0*c.render.z))-2.0,0.0,min(limit,f32(textureNumLevels(soft)-1u)));
 let step=select(max(4.0*c.render.z,1.0),min(max(4.0*c.render.z,1.0),0.1*shortest),c.camera.w==1.0);
 // The same reconstructed field supplies both color and its analytic slope.
 let g=wash_gradient(p,level)*(2.0*step);
 let broad_shift=select(c.render.z,min(c.render.z,0.1*shortest),c.camera.w==1.0);
 let broad=wash_value(p+g/(1.0+length(g))*broad_shift,level);
 let wash=ink(broad.a)*c.opts.w*select(1.0,smoothstep(16.0,32.0,shortest),c.camera.w==1.0);
 let fine=fine_ink(a.a,q);
 var rgb=mix(background,pigment(broad),wash*0.55*mix(1.0,0.3,c.structure.y));rgb=mix(rgb,pigment(a),fine*0.82);
 let center=height_at(q);let lap=(height_at(q+vec2f(c.render.z,0))+height_at(q-vec2f(c.render.z,0))+height_at(q+vec2f(0,c.render.z))+height_at(q-vec2f(0,c.render.z)))*0.25-center;
 rgb*=1.0-clamp(abs(lap)*c.look.y*4.0,0.0,0.22)*structure_visibility(q);
 let chosen=select(1.0,clamp(selected_at(q)/max(a.a,0.0000001),0.0,1.0),a.a>0.0000001);
 return Shaded(vec4f(rgb,chosen),pigment(a)*fine);
}
fn pixel_point(pix:vec2i)->vec2f {return ((vec2f(pix)+0.5)/c.canvas.zw)*c.canvas.zw+c.screen.zw;}
// Separation keeps exact subpixel evaluation: interpolating shaded pixels
// would soften fine grain in the displaced channels.
fn separate(color:vec4f,p:vec2f)->vec4f {
 if(c.look.z<=0.0){return color;}
 let d=c.look.z*c.render.z;var o=color;o.r=shade(p+vec2f(d,0)).color.r;o.b=shade(p-vec2f(d,0)).color.b;return o;
}
// Shade each pixel once. Sharpening reads neighbouring shades, so while it is
// active the pass also covers a margin around the image and writes an
// intermediate that finish() combines. Otherwise it writes the scene directly
// and finish() is not dispatched.
@compute @workgroup_size(8,8) fn shade_pass(@builtin(global_invocation_id) id:vec3u){
 let dims=vec2i(textureDimensions(shaded_out));if(any(vec2i(id.xy)>=dims)){return;}
 let size=vec2i(c.canvas.zw);let pix=vec2i(id.xy)-(dims-size)/2;
 let visible=all(pix>=vec2i(0)) && all(pix<size);let sharpen=c.look.w>0.0;
 if(!visible && !sharpen){return;}
 let p=pixel_point(pix);let s=shade(p);
 if(visible){
  textureStore(marks_out,pix,vec4f(s.marks,1));
  if(!sharpen){textureStore(scene_out,pix,separate(s.color,p));}
 }
 if(sharpen){textureStore(shaded_out,id.xy,s.color);}
}
// Bilinear in intermediate texel-center coordinates; integer offsets are exact loads.
fn shaded_at(t:vec2f)->vec4f {
 let res=vec2i(textureDimensions(shaded));let q=vec2i(floor(t));let f=t-floor(t);
 let a=textureLoad(shaded,clamp(q,vec2i(0),res-1),0);
 if(all(f==vec2f(0))){return a;}
 let b=textureLoad(shaded,clamp(q+vec2i(1,0),vec2i(0),res-1),0);
 let d=textureLoad(shaded,clamp(q+vec2i(0,1),vec2i(0),res-1),0);
 let e=textureLoad(shaded,clamp(q+vec2i(1,1),vec2i(0),res-1),0);
 return mix(mix(a,b,f.x),mix(d,e,f.x),f.y);
}
@compute @workgroup_size(8,8) fn finish(@builtin(global_invocation_id) id:vec3u){
 if(c.look.w<=0.0){return;}
 let size=vec2i(c.canvas.zw);if(any(vec2i(id.xy)>=size)){return;}
 let t=vec2f(vec2i(id.xy)+(vec2i(textureDimensions(shaded))-size)/2);
 var color=shaded_at(t);
 let r=max(c.render.z,1.0);let avg=(shaded_at(t+vec2f(r,0)).rgb+shaded_at(t-vec2f(r,0)).rgb+shaded_at(t+vec2f(0,r)).rgb+shaded_at(t-vec2f(0,r)).rgb)*0.25;color=vec4f(clamp(color.rgb+(color.rgb-avg)*c.look.w,vec3f(0),vec3f(1)),color.a);
 textureStore(scene_out,id.xy,separate(color,pixel_point(vec2i(id.xy))));
}
