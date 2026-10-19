@group(0) @binding(0) var source:texture_2d<f32>;
@group(0) @binding(1) var destination:texture_storage_2d<rgba16float,write>;
// Separable 8-tap Gaussian (sigma 1.5 source texels) per axis. A workgroup's
// 8x8 outputs read at most a 24x24 source footprint at the 2x reduction ratio;
// it is loaded once, filtered along rows, then along columns. Edge texels are
// clamped exactly as the direct form clamps them.
const SPAN:i32=24;
var<workgroup> tile:array<vec4f,576>;
var<workgroup> rows:array<vec4f,192>;
// Stored wash = mass-weighted color / WASH_RANGE (set by the first reduction),
// keeping dense piles inside the half-float range.
var<private> gain:f32=1.0;
fn gauss(delta:f32)->f32 {return exp(-delta*delta/(2.0*1.5*1.5));}
// Align actual texel centers, including odd and rectangular mip dimensions.
fn center(o:vec2i,res:vec2i,size:vec2i)->vec2f {return (vec2f(o)+0.5)*vec2f(res)/vec2f(size)-0.5;}
struct Footprint { first:vec2i, origin:vec2i, fits:bool }
fn footprint(wg:vec3u)->Footprint {
 let size=vec2i(textureDimensions(destination));let res=vec2i(textureDimensions(source));
 let first=vec2i(wg.xy)*8;let last=min(first+7,size-1);
 let origin=vec2i(floor(center(first,res,size)))-3;
 let extent=vec2i(floor(center(last,res,size)))+5-origin;
 return Footprint(first,origin,all(extent<=vec2i(SPAN)));
}
fn load_tile(origin:vec2i,li:u32){
 let res=vec2i(textureDimensions(source));
 for(var i=i32(li);i<SPAN*SPAN;i+=64){
  tile[i]=textureLoad(source,clamp(origin+vec2i(i%SPAN,i/SPAN),vec2i(0),res-1),0);
 }
 workgroupBarrier();
}
fn filter_tile(f:Footprint,lid:vec3u,li:u32){
 let size=vec2i(textureDimensions(destination));let res=vec2i(textureDimensions(source));
 for(var i=i32(li);i<SPAN*8;i+=64){
  let column=i%8;let row=i/8;
  let c=center(vec2i(f.first.x+column,0),res,size).x;let p=i32(floor(c));var sum=vec4f(0);
  for(var k=0;k<8;k++){let s=p+k-3;sum+=tile[row*SPAN+s-f.origin.x]*gauss(f32(s)-c);}
  rows[row*8+column]=sum;
 }
 workgroupBarrier();
 let o=f.first+vec2i(lid.xy);if(any(o>=size)){return;}
 let c=center(o,res,size);let p=vec2i(floor(c));var sum=vec4f(0);var wx=0.0;var wy=0.0;
 for(var k=0;k<8;k++){
  let s=p.y+k-3;let w=gauss(f32(s)-c.y);sum+=rows[(s-f.origin.y)*8+i32(lid.x)]*w;wy+=w;
  wx+=gauss(f32(p.x+k-3)-c.x);
 }
 textureStore(destination,o,sum*(gain/(wx*wy)));
}
// Small mips with odd ratios: direct 8x8 taps.
fn direct(first:vec2i,lid:vec3u){
 let size=vec2i(textureDimensions(destination));let res=vec2i(textureDimensions(source));
 let o=first+vec2i(lid.xy);if(any(o>=size)){return;}
 let c=center(o,res,size);let p=vec2i(floor(c));var color=vec4f(0);var total=0.0;
 var wx:array<f32,8>;var wy:array<f32,8>;
 for(var k=0;k<8;k++){
  let delta=vec2f(p+vec2i(k-3))-c;
  wx[k]=gauss(delta.x);
  wy[k]=gauss(delta.y);
 }
 for(var y=0;y<8;y++){for(var x=0;x<8;x++){
 let weight=wx[x]*wy[y];
 color+=textureLoad(source,clamp(p+vec2i(x-3,y-3),vec2i(0),res-1),0)*weight;total+=weight;
 }}
 textureStore(destination,o,color*(gain/total));
}
@compute @workgroup_size(8,8) fn main(@builtin(workgroup_id) wg:vec3u,@builtin(local_invocation_id) lid:vec3u,@builtin(local_invocation_index) li:u32){
 let f=footprint(wg);
 if(f.fits){load_tile(f.origin,li);filter_tile(f,lid,li);}else{direct(f.first,lid);}
}
