// Last displayed displacement, in reference pixels. Coarse bilinear interpolation
// with >=8 reference-pixel spacing and |component|<=0.9 has ||Jd||infinity<=0.45.
struct CellMapUniforms {canvas:vec4f,camera:vec4f,surface:vec4f,params:vec4f}
@group(2) @binding(0) var cell_map:texture_2d<f32>;
@group(2) @binding(1) var<uniform> cell_map_u:CellMapUniforms;
fn cell_map_delta(p:vec2f)->vec2f {
 let res=vec2i(textureDimensions(cell_map));let q=clamp(p/cell_map_u.params.xy,vec2f(0),vec2f(1))*vec2f(res-1);let base=vec2i(floor(q));let f=fract(q);var value=vec2f(0);
 for(var y=0;y<2;y++){for(var x=0;x<2;x++){value+=textureLoad(cell_map,min(base+vec2i(x,y),res-1),0).xy*select(1.0-f.x,f.x,x==1)*select(1.0-f.y,f.y,y==1);}}
 return value*cell_map_u.params.z;
}
fn cell_map_screen(p:vec2f)->vec2f {let n=world_to_screen_ndc(p,cell_map_u.canvas.xy,cell_map_u.surface.xy,cell_map_u.camera.xy,cell_map_u.camera.z);return vec2f(n.x*0.5+0.5,0.5-n.y*0.5)*cell_map_u.surface.xy+cell_map_u.surface.zw;}
fn cell_map_world(p:vec2f)->vec2f {let uv=(p-cell_map_u.surface.zw)/cell_map_u.surface.xy;return screen_ndc_to_world(vec2f(uv.x*2.0-1.0,1.0-uv.y*2.0),cell_map_u.canvas.xy,cell_map_u.surface.xy,cell_map_u.camera.xy,cell_map_u.camera.z);}
fn cell_display_to_world(raw:vec2f)->vec2f {if(cell_map_u.params.w<0.5){return raw;}let p=cell_map_screen(raw);return cell_map_world(p+cell_map_delta(p));}
fn cell_world_to_display(world:vec2f)->vec2f {if(cell_map_u.params.w<0.5){return world;}let q=cell_map_screen(world);var p=q;for(var i=0;i<10;i++){p=q-cell_map_delta(p);}return cell_map_world(p);}
