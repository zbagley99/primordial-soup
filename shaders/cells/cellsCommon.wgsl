#include "common.wgsl"
struct CellUniforms {
    canvas: vec4f, // world canvas, visible work resolution
    camera: vec4f, // pan, zoom, boundary
    screen: vec4f, // padded work resolution, padding
    grid: vec4f, // density resolution, displacement map resolution
    opts: vec4f, // size, density response, relief, wash
    look: vec4f, // reference palette, contrast, chromatic split, sharpening
    color: vec4f, // sensitivity, offset, cohort color, selected cohort
    render: vec4f, // inverse quantum, inverse samples, reference scale, quantum
    extra: vec4f, // exposure, softness, zoom response, allocated particle count
    background: vec4f, // linear background RGB, light theme
    structure: vec4f, // local density threshold, zoom-blended membranes, reserved
}
@group(0) @binding(0) var<uniform> c: CellUniforms;
fn cell_uv(p:vec2f)->vec2f { return world_to_uv(p,c.canvas.xy); }
fn cell_ndc(p:vec2f)->vec2f { return world_to_screen_ndc(p,c.canvas.xy,c.canvas.zw,c.camera.xy,c.camera.z); }
fn cell_hsv(h:f32)->vec3f {
    let p=abs(fract(vec3f(h)+vec3f(1.0,2.0/3.0,1.0/3.0))*6.0-3.0);
    return mix(vec3f(1.0),clamp(p-1.0,vec3f(0),vec3f(1)),0.8);
}

// Top-left padded framebuffer -> simulation world. Letterboxing uses the theme.
fn cell_world(p:vec2f)->vec2f {
 let screen=(p-c.screen.zw)/c.canvas.zw;
 return screen_ndc_to_world(vec2f(screen.x*2.0-1.0,1.0-screen.y*2.0),c.canvas.xy,c.canvas.zw,c.camera.xy,c.camera.z);
}
fn cell_inside(p:vec2f)->bool {
 let uv=cell_uv(cell_world(p));return all(uv>=vec2f(0)) && all(uv<=vec2f(1));
}

fn cell_period()->vec2f {
 let half_world=world_half_extent_from_res(c.canvas.xy);
 return abs(cell_ndc(half_world)-cell_ndc(-half_world))*c.canvas.zw*0.5;
}

// Reversing ordered positive-float bits makes zero the absent-site sentinel.
// This permits an exact clear_buffer reset and atomicMax top-two reduction.
struct CellSites { first:atomic<u32>, second:atomic<u32> }
fn cell_distance_key(rr:f32)->u32 {return 0x3f800000u-bitcast<u32>(clamp(rr,0.0,1.0));}
fn cell_key_distance(key:u32)->f32 {return sqrt(bitcast<f32>(0x3f800000u-key));}
