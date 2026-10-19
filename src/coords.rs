//! Source-faithful Fluoddity coordinate and camera math (MIT, Jesse Gelders, 2026).
use anyhow::{Result, ensure};
pub type Vec2 = [f64; 2];
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sizing {
    pub entity_count: u32,
    pub canvas: [u32; 2],
    pub sqrt_world_size: f32,
}
pub fn sizing_for(world_size: f64, aspect: f64) -> Result<Sizing> {
    ensure!(
        world_size.is_finite() && world_size >= 0.0,
        "world size must be finite and nonnegative"
    );
    ensure!(
        aspect.is_finite() && aspect > 0.0,
        "aspect must be finite and positive"
    );
    let count = (600_000.0 * world_size).trunc().max(1.0);
    let dim = (1024.0 * world_size.sqrt()).trunc().max(16.0);
    let canvas = canvas_dimensions(aspect, dim);
    ensure!(
        count <= u32::MAX as f64
            && canvas
                .iter()
                .all(|v| v.is_finite() && *v <= u32::MAX as f64),
        "sizing exceeds u32 limits"
    );
    Ok(Sizing {
        entity_count: count as u32,
        canvas: [canvas[0] as u32, canvas[1] as u32],
        sqrt_world_size: world_size.sqrt() as f32,
    })
}
pub fn canvas_dimensions(aspect: f64, dim: f64) -> Vec2 {
    let s = aspect.sqrt();
    [(dim * s).round().max(1.0), (dim / s).round().max(1.0)]
}
pub fn world_half_extent(canvas: Vec2) -> Vec2 {
    let s = (canvas[0] / canvas[1]).sqrt();
    [s, 1.0 / s]
}
pub fn world_to_uv(p: Vec2, canvas: Vec2) -> Vec2 {
    let e = world_half_extent(canvas);
    [p[0] / (2.0 * e[0]) + 0.5, p[1] / (2.0 * e[1]) + 0.5]
}
pub fn uv_to_world(p: Vec2, canvas: Vec2) -> Vec2 {
    let e = world_half_extent(canvas);
    [(p[0] - 0.5) * 2.0 * e[0], (p[1] - 0.5) * 2.0 * e[1]]
}
pub fn uv_radius_to_world(r: f64) -> f64 {
    r * 2.0
}
pub fn world_to_ndc(p: Vec2, canvas: Vec2) -> Vec2 {
    let e = world_half_extent(canvas);
    [p[0] / e[0], p[1] / e[1]]
}
pub fn ndc_to_world(p: Vec2, canvas: Vec2) -> Vec2 {
    let e = world_half_extent(canvas);
    [p[0] * e[0], p[1] * e[1]]
}
pub fn letterbox_scale(canvas: Vec2, window: Vec2) -> Vec2 {
    if window[0] <= 0.0 || window[1] <= 0.0 {
        return [1.0, 1.0];
    }
    let ca = canvas[0] / canvas[1];
    let wa = window[0] / window[1];
    if wa > ca {
        [ca / wa, 1.0]
    } else {
        [1.0, wa / ca]
    }
}
pub fn world_to_screen_ndc(p: Vec2, canvas: Vec2, window: Vec2, pan: Vec2, zoom: f64) -> Vec2 {
    let p = world_to_ndc([p[0] - pan[0], p[1] - pan[1]], canvas);
    let s = letterbox_scale(canvas, window);
    [p[0] * zoom * s[0], p[1] * zoom * s[1]]
}
pub fn screen_ndc_to_world(p: Vec2, canvas: Vec2, window: Vec2, pan: Vec2, zoom: f64) -> Vec2 {
    let s = letterbox_scale(canvas, window);
    let mut p = [
        if s[0] != 0.0 { p[0] / s[0] } else { 0.0 },
        if s[1] != 0.0 { p[1] / s[1] } else { 0.0 },
    ];
    if zoom != 0.0 {
        p[0] /= zoom;
        p[1] /= zoom;
    }
    let p = ndc_to_world(p, canvas);
    [p[0] + pan[0], p[1] + pan[1]]
}
pub fn screen_to_ndc(p: Vec2, window: Vec2) -> Vec2 {
    if window[0] <= 0.0 || window[1] <= 0.0 {
        [0.0, 0.0]
    } else {
        [2.0 * p[0] / window[0] - 1.0, 1.0 - 2.0 * p[1] / window[1]]
    }
}
pub fn ndc_to_screen(p: Vec2, window: Vec2) -> Vec2 {
    [
        (p[0] + 1.0) * 0.5 * window[0],
        (1.0 - p[1]) * 0.5 * window[1],
    ]
}
pub fn screen_to_world(p: Vec2, window: Vec2, canvas: Vec2, pan: Vec2, zoom: f64) -> Vec2 {
    screen_ndc_to_world(screen_to_ndc(p, window), canvas, window, pan, zoom)
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub pan: [f32; 2],
    pub zoom: f32,
    pub trail_mode: bool,
}
impl Default for Camera {
    fn default() -> Self {
        Self {
            pan: [0.0; 2],
            zoom: 1.0,
            trail_mode: false,
        }
    }
}
impl Camera {
    pub fn reset(&mut self) {
        self.pan = [0.0; 2];
        self.zoom = 1.0;
    }
    pub fn set_zoom(&mut self, zoom: f32) {
        if zoom.is_finite() {
            self.zoom = zoom.clamp(0.1, 100.0);
        }
    }
    pub fn zoom_by_factor(&mut self, factor: f32) {
        self.set_zoom(self.zoom * factor);
    }
    pub fn toggle_mode(&mut self) {
        self.trail_mode = !self.trail_mode;
    }
    pub fn screen_to_world(&self, p: Vec2, window: Vec2, canvas: Vec2) -> Vec2 {
        screen_to_world(p, window, canvas, self.pan.map(f64::from), self.zoom as f64)
    }
    pub fn zoom_at_pixel(&mut self, notches: f64, pixel: Vec2, window: Vec2, canvas: Vec2) {
        if notches == 0.0 {
            return;
        }
        let before = self.screen_to_world(pixel, window, canvas);
        self.set_zoom((self.zoom as f64 * 1.1f64.powf(notches)) as f32);
        let after = self.screen_to_world(pixel, window, canvas);
        for k in 0..2 {
            self.pan[k] = (self.pan[k] as f64 + before[k] - after[k]) as f32;
        }
    }
    pub fn pan_by_fraction(&mut self, fraction: Vec2, canvas: Vec2) {
        let step = 2.0 * world_half_extent(canvas)[1] / self.zoom as f64;
        for (k, v) in fraction.iter().enumerate() {
            self.pan[k] = (self.pan[k] as f64 + v * step) as f32;
        }
    }
    pub fn pan_by_pixels(&mut self, delta: Vec2, window: Vec2, canvas: Vec2) {
        let a = self.screen_to_world([0.0; 2], window, canvas);
        let b = self.screen_to_world(delta, window, canvas);
        for k in 0..2 {
            self.pan[k] = (self.pan[k] as f64 - (b[k] - a[k])) as f32;
        }
    }
}
