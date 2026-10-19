pub mod app;
pub mod archive;
pub mod calibration;
pub mod cells;
mod cells_tiled;
pub mod config;
pub mod controls;
pub mod coords;
pub mod export;
#[cfg(not(target_arch = "wasm32"))]
pub mod field_validation;
pub mod gpu;
pub mod history;
pub mod interaction;
pub mod network;
#[cfg(not(target_arch = "wasm32"))]
pub mod numerical;
pub mod performance;
pub mod profiling;
pub mod qr;
pub mod rate_calibration;
pub mod record;
pub mod render;
pub mod share;
pub mod simulation;
pub mod webcam;

#[cfg(target_arch = "wasm32")]
mod web;
