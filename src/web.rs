use crate::app::{self, SoupApp, StartOptions};
use wasm_bindgen::{JsCast, prelude::*};

/// eframe may skip UI and repaint requests at a visibility boundary. Observe both
/// transitions explicitly so the wall clock is suspended before any resume work.
pub struct VisibilityWake {
    document: web_sys::Document,
    callback: Closure<dyn FnMut()>,
    changed: std::sync::Arc<std::sync::atomic::AtomicBool>,
}
impl VisibilityWake {
    pub fn new(ctx: eframe::egui::Context) -> Result<Self, JsValue> {
        let document = web_sys::window()
            .and_then(|w| w.document())
            .ok_or_else(|| JsValue::from_str("No document"))?;
        let changed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let notification = changed.clone();
        let callback = Closure::wrap(Box::new(move || {
            notification.store(true, std::sync::atomic::Ordering::Relaxed);
            ctx.request_repaint();
        }) as Box<dyn FnMut()>);
        document.add_event_listener_with_callback(
            "visibilitychange",
            callback.as_ref().unchecked_ref(),
        )?;
        Ok(Self {
            document,
            callback,
            changed,
        })
    }
    pub fn take_transition(&self) -> bool {
        self.changed
            .swap(false, std::sync::atomic::Ordering::Relaxed)
    }
}
impl Drop for VisibilityWake {
    fn drop(&mut self) {
        let _ = self.document.remove_event_listener_with_callback(
            "visibilitychange",
            self.callback.as_ref().unchecked_ref(),
        );
    }
}

#[wasm_bindgen]
pub async fn start(canvas: web_sys::HtmlCanvasElement) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("No browser window"))?;
    let mut opts = StartOptions::default();
    match wasm_bindgen_futures::JsFuture::from(load_core()).await {
        Ok(value) => opts.persisted = value.as_string().filter(|s| !s.is_empty()),
        Err(error) => {
            opts.storage_error = Some(format!(
                "Project storage unavailable: {error:?}. Download JSON to preserve projects."
            ))
        }
    }
    let search = window.location().search().unwrap_or_default();
    for pair in search.trim_start_matches('?').split('&') {
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
        let value = value.replace('+', " ");
        let value = js_sys::decode_uri_component(&value)
            .ok()
            .and_then(|s| s.as_string())
            .unwrap_or(value);
        match key {
            "preset" => opts.preset = Some(value),
            "world" => opts.world_size = value.parse().ok(),
            "steps" => opts.physics_steps = value.parse().ok(),
            "ticks" => opts.ticks_per_second = value.parse().ok(),
            "fps" => opts.presentation_fps = value.parse().ok(),
            "debug" => opts.debug = true,
            "paused" => opts.paused = true,
            "nopanel" => opts.no_panels = true,
            "nocalibrate" => opts.no_calibrate = true,
            "camera" => opts.trail_mode = value == "trail",
            "name" => opts.name = Some(value),
            "splash" => {
                if ["welcome", "guide", "controls"].contains(&value.as_str()) {
                    opts.splash = Some(value)
                }
            }
            "worldSize" => {
                opts.proposed.world_size = value
                    .parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite() && (0.05..=27.0).contains(v))
            }
            "canvasAspect" => {
                opts.proposed.canvas_aspect = value
                    .parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite() && (0.1..=10.0).contains(v))
            }
            "brightness" => {
                opts.proposed.brightness = value
                    .parse::<f32>()
                    .ok()
                    .filter(|v| v.is_finite() && (0.1..=4.0).contains(v))
            }
            "tonemapSoftness" => {
                opts.proposed.tonemap_softness = value
                    .parse::<f32>()
                    .ok()
                    .filter(|v| v.is_finite() && (0.1..=5.0).contains(v))
            }
            "trailmap" => {
                opts.proposed.trail_mode = match value.as_str() {
                    "1" | "true" | "on" => Some(true),
                    "0" | "false" | "off" => Some(false),
                    _ => None,
                }
            }
            _ => {}
        }
    }
    opts.shared = window.location().hash().ok().filter(|h| !h.is_empty());
    let runner = eframe::WebRunner::new();
    runner
        .start(
            canvas,
            eframe::WebOptions {
                renderer: eframe::Renderer::Wgpu,
                wgpu_options: app::wgpu_options(),
                // The app paces presentation itself. eframe's limiter rounds every
                // repaint up to a 1/60 s grid that is not vsync-aligned, which beats
                // against the display and alternates one- and two-vsync intervals.
                max_fps: None,
                ..Default::default()
            },
            Box::new(move |cc| Ok(Box::new(SoupApp::new(cc, opts)?))),
        )
        .await?;
    // WebRunner owns event handlers which keep the runner alive.
    Ok(())
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(catch,js_namespace=globalThis,js_name=soupPickImage)]
    async fn pick_image() -> Result<JsValue, JsValue>;
    #[wasm_bindgen(catch,js_namespace=globalThis,js_name=soupDecodeImage)]
    async fn decode_image(file: JsValue) -> Result<JsValue, JsValue>;
}

#[wasm_bindgen(module = "/web/services.js")]
extern "C" {
    #[wasm_bindgen(js_name=copyPng)]
    fn copy_png(png: js_sys::Uint8Array, name: &str) -> js_sys::Promise;
    #[wasm_bindgen(js_name=archiveAppend)]
    fn archive_append(text: &str) -> js_sys::Promise;
    #[wasm_bindgen(js_name=archiveRead)]
    fn archive_read() -> js_sys::Promise;
    #[wasm_bindgen(js_name=archiveClear)]
    fn archive_clear() -> js_sys::Promise;
    #[wasm_bindgen(js_name=loadCore)]
    fn load_core() -> js_sys::Promise;
    #[wasm_bindgen(js_name=saveCore)]
    fn save_core(text: &str) -> js_sys::Promise;
}
pub type Notices = std::sync::Arc<std::sync::Mutex<Vec<String>>>;
fn notify_promise(
    promise: js_sys::Promise,
    notices: Notices,
    ctx: eframe::egui::Context,
    success: Option<&'static str>,
    failure: &'static str,
) {
    wasm_bindgen_futures::spawn_local(async move {
        let message = match wasm_bindgen_futures::JsFuture::from(promise).await {
            Ok(value) => value.as_string().or_else(|| success.map(str::to_owned)),
            Err(e) => Some(format!("{failure}: {e:?}")),
        };
        if let Some(message) = message {
            notices.lock().unwrap().push(message);
            ctx.request_repaint();
        }
    });
}
pub fn clipboard_png(bytes: &[u8], notices: Notices, ctx: eframe::egui::Context) {
    notify_promise(
        copy_png(js_sys::Uint8Array::from(bytes), "primordial-soup.png"),
        notices,
        ctx,
        None,
        "Screenshot export failed",
    );
}
pub fn save_state(text: &str, notices: Notices, ctx: eframe::egui::Context, explicit: bool) {
    notify_promise(
        save_core(text),
        notices,
        ctx,
        explicit.then_some("Saved projects and preferences"),
        "Project save failed; download JSON to preserve your work",
    );
}
pub fn append_archive(text: &str, notices: Notices, ctx: eframe::egui::Context) {
    notify_promise(
        archive_append(text),
        notices,
        ctx,
        None,
        "Archive storage failed",
    );
}
pub fn clear_archive(notices: Notices, ctx: eframe::egui::Context) {
    notify_promise(
        archive_clear(),
        notices,
        ctx,
        Some("State archive cleared"),
        "Archive clear failed",
    );
}
pub fn export_archive(notices: Notices, ctx: eframe::egui::Context) {
    wasm_bindgen_futures::spawn_local(async move {
        let result = async {
            let value = wasm_bindgen_futures::JsFuture::from(archive_read())
                .await
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let events: Vec<crate::archive::Event> =
                serde_json::from_str(&value.as_string().unwrap_or_default())?;
            let document = crate::archive::document(&events)?;
            download(
                "Primordial-Soup-State-Archive.json",
                document.as_bytes(),
                "application/json",
            )?;
            anyhow::Ok(())
        }
        .await;
        notices.lock().unwrap().push(match result {
            Ok(()) => "State archive exported".into(),
            Err(e) => format!("Archive export failed: {e}"),
        });
        ctx.request_repaint();
    });
}
pub fn import_image(
    incoming: std::sync::Arc<std::sync::Mutex<Vec<Result<String, String>>>>,
    ctx: eframe::egui::Context,
) {
    wasm_bindgen_futures::spawn_local(async move {
        match pick_image().await {
            Ok(text) => {
                if let Some(text) = text.as_string() {
                    incoming.lock().unwrap().push(Ok(text));
                }
            }
            Err(e) => incoming
                .lock()
                .unwrap()
                .push(Err(format!("Image import: {e:?}"))),
        }
        ctx.request_repaint();
    });
}
pub fn import_image_file(
    file: JsValue,
    incoming: std::sync::Arc<std::sync::Mutex<Vec<Result<String, String>>>>,
    ctx: eframe::egui::Context,
) {
    wasm_bindgen_futures::spawn_local(async move {
        match decode_image(file).await {
            Ok(text) => {
                if let Some(text) = text.as_string() {
                    incoming.lock().unwrap().push(Ok(text));
                }
            }
            Err(e) => incoming
                .lock()
                .unwrap()
                .push(Err(format!("Image import: {e:?}"))),
        }
        ctx.request_repaint();
    });
}

pub fn set_panels_hidden(hidden: bool) {
    if let Some(body) = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.body())
    {
        let _ = body.set_attribute("data-panels-hidden", if hidden { "true" } else { "false" });
    }
}

pub fn status(text: &str) {
    if let Some(element) = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id("status"))
    {
        element.set_text_content(Some(text));
    }
}

pub fn download(name: &str, bytes: &[u8], mime: &str) -> anyhow::Result<()> {
    let result = (|| -> Result<(), JsValue> {
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("No browser window"))?;
        let document = window
            .document()
            .ok_or_else(|| JsValue::from_str("No browser document"))?;
        let array = js_sys::Uint8Array::from(bytes);
        let parts = js_sys::Array::new();
        parts.push(&array);
        let options = web_sys::BlobPropertyBag::new();
        options.set_type(mime);
        let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&parts, &options)?;
        let url = web_sys::Url::create_object_url_with_blob(&blob)?;
        let link = document
            .create_element("a")?
            .dyn_into::<web_sys::HtmlAnchorElement>()?;
        link.set_href(&url);
        link.set_download(name);
        document
            .body()
            .ok_or_else(|| JsValue::from_str("No document body"))?
            .append_child(&link)?;
        link.click();
        link.remove();
        // Keep the object URL alive until the download has consumed it.
        let cleanup = Closure::once_into_js(move || {
            let _ = web_sys::Url::revoke_object_url(&url);
        });
        window.set_timeout_with_callback_and_timeout_and_arguments_0(
            cleanup.unchecked_ref(),
            30_000,
        )?;
        Ok(())
    })();
    result.map_err(|e| anyhow::anyhow!("Browser download failed: {e:?}"))
}
