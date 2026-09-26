//! tinypt のレンダリング途中ビューア（egui / eframe）。`--features viewer` でだけビルドされる。
//!
//! CLI（`tinypt`）と同じオプションを受け付け（解析は [`tinypt::cli`] を共有）、別スレッドでレンダーしながら
//! 蓄積バッファを約 10 Hz で表示する。範囲は「進行を見るだけ」: 中断・保存・読み込み直しはあるが、
//! カメラ操作や設定の編集は無い。
//!
//! 表示画像は保存と同じ経路（`resolve_pixels` → `ppm_bytes` = 露出 → トーンマップ → sRGB → 8bit）で作る。
//! 表示用に別のトーンマップは持たない。デノイズはプレビューには掛けず、保存時に CLI と同じ扱いで掛ける。
//!
//! 検証用フック（環境変数）: `TINYPT_VIEWER_AUTOSAVE=PATH` は完了・中断の後に保存ボタンと同じ処理で保存し、
//! `TINYPT_VIEWER_CANCEL_AT_TILES=N` は N タイルのマージ後に中断する。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui;
use tinypt::cli::{load_with_overrides, parse_args, CliOverrides, USAGE};
use tinypt::math::Color;
use tinypt::{
    denoise, ppm_bytes, render_observed, resolve_pixels, OutputFormat, OutputSettings, RenderConfig, RenderProbe,
};

/// レンダー用スレッドの状態（GUI が読む）。
enum Phase {
    Loading,
    Rendering { probe: Arc<RenderProbe>, config: Arc<RenderConfig>, start: Instant },
    Finished { probe: Arc<RenderProbe>, config: Arc<RenderConfig>, elapsed: Duration, cancelled: bool },
    Failed(String),
}

/// 1 回のレンダー（読み込み直しごとに作り直す）。
struct Job {
    phase: Arc<Mutex<Phase>>,
    cancel: Arc<AtomicBool>,
}

impl Job {
    fn start(config: RenderConfig, overrides: Arc<CliOverrides>) -> Self {
        let phase = Arc::new(Mutex::new(Phase::Loading));
        let cancel = Arc::new(AtomicBool::new(false));
        let (p, c) = (phase.clone(), cancel.clone());
        std::thread::spawn(move || {
            let mut config = config;
            let scene = match load_with_overrides(&mut config, &overrides) {
                Ok(s) => s,
                Err(e) => {
                    *p.lock().unwrap() = Phase::Failed(format!("scene load failed: {e}"));
                    return;
                }
            };
            // ビューアはチェックポイントを使わない（中断・読み込み直しで古い状態を拾わないため）
            config.checkpoint_enabled = false;
            let probe = Arc::new(RenderProbe::new(config.width, config.height));
            let config = Arc::new(config);
            let start = Instant::now();
            *p.lock().unwrap() = Phase::Rendering { probe: probe.clone(), config: config.clone(), start };
            // 読み込み中に押された中断は、ここで probe に引き継ぐ
            if c.load(Ordering::Relaxed) {
                probe.cancel.store(true, Ordering::Relaxed);
            }
            let result = render_observed(&scene, &config, "", Some(&probe));
            *p.lock().unwrap() = match result {
                Ok(_) => {
                    let cancelled = probe.cancel.load(Ordering::Relaxed);
                    Phase::Finished { probe, config, elapsed: start.elapsed(), cancelled }
                }
                Err(e) => Phase::Failed(format!("render failed: {e}")),
            };
        });
        Job { phase, cancel }
    }

    fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
        if let Phase::Rendering { probe, .. } = &*self.phase.lock().unwrap() {
            probe.cancel.store(true, Ordering::Relaxed);
        }
    }
}

/// 保存と同じ経路でリニア RGB を作る（蓄積を解決 → 必要ならデノイズ）。
fn resolved_pixels(probe: &RenderProbe, config: &RenderConfig, denoise_it: bool) -> Vec<Color> {
    let mut pixels = probe.with_buffers(|acc, acc_w| resolve_pixels(config.width, config.height, acc, acc_w));
    if denoise_it {
        pixels = denoise::denoise_oidn(&pixels, config.width, config.height);
    }
    pixels
}

/// 1 フレーム分のフェーズのスナップショット（ロックは短く）。
enum Snap {
    Loading,
    Live { probe: Arc<RenderProbe>, config: Arc<RenderConfig>, elapsed: Duration, finished: bool, cancelled: bool },
    Failed(String),
}

struct App {
    base: RenderConfig,
    overrides: Arc<CliOverrides>,
    job: Job,
    texture: Option<egui::TextureHandle>,
    /// 最後にテクスチャへ反映したタイル数（変化が無ければ作り直さない）
    shown_tiles: Option<usize>,
    save_path: String,
    message: String,
    one_to_one: bool,
    /// 検証用フック（環境変数）
    autosave: Option<String>,
    cancel_at_tiles: Option<usize>,
}

impl App {
    fn new(base: RenderConfig, overrides: CliOverrides) -> Self {
        let overrides = Arc::new(overrides);
        let job = Job::start(base.clone(), overrides.clone());
        Self {
            save_path: base.output_path.clone(),
            base,
            overrides,
            job,
            texture: None,
            shown_tiles: None,
            message: String::new(),
            one_to_one: false,
            autosave: std::env::var("TINYPT_VIEWER_AUTOSAVE").ok(),
            cancel_at_tiles: std::env::var("TINYPT_VIEWER_CANCEL_AT_TILES").ok().and_then(|v| v.parse().ok()),
        }
    }

    fn reload(&mut self) {
        self.job.cancel();
        self.job = Job::start(self.base.clone(), self.overrides.clone());
        self.texture = None;
        self.shown_tiles = None;
        self.message.clear();
    }

    fn snapshot(&self) -> Snap {
        match &*self.job.phase.lock().unwrap() {
            Phase::Loading => Snap::Loading,
            Phase::Rendering { probe, config, start } => Snap::Live {
                probe: probe.clone(),
                config: config.clone(),
                elapsed: start.elapsed(),
                finished: false,
                cancelled: false,
            },
            Phase::Finished { probe, config, elapsed, cancelled } => Snap::Live {
                probe: probe.clone(),
                config: config.clone(),
                elapsed: *elapsed,
                finished: true,
                cancelled: *cancelled,
            },
            Phase::Failed(m) => Snap::Failed(m.clone()),
        }
    }

    /// 現在のバッファを既存の出力経路で保存する（拡張子でフォーマット。デノイズは CLI と同じ設定に従う）。
    fn save(&mut self, probe: &RenderProbe, config: &RenderConfig) {
        let pixels = resolved_pixels(probe, config, config.denoise_enabled);
        let settings = OutputSettings { exposure: config.exposure, tonemap: config.tonemap };
        let path = self.save_path.clone();
        self.message = match OutputFormat::from_path(&path).write(&path, config.width, config.height, &pixels, settings) {
            Ok(()) => format!("saved {path}"),
            Err(e) => format!("save failed: {e}"),
        };
        eprintln!("{}", self.message);
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let snap = self.snapshot();
        let running = matches!(snap, Snap::Loading | Snap::Live { finished: false, .. });

        // 検証用フック
        if let Snap::Live { probe, config, finished, .. } = &snap {
            if let Some(n) = self.cancel_at_tiles {
                if !finished && probe.tiles().0 >= n {
                    self.job.cancel();
                    self.cancel_at_tiles = None;
                }
            }
            if *finished {
                if let Some(path) = self.autosave.take() {
                    self.save_path = path;
                    self.save(probe, config);
                }
            }
        }

        // 画像の更新: 進捗が進んだときだけ resolve → ppm_bytes（保存と同じ経路）でテクスチャを作り直す
        if let Snap::Live { probe, config, .. } = &snap {
            let tiles = probe.tiles().0;
            if self.shown_tiles != Some(tiles) {
                let pixels = resolved_pixels(probe, config, false);
                let settings = OutputSettings { exposure: config.exposure, tonemap: config.tonemap };
                let rgb = ppm_bytes(config.width, config.height, &pixels, settings);
                let img = egui::ColorImage::from_rgb([config.width, config.height], &rgb);
                self.texture = Some(ctx.load_texture("preview", img, egui::TextureOptions::NEAREST));
                self.shown_tiles = Some(tiles);
            }
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("scene:");
                ui.monospace(self.base.scene_path.as_deref().unwrap_or("(built-in)"));
                if ui.button("Reload").clicked() {
                    self.reload();
                }
                if ui.add_enabled(running, egui::Button::new("Abort")).clicked() {
                    self.job.cancel();
                }
                ui.checkbox(&mut self.one_to_one, "1:1");
            });
            ui.horizontal(|ui| match &snap {
                Snap::Loading => {
                    ui.label("loading scene…");
                }
                Snap::Failed(m) => {
                    ui.colored_label(egui::Color32::LIGHT_RED, m);
                }
                Snap::Live { probe, config, elapsed, finished, cancelled } => {
                    let px = (config.width * config.height) as f64;
                    let done = probe.samples_done();
                    let secs = elapsed.as_secs_f64().max(1e-9);
                    let state = if !*finished {
                        "rendering"
                    } else if *cancelled {
                        "aborted"
                    } else {
                        "done"
                    };
                    let (t, tt) = probe.tiles();
                    ui.label(format!(
                        "{state}: {:.1}/{} spp ({t}/{tt} tiles) · {:.1}s · {:.2} Msamples/s",
                        done / px,
                        config.spp,
                        secs,
                        done / secs / 1e6
                    ));
                }
            });
            ui.horizontal(|ui| {
                ui.label("save to:");
                ui.text_edit_singleline(&mut self.save_path);
                if let Snap::Live { probe, config, .. } = &snap {
                    if ui.button("Save").clicked() {
                        self.save(probe, config);
                    }
                }
                if self.base.denoise_enabled {
                    ui.label("(denoised on save)");
                }
                ui.label(&self.message);
            });
            ui.weak("Preview is the raw accumulation — not denoised. Denoise (if enabled) is applied only when saving.");
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let Some(tex) = &self.texture else { return };
            let [w, h] = tex.size();
            if self.one_to_one {
                egui::ScrollArea::both().show(ui, |ui| {
                    ui.image((tex.id(), egui::vec2(w as f32, h as f32)));
                });
            } else {
                let avail = ui.available_size();
                let scale = (avail.x / w as f32).min(avail.y / h as f32).max(0.01);
                ui.centered_and_justified(|ui| {
                    ui.image((tex.id(), egui::vec2(w as f32 * scale, h as f32 * scale)));
                });
            }
        });

        // 実行中は約 10 Hz で覗く。終了後は入力があるときだけ再描画（自動保存フックの待ちを除く）
        if running || self.autosave.is_some() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }
}

fn main() -> eframe::Result<()> {
    let mut config = RenderConfig::default();
    let (overrides, warnings) = parse_args(std::env::args().skip(1), &mut config);
    for w in &warnings {
        eprintln!("Warning: {}", w);
    }
    if overrides.help {
        println!("tinypt viewer {} — live preview of a render", env!("CARGO_PKG_VERSION"));
        println!("Accepts the same options as tinypt (-o sets the initial save path).\n");
        print!("{}", USAGE.replacen("Usage: tinypt [OPTIONS]", "Usage: viewer [OPTIONS]", 1));
        return Ok(());
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([960.0, 640.0]).with_title("tinypt viewer"),
        ..Default::default()
    };
    eframe::run_native("tinypt viewer", options, Box::new(move |_cc| Ok(Box::new(App::new(config, overrides)))))
}
