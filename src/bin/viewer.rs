//! tinypt のレンダリング途中ビューア（egui / eframe）。`--features viewer` でだけビルドされる。
//!
//! CLI（`tinypt`）と同じオプションを受け付け（解析は [`tinypt::cli`] を共有）、別スレッドでレンダーしながら
//! 蓄積バッファを約 10 Hz で表示する。範囲は「進行を見るだけ」: 中断・保存・読み込み直しはあるが、
//! カメラ操作や設定の編集は無い。
//!
//! 表示画像は保存と同じ経路（`resolve_pixels` → `ppm_bytes` = 露出 → トーンマップ → sRGB → 8bit）で作る。
//! 表示用に別のトーンマップは持たない。デノイズはプレビューには掛けず、保存時に CLI と同じ扱いで掛ける。
//!
//! シーンは `Open…`（ネイティブのダイアログ）、パス欄、同じディレクトリの一覧から開ける。切り替えは
//! 「走っているレンダーを止める → ワーカーの終了を待つ → 読み込む → 描く」を新しいスレッドの中で行うので、
//! UI スレッドは止まらず、古いワーカーが新しい画面に書き込むこともない。読み込みに失敗したら直前の表示を保つ。
//!
//! 設定は 2 種類に分けてある。**A（やり直し不要）**: トーンマップ・露出・デノイズの ON/OFF は蓄積バッファを
//! そのまま `resolve_pixels` / `OutputSettings` に渡し直すだけ（動かした瞬間に反映。レンダーには触らない）。
//! **B（やり直し）**: spp・解像度・シード・適応サンプリングは「適用」で、シーン切り替えと同じ
//! `open_scene`（止める → ワーカー終了を待つ → 読む → 描く）を通る。
//!
//! 検証用フック（環境変数）: `TINYPT_VIEWER_VIEW=exposure=1.5;tonemap=none;denoise=0` は
//! `TINYPT_VIEWER_VIEW_AT_TILES=N`（既定 0）タイルで A の値を GUI の状態に設定し、
//! `TINYPT_VIEWER_APPLY=spp=8;res=160x90;seed=2` は最初の描画開始後に B の欄へ入れて「適用」と同じ処理を呼ぶ。
//! `TINYPT_VIEWER_AUTOSAVE=PATH` は完了・中断の後に保存ボタンと同じ処理で保存し、
//! `TINYPT_VIEWER_CANCEL_AT_TILES=N` は N タイルのマージ後に中断する。
//! `TINYPT_VIEWER_OPEN=A;B;…` と `TINYPT_VIEWER_SAVE_DIR=DIR` は、最初のシーンの完了後に（ボタンと同じ
//! `open_scene` で）順に開き、完了するたびに `DIR/<番号>_<名前>.ppm` へ保存する。
//! `TINYPT_VIEWER_SWITCH_AT_TILES=N` を足すと、完了を待たず N タイルで次へ切り替える。

use std::sync::atomic::{AtomicBool, Ordering};

use std::thread::JoinHandle;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui;
use tinypt::cli::{load_with_overrides, parse_args, CliOverrides, USAGE};
use tinypt::math::Color;
use tinypt::{
    denoise, ppm_bytes, render_observed, resolve_pixels, OutputFormat, OutputSettings, RenderConfig, RenderProbe, Tonemap,
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
    handle: Option<JoinHandle<()>>,
    /// このジョブが開くシーン（`None` は組み込みシーン）
    path: Option<String>,
}

impl Job {
    /// `prev`（前のジョブのスレッド）が完全に終わるのを**新しいスレッドの中で**待ってから読み込む。
    fn start(config: RenderConfig, overrides: Arc<CliOverrides>, prev: Option<JoinHandle<()>>) -> Self {
        let path = config.scene_path.clone();
        let phase = Arc::new(Mutex::new(Phase::Loading));
        let cancel = Arc::new(AtomicBool::new(false));
        let (p, c) = (phase.clone(), cancel.clone());
        let handle = std::thread::spawn(move || {
            if let Some(h) = prev {
                let _ = h.join();
            }
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
        Job { phase, cancel, handle: Some(handle), path }
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

/// プレビュー（GL テクスチャ）に載せられる 1 辺の上限。GUI の解像度欄はこれ以下に丸める。
const PREVIEW_MAX: usize = 8192;

/// デノイズ済みプレビューの 1 世代。リニアの画素（トーンマップ前）で持つので、露出・トーンマップを
/// 変えても作り直さずに `ppm_bytes` を通し直せる（CLI と同じ順序: resolve → denoise → ppm_bytes）。
struct Dn {
    /// この結果が元にした蓄積バッファのタイル数
    tiles: usize,
    pixels: Arc<Vec<Color>>,
    /// どのジョブ（シーン・設定）の結果か。古いジョブの結果は捨てる
    serial: u64,
    /// 世代番号（テクスチャを作り直すかの判定用）
    id: u64,
    took: Duration,
}

/// A: 表示・保存の後処理（変えてもレンダーは続く）。
#[derive(Clone, Copy, PartialEq)]
struct View {
    tonemap: Tonemap,
    exposure: f64,
    denoise: bool,
}

/// B: 変えるとやり直しになる設定。`edit`（欄の値）と `active`（いま走っているレンダーの値）を分けて持つ。
#[derive(Clone, Copy, PartialEq)]
struct Opts {
    spp: usize,
    width: usize,
    height: usize,
    seed: u64,
    adaptive: bool,
    adaptive_min: usize,
    threshold: f64,
}

impl Opts {
    fn of(c: &RenderConfig) -> Self {
        Self {
            spp: c.spp,
            width: c.width,
            height: c.height,
            seed: c.seed,
            adaptive: c.adaptive_enabled,
            adaptive_min: c.adaptive_min_spp,
            threshold: c.adaptive_threshold,
        }
    }

    /// 不正な値を丸める。丸めたものを警告として返す（落とさない）。
    fn clamped(mut self) -> (Self, Vec<String>) {
        let mut w = Vec::new();
        if self.spp == 0 {
            self.spp = 1;
            w.push("spp 0 → 1".to_string());
        }
        for (v, name) in [(&mut self.width, "width"), (&mut self.height, "height")] {
            if *v == 0 || *v > PREVIEW_MAX {
                let c = (*v).clamp(1, PREVIEW_MAX);
                w.push(format!("{name} {v} → {c} (viewer accepts 1..={PREVIEW_MAX})"));
                *v = c;
            }
        }
        if self.adaptive_min == 0 {
            self.adaptive_min = 1;
            w.push("adaptive min spp 0 → 1".to_string());
        }
        if !self.threshold.is_finite() || self.threshold < 0.0 {
            self.threshold = 0.02;
            w.push("adaptive threshold → 0.02".to_string());
        }
        (self, w)
    }
}

struct App {
    base: RenderConfig,
    view: View,
    /// 最後にテクスチャへ反映した (トーンマップ, 露出) — 変わったら蓄積はそのままで作り直す
    shown_view: Option<(Tonemap, u64)>,
    edit: Opts,
    active: Opts,
    /// 現在のジョブの config で `edit` / `active` を同期したか
    opts_synced: bool,
    opts_note: String,
    view_hook: Option<String>,
    view_hook_at: usize,
    apply_hook: Option<String>,
    /// 検証フック: `TINYPT_VIEWER_EDIT`（B の欄を入れるだけ）と `TINYPT_VIEWER_SHOTS=N:path;…`
    /// （N タイルに達したらビューア自身のウィンドウを PNG に保存する。OS のスクリーンキャプチャは使わない）
    edit_hook: Option<String>,
    /// プレビューをデノイズ済みで表示するか（保存時デノイズ `view.denoise` とは別の設定）
    show_dn: bool,
    dn: Option<Dn>,
    dn_slot: Arc<Mutex<Option<Dn>>>,
    dn_busy: Arc<AtomicBool>,
    dn_last_end: Option<Instant>,
    dn_last_took: Duration,
    dn_counter: u64,
    /// 現在のジョブの通し番号（`open_scene` ごとに増やす）
    serial: u64,
    /// 最後に作ったテクスチャの元（種別, 番号）と、その画素（8bit RGB。検証フックが書き出す）
    shown_src: Option<(u8, u64)>,
    shown_rgb: Vec<u8>,
    dump_shown: Option<String>,
    shots: Vec<(usize, String)>,
    shot_path: Option<String>,
    overrides: Arc<CliOverrides>,
    job: Job,
    texture: Option<egui::TextureHandle>,
    /// 最後にテクスチャへ反映したタイル数（変化が無ければ作り直さない）
    /// 直前に表示できていたレンダー（読み込みに失敗したときの保存・表示に使う）
    last: Option<(Arc<RenderProbe>, Arc<RenderConfig>)>,
    path_input: String,
    /// 現在のシーンと同じディレクトリの `*.xml`
    scenes: Vec<String>,
    scenes_dir: Option<std::path::PathBuf>,
    save_path: String,
    message: String,
    one_to_one: bool,
    /// 検証用フック（環境変数）
    autosave: Option<String>,
    cancel_at_tiles: Option<usize>,
    /// 検証用: 順に開くシーン、保存先、途中で切り替えるタイル数、次に開く番号
    seq: Vec<String>,
    seq_next: usize,
    seq_dir: Option<String>,
    switch_at_tiles: Option<usize>,
    seq_saved: bool,
}

impl App {
    fn new(base: RenderConfig, overrides: CliOverrides) -> Self {
        let overrides = Arc::new(overrides);
        let job = Job::start(base.clone(), overrides.clone(), None);
        Self {
            view: View { tonemap: base.tonemap, exposure: base.exposure, denoise: base.denoise_enabled },
            shown_view: None,
            edit: Opts::of(&base),
            active: Opts::of(&base),
            opts_synced: false,
            opts_note: String::new(),
            view_hook: std::env::var("TINYPT_VIEWER_VIEW").ok(),
            view_hook_at: std::env::var("TINYPT_VIEWER_VIEW_AT_TILES").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
            apply_hook: std::env::var("TINYPT_VIEWER_APPLY").ok(),
            edit_hook: std::env::var("TINYPT_VIEWER_EDIT").ok(),
            show_dn: false,
            dn: None,
            dn_slot: Arc::new(Mutex::new(None)),
            dn_busy: Arc::new(AtomicBool::new(false)),
            dn_last_end: None,
            dn_last_took: Duration::ZERO,
            dn_counter: 0,
            serial: 0,
            shown_src: None,
            shown_rgb: Vec::new(),
            dump_shown: std::env::var("TINYPT_VIEWER_DUMP_SHOWN").ok(),
            shots: std::env::var("TINYPT_VIEWER_SHOTS")
                .map(|v| {
                    v.split(';')
                        .filter_map(|e| e.split_once(':').and_then(|(n, p)| Some((n.parse().ok()?, p.to_string()))))
                        .collect()
                })
                .unwrap_or_default(),
            shot_path: None,
            last: None,
            path_input: base.scene_path.clone().unwrap_or_default(),
            scenes: Vec::new(),
            scenes_dir: None,
            seq: std::env::var("TINYPT_VIEWER_OPEN")
                .map(|v| v.split(';').filter(|s| !s.is_empty()).map(String::from).collect())
                .unwrap_or_default(),
            seq_next: 0,
            seq_dir: std::env::var("TINYPT_VIEWER_SAVE_DIR").ok(),
            switch_at_tiles: std::env::var("TINYPT_VIEWER_SWITCH_AT_TILES").ok().and_then(|v| v.parse().ok()),
            seq_saved: false,
            save_path: base.output_path.clone(),
            base,
            overrides,
            job,
            texture: None,
            message: String::new(),
            one_to_one: false,
            autosave: std::env::var("TINYPT_VIEWER_AUTOSAVE").ok(),
            cancel_at_tiles: std::env::var("TINYPT_VIEWER_CANCEL_AT_TILES").ok().and_then(|v| v.parse().ok()),
        }
    }

    /// シーンを開いて描き始める（`Open…`・パス欄・一覧・`Reload`・検証フックが共通で使う唯一の入口）。
    ///
    /// 走っているジョブに中断を掛け、そのスレッドを新しいジョブのスレッドへ渡す（終了を待つのはそちら。
    /// UI は待たない）。CLI 由来の上書き（`--res` / `--spp` など）は `load_with_overrides` が同じ規則で適用する。
    /// 表示中のテクスチャは、新しいシーンが描き始めるまで（失敗したらそのまま）残す。
    fn open_scene(&mut self, path: Option<String>) {
        self.job.cancel();
        let prev = self.job.handle.take();
        let mut config = self.base.clone();
        config.scene_path = path.clone();
        self.job = Job::start(config, self.overrides.clone(), prev);
        if let Some(p) = &path {
            self.path_input = p.clone();
        }
        self.message.clear();
        self.opts_synced = false;
        self.serial += 1;
        self.dn = None;
        self.shown_src = None;
    }

    /// B の「適用」: 欄の値を丸めて検証し、スペックを上書きとして記録してからシーンを開き直す（= やり直し）。
    /// 変えた項目だけを CLI と同じ「上書き」にする（変えていない解像度・spp はシーンファイルの値に従い続ける）。
    fn apply_opts(&mut self) {
        let (e, warns) = self.edit.clamped();
        self.edit = e;
        self.opts_note = if warns.is_empty() { String::new() } else { format!("adjusted: {}", warns.join(", ")) };
        let mut ov = CliOverrides { spp: self.overrides.spp, width: self.overrides.width, height: self.overrides.height, help: false };
        if e.spp != self.active.spp {
            ov.spp = Some(e.spp);
            self.base.spp = e.spp;
        }
        if e.width != self.active.width || e.height != self.active.height {
            ov.width = Some(e.width);
            ov.height = Some(e.height);
            self.base.width = e.width;
            self.base.height = e.height;
        }
        self.base.seed = e.seed;
        self.base.adaptive_enabled = e.adaptive;
        self.base.adaptive_min_spp = e.adaptive_min;
        self.base.adaptive_threshold = e.threshold;
        self.overrides = Arc::new(ov);
        let p = self.job.path.clone();
        self.open_scene(p);
    }

    /// 検証フック用: `key=value;…` を A の状態（`view`）または B の欄（`edit`）へ入れる。GUI の欄と同じ変数を触る。
    fn set_from_spec(&mut self, spec: &str) {
        for kv in spec.split(';').filter(|s| !s.is_empty()) {
            let Some((k, v)) = kv.split_once('=') else { continue };
            match k {
                "exposure" => self.view.exposure = v.parse().ok().filter(|x: &f64| x.is_finite()).unwrap_or(self.view.exposure),
                "tonemap" => self.view.tonemap = Tonemap::from_str(v).unwrap_or(self.view.tonemap),
                "denoise" => self.view.denoise = v != "0",
                "show" => self.show_dn = v != "0" && cfg!(feature = "oidn"),
                "spp" => self.edit.spp = v.parse().unwrap_or(self.edit.spp),
                "seed" => self.edit.seed = v.parse().unwrap_or(self.edit.seed),
                "adaptive" => self.edit.adaptive = v != "0",
                "res" => {
                    if let Some((w, h)) = v.split_once('x') {
                        self.edit.width = w.parse().unwrap_or(self.edit.width);
                        self.edit.height = h.parse().unwrap_or(self.edit.height);
                    }
                }
                _ => {}
            }
        }
    }

    /// 一覧用に、`path` と同じディレクトリの `*.xml` を集める（ディレクトリが変わったときだけ）。
    fn refresh_scene_list(&mut self, path: &str) {
        let dir = std::path::Path::new(path).parent().map(|d| if d.as_os_str().is_empty() { ".".into() } else { d.to_path_buf() });
        if dir == self.scenes_dir {
            return;
        }
        self.scenes = dir
            .as_ref()
            .and_then(|d| std::fs::read_dir(d).ok())
            .map(|rd| {
                let mut v: Vec<String> = rd
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("xml")))
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect();
                v.sort();
                v
            })
            .unwrap_or_default();
        self.scenes_dir = dir;
    }

    /// 画面（テクスチャ）に出ているのが、`tiles` 時点のデノイズ結果そのものか（現在の表示設定で）。
    fn dn_on_screen(&self, tiles: usize) -> bool {
        let key = (self.view.tonemap, self.view.exposure.to_bits());
        self.dn.as_ref().is_some_and(|d| d.tiles == tiles && self.shown_src == Some((1, d.id)) && self.shown_view == Some(key))
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
        let pixels = resolved_pixels(probe, config, self.view.denoise);
        let settings = OutputSettings { exposure: self.view.exposure, tonemap: self.view.tonemap };
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
        if let Snap::Live { probe, config, .. } = &snap {
            self.last = Some((probe.clone(), config.clone()));
        }
        if let Some(p) = self.job.path.clone() {
            self.refresh_scene_list(&p);
        }
        // 走り始めたレンダーの config を B の欄へ反映（適用後は同じ値、シーンを開いたときはシーンの値）
        if let (Snap::Live { config, .. }, false) = (&snap, self.opts_synced) {
            self.active = Opts::of(config);
            self.edit = self.active;
            self.opts_synced = true;
            if let Some(spec) = self.edit_hook.take() {
                self.set_from_spec(&spec);
            }
            if let Some(spec) = self.apply_hook.take() {
                self.set_from_spec(&spec);
                self.apply_opts();
            }
        }
        // デノイズ済みプレビュー: 結果の受け取り → 間引いて別スレッドで起動
        if let Some(d) = self.dn_slot.lock().unwrap().take() {
            self.dn_last_end = Some(Instant::now());
            self.dn_last_took = d.took;
            if d.serial == self.serial {
                self.dn = Some(d);
            }
        }
        if let (Snap::Live { probe, config, finished, .. }, true) = (&snap, self.show_dn && cfg!(feature = "oidn")) {
            let cur = probe.tiles().0;
            let have = self.dn.as_ref().map(|d| d.tiles);
            if cur > 0 && have != Some(cur) && !self.dn_busy.load(Ordering::Relaxed) {
                // 間引き: 前回の終了から max(1 秒, 直近の所要時間の 4 倍) あける（デノイズの CPU 占有を約 2 割以下に抑える）。
                // 完了後の最後の 1 回だけは待たずに走らせる（最終結果が古いまま残らないように）
                let interval = Duration::from_secs(1).max(self.dn_last_took * 4);
                let due = *finished || self.dn_last_end.map_or(true, |t| t.elapsed() >= interval);
                if due {
                    self.dn_busy.store(true, Ordering::Relaxed);
                    self.dn_counter += 1;
                    let (probe, config, slot, busy) = (probe.clone(), config.clone(), self.dn_slot.clone(), self.dn_busy.clone());
                    let (serial, id) = (self.serial, self.dn_counter);
                    std::thread::spawn(move || {
                        let t0 = Instant::now();
                        // タイル数とバッファは同じロックの中で組で取る。CLI と同じ resolve_pixels → denoise_oidn
                        let (tiles, px) = probe.with_buffers_at(|t, acc, w| (t, resolve_pixels(config.width, config.height, acc, w)));
                        let px = denoise::denoise_oidn(&px, config.width, config.height);
                        *slot.lock().unwrap() = Some(Dn { tiles, pixels: Arc::new(px), serial, id, took: t0.elapsed() });
                        busy.store(false, Ordering::Relaxed);
                    });
                }
            }
        }

        let finished_now = matches!(snap, Snap::Live { finished: true, .. });
        // 検証フック: スクリーンショット（要求 → 次のフレームで Event::Screenshot が届く）
        if let Some(path) = self.shot_path.clone() {
            let got = ctx.input(|i| {
                i.raw.events.iter().find_map(|e| match e {
                    egui::Event::Screenshot { image, .. } => Some(image.clone()),
                    _ => None,
                })
            });
            if let Some(img) = got {
                let rgba: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_array()).collect();
                let r = image::save_buffer(&path, &rgba, img.size[0] as u32, img.size[1] as u32, image::ColorType::Rgba8);
                eprintln!("[hook] screenshot {path}: {:?}", r.map_err(|e| e.to_string()));
                self.shot_path = None;
            }
        } else if let (Snap::Live { probe, .. }, Some((n, _))) = (&snap, self.shots.first()) {
            let cur = probe.tiles().0;
            let dn_ready = !(self.show_dn && finished_now) || self.dn_on_screen(cur);
            if cur >= *n && dn_ready {
                let (_, path) = self.shots.remove(0);
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
                self.shot_path = Some(path);
            }
        }
        // 検証フック（A）: 指定タイル数で view を変える。レンダーは触らない
        if let (Snap::Live { probe, elapsed, .. }, true) = (&snap, self.view_hook.is_some()) {
            let t = probe.tiles().0;
            if t >= self.view_hook_at {
                let spec = self.view_hook.take().unwrap();
                eprintln!("[hook] A-change '{}' at tiles={} elapsed={:.2}s", spec, t, elapsed.as_secs_f64());
                self.set_from_spec(&spec);
            }
        }

        // 検証用フック: 順に開く。完了（または指定タイル数）で次へ。失敗したら直前の表示を保存して次へ
        if !self.seq.is_empty() || self.seq_dir.is_some() {
            let idx = self.seq_next;
            let mut advance = false;
            match &snap {
                Snap::Live { probe, config, finished, cancelled, .. } => {
                    if *finished && !*cancelled && !self.seq_saved {
                        if let Some(dir) = self.seq_dir.clone() {
                            let stem = std::path::Path::new(self.job.path.as_deref().unwrap_or("builtin"))
                                .file_stem()
                                .map(|s| s.to_string_lossy().into_owned())
                                .unwrap_or_default();
                            self.save_path = format!("{dir}/{idx}_{stem}.ppm");
                            self.save(probe, config);
                        }
                        self.seq_saved = true;
                    }
                    match self.switch_at_tiles {
                        Some(n) if !*finished && probe.tiles().0 >= n => advance = true,
                        _ => advance = *finished && self.seq_saved,
                    }
                }
                Snap::Failed(_) if !self.seq_saved => {
                    if let (Some(dir), Some((probe, config))) = (self.seq_dir.clone(), self.last.clone()) {
                        self.save_path = format!("{dir}/{idx}_failed_kept.ppm");
                        self.save(&probe, &config);
                    }
                    self.seq_saved = true;
                    advance = true;
                }
                _ => {}
            }
            if advance {
                if let Some(next) = self.seq.get(self.seq_next).cloned() {
                    self.seq_next += 1;
                    self.seq_saved = false;
                    self.open_scene(Some(next));
                } else if matches!(snap, Snap::Live { finished: true, .. }) {
                    self.seq_dir = None; // 全部済んだ
                }
            }
        }

        // 検証用フック
        if let Snap::Live { probe, config, finished, .. } = &snap {
            if let Some(n) = self.cancel_at_tiles {
                if !finished && probe.tiles().0 >= n {
                    self.job.cancel();
                    self.cancel_at_tiles = None;
                }
            }
            // デノイズ表示中は、表示が最終タイル数に追いついてから保存する（検証フック用）
            let dn_ready = !self.show_dn || self.dn_on_screen(probe.tiles().0);
            if *finished && dn_ready && self.shown_src.is_some() {
                if let Some(path) = self.autosave.take() {
                    self.save_path = path;
                    self.save(probe, config);
                    if let Some(dump) = self.dump_shown.take() {
                        let mut f = format!("P6\n{} {}\n255\n", config.width, config.height).into_bytes();
                        f.extend_from_slice(&self.shown_rgb);
                        let _ = std::fs::write(&dump, f);
                    }
                }
            }
        }

        // 画像の更新: 元（生の蓄積 / デノイズ済み）か (トーンマップ, 露出) が変わったときだけテクスチャを作り直す
        if let Snap::Live { probe, config, .. } = &snap {
            let dn_now = if self.show_dn { self.dn.as_ref() } else { None };
            let src = match dn_now {
                Some(d) => (1u8, d.id),
                None => (0u8, probe.tiles().0 as u64),
            };
            let view_key = (self.view.tonemap, self.view.exposure.to_bits());
            if config.width.max(config.height) > PREVIEW_MAX {
                self.texture = None; // GL のテクスチャ上限を超える（進捗と保存は使える）
            } else if self.shown_src != Some(src) || self.shown_view != Some(view_key) {
                // CLI と同じ順序: 蓄積 → resolve_pixels →（デノイズ）→ ppm_bytes（露出 + トーンマップ）
                let pixels: Vec<Color> = match dn_now {
                    Some(d) => d.pixels.as_ref().clone(),
                    None => resolved_pixels(probe, config, false),
                };
                let settings = OutputSettings { exposure: self.view.exposure, tonemap: self.view.tonemap };
                let rgb = ppm_bytes(config.width, config.height, &pixels, settings);
                let img = egui::ColorImage::from_rgb([config.width, config.height], &rgb);
                self.texture = Some(ctx.load_texture("preview", img, egui::TextureOptions::NEAREST));
                self.shown_rgb = rgb;
                self.shown_src = Some(src);
                self.shown_view = Some(view_key);
            }
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("scene:");
                let edit = ui.add(egui::TextEdit::singleline(&mut self.path_input).desired_width(360.0).hint_text("path to a .xml scene"));
                let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                if (ui.button("Load").clicked() || enter) && !self.path_input.trim().is_empty() {
                    let p = self.path_input.trim().to_string();
                    self.open_scene(Some(p));
                }
                if ui.button("Open…").clicked() {
                    let mut dlg = rfd::FileDialog::new().add_filter("Mitsuba XML scene", &["xml"]);
                    if let Some(d) = self.scenes_dir.as_ref().and_then(|d| d.canonicalize().ok()) {
                        dlg = dlg.set_directory(d);
                    }
                    if let Some(f) = dlg.pick_file() {
                        self.open_scene(Some(f.to_string_lossy().into_owned()));
                    }
                }
                if !self.scenes.is_empty() {
                    let mut picked = None;
                    let cur = self.job.path.clone().unwrap_or_default();
                    egui::ComboBox::from_id_salt("scenes").selected_text("scenes…").show_ui(ui, |ui| {
                        for sc in &self.scenes {
                            let name = std::path::Path::new(sc).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                            if ui.selectable_label(*sc == cur, name).clicked() {
                                picked = Some(sc.clone());
                            }
                        }
                    });
                    if let Some(sc) = picked {
                        self.open_scene(Some(sc));
                    }
                }
                if ui.button("Reload").clicked() {
                    let p = self.job.path.clone();
                    self.open_scene(p);
                }
                if ui.add_enabled(running, egui::Button::new("Abort")).clicked() {
                    self.job.cancel();
                }
                ui.checkbox(&mut self.one_to_one, "1:1");
            });
            ui.horizontal(|ui| match &snap {
                Snap::Loading => {
                    ui.label(format!("loading {}…", self.job.path.as_deref().unwrap_or("built-in scene")));
                }
                Snap::Failed(m) => {
                    ui.colored_label(egui::Color32::LIGHT_RED, m);
                    if self.last.is_some() {
                        ui.weak("(showing the previous scene)");
                    }
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
                let target = match &snap {
                    Snap::Live { probe, config, .. } => Some((probe.clone(), config.clone())),
                    _ => self.last.clone(),
                };
                if let Some((probe, config)) = target {
                    if ui.button("Save").clicked() {
                        self.save(&probe, &config);
                    }
                }
                if self.base.denoise_enabled {
                    ui.label("(denoised on save)");
                }
                ui.label(&self.message);
            });
            let note = if self.show_dn {
                "Preview: denoised (OIDN, throttled — see the age below). Saved file is denoised only if 'denoise on save' is on."
            } else {
                "Preview: raw accumulation. Denoise (if 'denoise on save' is on) is applied only when saving."
            };
            ui.weak(note);
        });

        egui::SidePanel::left("opts").resizable(false).default_width(230.0).show(ctx, |ui| {
            ui.colored_label(egui::Color32::from_rgb(120, 200, 120), egui::RichText::new("View — applies instantly").strong());
            ui.weak("Re-displays the current accumulation. The render keeps running.");
            ui.horizontal(|ui| {
                ui.label("tonemap");
                egui::ComboBox::from_id_salt("tm")
                    .selected_text(if self.view.tonemap == Tonemap::Aces { "aces" } else { "none" })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.view.tonemap, Tonemap::Aces, "aces");
                        ui.selectable_value(&mut self.view.tonemap, Tonemap::None, "none");
                    });
            });
            let prev = self.view.exposure;
            ui.horizontal(|ui| {
                ui.label("exposure EV");
                ui.add(egui::DragValue::new(&mut self.view.exposure).speed(0.05).range(-20.0..=20.0));
            });
            if !self.view.exposure.is_finite() {
                self.view.exposure = if prev.is_finite() { prev } else { 0.0 };
            }
            if cfg!(feature = "oidn") {
                ui.checkbox(&mut self.show_dn, "show denoised preview");
                ui.weak("(runs OIDN in the background, throttled)");
                ui.checkbox(&mut self.view.denoise, "denoise on save");
                if self.show_dn != self.view.denoise {
                    ui.colored_label(egui::Color32::from_rgb(230, 160, 70), "preview and saved file differ in denoising");
                }
            } else {
                ui.add_enabled(false, egui::Checkbox::new(&mut self.show_dn, "show denoised preview"));
                ui.add_enabled(false, egui::Checkbox::new(&mut self.view.denoise, "denoise on save"));
                ui.weak("built without the oidn feature");
            }
            ui.separator();
            let orange = egui::Color32::from_rgb(230, 160, 70);
            ui.colored_label(orange, egui::RichText::new("Render — Apply restarts").strong());
            ui.weak("Discards the accumulation and re-renders from scratch.");
            let e = &mut self.edit;
            egui::Grid::new("b").num_columns(2).show(ui, |ui| {
                ui.label("spp");
                ui.add(egui::DragValue::new(&mut e.spp).range(1..=1_000_000));
                ui.end_row();
                ui.label("width");
                ui.add(egui::DragValue::new(&mut e.width).range(1..=PREVIEW_MAX));
                ui.end_row();
                ui.label("height");
                ui.add(egui::DragValue::new(&mut e.height).range(1..=PREVIEW_MAX));
                ui.end_row();
                ui.label("seed");
                ui.add(egui::DragValue::new(&mut e.seed));
                ui.end_row();
                ui.label("adaptive");
                ui.checkbox(&mut e.adaptive, "");
                ui.end_row();
                ui.label("  min spp");
                ui.add_enabled(e.adaptive, egui::DragValue::new(&mut e.adaptive_min).range(1..=1_000_000));
                ui.end_row();
                ui.label("  threshold");
                ui.add_enabled(e.adaptive, egui::DragValue::new(&mut e.threshold).speed(0.001).range(0.0..=10.0));
                ui.end_row();
            });
            let dirty = self.edit != self.active;
            ui.horizontal(|ui| {
                if ui.add_enabled(dirty, egui::Button::new("Apply (restart)")).clicked() {
                    self.apply_opts();
                }
                if ui.add_enabled(dirty, egui::Button::new("Revert")).clicked() {
                    self.edit = self.active;
                }
            });
            if dirty {
                ui.colored_label(orange, "pending changes — not applied yet");
            }
            if !self.opts_note.is_empty() {
                ui.colored_label(egui::Color32::LIGHT_RED, &self.opts_note);
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let Some(tex) = &self.texture else { return };
            let [w, h] = tex.size();
            // 常に 1 行出す（生 / デノイズで画像の位置がずれないように）。古さもここに出す
            let cur = match &snap {
                Snap::Live { probe, .. } => probe.tiles(),
                _ => (0, 0),
            };
            let shown_dn = self.show_dn && self.dn.is_some();
            if shown_dn {
                let d = self.dn.as_ref().unwrap();
                let lag = cur.0.saturating_sub(d.tiles);
                let busy = self.dn_busy.load(Ordering::Relaxed);
                let txt = format!(
                    "DENOISED @ {}/{} tiles · render at {}/{}{} · denoise took {:.0} ms{}",
                    d.tiles,
                    cur.1,
                    cur.0,
                    cur.1,
                    if lag == 0 { " · current".to_string() } else { format!(" · STALE by {lag} tiles") },
                    d.took.as_secs_f64() * 1e3,
                    if busy { " · denoising…" } else { "" }
                );
                let col = if lag == 0 { egui::Color32::from_rgb(120, 200, 120) } else { egui::Color32::from_rgb(230, 160, 70) };
                ui.colored_label(col, txt);
            } else if self.show_dn {
                ui.colored_label(egui::Color32::from_rgb(230, 160, 70), format!("RAW @ {}/{} tiles · denoising… (first result pending)", cur.0, cur.1));
            } else {
                ui.label(format!("RAW @ {}/{} tiles", cur.0, cur.1));
            }
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
        if running || self.autosave.is_some() || !self.seq.is_empty() || !self.shots.is_empty() || self.shot_path.is_some() {
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
        viewport: egui::ViewportBuilder::default().with_inner_size([1180.0, 680.0]).with_title("tinypt viewer"),
        ..Default::default()
    };
    eframe::run_native("tinypt viewer", options, Box::new(move |_cc| Ok(Box::new(App::new(config, overrides)))))
}
