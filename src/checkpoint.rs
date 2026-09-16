//! レンダリング蓄積バッファのチェックポイント永続化。
//!
//! 長時間レンダリングの中断・再開を可能にする。
//! バイナリフォーマット: マジック → バージョン → シーンハッシュ → 解像度 → タスクID → acc → acc_w
//! 書き込みは一時ファイル経由の atomic rename で行い、データ破損を防ぐ。

use std::fs::File;
use std::io::{BufWriter, Write, Read, BufReader};
use std::path::Path;
use crate::config::RenderConfig;
use crate::math::{Vec3, Color};

const CKPT_MAGIC: &[u8; 8] = b"HYPCKPT\0";
const CKPT_VERSION: u32 = 1;

/// シーンハッシュからチェックポイントファイル名を生成する。
pub fn ckpt_path(scene_hash: u64) -> String {
    format!("checkpoint_{:016x}.bin", scene_hash)
}

/// FNV-1a 64bit。`DefaultHasher` と違い Rust のバージョンを跨いで値が安定するため、
/// ディスクに残るチェックポイントのキーに使える。
struct Fnv64(u64);

impl Fnv64 {
    fn new() -> Self {
        Fnv64(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, data: &[u8]) {
        for b in data {
            self.0 ^= *b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
    /// 長さ前置きで書き込み、隣接フィールドの境界の曖昧さを無くす。
    fn field(&mut self, data: &[u8]) {
        self.u64(data.len() as u64);
        self.bytes(data);
    }
    /// 外部ファイルのパスと内容。読めないファイルは「欠落」として区別して混ぜる。
    fn file(&mut self, path: &Path) {
        self.field(path.to_string_lossy().as_bytes());
        match std::fs::read(path) {
            Ok(data) => {
                self.u64(1);
                self.field(&data);
            }
            Err(_) => self.u64(0),
        }
    }
}

/// 最終 config（シーンファイル・CLI 上書き適用後）からチェックポイント用シーンハッシュを導出する。
///
/// 含めるもの（蓄積バッファの中身かタスク ID の並びを変えるもの）:
/// - シーン内容: XML ならファイル内容と参照ファイル（OBJ / 環境マップ）のパス＋内容、
///   組み込みシーンなら識別子と `--env` の環境マップ（パス＋内容）
/// - width / height / spp / max_bounces / rr_start / seed
/// - adaptive（有効フラグ・min_spp・threshold）
/// - tile / morton（タスク ID の割り当てが変わるとレジューム位置が狂うため）
///
/// 出力パス・デノイズ・トーンマップ・露出・チェックポイント間隔は後処理か保存頻度にしか
/// 影響しないので含めない。
pub fn scene_hash(config: &RenderConfig) -> std::io::Result<u64> {
    let mut h = Fnv64::new();
    h.field(b"tinypt-scene-hash-v1");
    match &config.scene_path {
        Some(path) => {
            let xml = std::fs::read_to_string(path)?;
            let base_dir = Path::new(path).parent().map(Path::to_path_buf).unwrap_or_default();
            h.field(b"xml");
            h.field(xml.as_bytes());
            for f in crate::mitsuba::referenced_files(&xml, &base_dir)? {
                h.file(&f);
            }
        }
        None => {
            h.field(b"builtin:default");
            match &config.env_map_path {
                Some(env) => {
                    h.u64(1);
                    h.file(Path::new(env));
                }
                None => h.u64(0),
            }
        }
    }
    for v in [
        config.width as u64,
        config.height as u64,
        config.spp as u64,
        config.max_bounces as u64,
        config.rr_start as u64,
        config.seed,
        config.adaptive_enabled as u64,
        config.adaptive_min_spp as u64,
        config.adaptive_threshold.to_bits(),
        config.tile as u64,
        config.morton_enabled as u64,
    ] {
        h.u64(v);
    }
    Ok(h.0)
}

fn write_u32_le<W: Write>(w: &mut W, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_u64_le<W: Write>(w: &mut W, v: u64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_f64_le<W: Write>(w: &mut W, v: f64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u32_le<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64_le<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn read_f64_le<R: Read>(r: &mut R) -> std::io::Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

/// 現在の蓄積バッファをチェックポイントファイルに書き出す。
/// 一時ファイルに書き込んでからリネームすることで、書き込み中の破損を防ぐ。
pub fn save_checkpoint(
    path: &str,
    scene_hash: u64,
    w: usize,
    h: usize,
    next_id: usize,
    acc: &[Color],
    acc_w: &[f64],
) -> std::io::Result<()> {
    // Atomic-ish write: temp file then rename.
    let tmp = format!("{}.tmp", path);
    {
        let f = File::create(&tmp)?;
        let mut out = BufWriter::new(f);

        out.write_all(CKPT_MAGIC)?;
        write_u32_le(&mut out, CKPT_VERSION)?;
        write_u64_le(&mut out, scene_hash)?;
        write_u32_le(&mut out, w as u32)?;
        write_u32_le(&mut out, h as u32)?;
        write_u64_le(&mut out, next_id as u64)?;

        // acc (x,y,z) then acc_w
        for p in acc {
            let v: Vec3 = (*p).into();
            write_f64_le(&mut out, v.x)?;
            write_f64_le(&mut out, v.y)?;
            write_f64_le(&mut out, v.z)?;
        }
        for v in acc_w {
            write_f64_le(&mut out, *v)?;
        }
        out.flush()?;
    }
    std::fs::rename(tmp, path)?;
    Ok(())
}

/// チェックポイントファイルを読み込む。シーンハッシュと解像度が一致しない場合は None を返す。
pub fn load_checkpoint(
    path: &str,
    scene_hash: u64,
    w: usize,
    h: usize,
) -> std::io::Result<Option<(usize, Vec<Color>, Vec<f64>)>> {
    if !Path::new(path).exists() {
        return Ok(None);
    }

    let f = File::open(path)?;
    let mut inp = BufReader::new(f);

    let mut magic = [0u8; 8];
    inp.read_exact(&mut magic)?;
    if &magic != CKPT_MAGIC {
        return Ok(None);
    }

    let ver = read_u32_le(&mut inp)?;
    if ver != CKPT_VERSION {
        return Ok(None);
    }

    let file_scene = read_u64_le(&mut inp)?;
    if file_scene != scene_hash {
        return Ok(None);
    }

    let fw = read_u32_le(&mut inp)? as usize;
    let fh = read_u32_le(&mut inp)? as usize;
    if fw != w || fh != h {
        return Ok(None);
    }

    let next_id = read_u64_le(&mut inp)? as usize;

    let n = w * h;
    let mut acc = Vec::with_capacity(n);
    for _ in 0..n {
        let x = read_f64_le(&mut inp)?;
        let y = read_f64_le(&mut inp)?;
        let z = read_f64_le(&mut inp)?;
        acc.push(Color::new(x, y, z));
    }
    let mut acc_w = Vec::with_capacity(n);
    for _ in 0..n {
        acc_w.push(read_f64_le(&mut inp)?);
    }

    Ok(Some((next_id, acc, acc_w)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("tinypt_ckpt_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn base() -> RenderConfig {
        RenderConfig::default()
    }

    /// 同じ config からは同じハッシュ（決定論的）。
    #[test]
    fn scene_hash_is_deterministic() {
        assert_eq!(scene_hash(&base()).unwrap(), scene_hash(&base()).unwrap());
    }

    /// レンダー結果/タスク並びに効く設定を変えるとハッシュが変わる。
    #[test]
    fn scene_hash_changes_with_render_affecting_settings() {
        let h0 = scene_hash(&base()).unwrap();
        let mutators: Vec<(&str, fn(&mut RenderConfig))> = vec![
            ("width", |c| c.width += 1),
            ("height", |c| c.height += 1),
            ("spp", |c| c.spp += 1),
            ("max_bounces", |c| c.max_bounces += 1),
            ("rr_start", |c| c.rr_start += 1),
            ("seed", |c| c.seed += 1),
            ("adaptive_enabled", |c| c.adaptive_enabled = !c.adaptive_enabled),
            ("adaptive_min_spp", |c| c.adaptive_min_spp += 1),
            ("adaptive_threshold", |c| c.adaptive_threshold *= 2.0),
            ("tile", |c| c.tile += 1),
            ("morton", |c| c.morton_enabled = !c.morton_enabled),
            ("env", |c| c.env_map_path = Some("nonexistent.hdr".into())),
        ];
        for (name, m) in mutators {
            let mut c = base();
            m(&mut c);
            assert_ne!(scene_hash(&c).unwrap(), h0, "{} should affect hash", name);
        }
    }

    /// 後処理だけの設定はハッシュに影響しない（同じ蓄積バッファを再利用できる）。
    #[test]
    fn scene_hash_ignores_post_process_settings() {
        let h0 = scene_hash(&base()).unwrap();
        let mut c = base();
        c.output_path = "other.exr".into();
        c.denoise_enabled = !c.denoise_enabled;
        c.exposure = 1.5;
        c.tonemap = crate::config::Tonemap::None;
        c.checkpoint_enabled = true;
        c.checkpoint_every_tasks = 7;
        assert_eq!(scene_hash(&c).unwrap(), h0);
    }

    /// XML シーン: XML 内容・参照 OBJ の内容が変わるとハッシュが変わる。組み込みシーンとも異なる。
    #[test]
    fn scene_hash_covers_xml_and_referenced_files() {
        let obj = tmp("hash_mesh.obj");
        let xml = tmp("hash_scene.xml");
        std::fs::write(&obj, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
        let body = |r: &str| format!(
            r#"<scene version="3.0.0"><shape type="obj"><string name="filename" value="hash_mesh.obj"/></shape><shape type="sphere"><float name="radius" value="{}"/></shape></scene>"#,
            r
        );
        std::fs::write(&xml, body("1")).unwrap();
        let mut c = base();
        c.scene_path = Some(xml.to_string_lossy().into_owned());
        let h_xml = scene_hash(&c).unwrap();
        assert_ne!(h_xml, scene_hash(&base()).unwrap(), "xml vs builtin");

        std::fs::write(&obj, "v 0 0 0\nv 2 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
        let h_obj = scene_hash(&c).unwrap();
        assert_ne!(h_obj, h_xml, "referenced obj content");

        std::fs::write(&xml, body("2")).unwrap();
        assert_ne!(scene_hash(&c).unwrap(), h_obj, "xml content");

        std::fs::remove_file(&obj).ok();
        std::fs::remove_file(&xml).ok();
    }
}
