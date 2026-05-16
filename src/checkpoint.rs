//! レンダリング蓄積バッファのチェックポイント永続化。
//!
//! 長時間レンダリングの中断・再開を可能にする。
//! バイナリフォーマット: マジック → バージョン → シーンハッシュ → 解像度 → タスクID → acc → acc_w
//! 書き込みは一時ファイル経由の atomic rename で行い、データ破損を防ぐ。

use std::fs::File;
use std::io::{BufWriter, Write, Read, BufReader};
use std::path::Path;
use crate::math::{Vec3, Color};

const CKPT_MAGIC: &[u8; 8] = b"HYPCKPT\0";
const CKPT_VERSION: u32 = 1;

/// シーンハッシュからチェックポイントファイル名を生成する。
pub fn ckpt_path(scene_hash: u64) -> String {
    format!("checkpoint_{:016x}.bin", scene_hash)
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
