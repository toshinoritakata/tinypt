//! 最小限の Wavefront OBJ ローダー（三角形化 + モーションブラー対応）。
//!
//! `v`（頂点）と `f`（フェース）のみを解析し、テクスチャ座標や法線は無視する。
//! N 角形フェースはファン三角形化で分割される。
//! モーションブラー用に 2 つの OBJ ファイル（シャッター開/閉）を読み込む機能もある。

use crate::math::Vec3;
use crate::geometry::Triangle;

/// OBJ のインデックスを 0-based に変換する。
/// OBJ は 1-based で、負のインデックスは末尾からの相対位置を表す。
fn parse_obj_index(i: i32, len: usize) -> Option<usize> {
    if i > 0 {
        let u = (i as usize).wrapping_sub(1);
        if u < len { Some(u) } else { None }
    } else if i < 0 {
        let u = (len as i32 + i) as isize; // i is negative
        if u >= 0 { Some(u as usize) } else { None }
    } else {
        None
    }
}

/// OBJ ファイルから頂点座標と三角形インデックスを解析する。
fn parse_obj_positions_and_tris(path: &str) -> std::io::Result<(Vec<Vec3>, Vec<[usize;3]>)> {
    let text = std::fs::read_to_string(path)?;
    let mut positions: Vec<Vec3> = Vec::new();
    let mut tris: Vec<[usize;3]> = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }

        if let Some(rest) = line.strip_prefix("v ") {
            let mut it = rest.split_whitespace();
            let x: f64 = it.next().unwrap_or("0").parse().unwrap_or(0.0);
            let y: f64 = it.next().unwrap_or("0").parse().unwrap_or(0.0);
            let z: f64 = it.next().unwrap_or("0").parse().unwrap_or(0.0);
            positions.push(Vec3::new(x, y, z));
        } else if let Some(rest) = line.strip_prefix("f ") {
            let mut face: Vec<usize> = Vec::new();
            for tok in rest.split_whitespace() {
                let first = tok.split('/').next().unwrap_or("");
                let idx_i: i32 = first.parse().unwrap_or(0);
                if let Some(pi) = parse_obj_index(idx_i, positions.len()) {
                    face.push(pi);
                }
            }
            if face.len() >= 3 {
                let i0 = face[0];
                for k in 1..(face.len() - 1) {
                    tris.push([i0, face[k], face[k + 1]]);
                }
            }
        }
    }

    Ok((positions, tris))
}

/// 2 つの OBJ ファイルをモーションブラー付き三角形メッシュとして読み込む。
/// 両ファイルは同一トポロジ（頂点数・面数・インデックス）である必要がある。
pub fn load_obj_triangles_mb(path0: &str, path1: &str, mat_id: usize) -> std::io::Result<Vec<Triangle>> {
    let (p0, t0) = parse_obj_positions_and_tris(path0)?;
    let (p1, t1) = parse_obj_positions_and_tris(path1)?;

    if p0.len() != p1.len() || t0.len() != t1.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Topology mismatch between {} and {} (verts/tris differ)", path0, path1),
        ));
    }

    // Ensure triangle index lists match.
    for (a, b) in t0.iter().zip(t1.iter()) {
        if a != b {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Topology mismatch between {} and {} (triangle indices differ)", path0, path1),
            ));
        }
    }

    let mut tris: Vec<Triangle> = Vec::with_capacity(t0.len());
    for [i0, i1, i2] in t0 {
        let e1_0 = p0[i1] - p0[i0];
        let e2_0 = p0[i2] - p0[i0];
        let e1_1 = p1[i1] - p1[i0];
        let e2_1 = p1[i2] - p1[i0];
        tris.push(Triangle {
            v0_0: p0[i0], v1_0: p0[i1], v2_0: p0[i2],
            v0_1: p1[i0], v1_1: p1[i1], v2_1: p1[i2],
            e1_0, e2_0, e1_1, e2_1,
            mat_id,
        });
    }
    Ok(tris)
}

/// 単一の OBJ ファイルを静的三角形メッシュとして読み込む。
/// シャッター開 = シャッター閉に同一頂点を設定（モーションブラーなし）。
pub fn load_obj_triangles(path: &str, mat_id: usize) -> std::io::Result<Vec<Triangle>> {
    let (p0, t0) = parse_obj_positions_and_tris(path)?;
    let mut tris: Vec<Triangle> = Vec::with_capacity(t0.len());
    for [i0, i1, i2] in t0 {
        let e1 = p0[i1] - p0[i0];
        let e2 = p0[i2] - p0[i0];
        tris.push(Triangle {
            v0_0: p0[i0], v1_0: p0[i1], v2_0: p0[i2],
            v0_1: p0[i0], v1_1: p0[i1], v2_1: p0[i2],
            e1_0: e1, e2_0: e2, e1_1: e1, e2_1: e2,
            mat_id,
        });
    }
    Ok(tris)
}
