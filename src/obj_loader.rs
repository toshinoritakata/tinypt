//! 最小限の Wavefront OBJ ローダー（三角形化 + 頂点法線 + モーションブラー対応）。
//!
//! `v`（頂点）、`vn`（頂点法線）、`f`（フェース）を解析し、テクスチャ座標は無視する。
//! N 角形フェースはファン三角形化で分割される。
//! モーションブラー用に 2 つの OBJ ファイル（シャッター開/閉）を読み込む機能もある。
//!
//! 頂点法線は三角形とは別の配列で持つ（[`MeshData`]）。三角形 1 個あたり法線 3 本を
//! 直に持たせると 72 バイト増えるが、OBJ の `vn` を共有して添字（u32×3）で参照すれば
//! 12 バイトで済む（Rungholt 670 万三角形で 483MB → 80MB）。

use crate::math::Vec3;
use crate::geometry::Triangle;

/// 頂点法線を持たない三角形を表す番兵（`MeshData::tri_vn` の要素）。
pub const NO_NORMAL: u32 = u32::MAX;

/// OBJ から読み込んだメッシュ。三角形と、（あれば）頂点法線。
pub struct MeshData {
    /// 三角形リスト
    pub tris: Vec<Triangle>,
    /// OBJ の `vn`（正規化済み）。頂点法線が無いファイルでは空
    pub vn: Vec<Vec3>,
    /// 三角形ごとの `vn` の添字（v0, v1, v2 の順）。3 つとも [`NO_NORMAL`] なら面法線を使う。
    /// `vn` が空のときはこの配列も空（= メッシュ全体が面法線）
    pub tri_vn: Vec<[u32; 3]>,
}

impl MeshData {
    /// 頂点法線を持たない（面法線だけの）メッシュ。
    pub fn flat(tris: Vec<Triangle>) -> Self {
        Self { tris, vn: Vec::new(), tri_vn: Vec::new() }
    }

    /// 頂点法線を捨てて面法線だけにする（シーンファイルの `face_normals=true`）。
    pub fn into_flat(mut self) -> Self {
        self.vn.clear();
        self.tri_vn.clear();
        self
    }

    /// 1 つでも頂点法線を持つ三角形があるか。
    pub fn has_normals(&self) -> bool {
        !self.vn.is_empty() && !self.tri_vn.is_empty()
    }
}

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

/// `f` の 1 トークン（`v`, `v/vt`, `v//vn`, `v/vt/vn`）から頂点・法線の添字を取り出す。
/// 添字は「そのトークンを読んだ時点の配列長」を基準に解決する（負の相対添字の意味）。
fn parse_face_token(tok: &str, nv: usize, nn: usize) -> (Option<usize>, Option<usize>) {
    let mut it = tok.split('/');
    let v = it
        .next()
        .and_then(|s| s.parse::<i32>().ok())
        .and_then(|i| parse_obj_index(i, nv));
    let _vt = it.next();
    let n = it
        .next()
        .and_then(|s| s.parse::<i32>().ok())
        .and_then(|i| parse_obj_index(i, nn));
    (v, n)
}

/// OBJ ファイルから頂点座標・頂点法線・三角形（頂点添字と法線添字）を解析する。
fn parse_obj(path: &str) -> std::io::Result<(Vec<Vec3>, Vec<Vec3>, Vec<[usize; 3]>, Vec<[u32; 3]>)> {
    let text = std::fs::read_to_string(path)?;
    let mut positions: Vec<Vec3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut tris: Vec<[usize; 3]> = Vec::new();
    let mut tri_vn: Vec<[u32; 3]> = Vec::new();
    // フェースごとの一時バッファはループの外で使い回す（数百万フェースで確保が効いてくる）
    let mut face: Vec<(usize, u32)> = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }

        if let Some(rest) = line.strip_prefix("v ") {
            let mut it = rest.split_whitespace();
            let x: f64 = it.next().unwrap_or("0").parse().unwrap_or(0.0);
            let y: f64 = it.next().unwrap_or("0").parse().unwrap_or(0.0);
            let z: f64 = it.next().unwrap_or("0").parse().unwrap_or(0.0);
            positions.push(Vec3::new(x, y, z));
        } else if let Some(rest) = line.strip_prefix("vn ") {
            let mut it = rest.split_whitespace();
            let x: f64 = it.next().unwrap_or("0").parse().unwrap_or(0.0);
            let y: f64 = it.next().unwrap_or("0").parse().unwrap_or(0.0);
            let z: f64 = it.next().unwrap_or("0").parse().unwrap_or(0.0);
            let n = Vec3::new(x, y, z);
            // 長さ 0 の vn（壊れたファイル）は面法線扱いにできるよう 0 のまま入れておき、
            // 参照側（補間）で長さを見て落とす
            let len = n.len();
            normals.push(if len > 0.0 { n / len } else { n });
        } else if let Some(rest) = line.strip_prefix("f ") {
            face.clear();
            for tok in rest.split_whitespace() {
                let (v, n) = parse_face_token(tok, positions.len(), normals.len());
                if let Some(vi) = v {
                    face.push((vi, n.map(|i| i as u32).unwrap_or(NO_NORMAL)));
                }
            }
            if face.len() >= 3 {
                // ファン三角形化。法線の添字も同じ並びで割り当てる
                let (i0, n0) = face[0];
                for k in 1..(face.len() - 1) {
                    let (i1, n1) = face[k];
                    let (i2, n2) = face[k + 1];
                    tris.push([i0, i1, i2]);
                    tri_vn.push([n0, n1, n2]);
                }
            }
        }
    }

    Ok((positions, normals, tris, tri_vn))
}

/// 三角形ごとの法線添字を**その場で**整理する。3 つ揃っていない三角形は面法線扱い
/// （`NO_NORMAL` 3 つ）にし、1 つも法線が残らなければ両方の配列を空にする（メッシュ全体が面法線）。
///
/// 新しい配列を作らずに書き換えるのは、大きなメッシュで読み込み時のピーク RSS を抑えるため
/// （Rungholt の 670 万三角形では添字配列だけで 80MB あり、コピーを作ると一時的に倍になる）。
fn normalize_tri_vn(normals: &mut Vec<Vec3>, tri_vn: &mut Vec<[u32; 3]>) {
    if normals.is_empty() {
        tri_vn.clear();
        tri_vn.shrink_to_fit();
        return;
    }
    let mut any = false;
    for t in tri_vn.iter_mut() {
        // 3 頂点とも有効な法線を持つ三角形だけ補間する（部分的な指定は面法線に落とす）
        let ok = t.iter().all(|&i| i != NO_NORMAL && normals[i as usize].len() > 0.0);
        if ok {
            any = true;
        } else {
            *t = [NO_NORMAL; 3];
        }
    }
    if !any {
        normals.clear();
        normals.shrink_to_fit();
        tri_vn.clear();
        tri_vn.shrink_to_fit();
    }
}

/// 2 つの OBJ ファイルをモーションブラー付き三角形メッシュとして読み込む。
/// 両ファイルは同一トポロジ（頂点数・面数・インデックス）である必要がある。
/// 頂点法線はシャッター開（`path0`）のものを使う（時間で回転する法線は未対応）。
pub fn load_obj_mesh_mb(path0: &str, path1: &str, mat_id: usize) -> std::io::Result<MeshData> {
    let (p0, n0, t0, vn0) = parse_obj(path0)?;
    let (p1, _n1, t1, _vn1) = parse_obj(path1)?;

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
    let (mut vn, mut tri_vn) = (n0, vn0);
    normalize_tri_vn(&mut vn, &mut tri_vn);
    Ok(MeshData { tris, vn, tri_vn })
}

/// 単一の OBJ ファイルを静的三角形メッシュとして読み込む。
/// シャッター開 = シャッター閉に同一頂点を設定（モーションブラーなし）。
pub fn load_obj_mesh(path: &str, mat_id: usize) -> std::io::Result<MeshData> {
    let (p0, n0, t0, vn0) = parse_obj(path)?;
    let mut tris: Vec<Triangle> = Vec::with_capacity(t0.len());
    for [i0, i1, i2] in t0 {
        tris.push(Triangle::new_static(p0[i0], p0[i1], p0[i2], mat_id));
    }
    let (mut vn, mut tri_vn) = (n0, vn0);
    normalize_tri_vn(&mut vn, &mut tri_vn);
    Ok(MeshData { tris, vn, tri_vn })
}

/// 単一の OBJ を三角形リストだけ読み込む（頂点法線は捨てる）。
pub fn load_obj_triangles(path: &str, mat_id: usize) -> std::io::Result<Vec<Triangle>> {
    Ok(load_obj_mesh(path, mat_id)?.tris)
}

/// 2 つの OBJ を三角形リストだけ読み込む（頂点法線は捨てる）。
pub fn load_obj_triangles_mb(path0: &str, path1: &str, mat_id: usize) -> std::io::Result<Vec<Triangle>> {
    Ok(load_obj_mesh_mb(path0, path1, mat_id)?.tris)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト用に一時 OBJ を書いて読み、後始末する。
    /// 名前はプロセス ID とグローバルな連番で決める（同じ内容の文字列リテラルは
    /// コンパイラに共有されうるので、内容から名前を作るとテスト間で衝突する）。
    fn with_obj<T>(body: &str, f: impl FnOnce(&str) -> T) -> T {
        static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::env::temp_dir();
        let id = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = dir.join(format!("tinypt_obj_{}_{}.obj", std::process::id(), id));
        std::fs::write(&path, body).unwrap();
        let out = f(path.to_string_lossy().as_ref());
        std::fs::remove_file(&path).ok();
        out
    }

    /// `v//vn` 形式の法線添字を読む。
    #[test]
    fn parses_double_slash_normal_indices() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nvn 0 0 1\nvn 0 0 1\nf 1//1 2//2 3//3\n";
        let m = with_obj(obj, |p| load_obj_mesh(p, 0).unwrap());
        assert_eq!(m.tris.len(), 1);
        assert!(m.has_normals());
        assert_eq!(m.tri_vn[0], [0, 1, 2]);
        assert_eq!(m.vn.len(), 3);
    }

    /// `v/vt/vn` 形式でも法線添字だけを拾う（テクスチャ座標は無視）。
    #[test]
    fn parses_slash_texture_normal_indices() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvn 1 0 0\nvn 0 1 0\nvn 0 0 1\nf 1/1/1 2/1/2 3/1/3\n";
        let m = with_obj(obj, |p| load_obj_mesh(p, 0).unwrap());
        assert_eq!(m.tri_vn[0], [0, 1, 2]);
    }

    /// 負のインデックス（末尾からの相対）は頂点にも法線にも効く。
    #[test]
    fn parses_negative_indices_for_positions_and_normals() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 1 0 0\nvn 0 1 0\nvn 0 0 1\nf -3//-3 -2//-2 -1//-1\n";
        let m = with_obj(obj, |p| load_obj_mesh(p, 0).unwrap());
        assert_eq!(m.tris.len(), 1);
        assert_eq!(m.tri_vn[0], [0, 1, 2]);
        // 頂点も正しく解決されている
        assert!((m.tris[0].v0_0 - Vec3::new(0.0, 0.0, 0.0)).len() < 1e-12);
        assert!((m.tris[0].v2_0 - Vec3::new(0.0, 1.0, 0.0)).len() < 1e-12);
    }

    /// `vn` が無いファイルは面法線だけのメッシュになる（配列は空＝メモリを食わない）。
    #[test]
    fn file_without_normals_has_no_normal_arrays() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
        let m = with_obj(obj, |p| load_obj_mesh(p, 0).unwrap());
        assert!(!m.has_normals());
        assert!(m.vn.is_empty() && m.tri_vn.is_empty());
    }

    /// ファン三角形化で、各三角形に対応する法線添字が割り当てられる。
    /// 4 角形 (1,2,3,4) → (1,2,3) と (1,3,4)。
    #[test]
    fn fan_triangulation_assigns_matching_normal_indices() {
        let obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n\
                   vn 1 0 0\nvn 0 1 0\nvn 0 0 1\nvn 1 1 0\n\
                   f 1//1 2//2 3//3 4//4\n";
        let m = with_obj(obj, |p| load_obj_mesh(p, 0).unwrap());
        assert_eq!(m.tris.len(), 2);
        assert_eq!(m.tri_vn[0], [0, 1, 2]);
        assert_eq!(m.tri_vn[1], [0, 2, 3]);
    }

    /// 一部のフェースだけ法線を持つ場合、持たないフェースは面法線（NO_NORMAL）になる。
    #[test]
    fn faces_without_normals_fall_back_to_face_normal() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nvn 0 0 1\n\
                   f 1//1 2//1 3//1\nf 2 4 3\n";
        let m = with_obj(obj, |p| load_obj_mesh(p, 0).unwrap());
        assert!(m.has_normals());
        assert_eq!(m.tri_vn[0], [0, 0, 0]);
        assert_eq!(m.tri_vn[1], [NO_NORMAL; 3]);
    }

    /// 長さ 0 の `vn`（壊れたファイル）を参照する三角形は面法線に落ちる。
    #[test]
    fn zero_length_normal_falls_back_to_face_normal() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 0\nf 1//1 2//1 3//1\n";
        let m = with_obj(obj, |p| load_obj_mesh(p, 0).unwrap());
        assert!(!m.has_normals(), "全ての三角形が面法線なら配列は空になる");
    }

    /// 読み込んだ `vn` は正規化されている（補間の重みが法線の長さに引きずられない）。
    #[test]
    fn normals_are_normalized_on_load() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 7\nvn 0 0 7\nvn 0 0 7\nf 1//1 2//2 3//3\n";
        let m = with_obj(obj, |p| load_obj_mesh(p, 0).unwrap());
        assert!((m.vn[0].len() - 1.0).abs() < 1e-12);
    }

    /// モーションブラー版もシャッター開の頂点法線を保持する。
    #[test]
    fn motion_blur_pair_keeps_open_shutter_normals() {
        let a = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1//1 2//1 3//1\n";
        let b = "v 0 0 1\nv 1 0 1\nv 0 1 1\nvn 0 0 1\nf 1//1 2//1 3//1\n";
        let m = with_obj(a, |pa| with_obj(b, |pb| load_obj_mesh_mb(pa, pb, 0).unwrap()));
        assert!(m.has_normals());
        assert_eq!(m.tri_vn[0], [0, 0, 0]);
    }

    /// `into_flat` は頂点法線を捨てる（シーンの face_normals=true）。
    #[test]
    fn into_flat_drops_normals() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1//1 2//1 3//1\n";
        let m = with_obj(obj, |p| load_obj_mesh(p, 0).unwrap()).into_flat();
        assert!(!m.has_normals());
        assert_eq!(m.tris.len(), 1);
    }
}
