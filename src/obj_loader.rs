//! 最小限の Wavefront OBJ ローダー（三角形化 + 頂点法線 + テクスチャ座標 + モーションブラー対応）。
//!
//! `v`（頂点）、`vn`（頂点法線）、`vt`（テクスチャ座標）、`f`（フェース）を解析する。
//! N 角形フェースはファン三角形化で分割される。
//! モーションブラー用に 2 つの OBJ ファイル（シャッター開/閉）を読み込む機能もある。
//!
//! 頂点法線と UV は三角形とは別の配列で持つ（[`MeshData`]）。三角形 1 個あたり法線 3 本を
//! 直に持たせると 72 バイト増えるが、OBJ の `vn` を共有して添字（u32×3）で参照すれば
//! 12 バイトで済む（Rungholt 670 万三角形で 483MB → 80MB）。UV も同じ設計
//! （`vt` を共有 + 三角形ごとに u32×3 = 12 B/三角形。UV の無いメッシュは増加ゼロ）。

use crate::math::Vec3;
use crate::geometry::Triangle;

/// 頂点法線を持たない三角形を表す番兵（`MeshData::tri_vn` の要素）。
pub const NO_NORMAL: u32 = u32::MAX;

/// UV を持たない三角形を表す番兵（`MeshData::tri_uv` の要素）。
pub const NO_UV: u32 = u32::MAX;

/// OBJ から読み込んだメッシュ。三角形と、（あれば）頂点法線・UV。
pub struct MeshData {
    /// 三角形リスト
    pub tris: Vec<Triangle>,
    /// OBJ の `vn`（正規化済み）。頂点法線が無いファイルでは空
    pub vn: Vec<Vec3>,
    /// 三角形ごとの `vn` の添字（v0, v1, v2 の順）。3 つとも [`NO_NORMAL`] なら面法線を使う。
    /// `vn` が空のときはこの配列も空（= メッシュ全体が面法線）
    pub tri_vn: Vec<[u32; 3]>,
    /// OBJ の `vt`（(u, v)。第 3 成分 w は無視）。UV が無いファイルでは空
    pub uv: Vec<[f64; 2]>,
    /// 三角形ごとの `vt` の添字。3 つとも [`NO_UV`] なら UV 無し（テクスチャは (0,0) を引く）。
    /// `uv` が空のときはこの配列も空
    pub tri_uv: Vec<[u32; 3]>,
    /// モーションブラーのシャッター閉頂点（PERF-4 P4b）。空なら全三角形が静止（閉 = 開）。
    /// 非空なら `tris` と同じ長さで添字がそのまま対応する（[`crate::world::Mesh::motion`] と同じ規約）。
    pub motion: Vec<[Vec3; 3]>,
}

impl MeshData {
    /// 頂点法線も UV も持たないメッシュ。
    pub fn flat(tris: Vec<Triangle>) -> Self {
        Self { tris, vn: Vec::new(), tri_vn: Vec::new(), uv: Vec::new(), tri_uv: Vec::new(), motion: Vec::new() }
    }

    /// UV だけを持つメッシュ（パラメトリック形状用）。
    pub fn with_uv(tris: Vec<Triangle>, uv: Vec<[f64; 2]>, tri_uv: Vec<[u32; 3]>) -> Self {
        Self { tris, vn: Vec::new(), tri_vn: Vec::new(), uv, tri_uv, motion: Vec::new() }
    }

    /// 1 つでも UV を持つ三角形があるか。
    pub fn has_uv(&self) -> bool {
        !self.uv.is_empty() && !self.tri_uv.is_empty()
    }

    /// 頂点法線を捨てて面法線だけにする（シーンファイルの `face_normals=true`）。
    /// UV は陰影の付け方とは無関係なので**残す**（テクスチャは引き続き効く）。
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

/// `f` の 1 トークン（`v`, `v/vt`, `v//vn`, `v/vt/vn`）から頂点・UV・法線の添字を取り出す。
/// 添字は「そのトークンを読んだ時点の配列長」を基準に解決する（負の相対添字の意味）。
fn parse_face_token(tok: &str, nv: usize, nt: usize, nn: usize) -> (Option<usize>, Option<usize>, Option<usize>) {
    let mut it = tok.split('/');
    let v = it
        .next()
        .and_then(|s| s.parse::<i32>().ok())
        .and_then(|i| parse_obj_index(i, nv));
    let t = it
        .next()
        .and_then(|s| s.parse::<i32>().ok())
        .and_then(|i| parse_obj_index(i, nt));
    let n = it
        .next()
        .and_then(|s| s.parse::<i32>().ok())
        .and_then(|i| parse_obj_index(i, nn));
    (v, t, n)
}

/// `parse_obj` の生の結果（添字はまだ番兵の整理をしていない）。
struct ParsedObj {
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Vec<[f64; 2]>,
    tris: Vec<[usize; 3]>,
    tri_vn: Vec<[u32; 3]>,
    tri_uv: Vec<[u32; 3]>,
    /// `usemtl` のラン（三角形列の開始位置, `mat_names` の添字）。三角形数に比例しない
    mat_runs: Vec<(usize, usize)>,
    /// 実際に面に使われた材質名（初出順）。空文字列は「`usemtl` 前の面」= 既定材質
    mat_names: Vec<String>,
    /// `mtllib` で指定された MTL ファイル名（出現順）
    mtllibs: Vec<String>,
}

/// `usemtl` でグループ分けした OBJ（1 メッシュのまま、三角形ごとに材質を持つ）。
pub struct ObjGroups {
    /// 三角形の `mat_id` は `mat_names` の添字（呼び出し側が実際の材質 ID に振り直す）
    pub mesh: MeshData,
    /// 面に使われた材質名（初出順）。空文字列は `usemtl` 前の面（既定材質）
    pub mat_names: Vec<String>,
    /// `mtllib` のファイル名（OBJ からの相対、出現順）
    pub mtllibs: Vec<String>,
}

/// OBJ ファイルから頂点座標・頂点法線・UV・三角形（各添字）を解析する。
fn parse_obj(path: &str) -> std::io::Result<ParsedObj> {
    let text = std::fs::read_to_string(path)?;
    let mut positions: Vec<Vec3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut uvs: Vec<[f64; 2]> = Vec::new();
    let mut tris: Vec<[usize; 3]> = Vec::new();
    let mut tri_vn: Vec<[u32; 3]> = Vec::new();
    let mut tri_uv: Vec<[u32; 3]> = Vec::new();
    // フェースごとの一時バッファはループの外で使い回す（数百万フェースで確保が効いてくる）
    let mut face: Vec<(usize, u32, u32)> = Vec::new();
    let mut mat_runs: Vec<(usize, usize)> = Vec::new();
    let mut mat_names: Vec<String> = Vec::new();
    let mut mtllibs: Vec<String> = Vec::new();
    // 現在の `usemtl` 名と、その `mat_names` 添字（最初の面が来るまで作らない）
    let mut cur_name = String::new();
    let mut cur_idx: Option<usize> = None;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }

        if let Some(rest) = line.strip_prefix("v ") {
            let mut it = rest.split_whitespace();
            let x: f64 = it.next().unwrap_or("0").parse().unwrap_or(0.0);
            let y: f64 = it.next().unwrap_or("0").parse().unwrap_or(0.0);
            let z: f64 = it.next().unwrap_or("0").parse().unwrap_or(0.0);
            positions.push(Vec3::new(x, y, z));
        } else if let Some(rest) = line.strip_prefix("vt ") {
            let mut it = rest.split_whitespace();
            let u: f64 = it.next().unwrap_or("0").parse().unwrap_or(0.0);
            let v: f64 = it.next().unwrap_or("0").parse().unwrap_or(0.0);
            // 第 3 成分 w は使わない（Wavefront の仕様上あってもよい）
            uvs.push([u, v]);
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
        } else if let Some(rest) = line.strip_prefix("usemtl ") {
            cur_name = rest.trim().to_string();
            cur_idx = None;
        } else if let Some(rest) = line.strip_prefix("mtllib ") {
            // 複数ファイルを空白区切りで並べる書式もあるが、空白入りのファイル名の方が現実的
            mtllibs.push(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("f ") {
            face.clear();
            for tok in rest.split_whitespace() {
                let (v, t, n) = parse_face_token(tok, positions.len(), uvs.len(), normals.len());
                if let Some(vi) = v {
                    face.push((
                        vi,
                        t.map(|i| i as u32).unwrap_or(NO_UV),
                        n.map(|i| i as u32).unwrap_or(NO_NORMAL),
                    ));
                }
            }
            if face.len() >= 3 {
                let mi = match cur_idx {
                    Some(i) => i,
                    None => {
                        let i = mat_names.iter().position(|n| *n == cur_name).unwrap_or_else(|| {
                            mat_names.push(cur_name.clone());
                            mat_names.len() - 1
                        });
                        cur_idx = Some(i);
                        i
                    }
                };
                if mat_runs.last().map(|r| r.1) != Some(mi) {
                    mat_runs.push((tris.len(), mi));
                }
                // ファン三角形化。法線と UV の添字も同じ並びで割り当てる
                let (i0, t0, n0) = face[0];
                for k in 1..(face.len() - 1) {
                    let (i1, t1, n1) = face[k];
                    let (i2, t2, n2) = face[k + 1];
                    tris.push([i0, i1, i2]);
                    tri_vn.push([n0, n1, n2]);
                    tri_uv.push([t0, t1, t2]);
                }
            }
        }
    }

    Ok(ParsedObj { positions, normals, uvs, tris, tri_vn, tri_uv, mat_runs, mat_names, mtllibs })
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

/// 三角形ごとの UV 添字を**その場で**整理する。3 つ揃っていない三角形は UV 無し
/// （`NO_UV` 3 つ）にし、1 つも残らなければ両方の配列を空にする（メッシュ全体が UV 無し）。
/// 法線側（[`normalize_tri_vn`]）と同じ方針。
fn normalize_tri_uv(uvs: &mut Vec<[f64; 2]>, tri_uv: &mut Vec<[u32; 3]>) {
    if uvs.is_empty() {
        tri_uv.clear();
        tri_uv.shrink_to_fit();
        return;
    }
    let mut any = false;
    for t in tri_uv.iter_mut() {
        if t.iter().all(|&i| i != NO_UV) {
            any = true;
        } else {
            *t = [NO_UV; 3];
        }
    }
    if !any {
        uvs.clear();
        uvs.shrink_to_fit();
        tri_uv.clear();
        tri_uv.shrink_to_fit();
    }
}

/// 2 つの OBJ ファイルをモーションブラー付き三角形メッシュとして読み込む。
/// 両ファイルは同一トポロジ（頂点数・面数・インデックス）である必要がある。
/// 頂点法線はシャッター開（`path0`）のものを使う（時間で回転する法線は未対応）。
pub fn load_obj_mesh_mb(path0: &str, path1: &str, mat_id: usize) -> std::io::Result<MeshData> {
    let a = parse_obj(path0)?;
    let b = parse_obj(path1)?;
    let (p0, n0, t0, vn0, uv0, tuv0) = (a.positions, a.normals, a.tris, a.tri_vn, a.uvs, a.tri_uv);
    let (p1, t1) = (b.positions, b.tris);

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
    let mut motion: Vec<[Vec3; 3]> = Vec::with_capacity(t0.len());
    for [i0, i1, i2] in t0 {
        tris.push(Triangle { v0_0: p0[i0], v1_0: p0[i1], v2_0: p0[i2], mat_id });
        motion.push([p1[i0], p1[i1], p1[i2]]);
    }
    let (mut vn, mut tri_vn) = (n0, vn0);
    normalize_tri_vn(&mut vn, &mut tri_vn);
    let (mut uv, mut tri_uv) = (uv0, tuv0);
    normalize_tri_uv(&mut uv, &mut tri_uv);
    Ok(MeshData { tris, vn, tri_vn, uv, tri_uv, motion })
}

/// 単一の OBJ ファイルを静的三角形メッシュとして読み込む。
/// シャッター開 = シャッター閉に同一頂点を設定（モーションブラーなし）。
/// `usemtl` は無視し、全三角形が `mat_id` になる。
pub fn load_obj_mesh(path: &str, mat_id: usize) -> std::io::Result<MeshData> {
    Ok(static_mesh(parse_obj(path)?, |_| mat_id).0)
}

/// [`load_obj_mesh`] の `usemtl` 対応版。1 メッシュのまま、三角形ごとに材質（`mat_names` の添字）を持つ。
pub fn load_obj_groups(path: &str) -> std::io::Result<ObjGroups> {
    let parsed = parse_obj(path)?;
    // ラン表から三角形 → 材質添字を引く（ランは三角形の並び順なので単調に進める）
    let runs = parsed.mat_runs.clone();
    let mut r = 0usize;
    let (mesh, mat_names, mtllibs) = static_mesh(parsed, |ti| {
        while r + 1 < runs.len() && runs[r + 1].0 <= ti {
            r += 1;
        }
        runs.get(r).map(|x| x.1).unwrap_or(0)
    });
    Ok(ObjGroups { mesh, mat_names, mtllibs })
}

/// 解析結果から静的メッシュを作る。`mat_of(三角形番号)`（昇順に呼ばれる）が各三角形の `mat_id`。
fn static_mesh(parsed: ParsedObj, mut mat_of: impl FnMut(usize) -> usize) -> (MeshData, Vec<String>, Vec<String>) {
    let p0 = parsed.positions;
    let mut tris: Vec<Triangle> = Vec::with_capacity(parsed.tris.len());
    for (ti, [i0, i1, i2]) in parsed.tris.into_iter().enumerate() {
        tris.push(Triangle::new_static(p0[i0], p0[i1], p0[i2], mat_of(ti)));
    }
    let (mut vn, mut tri_vn) = (parsed.normals, parsed.tri_vn);
    normalize_tri_vn(&mut vn, &mut tri_vn);
    let (mut uv, mut tri_uv) = (parsed.uvs, parsed.tri_uv);
    normalize_tri_uv(&mut uv, &mut tri_uv);
    (MeshData { tris, vn, tri_vn, uv, tri_uv, motion: Vec::new() }, parsed.mat_names, parsed.mtllibs)
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

    /// `f v/vt` 形式（法線なし）の UV 添字を読む。
    #[test]
    fn parses_texture_coordinate_indices() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n";
        let m = with_obj(obj, |p| load_obj_mesh(p, 0).unwrap());
        assert!(m.has_uv());
        assert!(!m.has_normals());
        assert_eq!(m.tri_uv[0], [0, 1, 2]);
        assert_eq!(m.uv.len(), 3);
        assert_eq!(m.uv[1], [1.0, 0.0]);
    }

    /// `f v/vt/vn` では UV と法線の両方を読む。
    #[test]
    fn parses_uv_and_normal_indices_together() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0.25 0.5\nvn 0 0 1\nf 1/1/1 2/1/1 3/1/1\n";
        let m = with_obj(obj, |p| load_obj_mesh(p, 0).unwrap());
        assert!(m.has_uv() && m.has_normals());
        assert_eq!(m.tri_uv[0], [0, 0, 0]);
        assert_eq!(m.tri_vn[0], [0, 0, 0]);
        assert_eq!(m.uv[0], [0.25, 0.5]);
    }

    /// `v//vn`（UV を飛ばす記法）では UV を持たない。
    #[test]
    fn double_slash_means_no_uv() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvn 0 0 1\nf 1//1 2//1 3//1\n";
        let m = with_obj(obj, |p| load_obj_mesh(p, 0).unwrap());
        assert!(!m.has_uv(), "v//vn は UV 添字を持たない");
        assert!(m.has_normals());
    }

    /// UV の負の相対添字も解決する。
    #[test]
    fn parses_negative_uv_indices() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/-3 2/-2 3/-1\n";
        let m = with_obj(obj, |p| load_obj_mesh(p, 0).unwrap());
        assert_eq!(m.tri_uv[0], [0, 1, 2]);
    }

    /// `vt` の第 3 成分 w は無視する。
    #[test]
    fn ignores_the_third_uv_component() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0.3 0.7 0.9\nf 1/1 2/1 3/1\n";
        let m = with_obj(obj, |p| load_obj_mesh(p, 0).unwrap());
        assert_eq!(m.uv[0], [0.3, 0.7]);
    }

    /// UV が無いファイルでは配列が空（メモリ増加ゼロ）。
    #[test]
    fn file_without_uv_has_no_uv_arrays() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
        let m = with_obj(obj, |p| load_obj_mesh(p, 0).unwrap());
        assert!(!m.has_uv());
        assert!(m.uv.is_empty() && m.tri_uv.is_empty());
    }

    /// 一部のフェースだけ UV を持つ場合、持たないフェースは `NO_UV` に落ちる。
    #[test]
    fn faces_without_uv_fall_back_to_no_uv() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nvt 0 0\n\
                   f 1/1 2/1 3/1\nf 2 4 3\n";
        let m = with_obj(obj, |p| load_obj_mesh(p, 0).unwrap());
        assert!(m.has_uv());
        assert_eq!(m.tri_uv[0], [0, 0, 0]);
        assert_eq!(m.tri_uv[1], [NO_UV; 3]);
    }

    /// 四角形のファン三角形化で UV 添字も同じ並びに割り当てられる。
    #[test]
    fn fan_triangulation_assigns_matching_uv_indices() {
        let obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n\
                   vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n\
                   f 1/1 2/2 3/3 4/4\n";
        let m = with_obj(obj, |p| load_obj_mesh(p, 0).unwrap());
        assert_eq!(m.tris.len(), 2);
        assert_eq!(m.tri_uv[0], [0, 1, 2]);
        assert_eq!(m.tri_uv[1], [0, 2, 3]);
    }

    /// `into_flat`（face_normals=true）は法線だけを捨てて **UV は残す**。
    #[test]
    fn into_flat_keeps_uv() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvn 0 0 1\nf 1/1/1 2/1/1 3/1/1\n";
        let m = with_obj(obj, |p| load_obj_mesh(p, 0).unwrap()).into_flat();
        assert!(!m.has_normals(), "法線は捨てる");
        assert!(m.has_uv(), "UV は陰影と無関係なので残す");
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

    /// `usemtl` ごとに三角形の材質添字が付く。`usemtl` 前の面は既定（空文字列）、
    /// 同じ名前に戻ると同じ添字、四角形のファン分割も同じ材質。
    #[test]
    fn usemtl_groups_assign_per_triangle_material() {
        let obj = "mtllib a.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\n\
                   f 1 2 3\nusemtl Red\nf 1 2 3\nf 1 2 4 3\nusemtl Blue\nf 1 2 3\nusemtl Red\nf 2 3 4\n";
        let g = with_obj(obj, |p| load_obj_groups(p).unwrap());
        assert_eq!(g.mat_names, vec!["".to_string(), "Red".to_string(), "Blue".to_string()]);
        assert_eq!(g.mtllibs, vec!["a.mtl".to_string()]);
        let ids: Vec<usize> = g.mesh.tris.iter().map(|t| t.mat_id).collect();
        assert_eq!(ids, vec![0, 1, 1, 1, 2, 1]);
    }

    /// `usemtl` の無い OBJ は既定 1 つだけ。`load_obj_mesh` は `usemtl` を無視して `mat_id` を使う。
    #[test]
    fn no_usemtl_is_single_default_and_plain_loader_ignores_usemtl() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl X\nf 1 2 3\n";
        let m = with_obj(obj, |p| load_obj_mesh(p, 7).unwrap());
        assert_eq!(m.tris[0].mat_id, 7);
        let g = with_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", |p| load_obj_groups(p).unwrap());
        assert_eq!(g.mat_names, vec!["".to_string()]);
    }
}
