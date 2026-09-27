//! 組み込みの解析的メッシュ（矩形・立方体・円盤）の三角形生成。

use crate::geometry::Triangle;
use crate::math::Vec3;
use crate::obj_loader::MeshData;

pub(super) fn unit_rectangle_tris(mat_id: usize) -> MeshData {
    let a = Vec3::new(-1.0, -1.0, 0.0);
    let b = Vec3::new(1.0, -1.0, 0.0);
    let c = Vec3::new(1.0, 1.0, 0.0);
    let d = Vec3::new(-1.0, 1.0, 0.0);
    let tris = vec![
        Triangle::new_static(a, b, c, mat_id),
        Triangle::new_static(a, c, d, mat_id),
    ];
    // Mitsuba の rectangle と同じ割り当て: uv = ((x+1)/2, (y+1)/2)
    // 頂点 a, b, c, d の順に [0,0] [1,0] [1,1] [0,1]
    let uv = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    MeshData::with_uv(tris, uv, vec![[0, 1, 2], [0, 2, 3]])
}

pub(super) fn unit_cube_tris(mat_id: usize) -> MeshData {
    let v = [
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(-1.0, 1.0, 1.0),
    ];
    let quads = [
        [0, 1, 2, 3],
        [4, 7, 6, 5],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0],
    ];
    let mut tris = Vec::with_capacity(12);
    let mut tri_uv = Vec::with_capacity(12);
    for q in quads {
        tris.push(Triangle::new_static(v[q[0]], v[q[1]], v[q[2]], mat_id));
        tris.push(Triangle::new_static(v[q[0]], v[q[2]], v[q[3]], mat_id));
        // Mitsuba の cube は面ごとに [0,1]² の UV を張る。四角形の 4 頂点を
        // [0,0] [1,0] [1,1] [0,1] の順に対応させる（quads の並びがその順）
        tri_uv.push([0, 1, 2]);
        tri_uv.push([0, 2, 3]);
    }
    let uv = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    MeshData::with_uv(tris, uv, tri_uv)
}

pub(super) fn unit_disk_tris(mat_id: usize) -> MeshData {
    let n = 64;
    let center = Vec3::new(0.0, 0.0, 0.0);
    let mut tris = Vec::with_capacity(n);
    let mut uv: Vec<[f64; 2]> = Vec::with_capacity(2 * n + 1);
    let mut tri_uv = Vec::with_capacity(n);
    // Mitsuba の disk は極座標を UV に割り当てる: u = 半径 r、v = 角度 φ/2π。
    // 中心は r = 0 なので u = 0（v は縮退するので扇の始端の角度に合わせる）。
    uv.push([0.0, 0.0]); // 中心（扇ごとに v を変えたいので下で差し替える）
    for i in 0..n {
        let a0 = std::f64::consts::TAU * (i as f64) / (n as f64);
        let a1 = std::f64::consts::TAU * ((i + 1) as f64) / (n as f64);
        tris.push(Triangle::new_static(
            center,
            Vec3::new(a0.cos(), a0.sin(), 0.0),
            Vec3::new(a1.cos(), a1.sin(), 0.0),
            mat_id,
        ));
        let c = uv.len() as u32;
        uv.push([0.0, (i as f64) / (n as f64)]); // この扇の中心（u=0）
        let e0 = uv.len() as u32;
        uv.push([1.0, (i as f64) / (n as f64)]);
        let e1 = uv.len() as u32;
        uv.push([1.0, ((i + 1) as f64) / (n as f64)]);
        tri_uv.push([c, e0, e1]);
    }
    MeshData::with_uv(tris, uv, tri_uv)
}
