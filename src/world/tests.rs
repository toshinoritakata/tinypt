use super::*;
use std::sync::Arc;

use crate::geometry::{face_forward, Hit, Triangle};
use crate::material::Material;
use crate::math::Color;
use crate::obj_loader::{MeshData, NO_NORMAL, NO_UV};
use crate::ray::Ray;
use crate::rng::Rng;
use crate::texture::AlphaMask;
use crate::transform::{AnimatedTransform, Transform};

use super::hit::{INSTANCE_RETRY_LIMIT, TLAS_MIN_PRIMS};
use super::lights::{orthonormal_basis, sphere_cone};

#[cfg(test)]
/// テスト用のメッシュ生成（頂点法線つき）。スムーズシェーディングの検証で world / integrator の
/// 両方から使う。
#[cfg(test)]
pub(crate) mod test_meshes {
    use super::*;
    use crate::obj_loader::MeshData;

    /// z=0 平面の四角形（2 三角形、面法線 +z）。頂点法線を `ns` で指定できる。
    /// 幾何法線とシェーディング法線を大きく食い違わせた「1 枚ポリゴン」を作るのに使う。
    pub fn tilted_quad(half: f64, ns: Vec3, mat_id: usize) -> MeshData {
        let v = |x: f64, y: f64| Vec3::new(x, y, 0.0);
        let tris = vec![
            Triangle::new_static(v(-half, -half), v(half, -half), v(half, half), mat_id),
            Triangle::new_static(v(-half, -half), v(half, half), v(-half, half), mat_id),
        ];
        MeshData { tris, vn: vec![ns.norm()], tri_vn: vec![[0, 0, 0], [0, 0, 0]], uv: Vec::new(), tri_uv: Vec::new(), motion: Vec::new() }
    }

    /// 経度 `nu` × 緯度 `nv` の UV 球。頂点法線は解析的な法線（中心からの単位ベクトル）。
    /// `smooth = false` なら頂点法線を付けない（面法線だけの同一形状）。
    pub fn uv_sphere(center: Vec3, radius: f64, nu: usize, nv: usize, mat_id: usize, smooth: bool) -> MeshData {
        let mut pos: Vec<Vec3> = Vec::new();
        let at = |iu: usize, iv: usize| -> Vec3 {
            let theta = std::f64::consts::PI * (iv as f64) / (nv as f64);
            let phi = 2.0 * std::f64::consts::PI * (iu as f64) / (nu as f64);
            Vec3::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin())
        };
        for iv in 0..=nv {
            for iu in 0..nu {
                pos.push(at(iu, iv));
            }
        }
        let idx = |iu: usize, iv: usize| iv * nu + (iu % nu);
        let mut tris = Vec::new();
        let mut tri_vn = Vec::new();
        let push = |a: usize, b: usize, c: usize, tris: &mut Vec<Triangle>, tri_vn: &mut Vec<[u32; 3]>, pos: &Vec<Vec3>| {
            let p = |i: usize| center + pos[i] * radius;
            tris.push(Triangle::new_static(p(a), p(b), p(c), mat_id));
            tri_vn.push([a as u32, b as u32, c as u32]);
        };
        for iv in 0..nv {
            for iu in 0..nu {
                let (a, b, c, d) = (idx(iu, iv), idx(iu + 1, iv), idx(iu + 1, iv + 1), idx(iu, iv + 1));
                push(a, b, c, &mut tris, &mut tri_vn, &pos);
                push(a, c, d, &mut tris, &mut tri_vn, &pos);
            }
        }
        // 頂点法線 = 単位球面上の位置（解析的な法線）
        let vn = pos.clone();
        if smooth { MeshData { tris, vn, tri_vn, uv: Vec::new(), tri_uv: Vec::new(), motion: Vec::new() } } else { MeshData::flat(tris) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::uniform_sphere_dir;
    use crate::geometry::Sphere;

    /// z=0 の [0,1]² の板（UV = xy）に、左半分が不透明・右半分が透明の 2x1 マスクを貼ったワールド。
    /// 板の奥 z=-1 に不透明の床（マスク無し）を置く。
    fn masked_plate_world(mask_values: [u8; 2]) -> World {
        let mut world = World::new();
        let (a, b, c, d) = (
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let tris = vec![Triangle::new_static(a, b, c, 0), Triangle::new_static(a, c, d, 0)];
        let uv = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let tri_uv = vec![[0, 1, 2], [0, 2, 3]];
        let mask = Arc::new(AlphaMask::from_u8(2, 1, mask_values.to_vec(), crate::texture::Wrap::Clamp));
        let data = MeshData::with_uv(tris, uv, tri_uv);
        world.add_mesh_data_instance_with_alpha(data, vec![mask], vec![(1, 1.0); 2], Transform::identity());
        // 奥の床（マスク無し）
        let f = |x: f64, y: f64| Vec3::new(x, y, -1.0);
        world.add_mesh_instance(
            vec![
                Triangle::new_static(f(-5.0, -5.0), f(5.0, -5.0), f(5.0, 5.0), 1),
                Triangle::new_static(f(-5.0, -5.0), f(5.0, 5.0), f(-5.0, 5.0), 1),
            ],
            Transform::identity(),
            None,
        );
        world
    }

    /// 不透明部分ではレイが止まり、透明部分は素通りして奥の面に当たる（シャドウレイも同じ経路）。
    #[test]
    fn alpha_mask_transparent_texels_are_passed_through() {
        // マスク: 左 255（不透明）、右 0（透明）。クランプなので 2 テクセルの境界は u=0.5 付近で補間される
        let world = masked_plate_world([255, 0]);
        let down = |x: f64, y: f64| Ray { o: Vec3::new(x, y, 2.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 };
        let opaque = world.hit(down(0.1, 0.5), 0.0, 1e30).unwrap();
        assert_eq!(opaque.mat_id, 0, "不透明部分で板に当たるはず");
        assert!((opaque.t - 2.0).abs() < 1e-9);
        let through = world.hit(down(0.9, 0.5), 0.0, 1e30).unwrap();
        assert_eq!(through.mat_id, 1, "透明部分は素通りして床に当たるはず");
        assert!((through.t - 3.0).abs() < 1e-9);
        // シャドウレイ: 板の手前から奥の点へ。不透明側は遮られ、穴側は抜ける
        let occluded = |x: f64| {
            let r = Ray { o: Vec3::new(x, 0.5, 2.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 };
            world.hit(r, 0.0, 2.9).is_some() // 床（t=3）の手前まで
        };
        assert!(occluded(0.1));
        assert!(!occluded(0.9), "穴を通る光が遮られている");
    }

    /// 透明部分を捨てた後の探索で、同じ板（および裏面側から）を再び拾わない = 自己交差しない。
    /// 板の透明部分の上に原点を置き、板を貫いて上下どちらへ撃っても板自身には当たらない。
    #[test]
    fn alpha_mask_rejected_surface_is_not_hit_again() {
        let world = masked_plate_world([255, 0]);
        for &dz in &[-1.0, 1.0] {
            // 原点は板の面上（透明部分）
            let r = Ray { o: Vec3::new(0.9, 0.5, 0.0), d: Vec3::new(0.0, 0.0, dz), time: 0.0 };
            let h = world.hit(r, 0.0, 1e30);
            match (dz < 0.0, h) {
                (true, Some(h)) => assert_eq!(h.mat_id, 1),
                (false, None) => {}
                (_, h) => panic!("板に自己交差した: {:?}", h.map(|h| (h.mat_id, h.t))),
            }
        }
    }

    /// しきい値ちょうどの扱いは決定的（`alpha >= 0.5` が不透明）。8bit で 127 は透明、128 は不透明。
    #[test]
    fn alpha_mask_threshold_is_inclusive_and_deterministic() {
        for (v, expect_plate) in [(127u8, false), (128u8, true)] {
            let world = masked_plate_world([v, v]);
            let r = Ray { o: Vec3::new(0.5, 0.5, 2.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 };
            let h = world.hit(r, 0.0, 1e30).unwrap();
            assert_eq!(h.mat_id == 0, expect_plate, "mask={}", v);
        }
    }

    // ---- PERF-1: シャドウレイの any-hit 化（`World::occluded`） ----

    /// `World::occluded` は `tmax` を守る（`tmax` を超えて存在する遮蔽物を遮蔽と数えない）。
    /// 1 つのメッシュに、真の tmax 内にある遮蔽物（近い三角形）と、tmax の外にある別の遮蔽物
    /// （遠い三角形、配列の先頭に置いて any-hit の探索順で先に見つかりやすくしてある）を同居させる。
    /// `tmax` を無視して探すと、遠い三角形を先に見つけて「範囲外だから」と探索を打ち切ってしまい、
    /// 本来見つかるはずの近い三角形を取りこぼして「遮蔽なし」と誤判定しうる。
    #[test]
    fn occluded_respects_tmax_even_when_a_farther_triangle_is_probed_first() {
        let far = Triangle::new_static(Vec3::new(-1.0, -1.0, 10.0), Vec3::new(1.0, -1.0, 10.0), Vec3::new(0.0, 1.0, 10.0), 0);
        let near = Triangle::new_static(Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), Vec3::new(0.0, 1.0, 1.0), 0);
        let mut world = World::new();
        // far を先に入れる（1 リーフに収まる三角形数なので、探索順に影響しうる）
        world.add_mesh_instance(vec![far, near], Transform::identity(), None);
        let ray = Ray { o: Vec3::new(0.0, 0.0, 0.0), d: Vec3::new(0.0, 0.0, 1.0), time: 0.0 };
        // tmax=2: near（t=1）は範囲内、far（t=10）は範囲外
        assert!(world.occluded(ray, 0.0, 2.0, None), "範囲内の近い三角形を遮蔽として見つけられるはず");
        // tmax=0.5: どちらも範囲外
        assert!(!world.occluded(ray, 0.0, 0.5, None), "どちらも範囲外なら遮蔽なしのはず");
        // 対照: tmax=20 なら両方範囲内（当然遮蔽）
        assert!(world.occluded(ray, 0.0, 20.0, None));
    }

    /// `World::occluded` の `tmin` 境界（`t_world <= tmin` で自己交差とみなして再探索する規律）と、
    /// 再探索ループ（インスタンスの近い面が帯の中で棄却されたとき、奥の面を諦めずに探す）が、
    /// `World::hit` と同じく効いていることを確認する（PERF-1 追補: check がこの領域の穴を発見）。
    ///
    /// `s=1`（恒等に近い変換）にして、手前の板のワールド距離がちょうど `tmin` になるようレイ原点を
    /// 置く。正しい実装なら「ちょうど `tmin`」は自己交差とみなして棄却・再探索し、奥の板（t≈1+tmin）
    /// を見つける。`tmax` を奥の板より手前に絞れば、手前の板を遮蔽物と誤認しない限り「遮蔽なし」になる
    /// はず — これが `<=`→`<` の境界緩和（手前の板を誤って採用してしまう）を検出する。
    /// `tmax` を奥の板まで届く値にすれば「遮蔽あり」になるはずで、これが再探索ループの欠落
    /// （奥の板を探しにいかない）を検出する。
    #[test]
    fn occluded_treats_exact_tmin_as_self_intersection_and_retries_for_the_far_plate() {
        let tmin = 1e-4;
        let world = two_plates_world(1.0, 1.0);
        let r = Ray { o: Vec3::new(0.3, -0.2, -tmin), d: Vec3::new(0.0, 0.0, 1.0), time: 0.0 };
        // 前提: 手前の板のワールド距離がちょうど tmin であること（境界値のすり替えを検出するための土台）
        let h = world.hit(r, 0.0, 1e30).expect("near plate must be hittable without a tmin floor");
        assert_eq!(h.t.to_bits(), tmin.to_bits(), "test setup: near-plate t must equal tmin exactly");

        // 手前の板だけが範囲内（tmax=0.5 < 奥の板の t≈1）でも、手前の板はちょうど tmin なので
        // 自己交差とみなし、遮蔽物として数えてはいけない
        assert!(
            !world.occluded(r, tmin, 0.5, None),
            "ちょうど tmin の手前の板を遮蔽物として採用してしまった（tmin 境界の緩和を検出）"
        );
        // tmax を奥の板まで伸ばせば、再探索で見つかって遮蔽ありになるはず
        assert!(
            world.occluded(r, tmin, 2.0, None),
            "再探索で奥の板を見つけられなかった（再探索ループの欠落を検出）"
        );
    }

    /// 再探索の上限回数（`INSTANCE_RETRY_LIMIT`）に達しても無限ループせず、諦めて「遮蔽なし」を返す
    /// （`World::hit` の `retry_limit_terminates_on_many_faces_inside_tmin_band` と同じ状況を
    /// `occluded` でも確認する）。
    #[test]
    fn occluded_terminates_when_the_retry_limit_is_reached() {
        let tmin = 1e-4;
        let quad = |z: f64| {
            vec![
                Triangle::new_static(Vec3::new(-1.0, -1.0, z), Vec3::new(1.0, -1.0, z), Vec3::new(1.0, 1.0, z), 0),
                Triangle::new_static(Vec3::new(-1.0, -1.0, z), Vec3::new(1.0, 1.0, z), Vec3::new(-1.0, 1.0, z), 0),
            ]
        };
        let mut world = World::new();
        let mut tris = Vec::new();
        for k in 0..20 {
            tris.extend(quad(k as f64 * 1e-12));
        }
        tris.extend(quad(0.5));
        world.add_mesh_instance(tris, Transform::identity(), None);
        let r = Ray { o: Vec3::new(0.1, 0.1, -tmin * (1.0 - 1e-6)), d: Vec3::new(0.0, 0.0, 1.0), time: 0.0 };
        // 終了すること自体が要件（タイムアウトしないこと）。tmax を極端に大きくしても壊れない
        let _ = world.occluded(r, tmin, 1e30, None);
    }

    /// `World::occluded` は `tmax` の境界でも `World::hit` と同じ規約（`t_world < closest` 相当。
    /// ちょうど `tmax` の面は範囲外）に従う。
    #[test]
    fn occluded_excludes_a_surface_exactly_at_tmax() {
        let world = two_plates_world(1.0, 1.0);
        let r = Ray { o: Vec3::new(0.3, -0.2, -0.5), d: Vec3::new(0.0, 0.0, 1.0), time: 0.0 };
        // 手前の板までの距離はちょうど 0.5
        let h = world.hit(r, 0.0, 1e30).expect("hit");
        assert_eq!(h.t, 0.5);
        assert!(!world.occluded(r, 0.0, 0.5, None), "ちょうど tmax の面は範囲外のはず（World::hit と同じ規約）");
        assert!(world.occluded(r, 0.0, 0.5 + 1e-9, None), "tmax をわずかに超えれば範囲内のはず");
    }

    /// `World::occluded` は「解決済みの `skip` に一致する三角形だけ」を除外し、アルファ不透明な三角形は
    /// `skip` の有無に関わらず遮蔽として扱う（除外条件の取り違えが無いことの確認）。
    #[test]
    fn occluded_skip_and_alpha_combine_correctly() {
        let world = masked_plate_world([255, 0]);
        let opaque_ray = Ray { o: Vec3::new(0.1, 0.5, 2.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 };
        let h = world.hit(opaque_ray, 0.0, 1e30).expect("hit");
        // skip は「指定した三角形番号だけ」を除外する。板は 2 枚の三角形でできているので、
        // 当たっていない方（prim_id が違う方）を指定しても、当たった三角形はまだ遮蔽扱いのはず
        let other_prim = 1 - h.prim_id;
        assert!(
            world.occluded(opaque_ray, 0.0, 2.9, Some((h.inst_id, other_prim))),
            "当たっていない三角形を skip しても不透明側はまだ遮蔽するはず"
        );
        // 当たった三角形そのものを skip すれば、遮蔽されない
        assert!(!world.occluded(opaque_ray, 0.0, 2.9, Some((h.inst_id, h.prim_id))), "skip した三角形自身は遮蔽しないはず");
    }

    /// `World::occluded` は `World::hit(...).is_some()` と常に一致する（遮蔽の有無について。
    /// アルファマスク付きの板で、不透明部分は遮蔽・透明部分は素通しになることを any-hit 経路でも確認する）。
    #[test]
    fn occluded_agrees_with_hit_through_alpha_mask() {
        let world = masked_plate_world([255, 0]);
        let down = |x: f64, y: f64| Ray { o: Vec3::new(x, y, 2.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 };
        // 不透明側: 板で遮蔽される（奥の床 t=3 より手前）
        assert!(world.occluded(down(0.1, 0.5), 0.0, 2.9, None), "不透明部分は遮蔽するはず");
        assert_eq!(world.occluded(down(0.1, 0.5), 0.0, 2.9, None), world.hit(down(0.1, 0.5), 0.0, 2.9).is_some());
        // 透明側: 板は素通しで、奥の床（t=3）より手前の tmax=2.9 では何にも当たらない
        assert!(!world.occluded(down(0.9, 0.5), 0.0, 2.9, None), "透明部分は遮蔽しないはず");
        assert_eq!(world.occluded(down(0.9, 0.5), 0.0, 2.9, None), world.hit(down(0.9, 0.5), 0.0, 2.9).is_some());
        // 透明側でも、奥の床まで届く tmax なら床で遮蔽される
        assert!(world.occluded(down(0.9, 0.5), 0.0, 3.5, None), "奥の床までは届くはず");
    }

    /// 光源自身（三角形光源）への交差は `World::occluded` の `skip` で遮蔽と数えない。
    /// ランダムサンプリングでの再現（元の `shadow_ray_does_not_self_hit_sampled_light` と同じ配置）に加え、
    /// 光源面をわざと突き抜ける tmax を渡して「必ず光源自身に当たる」状況を作り、`skip` の有無で
    /// 結果が変わることを直接確認する（除外を外すミューテーションで確実に落ちる）。
    #[test]
    fn occluded_excludes_the_sampled_light_itself() {
        use crate::transform::Transform;
        let mut world = World::new();
        let m = |x: f64, z: f64| Vec3::new(0.35 * x, 1.98, 0.35 * z);
        let tris = vec![
            Triangle::new_static(m(-1.0, -1.0), m(1.0, -1.0), m(1.0, 1.0), 0),
            Triangle::new_static(m(-1.0, -1.0), m(1.0, 1.0), m(-1.0, 1.0), 0),
        ];
        let inst_id = world.add_mesh_instance(tris, Transform::identity(), None);
        let mats = vec![Material::DiffuseLight { emit: Color::new(4.6, 3.9, 2.0) }];
        world.build_lights(&mats);

        let mut rng = Rng::new(123);
        let p = Vec3::new(0.0, 1.4, -1.0);
        let mut sampled = 0;
        for _ in 0..2000 {
            if let Some(ls) = world.sample_light(&mut rng, 0.0, p) {
                sampled += 1;
                let to = crate::geometry::offset_ray_origin(ls.position, ls.p_error, ls.normal, p - ls.position);
                let seg = to - p;
                let ray = Ray { o: p, d: seg / seg.len(), time: 0.0 };
                assert!(
                    !world.occluded(ray, 0.0, seg.len(), Some((ls.inst_id, ls.prim_id))),
                    "光源自身を遮蔽と数えてしまった"
                );
                assert_eq!(ls.inst_id, Some(inst_id));
            }
        }
        assert!(sampled > 0, "no light samples drawn");

        // 光源面を確実に突き抜ける tmax（真の距離 + 大きめの余白）で、必ず光源自身に当たる状況を作る
        let ls = world.sample_light(&mut rng, 0.0, p).expect("light sample");
        let to_light = ls.position - p;
        let dist = to_light.len();
        let ray = Ray { o: p, d: to_light / dist, time: 0.0 };
        assert!(!world.occluded(ray, 0.0, dist + 0.5, Some((ls.inst_id, ls.prim_id))), "skip 付きなら光源自身は遮蔽ではない");
        assert!(world.occluded(ray, 0.0, dist + 0.5, None), "skip 無しなら光源自身への交差も遮蔽扱いのはず（比較用）");
    }

    /// 光源自身への交差の除外は球光源でも効く（`inst_id = None`）。`default.xml` 実測（S3）で、
    /// この除外を外すと画像が約 6% 暗くなったのと同じ配置（床の真上の球光源）で確認する。
    /// ランダムサンプリングに加え、球面を確実に突き抜ける tmax で「必ず光源自身に当たる」状況も直接確認する
    /// （除外を外すミューテーションで確実に落ちる）。
    #[test]
    fn occluded_excludes_the_sampled_sphere_light_itself() {
        let mut world = World::new();
        let light_idx = 0usize;
        world.spheres.push(Sphere { c: Vec3::new(0.0, 3.0, 0.0), r: 1.0, mat_id: 0 });
        // 床（大きな四角形、光源の真下）
        let f = |x: f64, z: f64| Vec3::new(x, 0.0, z);
        world.add_mesh_instance(
            vec![
                Triangle::new_static(f(-5.0, -5.0), f(5.0, -5.0), f(5.0, 5.0), 1),
                Triangle::new_static(f(-5.0, -5.0), f(5.0, 5.0), f(-5.0, 5.0), 1),
            ],
            Transform::identity(),
            None,
        );
        let mats = vec![Material::DiffuseLight { emit: Color::new(4.0, 4.0, 4.0) }, Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5) }];
        world.build_lights(&mats);

        let mut rng = Rng::new(9);
        let p = Vec3::new(0.0, 0.0, 0.0);
        let mut sampled = 0;
        let mut last_ls = None;
        for _ in 0..2000 {
            if let Some(ls) = world.sample_light(&mut rng, 0.0, p) {
                if !ls.visible {
                    continue;
                }
                sampled += 1;
                assert_eq!(ls.inst_id, None);
                assert_eq!(ls.prim_id, light_idx);
                let to = crate::geometry::offset_ray_origin(ls.position, ls.p_error, ls.normal, p - ls.position);
                let seg = to - p;
                let ray = Ray { o: p, d: seg / seg.len(), time: 0.0 };
                assert!(
                    !world.occluded(ray, 0.0, seg.len(), Some((ls.inst_id, ls.prim_id))),
                    "球光源自身を遮蔽と数えてしまった"
                );
                last_ls = Some(ls);
            }
        }
        assert!(sampled > 0, "no light samples drawn");

        // 球面を確実に突き抜ける tmax で、必ず光源自身に当たる状況を作る
        let ls = last_ls.expect("at least one visible sample");
        let to_light = ls.position - p;
        let dist = to_light.len();
        let ray = Ray { o: p, d: to_light / dist, time: 0.0 };
        assert!(!world.occluded(ray, 0.0, dist + 0.5, Some((ls.inst_id, ls.prim_id))), "skip 付きなら球光源自身は遮蔽ではない");
        assert!(world.occluded(ray, 0.0, dist + 0.5, None), "skip 無しなら球光源自身への交差も遮蔽扱いのはず（比較用）");
    }

    fn emissive_sphere_world(c: Vec3, r: f64) -> World {
        let mut world = World::new();
        let mats = vec![Material::DiffuseLight { emit: Color::new(3.0, 4.0, 5.0) }];
        world.spheres.push(Sphere { c, r, mat_id: 0 });
        world.build_lights(&mats);
        world
    }

    /// 光源が多数・混在（球 5 個 + 向きの違う矩形 2 枚）のワールド。矩形の法線は +z（cross(v1−v0, v2−v0)）。
    fn many_lights_world() -> World {
        use crate::world::test_meshes::tilted_quad;
        let mut world = World::new();
        let mats = vec![Material::DiffuseLight { emit: Color::new(3.0, 4.0, 5.0) }];
        for (c, r) in [
            (Vec3::new(-6.0, 1.0, 2.0), 0.4),
            (Vec3::new(4.0, 0.5, -3.0), 1.5),
            (Vec3::new(0.0, 5.0, 0.0), 0.05),
            (Vec3::new(20.0, 2.0, 20.0), 3.0),
            (Vec3::new(0.0, 0.0, 0.0), 0.7),
        ] {
            world.add_sphere(Sphere { c, r, mat_id: 0 });
        }
        // +z を向く矩形と、y 軸まわりに 180° 回して −z を向く矩形
        world.add_mesh_data_instance(tilted_quad(1.0, Vec3::new(0.0, 0.0, 1.0), 0), Transform::translate(Vec3::new(2.0, 1.0, 6.0)), None);
        world.add_mesh_data_instance(
            tilted_quad(2.0, Vec3::new(0.0, 0.0, 1.0), 0),
            Transform::translate(Vec3::new(-3.0, 1.0, -6.0)).compose(Transform::rotate(Vec3::new(0.0, 1.0, 0.0), 180.0)),
            None,
        );
        world.build_lights(&mats);
        assert!(world.light_select != LightSelect::Power, "テスト設定: 光源のまとまりが複数なので重み付けになるはず");
        // 既定は線形（7 個）。光源 BVH の経路もこのワールドで通す
        world.set_light_select(LightSelect::Bvh);
        world
    }

    /// **MIS の前提**: `sample_light` が返した `pdf` は、同じ光源・同じ点に対する `light_pdf` とビット単位で一致する
    /// （選択確率が参照点に依存しても）。光源 1 個 / 多数 / 裏側 / 距離 0 近傍 / 重み 0 の光源を含む。
    #[test]
    fn light_pdf_is_bit_identical_to_the_sampled_pdf_with_position_dependent_selection() {
        let mut world = many_lights_world();
        let mut rng = Rng::new(77);
        // 既定（線形）と光源 BVH の両方でビット一致を確かめる
        for mode in [LightSelect::Linear, LightSelect::Bvh] {
        world.set_light_select(mode);
        let mut points = vec![
            Vec3::new(0.0, 0.0, 0.0),           // 球 [4] の中心（距離 0）
            Vec3::new(0.0, 5.0, 1e-9),          // 小さな球のすぐそば
            Vec3::new(2.0, 1.0, 6.0 + 1e-9),    // +z 矩形の面上すれすれ（表側）
            Vec3::new(2.0, 1.0, 6.0 - 1e-3),    // 同・裏側（この矩形の重みは 0）
            Vec3::new(-3.0, 1.0, -6.0 - 0.5),   // −z を向く矩形の裏側（−z 側 = 表側なので実は表）
            Vec3::new(-3.0, 1.0, -5.0),         // 同・裏側
            Vec3::new(300.0, 100.0, -200.0),    // 遠方
        ];
        for _ in 0..40 {
            points.push(Vec3::new(rng.next_f64() * 40.0 - 20.0, rng.next_f64() * 12.0 - 3.0, rng.next_f64() * 40.0 - 20.0));
        }
        let mut checked = 0;
        for from in points {
            for _ in 0..200 {
                let Some(ls) = world.sample_light(&mut rng, 0.0, from) else { continue };
                let hit = Hit {
                    t: 0.0, p: ls.position, ng: ls.normal, ns: ls.normal, mat_id: 0,
                    prim_id: ls.prim_id, inst_id: ls.inst_id, p_error: Vec3::new(0.0, 0.0, 0.0), bary: (0.0, 0.0), uv: (0.0, 0.0),
                };
                let pdf = world.light_pdf(from, 0.0, &hit);
                assert_eq!(pdf.to_bits(), ls.pdf.to_bits(), "from {:?}: light_pdf {} != sample pdf {}", from, pdf, ls.pdf);
                checked += 1;
            }
        }
        assert!(checked > 3000, "{mode:?}: too few samples: {checked}");
        }
    }

    /// 光源が 64 個を超える（重みを配列に持てず 2 回に分けて計算する経路）ワールドでも、`light_pdf` とビット一致する。
    #[test]
    fn light_pdf_is_bit_identical_when_there_are_more_lights_than_the_weight_cache() {
        let mut world = World::new();
        let mats = vec![Material::DiffuseLight { emit: Color::new(2.0, 2.0, 2.0) }];
        let mut rng = Rng::new(3);
        for _ in 0..150 {
            let c = Vec3::new(rng.next_f64() * 30.0 - 15.0, rng.next_f64() * 5.0, rng.next_f64() * 30.0 - 15.0);
            world.add_sphere(Sphere { c, r: 0.1 + rng.next_f64() * 0.4, mat_id: 0 });
        }
        world.build_lights(&mats);
        // 線形の重み付け（64 個超で重みを配列に持てない経路）を強制して通す
        world.force_light_weighting(true);
        assert!(world.lights.len() > 64);
        let mut checked = 0;
        for _ in 0..300 {
            let from = Vec3::new(rng.next_f64() * 30.0 - 15.0, rng.next_f64() * 6.0 - 1.0, rng.next_f64() * 30.0 - 15.0);
            let Some(ls) = world.sample_light(&mut rng, 0.0, from) else { continue };
            let hit = Hit { t: 0.0, p: ls.position, ng: ls.normal, ns: ls.normal, mat_id: 0, prim_id: ls.prim_id, inst_id: ls.inst_id, p_error: Vec3::new(0.0, 0.0, 0.0), bary: (0.0, 0.0), uv: (0.0, 0.0) };
            assert_eq!(world.light_pdf(from, 0.0, &hit).to_bits(), ls.pdf.to_bits());
            checked += 1;
        }
        assert!(checked > 200);
    }

    /// 選択確率は参照点ごとに全光源で足して 1（線形の重み付けも光源 BVH も。木が正しい直接の証拠）、
    /// 裏側の片面発光は選ばれず（確率 0）、近い光源ほど選ばれやすい。
    #[test]
    fn selection_probabilities_sum_to_one_and_respect_orientation_and_distance() {
        let mut world = many_lights_world();
        for mode in [LightSelect::Linear, LightSelect::Bvh] {
            world.set_light_select(mode);
            let mut rng = Rng::new(5);
            let mut full = 0;
            for _ in 0..300 {
                let from = Vec3::new(rng.next_f64() * 40.0 - 20.0, rng.next_f64() * 12.0 - 3.0, rng.next_f64() * 40.0 - 20.0);
                let sum: f64 = (0..world.lights.len()).map(|i| world.light_selection_prob(i, from)).sum();
                assert!(sum <= 1.0 + 1e-12, "{mode:?}: sum {sum}");
                if (sum - 1.0).abs() < 1e-12 {
                    full += 1;
                } else {
                    assert!(sum == 0.0 || mode == LightSelect::Bvh, "{mode:?}: sum {sum}");
                }
            }
            // 光源 BVH は向きの円錐が保守的なので、裏側の矩形だけを含むノードを引いて「行き止まり」になる分だけ総和が 1 に満たない点がある
            // （その確率は無駄になるだけで、選ばれる光源の確率は変わらない = 不偏）。球だけの構成は下のテストで厳密に 1
            assert!(full > 240, "{mode:?}: only {full} of 300 points sum to 1");
            // +z を向く矩形（インスタンス 0）の裏側からは、その 2 枚の三角形の選択確率が 0
            let behind = Vec3::new(2.0, 1.0, 3.0);
            for (i, info) in world.lights.iter().enumerate() {
                if let Light::Triangle { inst_id: 0, .. } = info.light {
                    assert_eq!(world.light_selection_prob(i, behind), 0.0, "{mode:?}");
                }
            }
            // 小さな球 [2]（0,5,0）は、すぐそばのほうが遠くからより選ばれやすい（出力パワーは同じ）
            let i_small = world.sphere_light_id[2].unwrap();
            assert!(world.light_selection_prob(i_small, Vec3::new(0.0, 5.3, 0.0)) > 5.0 * world.light_selection_prob(i_small, Vec3::new(0.0, 5.0, 12.0)), "{mode:?}");
        }
    }

    /// 光源が球だけ（向きの円錐が全方向で行き止まりが無い）の 2 / 33 / 150 / 500 個で、光源 BVH の選択確率の総和が
    /// **すべての参照点で 1**（木が正しい直接の証拠）、かつ `sample_light` の pdf と `light_pdf` がビット一致する。
    #[test]
    fn light_bvh_probabilities_sum_to_exactly_one_for_sphere_lights() {
        for n in [2usize, 33, 150, 500] {
            let mut world = World::new();
            let mats = vec![Material::DiffuseLight { emit: Color::new(2.0, 1.0, 3.0) }];
            let mut rng = Rng::new(n as u64);
            for _ in 0..n {
                let c = Vec3::new(rng.next_f64() * 40.0 - 20.0, rng.next_f64() * 6.0, rng.next_f64() * 40.0 - 20.0);
                world.add_sphere(Sphere { c, r: 0.05 + rng.next_f64() * 0.6, mat_id: 0 });
            }
            world.build_lights(&mats);
            world.set_light_select(LightSelect::Bvh);
            for _ in 0..200 {
                // 光源の中・ごく近く・遠方を含む
                let from = match (rng.next_f64() * 4.0) as usize {
                    0 => world.spheres[(rng.next_f64() * n as f64) as usize % n].c + Vec3::new(1e-9, 0.0, 0.0),
                    1 => Vec3::new(rng.next_f64() * 1e4, rng.next_f64() * 1e4, rng.next_f64() * 1e4),
                    _ => Vec3::new(rng.next_f64() * 40.0 - 20.0, rng.next_f64() * 8.0 - 1.0, rng.next_f64() * 40.0 - 20.0),
                };
                let sum: f64 = (0..n).map(|i| world.light_selection_prob(i, from)).sum();
                assert!((sum - 1.0).abs() < 1e-12, "n={n} from {from:?}: sum {sum}");
                if let Some(ls) = world.sample_light(&mut rng, 0.0, from) {
                    let hit = Hit { t: 0.0, p: ls.position, ng: ls.normal, ns: ls.normal, mat_id: 0, prim_id: ls.prim_id, inst_id: ls.inst_id, p_error: Vec3::new(0.0, 0.0, 0.0), bary: (0.0, 0.0), uv: (0.0, 0.0) };
                    assert_eq!(world.light_pdf(from, 0.0, &hit).to_bits(), ls.pdf.to_bits(), "n={n}");
                }
            }
        }
    }

    /// 枝刈りなし（インスタンス BVH に tmax = 1e30）の参照実装。tmin の写像と再探索の規則は World::hit と同じ。
    /// 枝刈り距離の誤りをビット単位で検出するために使う。
    fn hit_without_instance_pruning(world: &World, r: Ray, tmin: f64, tmax: f64) -> Option<Hit> {
        let mut closest = tmax;
        let mut best: Option<Hit> = None;
        for (inst_id, inst) in world.instances.iter().enumerate() {
            let mesh = &world.meshes[inst.mesh_id];
            let (o_obj, o_err) = inst.xform.apply_point_inv_with_error_linf(r.o);
            let d_raw = inst.xform.apply_vec_inv(r.d);
            let d_len = d_raw.len().max(1e-30);
            let d_obj = d_raw / d_len;
            let r_obj = Ray { o: o_obj + d_obj * (d_obj.l1() * o_err), d: d_obj, time: r.time };
            // World::hit と同じ tmin の写像・再探索の規則で、tmax だけ無制限（枝刈りなし）
            let mut search_from = tmin * d_len * (1.0 - 1e-9);
            for _ in 0..=INSTANCE_RETRY_LIMIT {
                let Some(h_obj) = mesh.hit(r_obj, search_from, 1e30) else { break };
                let (p_world, p_error) = inst.xform.apply_point_with_error(h_obj.p, h_obj.p_error);
                let t_world = (p_world - r.o).dot(r.d);
                if t_world <= tmin {
                    search_from = h_obj.t.next_up();
                    continue;
                }
                if t_world < closest {
                    closest = t_world;
                    let ng = inst.xform.apply_normal(h_obj.ng);
                    let ns = if h_obj.is_smooth() {
                        face_forward(inst.xform.apply_normal(h_obj.ns), ng)
                    } else {
                        ng
                    };
                    best = Some(Hit {
                        t: t_world,
                        p: p_world,
                        ng,
                        ns,
                        mat_id: inst.mat_override.unwrap_or(h_obj.mat_id),
                        prim_id: h_obj.prim_id,
                        inst_id: Some(inst_id),
                        p_error,
                        bary: h_obj.bary,
                        uv: h_obj.uv,
                    });
                }
                break;
            }
        }
        for (idx, s) in world.spheres.iter().enumerate() {
            if let Some(mut h) = s.hit(r, tmin, closest) {
                h.prim_id = idx;
                closest = h.t;
                best = Some(h);
            }
        }
        best
    }

    /// 2 つの Hit がビット単位で等しいか（参照実装との比較用）。
    fn assert_same_hit(a: Option<Hit>, b: Option<Hit>, what: &str) {
        match (a, b) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                let bits = |v: Vec3| (v.x.to_bits(), v.y.to_bits(), v.z.to_bits());
                assert_eq!(a.t.to_bits(), b.t.to_bits(), "{}: t {} vs {}", what, a.t, b.t);
                assert_eq!(bits(a.p), bits(b.p), "{}: p", what);
                assert_eq!(bits(a.ng), bits(b.ng), "{}: ng", what);
                assert_eq!(bits(a.ns), bits(b.ns), "{}: ns", what);
                assert_eq!((a.mat_id, a.prim_id, a.inst_id), (b.mat_id, b.prim_id, b.inst_id), "{}: ids", what);
            }
            (a, b) => panic!("{}: hit mismatch {:?} vs {:?}", what, a.map(|h| (h.t, h.inst_id)), b.map(|h| (h.t, h.inst_id))),
        }
    }

    /// ワールド空間の総当たり参照: 全インスタンスの全三角形をワールド座標に変換して直接交差判定する
    /// （インスタンス変換・物体空間の tmin/tmax の写像に依存しない独立な実装）。球も含む。
    fn hit_world_brute_force(world: &World, r: Ray, tmin: f64, tmax: f64) -> Option<(f64, Option<usize>, usize)> {
        let mut closest = tmax;
        let mut best = None;
        for (inst_id, inst) in world.instances.iter().enumerate() {
            for (tri_id, t) in world.meshes[inst.mesh_id].tris.iter().enumerate() {
                let w = Triangle::new_static(inst.xform.apply_point(t.v0_0), inst.xform.apply_point(t.v1_0), inst.xform.apply_point(t.v2_0), t.mat_id);
                if let Some(h) = w.hit(r, tmin, closest) {
                    closest = h.t;
                    best = Some((h.t, Some(inst_id), tri_id));
                }
            }
        }
        for (idx, s) in world.spheres.iter().enumerate() {
            if let Some(h) = s.hit(r, tmin, closest) {
                closest = h.t;
                best = Some((h.t, None, idx));
            }
        }
        best
    }

    /// World::hit とワールド空間総当たりが一致するか（計算経路が違うので t は相対 1e-7 で比較）。
    /// 片方だけがヒットする場合は、そのヒットが tmin / tmax の境界（相対 1e-6 以内）にあるときだけ許す。
    /// ID が違う場合は、t が一致する重なり面（同一平面・一致インスタンス）なら許す。
    fn check_against_brute_force(world: &World, r: Ray, tmin: f64, tmax: f64, what: &str) {
        let a = world.hit(r, tmin, tmax);
        let b = hit_world_brute_force(world, r, tmin, tmax);
        let near_bound = |t: f64| (t - tmin).abs() <= 1e-6 * tmin.max(t) || (t - tmax).abs() <= 1e-6 * tmax.max(t);
        match (a, b) {
            (None, None) => {}
            (Some(h), None) => assert!(near_bound(h.t), "{}: World::hit found t={} (inst {:?}) but brute force found nothing", what, h.t, h.inst_id),
            (None, Some((t, inst, _))) => assert!(near_bound(t), "{}: brute force found t={} (inst {:?}) but World::hit found nothing", what, t, inst),
            (Some(h), Some((t, inst, prim))) => {
                assert!((h.t - t).abs() <= 1e-7 * t.max(1e-3), "{}: t {} vs brute force {} (inst {:?} vs {:?})", what, h.t, t, h.inst_id, inst);
                let _ = prim;
            }
        }
    }

    /// インスタンス BVH を最近接距離で枝刈りしても、結果（t・点・法線・ID）はビット単位で不変で、
    /// かつワールド空間の総当たりと一致する（tmin の写像・再探索の正しさ）。
    ///
    /// 枝刈り距離の誤り（相対マージンの撤去・`|A⁻¹d|` の掛け忘れ・`|A⁻¹d|` で割る）を
    /// 検出できるよう、次を含める:
    /// - 先に走査される拡大インスタンス（×100）の手前に、後から走査される縮小インスタンス（×0.01）
    /// - 完全一致・ほぼ一致（相対 1e-8 / 1e-12 のスケール差）のインスタンス対
    /// - 一次レイに加え、ヒット点からの二次レイと有限 tmax のシャドウレイ
    #[test]
    fn instance_pruning_preserves_hits() {
        // 板 20 枚（z = 0, 0.1, …, 1.9）を重ねたメッシュ。1 インスタンス内でも奥の板が枝刈り対象になる
        let plates = || -> Vec<Triangle> {
            (0..20)
                .flat_map(|k| {
                    let z = k as f64 * 0.1;
                    vec![
                        Triangle::new_static(Vec3::new(-1.0, -1.0, z), Vec3::new(1.0, -1.0, z), Vec3::new(1.0, 1.0, z), 0),
                        Triangle::new_static(Vec3::new(-1.0, -1.0, z), Vec3::new(1.0, 1.0, z), Vec3::new(-1.0, 1.0, z), 0),
                    ]
                })
                .collect()
        };
        let mut rng = Rng::new(77);
        let rnd = |rng: &mut Rng, s: f64| Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, rng.next_f64() - 0.5) * s;
        let mut world = World::new();
        let mut xforms: Vec<Transform> = Vec::new();
        let add = |world: &mut World, xforms: &mut Vec<Transform>, xf: Transform| {
            let id = world.add_mesh_instance(plates(), xf, Some(xforms.len()));
            xforms.push(xf);
            id
        };

        // 1. 拡大（×100、非一様）を先に。原点付近を奥行き方向に大きく覆う
        for k in 0..3 {
            let xf = Transform::translate(Vec3::new(0.0, 0.0, -150.0 + k as f64 * 7.0))
                .compose(Transform::rotate(Vec3::new(0.2, 1.0, 0.1), 11.0 * k as f64))
                .compose(Transform::scale(Vec3::new(100.0, 80.0, 100.0)));
            add(&mut world, &mut xforms, xf);
        }
        // 2. 縮小（×0.01）を後に、拡大インスタンスの手前に多数
        for _ in 0..40 {
            let xf = Transform::translate(rnd(&mut rng, 6.0))
                .compose(Transform::rotate(rnd(&mut rng, 2.0) + Vec3::new(0.0, 1.0, 0.0), rng.next_f64() * 360.0))
                .compose(Transform::scale(Vec3::new(0.01, 0.013, 0.008) * (1.0 + 30.0 * rng.next_f64())));
            add(&mut world, &mut xforms, xf);
        }
        // 3. 完全一致・ほぼ一致の対（同じメッシュ・ほぼ同じ変換）
        for &(s, eps) in &[(0.8, 0.0), (0.8, 1e-8), (1.0, 1e-12), (0.01, 1e-8), (100.0, 1e-12), (3.0, 1e-8)] {
            let base = Transform::translate(rnd(&mut rng, 4.0))
                .compose(Transform::rotate(rnd(&mut rng, 2.0) + Vec3::new(1.0, 0.0, 0.0), rng.next_f64() * 360.0));
            add(&mut world, &mut xforms, base.compose(Transform::scale(Vec3::new(s, s, s))));
            let s2 = s * (1.0 + eps);
            add(&mut world, &mut xforms, base.compose(Transform::scale(Vec3::new(s2, s2, s2))));
        }
        // 4. 一般的なスケールと回転の混在
        for i in 0..20 {
            let s = [0.01, 0.3, 1.0, 3.0, 100.0][i % 5];
            let xf = Transform::translate(rnd(&mut rng, 8.0))
                .compose(Transform::rotate(rnd(&mut rng, 2.0) + Vec3::new(0.0, 0.0, 1.0), 37.0 * i as f64))
                .compose(Transform::scale(Vec3::new(s, s * 0.7, s * 1.3)));
            add(&mut world, &mut xforms, xf);
        }
        world.spheres.push(Sphere { c: Vec3::new(0.0, 0.0, 0.0), r: 0.5, mat_id: 999 });

        let n_inst = xforms.len();
        let (mut primary_hits, mut queries) = (0usize, 0usize);
        for _ in 0..20_000 {
            // インスタンス内の点を狙った一次レイ（ヒットが多くなるように）
            let target_inst = (rng.next_f64() * n_inst as f64) as usize % n_inst;
            let local = Vec3::new(rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 1.9);
            let target = xforms[target_inst].apply_point(local);
            let o = target + uniform_sphere_dir(&mut rng) * (0.05 + 30.0 * rng.next_f64());
            let r = Ray { o, d: (target - o).norm(), time: 0.0 };
            let a = world.hit(r, 1e-4, 1e30);
            assert_same_hit(a, hit_without_instance_pruning(&world, r, 1e-4, 1e30), "primary");
            // 総当たりは重いので最初の 4000 本だけ（二次レイ・シャドウレイも同様）
            let brute = queries < 12_000;
            if brute {
                check_against_brute_force(&world, r, 1e-4, 1e30, "primary/brute");
            }
            queries += 1;
            let Some(h) = a else { continue };
            primary_hits += 1;

            // 二次レイ（ヒット点から任意方向）
            let d2 = uniform_sphere_dir(&mut rng);
            let r2 = Ray { o: h.p + 1e-4 * d2, d: d2, time: 0.0 };
            assert_same_hit(world.hit(r2, 1e-4, 1e30), hit_without_instance_pruning(&world, r2, 1e-4, 1e30), "secondary");
            if brute {
                check_against_brute_force(&world, r2, 1e-4, 1e30, "secondary/brute");
            }

            // シャドウレイ（有限 tmax。別インスタンス内の点へ）
            let other = (rng.next_f64() * n_inst as f64) as usize % n_inst;
            let lp = xforms[other].apply_point(Vec3::new(rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 2.0 - 1.0, 1.9 * rng.next_f64()));
            let to = lp - h.p;
            let dist = to.len();
            if dist > 1e-3 {
                let d3 = to / dist;
                let r3 = Ray { o: h.p + 1e-4 * d3, d: d3, time: 0.0 };
                let tmax = (dist - 2e-4).max(1e-4);
                assert_same_hit(world.hit(r3, 1e-4, tmax), hit_without_instance_pruning(&world, r3, 1e-4, tmax), "shadow");
                if brute {
                    check_against_brute_force(&world, r3, 1e-4, tmax, "shadow/brute");
                }
            }
            queries += 2;
        }
        assert!(primary_hits > 10_000, "too few hits to be meaningful: {} of {}", primary_hits, queries);
    }

    /// z = 0 と z = `gap` の 2 枚の板（xy は [-1, 1]）を `s` 倍に拡大縮小したインスタンスだけのワールド。
    fn two_plates_world(gap: f64, s: f64) -> World {
        let quad = |z: f64| {
            vec![
                Triangle::new_static(Vec3::new(-1.0, -1.0, z), Vec3::new(1.0, -1.0, z), Vec3::new(1.0, 1.0, z), 0),
                Triangle::new_static(Vec3::new(-1.0, -1.0, z), Vec3::new(1.0, 1.0, z), Vec3::new(-1.0, 1.0, z), 0),
            ]
        };
        let mut world = World::new();
        let tris: Vec<Triangle> = quad(0.0).into_iter().chain(quad(gap)).collect();
        world.add_mesh_instance(tris, Transform::scale(Vec3::new(s, s, s)), None);
        world
    }

    /// ケース A（verify_batch1 の tminprobe）: ×100 の拡大インスタンスで、ワールド距離 0.005（> tmin）にある
    /// 手前の板に当たる。旧実装は物体空間の t = 5e-5 < tmin で手前の板を取りこぼし、奥の板（t ≈ 100）に当たっていた。
    #[test]
    fn scaled_up_instance_keeps_near_surface_beyond_tmin() {
        let world = two_plates_world(1.0, 100.0);
        let r = Ray { o: Vec3::new(0.3, -0.2, -0.005), d: Vec3::new(0.0, 0.0, 1.0), time: 0.0 };
        let h = world.hit(r, 1e-4, 1e30).expect("hit");
        assert!((h.t - 0.005).abs() < 1e-9, "t = {}", h.t);
    }

    /// ケース B（verify_batch1 の tminprobe）: ×0.01 の縮小インスタンス（板はワールド z = 0 と z = 1）で、
    /// 始点が手前の板の 5e-6（< tmin）手前。手前の板は自己交差回避の帯の中なので無視し、奥の板（t ≈ 1）に
    /// 当たる。旧実装は物体空間で手前の板を返し、ワールド判定で却下してインスタンスごと None になっていた。
    #[test]
    fn scaled_down_instance_skips_surface_inside_tmin_and_finds_far_one() {
        let world = two_plates_world(100.0, 0.01);
        let r = Ray { o: Vec3::new(0.001, 0.002, -5e-6), d: Vec3::new(0.0, 0.0, 1.0), time: 0.0 };
        let h = world.hit(r, 1e-4, 1e30).expect("the far plate must be found");
        assert!((h.t - (1.0 + 5e-6)).abs() < 1e-9, "t = {}", h.t);
    }

    /// ケース C: 丸めで帯の境界に落ちる面。手前の板がワールド距離 tmin·(1 − 5e-10) にあると、物体空間では
    /// 広げた下限 tmin_obj を超えるので BVH が返すが、ワールド判定では t <= tmin で却下される。この場合も
    /// 再探索で同じインスタンスの奥の板が見つかる（再探索がないと None になる）。
    #[test]
    fn rejected_near_surface_retries_same_instance() {
        let tmin = 1e-4;
        let world = two_plates_world(100.0, 0.01);
        let r = Ray { o: Vec3::new(0.001, 0.002, -tmin * (1.0 - 5e-10)), d: Vec3::new(0.0, 0.0, 1.0), time: 0.0 };
        // 前提の確認: 物体空間では手前の板が下限を超える
        let inst = &world.instances[0];
        let d_len = inst.xform.apply_vec_inv(r.d).len();
        let t_obj_near = (0.0 - inst.xform.apply_point_inv(r.o).z) / (inst.xform.apply_vec_inv(r.d).z / d_len);
        assert!(t_obj_near > tmin * d_len * (1.0 - 1e-9), "test setup: the near plate must pass the object-space lower bound");
        let h = world.hit(r, tmin, 1e30).expect("the far plate must be found after retrying");
        assert!((h.t - (1.0 + tmin * (1.0 - 5e-10))).abs() < 1e-9, "t = {}", h.t);
    }

    /// 同一平面の面が自己交差回避の帯の中に多数重なっていても、再探索は上限回数で止まる（無限ループしない）。
    /// その場合、帯の先にある面は諦めて None を返しうる（退化した入力に対する割り切り）。
    #[test]
    fn retry_limit_terminates_on_many_faces_inside_tmin_band() {
        let tmin = 1e-4;
        let quad = |z: f64| {
            vec![
                Triangle::new_static(Vec3::new(-1.0, -1.0, z), Vec3::new(1.0, -1.0, z), Vec3::new(1.0, 1.0, z), 0),
                Triangle::new_static(Vec3::new(-1.0, -1.0, z), Vec3::new(1.0, 1.0, z), Vec3::new(-1.0, 1.0, z), 0),
            ]
        };
        let mut world = World::new();
        // 帯の中（ワールド距離 tmin の直前）に、わずかに z がずれた 20 枚。奥に 1 枚。
        let mut tris = Vec::new();
        for k in 0..20 {
            tris.extend(quad(k as f64 * 1e-12));
        }
        tris.extend(quad(0.5));
        world.add_mesh_instance(tris, Transform::identity(), None);
        let r = Ray { o: Vec3::new(0.1, 0.1, -tmin * (1.0 - 1e-6)), d: Vec3::new(0.0, 0.0, 1.0), time: 0.0 };
        // 終了すること自体が要件。結果は奥の板か None のどちらか
        match world.hit(r, tmin, 1e30) {
            None => {}
            Some(h) => assert!((h.t - (0.5 + tmin * (1.0 - 1e-6))).abs() < 1e-9, "t = {}", h.t),
        }
    }

    /// 12 枚の三角形で作る [-1, 1]³ の立方体。
    fn unit_box() -> Vec<Triangle> {
        let c = |x: f64, y: f64, z: f64| Vec3::new(x, y, z);
        let faces = [
            [c(-1., -1., -1.), c(1., -1., -1.), c(1., 1., -1.), c(-1., 1., -1.)],
            [c(-1., -1., 1.), c(1., -1., 1.), c(1., 1., 1.), c(-1., 1., 1.)],
            [c(-1., -1., -1.), c(1., -1., -1.), c(1., -1., 1.), c(-1., -1., 1.)],
            [c(-1., 1., -1.), c(1., 1., -1.), c(1., 1., 1.), c(-1., 1., 1.)],
            [c(-1., -1., -1.), c(-1., 1., -1.), c(-1., 1., 1.), c(-1., -1., 1.)],
            [c(1., -1., -1.), c(1., 1., -1.), c(1., 1., 1.), c(1., -1., 1.)],
        ];
        faces.iter().flat_map(|q| [Triangle::new_static(q[0], q[1], q[2], 0), Triangle::new_static(q[0], q[2], q[3], 0)]).collect()
    }

    /// 二次レイの自己再ヒットなし: 球と、回転・非一様スケールした立方体インスタンスを、大きさ 1e-3〜1e3、
    /// 原点からの距離 0・1e3 倍・**絶対 1e8** に置き、表面の点から `offset_ray_origin`（誤差上界ぶん法線方向に
    /// ずらす）で二次レイを出す（tmin = 0）。外向き（幾何法線側）のレイは凸な自分自身に当たらず、内向きのレイ
    /// （透過）は入射した面に再ヒットせず（t が交差点の誤差上界より十分大きい）物体の反対側へ抜ける。
    #[test]
    fn secondary_rays_do_not_rehit_their_own_surface_at_any_scale() {
        use crate::geometry::offset_ray_origin;
        let mut rng = Rng::new(31);
        for &k in &[1e-3, 1.0, 1e3] {
            for &dist in &[0.0, 1e3 * k, 1e8] {
                let center = Vec3::new(dist, 0.5 * dist, -0.3 * dist);
                let mut world = World::new();
                world.add_sphere(Sphere { c: center, r: k, mat_id: 0 });
                let box_center = center + Vec3::new(4.0 * k, 0.0, 0.0);
                let xf = Transform::translate(box_center)
                    .compose(Transform::rotate(Vec3::new(0.3, 1.0, 0.2), 33.0))
                    .compose(Transform::scale(Vec3::new(k, 0.7 * k, 1.3 * k)));
                world.add_mesh_instance(unit_box(), xf, None);
                let mut checked = 0;
                for i in 0..2000 {
                    let target = if i % 2 == 0 { center } else { box_center };
                    let o = target + uniform_sphere_dir(&mut rng) * (5.0 * k);
                    let Some(h) = world.hit(Ray { o, d: (target - o).norm(), time: 0.0 }, 0.0, 1e30) else { continue };
                    // 三角形の法線の向き（巻き順）は保証されないので、実際に当たった物体の中心から外向きにそろえる
                    let hit_center = if h.inst_id.is_some() { box_center } else { center };
                    let n = if h.ng.dot(h.p - hit_center) < 0.0 { -h.ng.norm() } else { h.ng.norm() };
                    for _ in 0..4 {
                        let mut d = uniform_sphere_dir(&mut rng);
                        if d.dot(n) < 0.0 { d = -d; }
                        // 外向き
                        if let Some(h2) = world.hit(Ray { o: offset_ray_origin(h.p, h.p_error, h.ng, d), d, time: 0.0 }, 0.0, 1e30) {
                            assert!(!(h2.inst_id == h.inst_id && (h.inst_id.is_some() || h2.prim_id == h.prim_id)),
                                "k={} dist={}: outward ray re-hit its own object at t={} (p_error {:?})", k, dist, h2.t, h.p_error.max_abs());
                        }
                        // 内向き（浅すぎる角度は除く）
                        let di = -d;
                        if di.dot(-n) > 0.1 {
                            let h2 = world.hit(Ray { o: offset_ray_origin(h.p, h.p_error, h.ng, di), d: di, time: 0.0 }, 0.0, 1e30)
                                .unwrap_or_else(|| panic!("k={} dist={}: inward ray escaped its object", k, dist));
                            // 入射面への再ヒットかどうか: 立方体なら新しい交点の面が入射面と平行で、入射面の上（法線方向の
                            // 変位が誤差上界程度）に残る。球なら新しい交点が入射点とほぼ一致する。立方体の辺の近くでは
                            // 隣の面から抜ける正当な短い弦（原点から遠い配置では誤差上界の数倍程度）もあるので、
                            // 弦の長さでは判定しない
                            let err = h.p_error.max_abs().max(h2.p_error.max_abs());
                            let same_object = h2.inst_id == h.inst_id;
                            let rehit = same_object
                                && if h.inst_id.is_some() {
                                    h2.ng.norm().dot(h.ng.norm()).abs() > 0.999 && (h2.p - h.p).dot(n).abs() <= 10.0 * err
                                } else {
                                    (h2.p - h.p).len() <= 100.0 * err
                                };
                            assert!(!rehit, "k={} dist={}: inward ray re-hit the entry surface at t={} (p_error {})", k, dist, h2.t, err);
                        }
                        checked += 1;
                    }
                }
                assert!(checked > 4000, "k={} dist={}: too few checks ({})", k, dist, checked);
            }
        }
    }

    /// 大きな床と薄い遮蔽物の混在（バッチ 3b では「期待される失敗」だったもの、3c の合格条件）:
    /// 1 万単位四方の床と、その上の非常に薄い壁（厚さ 5e-5、高さ 1e-2）。床上で壁から 5e-4 離れた点から
    /// 壁越しに出すシャドウレイは遮蔽される。シーンの大きさに比例したオフセット（≈ 1.4e-3）では原点が壁を
    /// 飛び越えていたが、交差点ごとの誤差上界に基づくオフセット（ここでは ~1e-12）なら壁を検出できる。
    #[test]
    fn large_floor_and_thin_occluder() {
        use crate::geometry::offset_ray_origin;
        let mut world = World::new();
        let q = |x: f64, z: f64| Vec3::new(x, 0.0, z);
        let floor = vec![
            Triangle::new_static(q(-5e3, -5e3), q(5e3, -5e3), q(5e3, 5e3), 0),
            Triangle::new_static(q(-5e3, -5e3), q(5e3, 5e3), q(-5e3, 5e3), 0),
        ];
        world.add_mesh_instance(floor, Transform::identity(), None);
        // 薄い壁: x ∈ [0, 5e-5]、y ∈ [0, 1e-2]、z ∈ [-5e-3, 5e-3]
        let wall = Transform::translate(Vec3::new(2.5e-5, 5e-3, 0.0)).compose(Transform::scale(Vec3::new(2.5e-5, 5e-3, 5e-3)));
        world.add_mesh_instance(unit_box(), wall, None);
        // 床の点（壁の手前 5e-4）を上から見て交差情報を得る
        let p = Vec3::new(-5e-4, 0.0, 0.0);
        let h = world.hit(Ray { o: p + Vec3::new(0.0, 1.0, 0.0), d: Vec3::new(0.0, -1.0, 0.0), time: 0.0 }, 0.0, 1e30).expect("floor hit");
        assert!((h.p - p).len() < 1e-9 && h.inst_id == Some(0));
        let d = Vec3::new(1.0, 0.002, 0.0).norm();
        let o = offset_ray_origin(h.p, h.p_error, h.ng, d);
        assert!((o - h.p).len() < 1e-9, "offset {} must be far below the distance to the wall", (o - h.p).len());
        let occluded = world.hit(Ray { o, d, time: 0.0 }, 0.0, 10.0);
        assert!(matches!(occluded, Some(w) if w.inst_id == Some(1)), "thin occluder not detected");
    }

    /// `World::bounds` は回転したインスタンスでも、変換後の頂点を包む（8 頂点の変換）。
    #[test]
    fn bounds_contain_rotated_instance() {
        let mut world = World::new();
        let xf = Transform::translate(Vec3::new(10.0, -2.0, 3.0)).compose(Transform::rotate(Vec3::new(0.0, 0.0, 1.0), 45.0));
        world.add_mesh_instance(unit_box(), xf, None);
        let b = world.bounds();
        let s = 2f64.sqrt();
        let expect_min = Vec3::new(10.0 - s, -2.0 - s, 2.0);
        let expect_max = Vec3::new(10.0 + s, -2.0 + s, 4.0);
        // BVH の AABB はわずかに（~1e-9）広げてある
        assert!((b.min - expect_min).len() < 1e-8 && (b.max - expect_max).len() < 1e-8, "{:?} {:?}", (b.min.x, b.min.y, b.min.z), (b.max.x, b.max.y, b.max.z));
    }

    /// インスタンスのワールド空間の境界ボックス（`Instance::world_bounds`）による事前棄却は、ヒットを落とさない。
    /// 事前棄却を使わない参照実装（`hit_without_instance_pruning`）と、ビット単位で同じ結果になることを確かめる。
    ///
    /// 箱の境界ちょうどに当たるレイを多く含める。立方体の頂点は箱の角・辺・面の上にあり、軸平行の配置では
    /// 立方体の面が箱の面と重なる。配置は大きさ 1e-3〜1e3、原点からの距離 0 と**絶対 1e8**、軸平行と
    /// 回転＋非一様スケール。レイは頂点・辺・面を狙い、すれすれの方向（面にほぼ平行）も含める。
    /// 箱のパディング（角の誤差上界と γ(3)·|座標|）を両方外すと、1e8 の配置でヒットを落として失敗する。
    #[test]
    fn instance_world_bounds_never_drop_hits() {
        let mut rng = Rng::new(97);
        let (mut hits, mut checked) = (0usize, 0usize);
        for case in 0..160 {
            let k = [1e-3, 1.0, 1e3][case % 3];
            let dist = if case % 2 == 0 { 0.0 } else { 1e8 };
            let rotated = case % 4 >= 2;
            let offset = Vec3::new(dist, -0.6 * dist, 0.8 * dist) + uniform_sphere_dir(&mut rng) * k;
            let mut xf = Transform::translate(offset);
            if rotated {
                xf = xf.compose(Transform::rotate(uniform_sphere_dir(&mut rng), 360.0 * rng.next_f64()));
            }
            let xf = xf.compose(Transform::scale(Vec3::new(k, (0.5 + rng.next_f64()) * k, (0.5 + rng.next_f64()) * k)));
            let mut world = World::new();
            world.add_mesh_instance(unit_box(), xf, None);
            for i in 0..150 {
                // 狙う点（物体空間）: 頂点・辺上・面上（座標の成分を ±1 に固定する数で切り替え）
                let mut q = [rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 2.0 - 1.0];
                let fixed = 3 - i % 3; // 3 = 頂点, 2 = 辺, 1 = 面
                let first = rng.next_u32() as usize % 3;
                for j in 0..fixed {
                    q[(first + j) % 3] = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
                }
                let target = xf.apply_point(Vec3::new(q[0], q[1], q[2]));
                // 方向: 一様な向き、または軸方向にほぼ平行（すれすれ）
                let mut d = uniform_sphere_dir(&mut rng);
                if i % 2 == 1 {
                    let axis = rng.next_u32() as usize % 3;
                    let mut a = [d.x * 1e-3, d.y * 1e-3, d.z * 1e-3];
                    a[axis] = if rng.next_f64() < 0.5 { 1.0 } else { -1.0 };
                    d = Vec3::new(a[0], a[1], a[2]).norm();
                }
                let o = target - d * (4.0 * k);
                let r = Ray { o, d, time: 0.0 };
                for &(tmin, tmax) in &[(0.0, 1e30), (0.0, 4.0 * k * (1.0 + 1e-12))] {
                    let expect = hit_without_instance_pruning(&world, r, tmin, tmax);
                    hits += expect.is_some() as usize;
                    checked += 1;
                    assert_same_hit(world.hit(r, tmin, tmax), expect, &format!("case {} (k={} dist={} rotated={}) ray {}", case, k, dist, rotated, i));
                }
            }
        }
        assert!(checked == 48_000 && hits > 10_000, "too few hits checked ({} of {})", hits, checked);
    }

    /// 中心頂点 `c` の周りに 6 枚の三角形を並べた扇（共有辺 6 本と共有頂点 1 つ）。法線 `n` の平面上、半径 `rad`。
    fn triangle_fan(c: Vec3, n: Vec3, rad: f64) -> (Vec<Triangle>, Vec<Vec3>) {
        let a = if n.x.abs() > 0.9 { Vec3::new(0.0, 1.0, 0.0) } else { Vec3::new(1.0, 0.0, 0.0) };
        let t = n.cross(a).norm();
        let b = n.cross(t);
        let rim: Vec<Vec3> = (0..6)
            .map(|i| {
                let ang = std::f64::consts::TAU * i as f64 / 6.0 + 0.3;
                c + (t * ang.cos() + b * ang.sin()) * rad
            })
            .collect();
        let tris = (0..6).map(|i| Triangle::new_static(c, rim[i], rim[(i + 1) % 6], 0)).collect();
        (tris, rim)
    }

    /// 水密性の回帰テスト: 三角形の扇の**共有頂点ちょうど**と**共有辺上の点**を狙うレイは、隣り合う三角形の
    /// どちらかに必ず当たる（すり抜け 0 件）。大きさ ×1e-3 / 等倍 / ×1e3 と、原点から 1e8 離した配置、
    /// ランダムな向きの平面、平面の両側からのレイで確認する。三角形単体（ワールド座標）と、回転・非一様
    /// スケールしたインスタンス経由（`World::hit`）の両方を調べる。
    /// Möller–Trumbore では共有辺ちょうどを狙うレイの約 2% がすり抜けていた。
    #[test]
    fn shared_edges_and_vertices_are_watertight() {
        let mut rng = Rng::new(41);
        let mut total = 0usize;
        for &(k, dist) in &[(1.0, 0.0), (1e-3, 0.0), (1e3, 0.0), (1.0, 1e8), (1e-3, 1e8)] {
            for _ in 0..20 {
                let c = Vec3::new(dist, -0.7 * dist, 0.4 * dist) + uniform_sphere_dir(&mut rng) * (0.3 * k);
                let n = uniform_sphere_dir(&mut rng);
                let (tris, rim) = triangle_fan(c, n, k);
                // 同じ扇をインスタンスとしても置く（物体空間では原点中心・等倍、ワールドへ回転・非一様スケール・平行移動）
                let (obj_tris, obj_rim) = triangle_fan(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0);
                let xf = Transform::translate(c)
                    .compose(Transform::rotate(uniform_sphere_dir(&mut rng), 360.0 * rng.next_f64()))
                    .compose(Transform::scale(Vec3::new(k, 0.8 * k, 1.3 * k)));
                let mut world = World::new();
                world.add_mesh_instance(obj_tris, xf, None);
                for i in 0..120 {
                    // 狙う点: 共有頂点（中心）、または共有辺（中心 → 外周の頂点）上の点。外周の頂点・辺はメッシュの
                    // 境界なので、丸めで外側に出た点を狙うレイが外れるのは正当（水密性の対象外）
                    let edge = i % 6;
                    let s = if i % 4 == 0 { 0.0 } else { rng.next_f64() };
                    let target = c + (rim[edge] - c) * s;
                    let obj_target = obj_rim[edge] * s;
                    let world_target = xf.apply_point(obj_target);
                    for (tgt, is_instance) in [(target, false), (world_target, true)] {
                        let o = tgt + uniform_sphere_dir(&mut rng) * (3.0 * k);
                        let d = (tgt - o).norm();
                        let r = Ray { o, d, time: 0.0 };
                        let hit = if is_instance {
                            world.hit(r, 0.0, 1e30).is_some()
                        } else {
                            tris.iter().any(|t| t.hit(r, 0.0, 1e30).is_some())
                        };
                        // 平面にほぼ平行なレイは除く（平面に届く前に扇の外を通りうる）
                        let plane_n = if is_instance { xf.apply_normal(Vec3::new(0.0, 0.0, 1.0)) } else { n };
                        if d.dot(plane_n).abs() < 0.05 {
                            continue;
                        }
                        total += 1;
                        assert!(hit, "k={} dist={} instance={} s={}: ray aimed at a shared {} slipped through", k, dist, is_instance, s, if s == 0.0 { "vertex" } else { "edge" });
                    }
                }
            }
        }
        assert!(total > 20_000, "too few rays checked ({})", total);
    }

    /// build_lights の重み = 面積 × 輝度（Light::area と共有された面積計算）。
    #[test]
    fn sphere_light_weight_is_area_times_luminance() {
        let r = 2.0;
        let world = emissive_sphere_world(Vec3::new(0.0, 0.0, 0.0), r);
        assert_eq!(world.lights.len(), 1);
        let area = 4.0 * std::f64::consts::PI * r * r;
        let lum = Color::new(3.0, 4.0, 5.0).luminance();
        assert!((world.light_total - area * lum).abs() < 1e-9);
    }

    /// sample_light のサンプルは球面上にあり、法線は外向き、PDF は有限正値。
    #[test]
    fn sphere_light_samples_lie_on_surface() {
        let c = Vec3::new(0.0, 0.0, 0.0);
        let r = 1.5;
        let world = emissive_sphere_world(c, r);
        let mut rng = Rng::new(5);
        let p = Vec3::new(5.0, 0.0, 0.0);
        let mut got = 0;
        for _ in 0..2000 {
            if let Some(ls) = world.sample_light(&mut rng, 0.0, p) {
                got += 1;
                assert!(((ls.position - c).len() - r).abs() < 1e-9, "off surface");
                assert!(ls.normal.dot(ls.position - c) > 0.0, "normal not outward");
                assert!(ls.pdf > 0.0 && ls.pdf.is_finite(), "bad pdf");
            }
        }
        assert!(got > 0, "no valid light samples");
    }

    /// `light_pdf` は `sample_light` の逆演算: サンプルされた点への Hit を作って
    /// `light_pdf` に渡すと、`sample_light` が返した pdf と一致しなければならない。
    /// これは BSDF サンプリングが発光体に命中した際の MIS 重み付けが正しく機能する
    /// ための前提条件で、この一致が壊れると面光源の寄与が二重計上/過小評価される。
    #[test]
    fn light_pdf_matches_sample_light_pdf() {
        let c = Vec3::new(0.0, 0.0, 0.0);
        let r = 1.5;
        let world = emissive_sphere_world(c, r);
        let mut rng = Rng::new(7);
        let from = Vec3::new(5.0, 0.0, 0.0);
        let mut checked = 0;
        for _ in 0..500 {
            if let Some(ls) = world.sample_light(&mut rng, 0.0, from) {
                let hit = Hit {
                    t: 0.0,
                    p: ls.position,
                    ng: ls.normal,
                    ns: ls.normal,
                    mat_id: 0,
                    prim_id: 0,
                    inst_id: None,
                    p_error: Vec3::new(0.0, 0.0, 0.0),
                    bary: (0.0, 0.0),
                    uv: (0.0, 0.0),
                };
                let pdf = world.light_pdf(from, 0.0, &hit);
                assert!((pdf - ls.pdf).abs() < 1e-9 * ls.pdf.max(1.0), "light_pdf {} != sample_light pdf {}", pdf, ls.pdf);
                checked += 1;
            }
        }
        assert!(checked > 0, "no valid light samples");
    }

    /// 参照点のバリエーション: 近い外部・遠い外部（sin²θmax が小円錐近似の閾値未満）・
    /// 表面すれすれの外部・内部（中心付近と表面付近）。
    fn sphere_light_reference_points() -> Vec<(&'static str, Vec3)> {
        vec![
            ("near outside", Vec3::new(2.5, 0.7, -0.4)),
            ("far outside (small cone)", Vec3::new(300.0, -50.0, 120.0)),
            ("just outside", Vec3::new(1.5 + 1e-3, 0.0, 0.0)),
            ("inside center", Vec3::new(0.1, -0.2, 0.05)),
            // 表面から 0.2。表面ごく近傍（例 0.01）だと面積サンプリングの 1/pdf の分散が
            // 対数発散し、有限サンプルの平均推定が安定しないため
            ("inside near surface", Vec3::new(0.0, 1.3, 0.0)),
        ]
    }

    /// 円錐サンプリング／内部フォールバックの両方で、sample_light の pdf は同じ点への light_pdf と一致し、
    /// サンプル点は球面上にある。外部からのサンプルは参照点から見える側（cos_light > 0）にある。
    #[test]
    fn sphere_light_pdf_matches_for_cone_and_inside_fallback() {
        let c = Vec3::new(0.0, 0.0, 0.0);
        let r = 1.5;
        let world = emissive_sphere_world(c, r);
        let mut rng = Rng::new(21);
        for (name, from) in sphere_light_reference_points() {
            let inside = (from - c).len() <= r;
            let mut got = 0;
            for _ in 0..4000 {
                let Some(ls) = world.sample_light(&mut rng, 0.0, from) else { continue };
                got += 1;
                assert!(((ls.position - c).len() - r).abs() < 1e-9, "{}: off surface", name);
                if !inside {
                    let wi = (ls.position - from).norm();
                    assert!(ls.normal.dot(-wi) > 0.0, "{}: sampled a point not visible from outside", name);
                }
                let hit = Hit { t: 0.0, p: ls.position, ng: ls.normal, ns: ls.normal, mat_id: 0, prim_id: 0, inst_id: None, p_error: Vec3::new(0.0, 0.0, 0.0), bary: (0.0, 0.0), uv: (0.0, 0.0) };
                let pdf = world.light_pdf(from, 0.0, &hit);
                assert!((pdf - ls.pdf).abs() <= 1e-9 * ls.pdf, "{}: light_pdf {} != sample pdf {}", name, pdf, ls.pdf);
            }
            assert!(got > 3900, "{}: too many rejected samples ({} / 4000)", name, got);
        }
    }

    /// light_pdf を立体角で積分すると 1（BSDF 側から見た光源の方向分布が正規化されている）。
    /// 全球一様な方向にレイを飛ばし、光源に当たった点の light_pdf の平均 × 4π で推定する。
    #[test]
    fn sphere_light_pdf_integrates_to_one_over_solid_angle() {
        let c = Vec3::new(0.0, 0.0, 0.0);
        let r = 1.5;
        let world = emissive_sphere_world(c, r);
        let mut rng = Rng::new(99);
        for (name, from) in sphere_light_reference_points() {
            // 外部では円錐を少し広げたキャップ（立体角は円錐の 1.5 倍）、内部では全球に
            // 一様な方向で推定する（遠い小円錐は全球一様だとほとんど当たらないため）
            let axis = (c - from).norm();
            let dc = (c - from).len();
            let cap_cos = if dc > r {
                let sin2 = (r / dc).powi(2);
                let one_minus_cos = if sin2 < 1e-3 { 0.5 * sin2 } else { 1.0 - (1.0 - sin2).sqrt() };
                1.0 - 1.5 * one_minus_cos
            } else {
                -1.0
            };
            let omega_cap = std::f64::consts::TAU * (1.0 - cap_cos);
            let (t, b) = orthonormal_basis(axis);
            let n = 400_000;
            let mut sum = 0.0;
            for _ in 0..n {
                let z = 1.0 - rng.next_f64() * (1.0 - cap_cos);
                let rr = (1.0 - z * z).max(0.0).sqrt();
                let phi = std::f64::consts::TAU * rng.next_f64();
                let d = t * (rr * phi.cos()) + b * (rr * phi.sin()) + axis * z;
                if let Some(h) = world.hit(Ray { o: from, d, time: 0.0 }, 1e-9, 1e30) {
                    sum += world.light_pdf(from, 0.0, &h);
                }
            }
            let integral = sum / n as f64 * omega_cap;
            assert!((integral - 1.0).abs() < 0.01, "{}: ∫pdf dω = {}", name, integral);
        }
    }

    /// 円錐サンプリングの推定は不偏: E[cosθ / pdf] = ∫_cone cosθ dω = π·sin²θmax
    /// （θ は球中心方向からの角）。内部フォールバックでは E[1/pdf] = 4π。
    #[test]
    fn sphere_light_estimates_are_unbiased() {
        let c = Vec3::new(0.0, 0.0, 0.0);
        let r = 1.5;
        let world = emissive_sphere_world(c, r);
        let mut rng = Rng::new(3);
        let n = 200_000;
        for (name, from) in sphere_light_reference_points() {
            let dc = (c - from).len();
            let axis = (c - from) / dc;
            let mut sum = 0.0;
            let inside = dc <= r;
            for _ in 0..n {
                if let Some(ls) = world.sample_light(&mut rng, 0.0, from) {
                    let f = if inside { 1.0 } else { (ls.position - from).norm().dot(axis) };
                    sum += f / ls.pdf;
                }
            }
            let est = sum / n as f64;
            let exact = if inside { 4.0 * std::f64::consts::PI } else { std::f64::consts::PI * (r / dc).powi(2) };
            assert!((est / exact - 1.0).abs() < 0.01, "{}: estimate {} vs exact {}", name, est, exact);
        }
    }

    /// 旧実装の小円錐近似の閾値（sin²θmax）。この前後で推定値が不連続にならないことを確かめる。
    const OLD_SMALL_CONE_SIN2: f64 = 0.00068523;

    /// 中心 c・半径 r の球に対し、sin²θmax が `sin2` になる外部の参照点。
    fn from_for_sin2(c: Vec3, r: f64, sin2: f64) -> Vec3 {
        c + Vec3::new(0.3, 0.8, -0.52).norm() * (r / sin2.sqrt())
    }

    /// 1 − cosθmax は小さな円錐でも厳密（桁落ち・一次近似の誤差がない）。
    #[test]
    fn cone_one_minus_cos_max_is_exact() {
        let s = Sphere { c: Vec3::new(0.3, -0.2, 0.1), r: 1.3, mat_id: 0 };
        let t = OLD_SMALL_CONE_SIN2;
        for sin2 in [0.5, 1e-2, t * (1.0 + 1e-3), t * (1.0 + 1e-9), t * (1.0 - 1e-9), t * (1.0 - 1e-3), 1e-5, 1e-8, 1e-12] {
            let cone = sphere_cone(&s, from_for_sin2(s.c, s.r, sin2)).expect("outside");
            let sin2 = cone.sin2_max; // 参照点の丸め後の実際の値で比較する
            let reference = if sin2 >= 1e-4 {
                1.0 - (1.0 - sin2).sqrt()
            } else {
                // 1 − √(1 − x) = x/2 + x²/8 + x³/16 + 5x⁴/128 + …
                sin2 / 2.0 + sin2 * sin2 / 8.0 + sin2.powi(3) / 16.0 + 5.0 * sin2.powi(4) / 128.0
            };
            let rel = (cone.one_minus_cos_max / reference - 1.0).abs();
            assert!(rel < 1e-11, "sin²θmax={:e}: 1−cosθmax {:e} vs reference {:e} (rel {:e})", sin2, cone.one_minus_cos_max, reference, rel);
        }
    }

    /// 小さな円錐でも推定が偏らず、旧近似の閾値の前後で連続:
    /// E[cosθ/pdf] = π·sin²θmax を相対 2e-5 で満たす（旧実装は閾値未満で −1.7e-4 の偏り）。
    /// cosθ/pdf = (1 − u·(1−cosθmax))·2π(1−cosθmax) は u に線形なので、統計誤差は (1−cosθmax) 倍に縮み
    /// 小さな円錐では 1e-6 未満になる。
    #[test]
    fn small_cone_estimate_is_unbiased_and_continuous() {
        let c = Vec3::new(0.3, -0.2, 0.1);
        let r = 1.3;
        let world = emissive_sphere_world(c, r);
        let t = OLD_SMALL_CONE_SIN2;
        let mut pdfs = Vec::new();
        for sin2 in [t * (1.0 - 1e-3), t * (1.0 - 1e-9), t * (1.0 + 1e-9), t * (1.0 + 1e-3), 1e-6, 1e-9] {
            let from = from_for_sin2(c, r, sin2);
            let dc = (c - from).len();
            let axis = (c - from) / dc;
            let sin2_actual = (r / dc).powi(2);
            let mut rng = Rng::new(17);
            let n = 200_000;
            let mut sum = 0.0;
            let mut pdf0 = 0.0;
            for _ in 0..n {
                let ls = world.sample_light(&mut rng, 0.0, from).expect("cone sample must not be rejected");
                sum += (ls.position - from).norm().dot(axis) / ls.pdf;
                pdf0 = ls.pdf;
            }
            let est = sum / n as f64;
            let exact = std::f64::consts::PI * sin2_actual;
            assert!((est / exact - 1.0).abs() < 2e-5, "sin²θmax={:e}: E[cosθ/pdf] {:e} vs π·sin² {:e} (rel {:e})", sin2, est, exact, est / exact - 1.0);
            pdfs.push(pdf0 * sin2_actual); // pdf ∝ 1/sin²（小円錐）なので正規化して連続性を見る
        }
        // 閾値のすぐ下とすぐ上（sin² の相対差 2e-9）で正規化 pdf が連続
        assert!((pdfs[1] / pdfs[2] - 1.0).abs() < 1e-6, "discontinuity at the old threshold: {} vs {}", pdfs[1], pdfs[2]);
    }

    /// 表面すれすれの外部の参照点でもサンプルがほぼ棄却されず（旧実装は r(1+1e-9) で 99.9% 棄却）、
    /// pdf は light_pdf と一致する。円錐サンプリングの範囲では、全サンプルが参照点から見える側にあり
    /// E[1/pdf] = Ω（円錐の立体角）が厳密に成り立つ（面積フォールバックに落ちると見える側をほぼ引けない）。
    #[test]
    fn near_surface_reference_points_keep_light_samples() {
      // 標準的な球・原点から離れた小さな球・大きな球（丸めの効き方が座標の大きさで変わるため）
      for (c, r) in [(Vec3::new(0.3, -0.2, 0.1), 1.3), (Vec3::new(1000.0, 500.0, -300.0), 0.05), (Vec3::new(-20.0, 3.0, 7.0), 1000.0)] {
        let world = emissive_sphere_world(c, r);
        let dir = Vec3::new(0.3, 0.8, -0.52).norm();
        for eps in [1e-3, 1e-5, 1e-6, 6e-7, 1e-7, 1e-9, 1e-11, 1e-12, 1e-14, 0.0, -1e-9] {
            let from = c + dir * (r * (1.0 + eps));
            let cone = sphere_cone(&world.spheres[0], from);
            let mut rng = Rng::new(5);
            let n = 20_000;
            let (mut got, mut sum_inv) = (0usize, 0.0);
            for _ in 0..n {
                let Some(ls) = world.sample_light(&mut rng, 0.0, from) else { continue };
                got += 1;
                sum_inv += 1.0 / ls.pdf;
                if cone.is_some() {
                    let wi = (ls.position - from).norm();
                    assert!(ls.normal.dot(-wi) > 0.0, "r={} eps={:e}: cone sample on the hidden side", r, eps);
                }
                let hit = Hit { t: 0.0, p: ls.position, ng: ls.normal, ns: ls.normal, mat_id: 0, prim_id: 0, inst_id: None, p_error: Vec3::new(0.0, 0.0, 0.0), bary: (0.0, 0.0), uv: (0.0, 0.0) };
                let pdf = world.light_pdf(from, 0.0, &hit);
                assert!((pdf - ls.pdf).abs() <= 1e-9 * ls.pdf, "r={} eps={:e}: light_pdf {} != sample pdf {}", r, eps, pdf, ls.pdf);
            }
            assert!(got as f64 >= 0.99 * n as f64, "r={} eps={:e}: {} / {} samples rejected (cone: {})", r, eps, n - got, n, cone.is_some());
            if let Some(cone) = cone {
                let omega = std::f64::consts::TAU * cone.one_minus_cos_max;
                assert!((sum_inv / got as f64 / omega - 1.0).abs() < 1e-9, "eps={:e}: E[1/pdf] {} vs Ω {}", eps, sum_inv / got as f64, omega);
            }
        }
      }
    }

    /// 発光体でないヒット（`inst_id`/`prim_id` が既知の発光体と一致しない）に対しては 0 を返す。
    #[test]
    fn light_pdf_is_zero_for_non_emitting_hit() {
        let world = emissive_sphere_world(Vec3::new(0.0, 0.0, 0.0), 1.5);
        let hit = Hit {
            t: 0.0,
            p: Vec3::new(10.0, 0.0, 0.0),
            ng: Vec3::new(1.0, 0.0, 0.0),
            ns: Vec3::new(1.0, 0.0, 0.0),
            mat_id: 0,
            prim_id: 3, // no sphere at this index
            inst_id: None,
            p_error: Vec3::new(0.0, 0.0, 0.0),
            bary: (0.0, 0.0),
            uv: (0.0, 0.0),
        };
        assert_eq!(world.light_pdf(Vec3::new(5.0, 0.0, 0.0), 0.0, &hit), 0.0);
    }

    /// `Mesh::hit`（BVH 経由）は全三角形を線形探索するブルートフォースと同じ最近接ヒットを返す。
    /// `Mesh` が三角形配列と BVH の対応関係を自分で保証しているからこそ書ける回帰テスト。
    #[test]
    fn mesh_hit_matches_brute_force() {
        let mut rng = Rng::new(42);
        let mut rand_range = |lo: f64, hi: f64| lo + rng.next_f64() * (hi - lo);

        let mut tris = Vec::new();
        for _ in 0..200 {
            let center = Vec3::new(rand_range(-5.0, 5.0), rand_range(-5.0, 5.0), rand_range(-5.0, 5.0));
            let v0 = center + Vec3::new(rand_range(-1.0, 1.0), rand_range(-1.0, 1.0), rand_range(-1.0, 1.0));
            let v1 = center + Vec3::new(rand_range(-1.0, 1.0), rand_range(-1.0, 1.0), rand_range(-1.0, 1.0));
            let v2 = center + Vec3::new(rand_range(-1.0, 1.0), rand_range(-1.0, 1.0), rand_range(-1.0, 1.0));
            tris.push(Triangle::new_static(v0, v1, v2, 0));
        }
        let mesh = Mesh::new(tris.clone());

        let mut checked_hits = 0;
        for _ in 0..500 {
            let o = Vec3::new(rand_range(-8.0, 8.0), rand_range(-8.0, 8.0), rand_range(-8.0, 8.0));
            let d = Vec3::new(rand_range(-1.0, 1.0), rand_range(-1.0, 1.0), rand_range(-1.0, 1.0)).norm();
            let r = Ray { o, d, time: 0.0 };

            let via_bvh = mesh.hit(r, 1e-6, 1e30);

            let mut brute: Option<Hit> = None;
            let mut closest = 1e30;
            for tri in &tris {
                if let Some(h) = tri.hit(r, 1e-6, closest) {
                    closest = h.t;
                    brute = Some(h);
                }
            }

            match (via_bvh, brute) {
                (Some(a), Some(b)) => {
                    checked_hits += 1;
                    assert!((a.t - b.t).abs() < 1e-9, "t mismatch: bvh={} brute={}", a.t, b.t);
                    assert!((a.p - b.p).len() < 1e-9, "p mismatch");
                }
                (None, None) => {}
                (a, b) => panic!("hit disagreement: bvh={:?}, brute={:?}", a.map(|h| h.t), b.map(|h| h.t)),
            }
        }
        assert!(checked_hits > 0, "no rays hit any triangle; test is vacuous");
    }

    /// 回帰テスト: NEE のシャドウレイは、原点を ε だけライト方向へ前進させても
    /// tmax を dist−2ε に取っておけば、サンプルした光源自身を誤って遮蔽物として
    /// 検出しない（tmax が dist−ε のままだと丸め次第で約半数が自己遮蔽してしまい、
    /// Cornell box が暗くなる/バンディングが出るバグがあった）。
    #[test]
    fn shadow_ray_does_not_self_hit_sampled_light() {
        use crate::transform::Transform;

        let mut world = World::new();
        // y=1.98 に、下向き法線（-Y）の矩形光源を三角形2枚で構成する。
        let m = |x: f64, z: f64| Vec3::new(0.35 * x, 1.98, 0.35 * z);
        let tris = vec![
            Triangle::new_static(m(-1.0, -1.0), m(1.0, -1.0), m(1.0, 1.0), 0),
            Triangle::new_static(m(-1.0, -1.0), m(1.0, 1.0), m(-1.0, 1.0), 0),
        ];
        world.add_mesh_instance(tris, Transform::identity(), None);
        let mats = vec![Material::DiffuseLight { emit: Color::new(4.6, 3.9, 2.0) }];
        world.build_lights(&mats);

        let mut rng = Rng::new(123);
        let p = Vec3::new(0.0, 1.4, -1.0);
        let mut sampled = 0;
        for _ in 0..2000 {
            if let Some(ls) = world.sample_light(&mut rng, 0.0, p) {
                sampled += 1;
                let to = ls.position - p;
                let dist = to.dot(to).sqrt();
                let wi = to / dist;
                // 終点を光源面の誤差の箱の外へずらした線分。途中で当たるのは光源自身だけ（それ以外の遮蔽物はない）
                let _ = wi;
                let to = crate::geometry::offset_ray_origin(ls.position, ls.p_error, ls.normal, p - ls.position);
                let seg = to - p;
                if let Some(h) = world.hit(Ray { o: p, d: seg / seg.len(), time: 0.0 }, 0.0, seg.len()) {
                    assert!(ls.is_light_itself(&h), "shadow ray hit something other than the light");
                }
            }
        }
        assert!(sampled > 0, "no light samples drawn");
    }

    /// 回帰テスト: `light_cdf` は先頭に 0.0 を持つ規約（cdf_search が前提とする
    /// [0, w0, w0+w1, …]）で構築されなければならない。先頭の 0.0 を欠くと
    /// cdf_search が常にインデックス 0 を返し、2 光源の場合は 2 番目の光源が
    /// 一切選ばれなくなる（面積比によらない偏ったサンプリングになる）。
    #[test]
    fn sample_light_selects_lights_area_proportionally() {
        let mut world = World::new();
        let c1 = Vec3::new(-5.0, 0.0, 0.0);
        let r1 = 1.0;
        let c2 = Vec3::new(5.0, 0.0, 0.0);
        let r2 = 2.0;
        world.spheres.push(Sphere { c: c1, r: r1, mat_id: 0 });
        world.spheres.push(Sphere { c: c2, r: r2, mat_id: 0 });
        let mats = vec![Material::DiffuseLight { emit: Color::new(1.0, 1.0, 1.0) }];
        world.build_lights(&mats);

        let mut rng = Rng::new(123);
        let p = Vec3::new(0.0, 0.0, 10.0);
        let n = 10_000;
        let mut count1 = 0;
        let mut count2 = 0;
        for _ in 0..n {
            if let Some(ls) = world.sample_light(&mut rng, 0.0, p) {
                if (ls.position - c1).len() < (ls.position - c2).len() {
                    count1 += 1;
                } else {
                    count2 += 1;
                }
            }
        }
        // sample_light は法線が p を向いていないサンプルを内部で棄却する（可視半球の
        // みを受理）ため、分母は総試行回数 n ではなく採択されたサンプル数にする。
        // 棄却率は両光源でほぼ等しいため、採択後の内訳は面積比をそのまま反映する。
        let accepted = count1 + count2;
        assert!(accepted > n / 4, "too few accepted samples ({}) to be meaningful", accepted);
        let frac1 = count1 as f64 / accepted as f64;
        let frac2 = count2 as f64 / accepted as f64;
        // 面積比: 4π·1² : 4π·2² = 1 : 4 -> 選択確率 0.2 : 0.8
        assert!((frac1 - 0.2).abs() < 0.05, "sphere1 fraction {} not near 0.2", frac1);
        assert!((frac2 - 0.8).abs() < 0.05, "sphere2 fraction {} not near 0.8", frac2);
    }

    // ---- スムーズシェーディング（頂点法線の補間） ----

    use test_meshes::uv_sphere;

    /// 全頂点の法線が同じなら、補間したシェーディング法線は面法線と（向きも含めて）一致する。
    /// 頂点法線を持たないメッシュとの差が出ないことの最小確認。
    #[test]
    fn uniform_vertex_normals_reproduce_the_face_normal() {
        let n = Vec3::new(0.0, 0.0, 1.0);
        let tris = vec![Triangle::new_static(
            Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0)];
        let mesh = Mesh::with_normals(tris, vec![n, n, n], vec![[0, 1, 2]]);
        for (x, y) in [(0.0, 0.0), (0.4, -0.3), (-0.3, -0.5), (0.0, 0.8)] {
            let o = Vec3::new(x, y, 3.0);
            let h = mesh.hit(Ray { o, d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30).unwrap();
            assert!((h.ns - h.ng).len() < 1e-15, "ns {:?} != ng {:?}", h.ns, h.ng);
            assert!(!h.is_smooth(), "一様な頂点法線は面法線と同一なので is_smooth は false");
        }
    }

    /// 球メッシュの補間法線は解析的な法線と一致し、面法線の誤差は分割を上げるまで大きい。
    ///
    /// 中心が原点の球で頂点法線 = 頂点位置（単位ベクトル）にすると、交点 p は三角形上の
    /// 重心座標の線形結合なので `normalize(Σ bᵢ·vᵢ) == normalize(p)`、つまり補間法線は
    /// **分割によらず解析解と一致する**（丸め誤差のみ）。面法線の方は三角形の大きさぶんずれ、
    /// 分割を上げると 1/nu のオーダーで減る。この 2 つの差が「補間が効いている」ことの証拠。
    #[test]
    fn interpolated_normals_match_the_analytic_sphere_normal() {
        let center = Vec3::new(0.0, 0.0, 0.0);
        let mut prev_flat = f64::INFINITY;
        for &nu in &[8usize, 16, 32, 64] {
            let smooth = Mesh::with_normals_from(uv_sphere(center, 1.0, nu, nu / 2, 0, true));
            let flat = Mesh::with_normals_from(uv_sphere(center, 1.0, nu, nu / 2, 0, false));
            let (mut worst_smooth, mut worst_flat) = (0.0f64, 0.0f64);
            let mut rng = Rng::new(7);
            for _ in 0..300 {
                let dir = uniform_sphere_dir(&mut rng);
                let o = center + dir * 4.0;
                let r = Ray { o, d: -dir, time: 0.0 };
                let (Some(hs), Some(hf)) = (smooth.hit(r, 0.0, 1e30), flat.hit(r, 0.0, 1e30)) else { continue };
                let ang = |n: Vec3, p: Vec3| n.dot((p - center).norm()).clamp(-1.0, 1.0).acos();
                worst_smooth = worst_smooth.max(ang(hs.ns, hs.p));
                worst_flat = worst_flat.max(ang(hf.ns, hf.p));
            }
            assert!(worst_smooth < 1e-6, "nu={}: 補間法線は解析解と一致するはず（最大 {} rad）", nu, worst_smooth);
            assert!(worst_flat > 20.0 * worst_smooth.max(1e-9), "nu={}: 面法線 {} は補間 {} より明確に大きいはず", nu, worst_flat, worst_smooth);
            assert!(worst_flat < prev_flat, "nu={}: 面法線の誤差は分割を上げると減るはず（{} -> {}）", nu, prev_flat, worst_flat);
            prev_flat = worst_flat;
        }
        // 面法線は 64 分割でもまだ 1 度以上ずれている（補間の 1e-6 rad とは桁が違う）
        assert!(prev_flat > 0.02, "面法線の最大誤差 {} rad", prev_flat);
    }

    /// 補間しても `Hit::ng` は面法線のまま（＝自己交差回避の基準が動かない）。
    /// さらに、原点ずらしを**シェーディング法線で行うミューテーション**では自己交差が起きることを
    /// 同じテストの中で示す（分離が効いていることの証拠）。
    #[test]
    fn geometric_normal_is_kept_for_ray_offsets() {
        // 大きく傾けた頂点法線を持つ 1 枚の三角形（z=0 平面、面法線は +z）
        let tris = vec![Triangle::new_static(
            Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0)];
        let tilted = Vec3::new(0.9, 0.0, 0.436).norm(); // 面法線から約 64 度
        let mesh = Mesh::with_normals(tris, vec![tilted; 3], vec![[0, 1, 2]]);
        let h = mesh.hit(Ray { o: Vec3::new(0.0, 0.0, 3.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30).unwrap();
        assert!((h.ng - Vec3::new(0.0, 0.0, 1.0)).len() < 1e-15, "ng は面法線のまま: {:?}", h.ng);
        assert!(h.is_smooth() && (h.ns - tilted).len() < 1e-12);

        // 面に沿って浅く出ていく方向。幾何法線基準なら誤差の箱の外に出るので自己交差しない
        let d = Vec3::new(1.0, 0.0, 1e-9).norm();
        let o_geom = crate::geometry::offset_ray_origin(h.p, h.p_error, h.ng, d);
        assert!(mesh.hit(Ray { o: o_geom, d, time: 0.0 }, 0.0, 1e30).is_none(), "幾何法線でずらせば自分に当たらない");
    }

    /// 非一様スケールと鏡像（負のスケール）を含むインスタンスでも、
    /// シェーディング法線は幾何法線と同じ側を向き、単位長のままになる。
    #[test]
    fn instance_transform_keeps_shading_normal_on_the_geometric_side() {
        for scale in [
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(2.0, 0.5, 1.0),   // 非一様
            Vec3::new(-1.0, 1.0, 1.0),  // 鏡像
            Vec3::new(-2.0, 0.5, 3.0),  // 鏡像 + 非一様
        ] {
            let mats = vec![Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5) }];
            let mut world = World::new();
            let xform = Transform::scale(scale);
            world.add_mesh_data_instance(uv_sphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 24, 12, 0, true), xform, None);
            world.build_lights(&mats);

            let mut rng = Rng::new(11);
            let mut checked = 0;
            for _ in 0..200 {
                let dir = uniform_sphere_dir(&mut rng);
                let o = dir * 20.0;
                let Some(h) = world.hit(Ray { o, d: -dir, time: 0.0 }, 0.0, 1e30) else { continue };
                assert!((h.ns.len() - 1.0).abs() < 1e-12, "scale {:?}: ns が単位長でない ({})", scale, h.ns.len());
                assert!((h.ng.len() - 1.0).abs() < 1e-12, "scale {:?}: ng が単位長でない ({})", scale, h.ng.len());
                assert!(h.ns.dot(h.ng) > 0.0, "scale {:?}: ns が ng の反対を向いた（{:?} vs {:?}）", scale, h.ns, h.ng);
                checked += 1;
            }
            assert!(checked > 100, "scale {:?}: ヒットが少なすぎる ({})", scale, checked);
        }
    }

    /// `MeshData` の頂点法線を捨てた（face_normals=true 相当）メッシュは、
    /// 頂点法線を最初から持たないメッシュと完全に同じヒットを返す。
    #[test]
    fn face_normals_flag_matches_a_mesh_without_normals() {
        let smooth_dropped = Mesh::with_normals_from(uv_sphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 16, 8, 0, true).into_flat());
        let never_had = Mesh::with_normals_from(uv_sphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 16, 8, 0, false));
        assert!(!smooth_dropped.is_smooth() && !never_had.is_smooth());
        let mut rng = Rng::new(3);
        for _ in 0..200 {
            let dir = uniform_sphere_dir(&mut rng);
            let r = Ray { o: dir * 5.0, d: -dir, time: 0.0 };
            match (smooth_dropped.hit(r, 0.0, 1e30), never_had.hit(r, 0.0, 1e30)) {
                (Some(a), Some(b)) => {
                    assert_eq!(a.t.to_bits(), b.t.to_bits());
                    assert_eq!(a.ng.x.to_bits(), b.ng.x.to_bits());
                    assert_eq!(a.ns.x.to_bits(), b.ns.x.to_bits());
                }
                (None, None) => {}
                _ => panic!("ヒットの有無が食い違う"),
            }
        }
    }

    /// 頂点法線の配列長が三角形数と合わない壊れた入力は、面法線メッシュとして扱う（添字ずれで落ちない）。
    #[test]
    fn mismatched_normal_index_array_is_ignored() {
        let tris = vec![
            Triangle::new_static(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0),
            Triangle::new_static(Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0),
        ];
        let mesh = Mesh::with_normals(tris, vec![Vec3::new(0.0, 0.0, 1.0)], vec![[0, 0, 0]]); // 1 個しかない
        assert!(!mesh.is_smooth());
    }

    // ---- スムーズシェーディング: ng / ns の取り違えを検出するテスト ----

    /// UV = (x, y) を張った直角三角形（2 枚で正方形）: ∂p/∂u = +x、∂p/∂v = +y。
    fn uv_plane_world(xform: Transform) -> World {
        let mut world = World::new();
        let (a, b, c, d) = (
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let tris = vec![Triangle::new_static(a, b, c, 0), Triangle::new_static(a, c, d, 0)];
        let uv = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        world.add_mesh_data_instance(MeshData::with_uv(tris, uv, vec![[0, 1, 2], [0, 2, 3]]), xform, None);
        world
    }

    fn hit_plane_at(world: &World, x: f64, y: f64, from: Vec3) -> Hit {
        let target = Vec3::new(x, y, 0.0);
        let world_target = {
            // 変換後の板の位置へ撃つため、インスタンスの変換を通した点へ向ける
            let inst = &world.instances[0];
            inst.xform.apply_point_with_error(target, Vec3::new(0.0, 0.0, 0.0)).0
        };
        let d = (world_target - from).norm();
        world.hit(Ray { o: from, d, time: 0.0 }, 0.0, 1e30).expect("plane not hit")
    }

    #[test]
    fn uv_derivatives_of_an_axis_aligned_uv_plane() {
        let world = uv_plane_world(Transform::identity());
        let h = hit_plane_at(&world, 0.7, 0.2, Vec3::new(0.7, 0.2, 3.0));
        let (dpdu, dpdv) = world.surface_tangents(&h, 0.0).unwrap();
        assert!((dpdu - Vec3::new(1.0, 0.0, 0.0)).len() < 1e-12, "{:?}", (dpdu.x, dpdu.y, dpdu.z));
        assert!((dpdv - Vec3::new(0.0, 1.0, 0.0)).len() < 1e-12, "{:?}", (dpdv.x, dpdv.y, dpdv.z));
    }

    /// 接空間は共有辺をまたいでも連続（同じ平面の 2 三角形で向きが一致）。
    #[test]
    fn tangents_are_continuous_across_a_shared_edge() {
        let world = uv_plane_world(Transform::identity());
        // 対角線 (0,0)-(1,1) をまたぐ 2 点
        let h1 = hit_plane_at(&world, 0.6, 0.4, Vec3::new(0.6, 0.4, 3.0));
        let h2 = hit_plane_at(&world, 0.4, 0.6, Vec3::new(0.4, 0.6, 3.0));
        assert_ne!(h1.prim_id, h2.prim_id);
        let (a, _) = world.surface_tangents(&h1, 0.0).unwrap();
        let (b, _) = world.surface_tangents(&h2, 0.0).unwrap();
        assert!(a.norm().dot(b.norm()) > 1.0 - 1e-9);
    }

    /// インスタンス変換: 接ベクトルは順方向の `A·t`。せん断・鏡像・非一様スケールでも、
    /// 変換後の 2 点の差（板の上の実際の方向）と一致する。
    ///
    /// ミューテーション検出: `surface_tangents` の `apply_vec` を `apply_normal`（逆転置 + 正規化）に
    /// 差し替えると、せん断変換で dpdu が実際の面上の方向からずれてこのテストが落ちる。
    #[test]
    fn tangents_follow_the_instance_transform_including_shear() {
        for (name, xform) in tricky_transforms() {
            let world = uv_plane_world(xform);
            let (u0, v0) = (0.3, 0.2);
            let du = 1e-3;
            let (pu0, pu1) = (Vec3::new(u0, v0, 0.0), Vec3::new(u0 + du, v0, 0.0));
            let (pv1, w) = (Vec3::new(u0, v0 + du, 0.0), Vec3::new(0.0, 0.0, 0.0));
            let map = |p: Vec3| xform.apply_point_with_error(p, w).0;
            let expect_u = (map(pu1) - map(pu0)) / du;
            let expect_v = (map(pv1) - map(pu0)) / du;
            // 板を撃つ点はワールド側の任意の位置から（変換後の板に向けて）
            let from = map(Vec3::new(u0, v0, 0.0)) + xform.apply_normal(Vec3::new(0.0, 0.0, 1.0)) * 3.0;
            let h = hit_plane_at(&world, u0, v0, from);
            let (dpdu, dpdv) = world.surface_tangents(&h, 0.0).unwrap();
            assert!((dpdu - expect_u).len() < 1e-9 * (1.0 + expect_u.len()), "{}: dpdu {:?} != {:?}", name, (dpdu.x, dpdu.y, dpdu.z), (expect_u.x, expect_u.y, expect_u.z));
            assert!((dpdv - expect_v).len() < 1e-9 * (1.0 + expect_v.len()), "{}: dpdv", name);
        }
    }

    /// UV 三角形が縮退（3 頂点とも同じ UV、または一直線）なら `None`。UV の無い三角形・球も `None`。
    #[test]
    fn degenerate_uv_and_uvless_hits_have_no_tangents() {
        for uv in [vec![[0.5, 0.5]; 3], vec![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]] {
            let mut world = World::new();
            let tris = vec![Triangle::new_static(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0)];
            let data = MeshData::with_uv(tris, uv, vec![[0, 1, 2]]);
            world.add_mesh_data_instance(data, Transform::identity(), None);
            let h = world.hit(Ray { o: Vec3::new(0.2, 0.2, 2.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30).unwrap();
            assert!(world.surface_tangents(&h, 0.0).is_none());
            assert_eq!(world.meshes()[0].degenerate_uv_count(), 1);
        }
        // UV 無しのメッシュ
        let mut world = World::new();
        world.add_mesh_instance(
            vec![Triangle::new_static(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0)],
            Transform::identity(),
            None,
        );
        let h = world.hit(Ray { o: Vec3::new(0.2, 0.2, 2.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30).unwrap();
        assert!(world.surface_tangents(&h, 0.0).is_none());
        // 球
        let mut world = World::new();
        world.add_sphere(Sphere { c: Vec3::new(0.0, 0.0, 0.0), r: 1.0, mat_id: 0 });
        let h = world.hit(Ray { o: Vec3::new(0.0, 0.0, 3.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30).unwrap();
        assert!(world.surface_tangents(&h, 0.0).is_none());
    }

    /// モーションブラー: `time` で辺を補間した接ベクトルは、その時刻の頂点から直接求めたものと一致する。
    #[test]
    fn tangents_interpolate_with_shutter_time() {
        let (a0, b0, c0) = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        // 閉時刻では x 方向に 3 倍に伸ばす
        let s = |v: Vec3| Vec3::new(v.x * 3.0, v.y, v.z);
        let tri = Triangle { v0_0: a0, v1_0: b0, v2_0: c0, mat_id: 0 };
        let mut data = MeshData::with_uv(vec![tri], vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], vec![[0, 1, 2]]);
        data.motion = vec![[s(a0), s(b0), s(c0)]];
        let mesh = Mesh::with_normals_from(data);
        for &t in &[0.0, 0.25, 1.0] {
            let (dpdu, dpdv) = mesh.uv_derivatives(0, t).unwrap();
            let (v0, v1, v2) = mesh.vertices_at(0, t).unwrap();
            assert!((dpdu - (v1 - v0)).len() < 1e-12, "t={}", t);
            assert!((dpdv - (v2 - v0)).len() < 1e-12, "t={}", t);
        }
    }

    /// せん断・鏡像を含む「意地悪な」変換の一覧（法線の逆転置がもっとも効く形）。
    fn tricky_transforms() -> Vec<(&'static str, Transform)> {
        let m = |a: [[f64; 4]; 4]| Transform::from_matrix4(a);
        vec![
            ("shear x+=2z", m([[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])),
            ("shear+mirror x", m([[-1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])),
            ("shear+mirror y", m([[1.0, 3.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])),
            ("shear xyz + mirror", m([[-1.0, 1.5, 2.5, 0.0], [0.5, 1.0, 0.0, 0.0], [2.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])),
            ("anisotropic + shear + mirror", m([[-4.0, 2.0, 0.0, 0.0], [0.0, 0.25, 3.0, 0.0], [1.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]])),
            ("mirror xyz", Transform::scale(Vec3::new(-1.0, -1.0, -1.0))),
            ("anisotropic", Transform::scale(Vec3::new(-5.0, 0.2, 2.0))),
        ]
    }

    /// インスタンス変換の後も `ns · ng > 0` が保たれる（せん断 + 鏡像を含む）。
    ///
    /// ミューテーション検出: `World::hit` の変換後の `face_forward` を外すと、
    /// せん断と鏡像を組み合わせた変換で `ns` が `ng` の反対側へ回り、このテストが落ちる。
    /// 粗い球（8x4 = 48 三角形）を使うのは、オブジェクト空間での `ns` と `ng` の開きが
    /// 大きいほど逆転置による向きの入れ替わりが起きやすいから。
    #[test]
    fn shading_normal_stays_on_the_geometric_side_through_shearing_transforms() {
        let mats = vec![Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5) }];
        for (name, xform) in tricky_transforms() {
            let mut world = World::new();
            world.add_mesh_data_instance(uv_sphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 8, 4, 0, true), xform, None);
            world.build_lights(&mats);
            let mut rng = Rng::new(17);
            let (mut checked, mut worst) = (0usize, f64::INFINITY);
            for _ in 0..4000 {
                let dir = uniform_sphere_dir(&mut rng);
                let o = dir * 40.0;
                let Some(h) = world.hit(Ray { o, d: -dir, time: 0.0 }, 0.0, 1e30) else { continue };
                let dot = h.ns.dot(h.ng);
                worst = worst.min(dot);
                assert!(dot > 0.0, "{}: ns が ng の裏へ回った（ns·ng = {}）", name, dot);
                assert!((h.ns.len() - 1.0).abs() < 1e-12, "{}: ns が単位長でない", name);
                checked += 1;
            }
            assert!(checked > 500, "{}: ヒットが少なすぎる ({})", name, checked);
            let _ = worst;
        }
    }

    /// **シェーディング法線はインスタンス変換で実際に変換される**（オブジェクト空間のまま使われない）。
    ///
    /// 頂点法線が一様なメッシュ（面ごとに補間値が定数）なら、変換後の `ns` は
    /// `face_forward(apply_normal(ns_obj), ng)` と一致するはず。正しい実装を再実装せずに書ける形にしてある。
    /// 重心座標での補間（`Σ bᵢ·vᵢ` を計算してから正規化）が入るぶん完全なビット一致にはならないので、
    /// 許容差 1e-12 で比べる（取り違えたときの差は 0.1 rad 以上なので、これで十分に鋭い）。
    ///
    /// ミューテーション検出: `World::hit` で `ns` を変換せず `h_obj.ns` のまま使うと落ちる。
    /// `ns·ng > 0` と単位長だけを見るテストでは、この取り違えは通り抜ける
    /// （オブジェクト空間の法線でも両方の条件を満たしてしまうため）。
    #[test]
    fn instance_transform_actually_transforms_the_shading_normal() {
        let mats = vec![Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5) }];
        // 面法線 +z から 50 度傾けた一様な頂点法線。回転・非一様・せん断で「変換しない」と
        // 明確に違う向きになる。
        let ns_obj = Vec3::new(0.766, 0.0, 0.643).norm();
        let mut meaningful_transforms = 0usize;
        for (name, xform) in tricky_transforms() {
            let mut world = World::new();
            world.add_mesh_data_instance(test_meshes::tilted_quad(4.0, ns_obj, 0), xform, None);
            world.build_lights(&mats);

            let expected_raw = xform.apply_normal(ns_obj);
            // 点対称（-I）のように、法線の向きが符号だけしか変わらない変換では
            // 「変換しない」ミューテーションと区別できない。その変換では下の空回り判定を外す。
            let direction_changes = expected_raw.dot(ns_obj).abs() < 1.0 - 1e-12;
            if direction_changes {
                meaningful_transforms += 1;
            }
            let mut rng = Rng::new(37);
            let mut checked = 0usize;
            let mut differs_from_object_space = 0usize;
            for _ in 0..2000 {
                let dir = uniform_sphere_dir(&mut rng);
                let o = dir * 30.0;
                let Some(h) = world.hit(Ray { o, d: -dir, time: 0.0 }, 0.0, 1e30) else { continue };
                let expected = face_forward(expected_raw, h.ng);
                assert!(
                    (h.ns - expected).len() < 1e-12,
                    "{}: ns が変換後の値と違う（ns = {:?}, 期待 {:?}）", name, h.ns, expected
                );
                // オブジェクト空間の法線とは実際に違う（テストが空回りしていないこと）
                if (h.ns - face_forward(ns_obj, h.ng)).len() > 1e-9 {
                    differs_from_object_space += 1;
                }
                checked += 1;
            }
            assert!(checked > 200, "{}: ヒットが少なすぎる ({})", name, checked);
            if direction_changes {
                assert!(
                    differs_from_object_space > checked / 2,
                    "{}: 変換前後で ns が変わらない配置ばかり（テストが空回り）", name
                );
            }
        }
        assert!(meaningful_transforms >= 4, "向きを変える変換が少なすぎる（{}）", meaningful_transforms);
    }

    /// **幾何法線はスムーズ化の影響を受けない**: 同じ形状・同じ変換で、頂点法線の有無だけが違う
    /// 2 つのインスタンスは、同じレイに対して**ビット単位で同じ `ng`** を返す（`ns` だけが違う）。
    ///
    /// ミューテーション検出: `World::hit` で `ng` と `ns` を取り違える（入れ替える）と落ちる。
    /// 幾何法線は自己交差回避・表裏判定・光源 pdf の基準なので、ここが補間値に化けると静かに壊れる。
    #[test]
    fn instance_geometric_normal_is_unaffected_by_vertex_normals() {
        let mats = vec![Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5) }];
        for (name, xform) in tricky_transforms() {
            let mut smooth_world = World::new();
            smooth_world.add_mesh_data_instance(uv_sphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 12, 6, 0, true), xform, None);
            smooth_world.build_lights(&mats);
            let mut flat_world = World::new();
            flat_world.add_mesh_data_instance(uv_sphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 12, 6, 0, false), xform, None);
            flat_world.build_lights(&mats);

            let mut rng = Rng::new(23);
            let (mut checked, mut smooth_seen) = (0usize, 0usize);
            for _ in 0..2000 {
                let dir = uniform_sphere_dir(&mut rng);
                let o = dir * 40.0;
                let r = Ray { o, d: -dir, time: 0.0 };
                match (smooth_world.hit(r, 0.0, 1e30), flat_world.hit(r, 0.0, 1e30)) {
                    (Some(a), Some(b)) => {
                        assert_eq!(a.t.to_bits(), b.t.to_bits(), "{}: 交差距離が違う", name);
                        assert_eq!(a.ng.x.to_bits(), b.ng.x.to_bits(), "{}: ng が頂点法線に汚染されている", name);
                        assert_eq!(a.ng.y.to_bits(), b.ng.y.to_bits(), "{}: ng が頂点法線に汚染されている", name);
                        assert_eq!(a.ng.z.to_bits(), b.ng.z.to_bits(), "{}: ng が頂点法線に汚染されている", name);
                        assert_eq!(b.ns.x.to_bits(), b.ng.x.to_bits(), "{}: 面法線メッシュは ns == ng", name);
                        if a.is_smooth() {
                            smooth_seen += 1;
                        }
                        checked += 1;
                    }
                    (None, None) => {}
                    _ => panic!("{}: ヒットの有無が食い違う", name),
                }
            }
            assert!(checked > 300, "{}: ヒットが少なすぎる ({})", name, checked);
            assert!(smooth_seen > checked / 2, "{}: 補間が効いているヒットが少なすぎる（テストが空回り）", name);
        }
    }

    // ---- テクスチャ座標（UV）の配管 ----

    /// 頂点 UV は重心座標で補間される（三角形の各頂点で厳密に頂点 UV になる）。
    #[test]
    fn vertex_uv_is_interpolated_by_barycentric_coordinates() {
        // z=0 平面の直角三角形。v0=(0,0) v1=(1,0) v2=(0,1) に UV [0,0] [1,0] [0,1] を割り当てる
        let tris = vec![Triangle::new_static(
            Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0)];
        let mesh = Mesh::build(
            tris, Vec::new(), Vec::new(), Vec::new(),
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], vec![[0, 1, 2]],
        );
        assert!(mesh.has_uv());
        // この割り当てでは UV は (x, y) と一致するので、当てた位置から期待値が直に決まる
        for (x, y) in [(0.05, 0.05), (0.5, 0.25), (0.25, 0.5), (0.3, 0.3), (0.8, 0.1)] {
            let h = mesh
                .hit(Ray { o: Vec3::new(x, y, 3.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30)
                .unwrap_or_else(|| panic!("({}, {}) に当たらない", x, y));
            assert!((h.uv.0 - x).abs() < 1e-12 && (h.uv.1 - y).abs() < 1e-12,
                    "uv = {:?}, 期待 ({}, {})", h.uv, x, y);
        }
    }

    /// UV を持たないメッシュのヒットは uv = (0, 0)（テクスチャを引かない既定値）。
    #[test]
    fn mesh_without_uv_reports_zero_uv() {
        let tris = vec![Triangle::new_static(
            Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0)];
        let mesh = Mesh::new(tris);
        assert!(!mesh.has_uv());
        let h = mesh.hit(Ray { o: Vec3::new(0.0, 0.0, 3.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30).unwrap();
        assert_eq!(h.uv, (0.0, 0.0));
    }

    /// UV 添字の配列長が三角形数と合わない壊れた入力は、UV 無しとして扱う（添字ずれで落ちない）。
    #[test]
    fn mismatched_uv_index_array_is_ignored() {
        let tris = vec![
            Triangle::new_static(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0),
            Triangle::new_static(Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0),
        ];
        let mesh = Mesh::build(tris, Vec::new(), Vec::new(), Vec::new(), vec![[0.0, 0.0]], vec![[0, 0, 0]]); // 1 個しかない
        assert!(!mesh.has_uv());
    }

    /// UV あり／なしが混在するメッシュでも、UV なしの三角形は (0, 0) になる（`NO_UV` の分岐）。
    #[test]
    fn mesh_with_mixed_uv_falls_back_per_triangle() {
        let tris = vec![
            Triangle::new_static(Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(0.0, -0.05, 0.0), 0),
            Triangle::new_static(Vec3::new(-1.0, 1.0, 0.0), Vec3::new(0.0, 0.05, 0.0), Vec3::new(1.0, 1.0, 0.0), 0),
        ];
        let mesh = Mesh::build(
            tris, Vec::new(), Vec::new(), Vec::new(),
            vec![[0.3, 0.4]], vec![[0, 0, 0], [NO_UV; 3]],
        );
        let shoot = |y: f64| mesh.hit(Ray { o: Vec3::new(0.0, y, 3.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30);
        let a = shoot(-0.5).expect("UV つきの三角形に当たらない");
        assert!((a.uv.0 - 0.3).abs() < 1e-12 && (a.uv.1 - 0.4).abs() < 1e-12, "uv = {:?}", a.uv);
        let b = shoot(0.5).expect("UV なしの三角形に当たらない");
        assert_eq!(b.uv, (0.0, 0.0));
    }

    /// インスタンス変換は UV を変えない（UV はオブジェクト空間の属性）。
    #[test]
    fn instance_transform_does_not_change_uv() {
        let mats = vec![Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5) }];
        let tris = vec![Triangle::new_static(
            Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0)];
        let data = MeshData::with_uv(
            tris, vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], vec![[0, 1, 2]]);
        let mut world = World::new();
        world.add_mesh_data_instance(data, Transform::scale(Vec3::new(4.0, 0.5, 2.0)), None);
        world.build_lights(&mats);
        // 変換後の (x, y) = (4·u, 0.5·v) に当たるレイを撃つ
        for (u, v) in [(0.2, 0.3), (0.5, 0.25), (0.1, 0.8)] {
            let r = Ray { o: Vec3::new(4.0 * u, 0.5 * v, 3.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 };
            let h = world.hit(r, 0.0, 1e30).unwrap_or_else(|| panic!("({}, {}) に当たらない", u, v));
            assert!((h.uv.0 - u).abs() < 1e-12 && (h.uv.1 - v).abs() < 1e-12,
                    "uv = {:?}, 期待 ({}, {})", h.uv, u, v);
        }
    }

    /// 球の UV は Mitsuba の球面座標（u = φ/2π、v = θ/π、極は ±z）。
    #[test]
    fn sphere_uv_follows_the_mitsuba_spherical_parameterisation() {
        let s = Sphere { c: Vec3::new(0.0, 0.0, 0.0), r: 1.0, mat_id: 0 };
        let hit_from = |dir: Vec3| {
            s.hit(Ray { o: dir * 5.0, d: -dir, time: 0.0 }, 0.0, 1e30).expect("球に当たらない")
        };
        // +x 方向（φ = 0, θ = π/2）
        let h = hit_from(Vec3::new(1.0, 0.0, 0.0));
        assert!(h.uv.0.abs() < 1e-12 && (h.uv.1 - 0.5).abs() < 1e-12, "+x: {:?}", h.uv);
        // +y 方向（φ = π/2）
        let h = hit_from(Vec3::new(0.0, 1.0, 0.0));
        assert!((h.uv.0 - 0.25).abs() < 1e-12 && (h.uv.1 - 0.5).abs() < 1e-12, "+y: {:?}", h.uv);
        // −x 方向（φ = π）
        let h = hit_from(Vec3::new(-1.0, 0.0, 0.0));
        assert!((h.uv.0 - 0.5).abs() < 1e-12, "−x: {:?}", h.uv);
        // +z（北極、θ = 0）と −z（南極、θ = π）
        let h = hit_from(Vec3::new(0.0, 0.0, 1.0));
        assert!(h.uv.1.abs() < 1e-9, "+z 極: {:?}", h.uv);
        let h = hit_from(Vec3::new(0.0, 0.0, -1.0));
        assert!((h.uv.1 - 1.0).abs() < 1e-9, "−z 極: {:?}", h.uv);
        // u は常に [0, 1]
        let mut rng = Rng::new(5);
        for _ in 0..200 {
            let d = uniform_sphere_dir(&mut rng);
            let h = hit_from(d);
            assert!((0.0..=1.0).contains(&h.uv.0) && (0.0..=1.0).contains(&h.uv.1), "uv = {:?}", h.uv);
        }
    }

    /// **法線あり／なしが混在するメッシュ**を交差判定（描画が通る経路）で扱える。
    ///
    /// `f 1//1 2//2 3//3` と `f 1 2 3` が混ざった OBJ は実在する。ローダーは法線を持たない
    /// 三角形に番兵 `NO_NORMAL` を入れ、`Mesh::shading_normal` がそれを見て面法線に落とす。
    ///
    /// ミューテーション検出: 番兵チェック（`if idx[0] == NO_NORMAL { return None }`）を外すと、
    /// 法線を持たない三角形に当たった瞬間に `vn[u32::MAX]` で**添字外アクセスのパニック**になる。
    /// `obj_loader` 側には混在のパーステストがあるが、そこはヒットを通らないので気づけない。
    #[test]
    fn mesh_with_mixed_vertex_normals_is_hit_without_panicking() {
        // 三角形 0（z=0 平面、y<0 側）は傾いた頂点法線つき、三角形 1（y>0 側）は法線なし
        let tilted = Vec3::new(0.6, 0.0, 0.8).norm();
        let tris = vec![
            Triangle::new_static(Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(0.0, -0.05, 0.0), 0),
            Triangle::new_static(Vec3::new(-1.0, 1.0, 0.0), Vec3::new(0.0, 0.05, 0.0), Vec3::new(1.0, 1.0, 0.0), 0),
        ];
        let mesh = Mesh::with_normals(tris, vec![tilted], vec![[0, 0, 0], [NO_NORMAL; 3]]);
        assert!(mesh.is_smooth(), "混在メッシュはスムーズ扱い（三角形ごとに分岐する）");

        let shoot = |y: f64| {
            mesh.hit(Ray { o: Vec3::new(0.0, y, 3.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30)
        };
        // 頂点法線を持つ側: 補間法線が使われる
        let h_smooth = shoot(-0.5).expect("法線つきの三角形に当たらない");
        assert!(h_smooth.is_smooth(), "頂点法線を持つ三角形で補間されていない");
        assert!((h_smooth.ns - tilted).len() < 1e-12, "補間法線が頂点法線と違う: {:?}", h_smooth.ns);
        // 頂点法線を持たない側: 面法線のまま（ここで番兵の分岐を踏む）
        let h_flat = shoot(0.5).expect("法線なしの三角形に当たらない");
        assert!(!h_flat.is_smooth(), "法線なしの三角形が補間されている");
        assert_eq!(h_flat.ns.z.to_bits(), h_flat.ng.z.to_bits(), "法線なしの三角形は ns == ng");

        // インスタンス経由（World::hit）でも同じ。変換つきでも番兵の分岐を踏む
        let mats = vec![Material::Lambert { albedo: Color::new(0.5, 0.5, 0.5) }];
        let mut world = World::new();
        world.add_mesh_data_instance(
            MeshData {
                tris: mesh.tris.clone(),
                vn: vec![tilted],
                tri_vn: vec![[0, 0, 0], [NO_NORMAL; 3]],
                uv: Vec::new(),
                tri_uv: Vec::new(),
                motion: Vec::new(),
            },
            Transform::scale(Vec3::new(2.0, 0.5, 1.5)),
            None,
        );
        world.build_lights(&mats);
        let mut smooth_hits = 0;
        let mut flat_hits = 0;
        for i in 0..40 {
            let y = -1.2 + 2.4 * (i as f64) / 39.0;
            let r = Ray { o: Vec3::new(0.0, y, 5.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 };
            if let Some(h) = world.hit(r, 0.0, 1e30) {
                if h.is_smooth() { smooth_hits += 1 } else { flat_hits += 1 }
            }
        }
        assert!(smooth_hits > 0 && flat_hits > 0,
                "両方の三角形を通っていない（smooth {} / flat {}）", smooth_hits, flat_hits);
    }

    /// 頂点法線が面法線と逆を向いている（壊れた、あるいは巻き順の違う）メッシュでも、
    /// `Hit::ns` は `ng` と同じ側に揃う。
    ///
    /// ミューテーション検出: `Mesh::shading_normal` の `face_forward` を外すと落ちる。
    #[test]
    fn shading_normal_is_flipped_to_the_geometric_side_at_the_mesh() {
        // 面法線は +z（反時計回り）だが、頂点法線は全て -z を向いている
        let tris = vec![Triangle::new_static(
            Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0)];
        let back = Vec3::new(0.0, 0.0, -1.0);
        let mesh = Mesh::with_normals(tris, vec![back; 3], vec![[0, 1, 2]]);
        let h = mesh.hit(Ray { o: Vec3::new(0.0, 0.0, 3.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30).unwrap();
        assert!(h.ns.dot(h.ng) > 0.0, "ns が ng の裏を向いたまま: ns={:?} ng={:?}", h.ns, h.ng);
    }

    /// **スムーズな発光メッシュの `light_pdf` は幾何法線で計算される**。
    ///
    /// 光源上の点の pdf は「その点の面が参照点をどれだけ斜めに見るか」で決まるので、
    /// シェーディング法線を使うと MIS の重みが狂う（BSDF サンプリングで光源に当たった経路の重み）。
    /// 頂点法線の有無だけが違う 2 つの発光メッシュで、同じレイのヒットに対する `light_pdf` が
    /// ビット単位で一致することを確かめる。
    ///
    /// ミューテーション検出: `World::light_pdf` の `hit.ng` を `hit.ns` にすると落ちる。
    /// （既存の `light_pdf` のテストは球・矩形の光源しか使っておらず、`ns == ng` なので素通りする。）
    #[test]
    fn light_pdf_on_a_smooth_emissive_mesh_uses_the_geometric_normal() {
        let mats = vec![Material::DiffuseLight { emit: Color::new(5.0, 5.0, 5.0) }];
        let build = |smooth: bool| {
            let mut w = World::new();
            w.add_mesh_data_instance(
                uv_sphere(Vec3::new(0.0, 0.0, 0.0), 1.0, 12, 6, 0, smooth), Transform::identity(), None);
            w.build_lights(&mats);
            w
        };
        let (smooth_world, flat_world) = (build(true), build(false));
        let mut rng = Rng::new(29);
        let (mut checked, mut smooth_hits) = (0usize, 0usize);
        for _ in 0..1500 {
            let dir = uniform_sphere_dir(&mut rng);
            let from = dir * 6.0;
            let r = Ray { o: from, d: -dir, time: 0.0 };
            let (Some(hs), Some(hf)) = (smooth_world.hit(r, 0.0, 1e30), flat_world.hit(r, 0.0, 1e30)) else { continue };
            let ps = smooth_world.light_pdf(from, 0.0, &hs);
            let pf = flat_world.light_pdf(from, 0.0, &hf);
            assert!(ps > 0.0, "発光メッシュへのヒットなのに pdf が 0");
            assert_eq!(ps.to_bits(), pf.to_bits(), "light_pdf が頂点法線に影響されている（{} vs {}）", ps, pf);
            if hs.is_smooth() {
                smooth_hits += 1;
            }
            checked += 1;
        }
        assert!(checked > 300, "ヒットが少なすぎる ({})", checked);
        assert!(smooth_hits > checked / 2, "補間が効いているヒットが少なすぎる（テストが空回り）");
    }
    // ---- トップレベル BVH（TLAS）----

    /// 乱数のインスタンス（大小・回転・非一様スケール・同じ位置の重なりを含む）と球からなるワールド。
    fn random_tlas_world(n_inst: usize, n_sph: usize, seed: u64, same_place: bool) -> World {
        let mut rng = Rng::new(seed);
        let mut u = |a: f64, b: f64| a + (b - a) * rng.next_f64();
        let mut world = World::new();
        let tri = |k: f64| vec![
            Triangle::new_static(Vec3::new(-k, -k, 0.0), Vec3::new(k, -k, 0.0), Vec3::new(0.0, k, 0.5), 0),
            Triangle::new_static(Vec3::new(-k, -k, 0.5), Vec3::new(k, -k, 0.5), Vec3::new(0.0, k, 0.0), 0),
        ];
        world.add_mesh_instance(tri(1.0), Transform::identity(), None);
        let cube_like = world.instance_mesh_id(0);
        for i in 0..n_inst {
            let (p, sc) = if same_place {
                (Vec3::new(0.0, 0.0, 0.0), 1.0)
            } else if i % 5 == 0 {
                (Vec3::new(u(-30.0, 30.0), u(-30.0, 30.0), u(-30.0, 30.0)), u(20.0, 60.0)) // 巨大
            } else if i % 5 == 1 {
                (Vec3::new(u(-8.0, 8.0), u(-8.0, 8.0), u(-8.0, 8.0)), u(1e-4, 1e-3)) // 極小
            } else {
                (Vec3::new(u(-8.0, 8.0), u(-8.0, 8.0), u(-8.0, 8.0)), u(0.3, 2.0))
            };
            let x = Transform::translate(p).compose(
                Transform::rotate(Vec3::new(u(-1.0, 1.0), u(-1.0, 1.0), u(0.1, 1.0)), u(0.0, 360.0))
                    .compose(Transform::scale(Vec3::new(sc, sc * u(0.5, 1.5), sc))),
            );
            world.add_instance_of(cube_like, x, Some(i % 3));
        }
        for j in 0..n_sph {
            let (c, r) = if same_place { (Vec3::new(0.0, 0.0, 0.0), 1.0) } else { (Vec3::new(u(-8.0, 8.0), u(-8.0, 8.0), u(-8.0, 8.0)), u(0.05, 2.0)) };
            world.add_sphere(Sphere { c, r, mat_id: j % 4 });
        }
        world
    }

    fn assert_tlas_hit(a: &Option<Hit>, b: &Option<Hit>, what: &str) {
        match (a, b) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                assert_eq!(a.t.to_bits(), b.t.to_bits(), "{what}: t");
                assert_eq!((a.inst_id, a.prim_id, a.mat_id), (b.inst_id, b.prim_id, b.mat_id), "{what}: ids");
                assert_eq!((a.p.x.to_bits(), a.ng.z.to_bits(), a.uv.0.to_bits()), (b.p.x.to_bits(), b.ng.z.to_bits(), b.uv.0.to_bits()), "{what}: geometry");
            }
            _ => panic!("{what}: {:?} vs {:?}", a.map(|h| h.t), b.map(|h| h.t)),
        }
    }

    /// TLAS 経由の `hit` / `occluded` は線形総当たりと完全に一致する（t のビットまで、`inst_id`・`prim_id`・`mat_id` も）。
    #[test]
    fn tlas_matches_linear_bruteforce() {
        let mut rng = Rng::new(99);
        for (n_inst, n_sph, same) in [(30, 30, false), (60, 0, false), (0, 60, false), (25, 5, false), (40, 40, true), (1, 30, false)] {
            let world = random_tlas_world(n_inst, n_sph, 7 + n_inst as u64, same);
            assert!(world.tlas().is_some(), "{} prims で TLAS が使われるはず", n_inst + n_sph + 1);
            for k in 0..3000 {
                let o = Vec3::new(rng.next_f64() * 40.0 - 20.0, rng.next_f64() * 40.0 - 20.0, rng.next_f64() * 40.0 - 20.0);
                let d = uniform_sphere_dir(&mut rng);
                let r = Ray { o, d, time: 0.0 };
                let tmin = if k % 7 == 0 { 1e-3 } else { 0.0 };
                let tmax = if k % 5 == 0 { 15.0 } else { 1e30 };
                let what = format!("case ({n_inst},{n_sph},{same}) ray {k}");
                assert_tlas_hit(&world.hit(r, tmin, tmax), &world.hit_linear(r, tmin, tmax), &what);
                for skip in [None, Some((None, 0usize)), Some((Some(1usize), 0usize))] {
                    assert_eq!(world.occluded(r, tmin, tmax, skip), world.occluded_linear(r, tmin, tmax, skip), "{what}: occluded {:?}", skip);
                }
            }
        }
    }

    /// 退化ケース: 空、インスタンスだけ／球だけ、1 個だけ、しきい値の前後。どれも落ちず線形と一致する。
    #[test]
    fn tlas_degenerate_worlds() {
        let r = Ray { o: Vec3::new(0.3, 0.2, 9.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 };
        let empty = World::new();
        assert!(empty.hit(r, 0.0, 1e30).is_none() && !empty.occluded(r, 0.0, 1e30, None));
        for (ni, ns) in [(0, 1), (1, 0), (1, 1), (0, TLAS_MIN_PRIMS - 1), (0, TLAS_MIN_PRIMS), (0, TLAS_MIN_PRIMS + 1), (TLAS_MIN_PRIMS, 0)] {
            let world = random_tlas_world(ni, ns, 3, false);
            let used = world.tlas().is_some();
            assert_eq!(used, ni + ns + 1 >= TLAS_MIN_PRIMS, "({ni},{ns})");
            assert_tlas_hit(&world.hit(r, 0.0, 1e30), &world.hit_linear(r, 0.0, 1e30), "degenerate");
            assert_eq!(world.occluded(r, 0.0, 1e30, None), world.occluded_linear(r, 0.0, 1e30, None));
        }
        // 構築後にジオメトリを足しても古い TLAS を使わない（足した球にも当たる）
        let mut world = random_tlas_world(TLAS_MIN_PRIMS, 0, 5, false);
        let _ = world.hit(r, 0.0, 1e30);
        world.add_sphere(Sphere { c: Vec3::new(0.3, 0.2, 50.0), r: 0.5, mat_id: 0 });
        assert!(world.hit(r, 0.0, 1e30).is_some_and(|h| h.t < 9.0 + 1e-9 || h.mat_id == 0));
        assert_tlas_hit(&world.hit(r, 0.0, 1e30), &world.hit_linear(r, 0.0, 1e30), "after add");
    }

    // ---- アニメーション変換 ----

    fn tri_mesh(k: f64) -> Vec<Triangle> {
        vec![
            Triangle::new_static(Vec3::new(-k, -k, -0.2), Vec3::new(k, -k, 0.2), Vec3::new(0.0, k, 0.0), 0),
            Triangle::new_static(Vec3::new(-k, -k, 0.3), Vec3::new(k, -k, -0.3), Vec3::new(0.0, k, 0.4), 0),
        ]
    }

    fn down_ray(x: f64, y: f64, time: f64) -> Ray {
        Ray { o: Vec3::new(x, y, 20.0), d: Vec3::new(0.0, 0.0, -1.0), time }
    }

    /// `time = 0` で開の位置、`time = 1` で閉の位置、`time = 0.5` でちょうど中間に交差する（平行移動）。
    #[test]
    fn animated_instance_is_at_open_middle_and_close_positions() {
        let mut world = World::new();
        let id = world.add_mesh_instance(tri_mesh(0.5), Transform::translate(Vec3::new(-2.0, 0.0, 0.0)), None);
        assert!(world.set_instance_end_transform(id, Transform::translate(Vec3::new(2.0, 1.0, 0.0))));
        for (time, cx, cy) in [(0.0, -2.0, 0.0), (0.5, 0.0, 0.5), (1.0, 2.0, 1.0)] {
            let h = world.hit(down_ray(cx, cy - 0.2, time), 0.0, 1e30).unwrap_or_else(|| panic!("time {time}: 当たるはず"));
            assert!((h.p.x - cx).abs() < 1e-12 && (h.p.y - (cy - 0.2)).abs() < 1e-12, "time {time}: {:?}", h.p);
            assert!(world.hit(down_ray(cx + 3.0, cy, time), 0.0, 1e30).is_none());
        }
        // 開の位置のままで時刻 1 を撃つと外れる（動いている）
        assert!(world.hit(down_ray(-2.0, -0.2, 1.0), 0.0, 1e30).is_none());
        // 遮蔽判定も同じ時刻の位置で判定する
        let occ = |x: f64, y: f64, time: f64| world.occluded(down_ray(x, y, time), 0.0, 1e30, None);
        assert!(occ(0.0, 0.3, 0.5) && !occ(0.0, 0.3, 0.0));
    }

    /// 掃過ボリュームの保守性: 回転・平行移動・非一様スケールのアニメーション変換に、ランダムなレイ × 時刻を
    /// 2 万本投げ、箱で棄却する通常の経路と、箱の棄却を外した総当たり（箱を巨大にした同じワールド）が
    /// ビット単位で一致する。取りこぼしがあれば保守的でない（TLAS の有無も両方: 24 個 = 閾値超）。
    #[test]
    fn swept_bounds_are_conservative_against_no_culling() {
        for n_inst in [3usize, 24] {
            let rng = std::cell::RefCell::new(Rng::new(31 + n_inst as u64));
            let u = |a: f64, b: f64| a + (b - a) * rng.borrow_mut().next_f64();
            let mut world = World::new();
            for i in 0..n_inst {
                let p0 = Vec3::new(u(-6.0, 6.0), u(-6.0, 6.0), u(-2.0, 2.0));
                let p1 = Vec3::new(u(-6.0, 6.0), u(-6.0, 6.0), u(-2.0, 2.0));
                let ax = Vec3::new(u(-1.0, 1.0), u(-1.0, 1.0), u(0.2, 1.0));
                let s0 = u(0.3, 1.5);
                let start = Transform::translate(p0).compose(Transform::rotate(ax, u(0.0, 360.0))).compose(Transform::scale(Vec3::new(s0, s0 * u(0.5, 2.0), s0)));
                let end = Transform::translate(p1)
                    .compose(Transform::rotate(Vec3::new(u(-1.0, 1.0), u(-1.0, 1.0), u(0.2, 1.0)), u(0.0, 360.0)))
                    .compose(Transform::scale(Vec3::new(u(0.3, 2.5), u(0.3, 2.5), u(0.3, 2.5))));
                let id = world.add_mesh_instance(tri_mesh(1.5), start, Some(i % 3));
                assert!(world.set_instance_end_transform(id, end));
            }
            let mut reference = World::new();
            reference.meshes = world.meshes.iter().map(|m| Mesh::new(m.tris.clone())).collect();
            reference.instances = world.instances.clone();
            let huge = Aabb { min: Vec3::new(-1e9, -1e9, -1e9), max: Vec3::new(1e9, 1e9, 1e9) };
            for inst in &mut reference.instances {
                inst.world_bounds = huge;
            }
            let mut hits = 0;
            for _ in 0..20_000 {
                let o = Vec3::new(u(-14.0, 14.0), u(-14.0, 14.0), u(-14.0, 14.0));
                let d = uniform_sphere_dir(&mut rng.borrow_mut());
                let r = Ray { o, d, time: u(0.0, 1.0) };
                let (a, b) = (world.hit(r, 0.0, 1e30), reference.hit(r, 0.0, 1e30));
                assert_eq!(a.is_some(), b.is_some(), "取りこぼし: t={:?} time={}", b.map(|h| h.t), r.time);
                if let (Some(a), Some(b)) = (a, b) {
                    hits += 1;
                    assert_eq!((a.t.to_bits(), a.inst_id, a.prim_id), (b.t.to_bits(), b.inst_id, b.prim_id));
                }
                assert_eq!(world.occluded(r, 0.0, 30.0, None), reference.occluded(r, 0.0, 30.0, None));
            }
            assert!(hits > 200, "テストが自明でない程度に当たる: {hits}");
        }
    }

    /// 頂点モーションとアニメーション変換の併用: どちらも効く。
    #[test]
    fn vertex_motion_and_animated_transform_combine() {
        let tri = Triangle::new_static(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0);
        let data = MeshData {
            tris: vec![tri],
            vn: Vec::new(),
            tri_vn: Vec::new(),
            uv: Vec::new(),
            tri_uv: Vec::new(),
            // 閉: 頂点が x に +2
            motion: vec![[Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0), Vec3::new(2.0, 1.0, 0.0)]],
        };
        let mut world = World::new();
        let id = world.add_mesh_data_instance(data, Transform::identity(), None);
        // 変換は y に +5
        assert!(world.set_instance_end_transform(id, Transform::translate(Vec3::new(0.0, 5.0, 0.0))));
        for time in [0.0, 0.25, 0.5, 1.0] {
            let (x, y) = (0.2 + 2.0 * time, 0.2 + 5.0 * time);
            assert!(world.hit(down_ray(x, y, time), 0.0, 1e30).is_some(), "time {time}: 両方の動きを足した位置に当たる");
            assert!(world.hit(down_ray(0.2, y, time), 0.0, 1e30).is_none() == (time > 0.0), "頂点モーションだけ無視した位置は外れる");
            assert!(world.hit(down_ray(x, 0.2, time), 0.0, 1e30).is_none() == (time > 0.0), "変換のモーションだけ無視した位置は外れる");
        }
    }

    /// アニメーション変換を持たないインスタンスは、他にアニメーションするインスタンスがあっても、単独のときとビット一致する
    /// （静止は従来の経路: `anim` が `None`）。
    #[test]
    fn static_instance_is_bit_identical_next_to_animated_ones() {
        let xf = Transform::translate(Vec3::new(0.3, 0.1, 0.0)).compose(Transform::rotate(Vec3::new(0.2, 1.0, 0.4), 33.0));
        let mut alone = World::new();
        alone.add_mesh_instance(tri_mesh(1.0), xf, None);
        let mut mixed = World::new();
        mixed.add_mesh_instance(tri_mesh(1.0), xf, None);
        let other = mixed.add_mesh_instance(tri_mesh(1.0), Transform::translate(Vec3::new(5.0, 0.0, 0.0)), None);
        assert!(mixed.set_instance_end_transform(other, Transform::translate(Vec3::new(9.0, 0.0, 0.0))));
        assert!(mixed.instances()[0].anim.is_none());
        let mut rng = Rng::new(4);
        for _ in 0..3000 {
            let r = Ray { o: Vec3::new(rng.next_f64() * 2.0 - 1.0, rng.next_f64() * 2.0 - 1.0, 9.0), d: Vec3::new(0.0, 0.0, -1.0), time: rng.next_f64() };
            let (a, b) = (alone.hit(r, 0.0, 1e30), mixed.hit(r, 0.0, 1e30));
            match (a, b) {
                (Some(a), Some(b)) if b.inst_id == Some(0) => {
                    assert_eq!((a.t.to_bits(), a.p.x.to_bits(), a.p_error.x.to_bits(), a.ng.z.to_bits()), (b.t.to_bits(), b.p.x.to_bits(), b.p_error.x.to_bits(), b.ng.z.to_bits()));
                }
                (Some(_), _) => panic!("静止インスタンスに当たらない"),
                _ => {}
            }
        }
    }


    // ---- ローカル座標（object_space_point）----

    fn local_noise() -> crate::noise::NoiseTexture {
        crate::noise::NoiseTexture {
            pattern: crate::noise::Pattern::Marble, scale: 3.0, octaves: 4, lacunarity: 2.0, gain: 0.5, strength: 4.0,
            color0: Color::new(0.0, 0.0, 0.0), color1: Color::new(1.0, 1.0, 1.0), local: true, offset: Vec3::new(0.0, 0.0, 0.0),
        }
    }

    /// 動く球: 時刻 0 / 0.5 / 1 で、球上の同じ点（中心からの相対位置）のローカル座標は同じで、ノイズの値も同じ
    /// （模様が球に追従する）。ワールド座標で評価すると値が違う。
    #[test]
    fn local_point_follows_a_moving_sphere() {
        let mut world = World::new();
        let idx = world.add_sphere(Sphere { c: Vec3::new(0.0, 0.0, 0.0), r: 0.5, mat_id: 0 });
        world.set_sphere_end(idx, Vec3::new(4.0, 1.0, 0.0));
        let n = local_noise();
        let mut local = Vec::new();
        let mut worldv = Vec::new();
        for time in [0.0, 0.5, 1.0] {
            let c = Vec3::new(4.0, 1.0, 0.0) * time;
            let h = world.hit(ray_at(c + Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0), time), 1e-9, 1e30).expect("hit");
            let p = world.object_space_point(&h, time);
            assert!((p - Vec3::new(0.0, 0.0, 0.5)).len() < 1e-9, "time {time}: {p:?}");
            local.push(n.factor(p));
            worldv.push(n.factor(h.p));
        }
        assert!(local.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9), "{local:?}");
        assert!((worldv[0] - worldv[1]).abs() > 1e-3, "ワールド評価は泳ぐ: {worldv:?}");
    }

    /// アニメーションするインスタンス: 補間した変換の逆を使う（開き姿勢の逆だと time 0.5 / 1 でずれる）。
    #[test]
    fn local_point_follows_an_animated_instance() {
        let tri = Triangle::new_static(Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0);
        let mut world = World::new();
        let start = Transform::translate(Vec3::new(0.0, 0.0, 0.0));
        let end = Transform::translate(Vec3::new(4.0, 1.0, 0.0)).compose(Transform::rotate(Vec3::new(0.0, 1.0, 0.0), 70.0));
        let inst = world.add_mesh_instance(vec![tri], start, None);
        assert!(world.set_instance_end_transform(inst, end));
        let anim = AnimatedTransform::new(start, end).unwrap();
        let local = Vec3::new(0.1, 0.0, 0.0);
        for time in [0.0, 0.3, 0.5, 1.0] {
            let xf = anim.at(time);
            let (pw, nw) = (xf.apply_point(local), xf.apply_normal(Vec3::new(0.0, 0.0, 1.0)));
            let h = world.hit(ray_at(pw + nw * 2.0, -nw, time), 1e-9, 1e30).expect("hit");
            let p = world.object_space_point(&h, time);
            assert!((p - local).len() < 1e-9, "time {time}: {p:?}");
        }
    }

    // ---- 動く球（center_end）----

    fn ray_at(o: Vec3, d: Vec3, time: f64) -> Ray {
        Ray { o, d, time }
    }

    /// time = 0 / 0.5 / 1 で開・中間・閉の位置に当たり、他の位置では外れる。
    #[test]
    fn moving_sphere_hits_at_interpolated_center() {
        let mut world = World::new();
        let idx = world.add_sphere(Sphere { c: Vec3::new(0.0, 0.0, 0.0), r: 0.5, mat_id: 0 });
        assert!(world.set_sphere_end(idx, Vec3::new(4.0, 0.0, 0.0)));
        let down = Vec3::new(0.0, 0.0, -1.0);
        let shoot = |x: f64, time: f64| world.hit(ray_at(Vec3::new(x, 0.0, 5.0), down, time), 1e-9, 1e30).is_some();
        for (time, cx) in [(0.0, 0.0), (0.5, 2.0), (1.0, 4.0)] {
            assert!(shoot(cx, time) && shoot(cx + 0.4, time) && shoot(cx - 0.4, time), "time {time}");
            assert!(!shoot(cx + 0.6, time) && !shoot(cx - 0.6, time), "time {time}");
        }
        // 別の時刻の位置には当たらない
        assert!(!shoot(4.0, 0.0) && !shoot(0.0, 1.0) && !shoot(2.0, 0.0));
        // 非有限の終点は拒否して静止のまま
        assert!(!world.set_sphere_end(idx, Vec3::new(f64::NAN, 0.0, 0.0)));
    }

    /// 掃過ボリュームの保守性: 動く球を含むシーンで、箱（TLAS）で棄却する通常経路と、箱の無い総当たり
    /// （`hit_linear` / `occluded_linear`）が、ランダムなレイ × 時刻でビット単位で一致する。
    /// プリミティブ数は TLAS の閾値（20）の前後、移動距離は半径より大きいものを含む。
    #[test]
    fn moving_sphere_sweep_bounds_are_conservative() {
        let mut rng = Rng::new(2024);
        let mut checked = 0;
        for n in [3usize, 19, 20, 21, 60] {
            let mut world = World::new();
            for i in 0..n {
                let c = Vec3::new(rng.next_f64() * 12.0 - 6.0, rng.next_f64() * 4.0 - 2.0, rng.next_f64() * 12.0 - 6.0);
                let r = 0.2 + rng.next_f64() * 0.6;
                let idx = world.add_sphere(Sphere { c, r, mat_id: 0 });
                if i % 2 == 0 {
                    let d = Vec3::new(rng.next_f64() * 8.0 - 4.0, rng.next_f64() * 4.0 - 2.0, rng.next_f64() * 8.0 - 4.0);
                    world.set_sphere_end(idx, c + d);
                }
            }
            world.set_shutter(0.1, 0.9);
            for _ in 0..2500 {
                let o = Vec3::new(rng.next_f64() * 20.0 - 10.0, rng.next_f64() * 8.0 - 4.0, rng.next_f64() * 20.0 - 10.0);
                let target = Vec3::new(rng.next_f64() * 12.0 - 6.0, rng.next_f64() * 4.0 - 2.0, rng.next_f64() * 12.0 - 6.0);
                let time = 0.1 + 0.8 * rng.next_f64();
                let r = ray_at(o, (target - o).norm(), time);
                assert_same_hit(world.hit(r, 1e-9, 1e30), world.hit_linear(r, 1e-9, 1e30), "sphere sweep");
                let tmax = (target - o).len() * 1.2;
                assert_eq!(world.occluded(r, 1e-9, tmax, None), world.occluded_linear(r, 1e-9, tmax, None), "occluded");
                checked += 1;
            }
        }
        assert!(checked >= 10_000);
    }

    /// `center_end` の無い球は、動く球が同じワールドにあってもビット一致（従来と同じ `Sphere::hit`）。
    /// 開と閉が同じ中心なら、どの時刻でも静止とほぼ一致（補間の丸めだけ）。
    #[test]
    fn static_sphere_is_bit_identical_and_degenerate_move_is_static() {
        let s = Sphere { c: Vec3::new(0.3, -0.2, 0.1), r: 0.9, mat_id: 0 };
        let mut world = World::new();
        let a = world.add_sphere(s);
        let b = world.add_sphere(Sphere { c: Vec3::new(5.0, 0.0, 0.0), r: 1.0, mat_id: 0 });
        world.set_sphere_end(b, Vec3::new(6.0, 1.0, 0.0));
        let c = world.add_sphere(s);
        world.set_sphere_end(c, s.c); // 開 = 閉
        let mut rng = Rng::new(9);
        for _ in 0..500 {
            let o = Vec3::new(rng.next_f64() * 6.0 - 3.0, rng.next_f64() * 6.0 - 3.0, 4.0);
            let r = ray_at(o, (Vec3::new(0.3, -0.2, 0.1) - o + Vec3::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5, 0.0)).norm(), rng.next_f64());
            assert_same_hit(world.sphere_hit(a, r, 1e-9, 1e30), s.hit(r, 1e-9, 1e30), "static");
            match (world.sphere_hit(c, r, 1e-9, 1e30), s.hit(r, 1e-9, 1e30)) {
                (None, None) => {}
                (Some(x), Some(y)) => assert!((x.t - y.t).abs() < 1e-12 && (x.p - y.p).len() < 1e-12),
                (x, y) => panic!("degenerate move differs: {:?} {:?}", x.map(|h| h.t), y.map(|h| h.t)),
            }
        }
    }
}
