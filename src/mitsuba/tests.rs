use super::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::normal_map::MTL_BUMP_K;
    use crate::transform::Transform;
    use crate::ray::Ray;
    use crate::rng::Rng;

    fn cfg() -> RenderConfig {
        let mut c = RenderConfig::default();
        c.width = 16;
        c.height = 9;
        c
    }

    /// referenced_files は入れ子の filename を文書順・base_dir 基準で列挙する。
    #[test]
    fn referenced_files_lists_filenames_in_document_order() {
        let xml = r#"<scene version="3.0.0">
              <shape type="obj"><string name="filename" value="a.obj"/></shape>
              <shape type="sphere"><float name="radius" value="1"/></shape>
              <emitter type="envmap"><string name="filename" value="/abs/env.exr"/></emitter>
            </scene>"#;
        let files = referenced_files(xml, Path::new("dir")).unwrap();
        assert_eq!(files, vec![PathBuf::from("dir/a.obj"), PathBuf::from("/abs/env.exr")]);
    }

    fn load(xml: &str) -> Scene {
        load_scene_from_str(xml, Path::new("."), &cfg(), (None, None)).unwrap().0
    }

    #[test]
    fn parses_shapes_and_materials() {
        let scene = load(
            r#"<scene version="3.0.0">
              <sensor type="perspective">
                <float name="fov" value="40"/>
                <string name="fov_axis" value="y"/>
                <transform name="to_world">
                  <lookat origin="0,1.2,4" target="0,0.5,0" up="0,1,0"/>
                </transform>
              </sensor>
              <shape type="sphere">
                <point name="center" x="-1" y="0.5" z="0"/>
                <float name="radius" value="0.5"/>
                <bsdf type="diffuse"><srgb name="reflectance" value="0.8,0.3,0.3"/></bsdf>
              </shape>
              <shape type="sphere">
                <point name="center" x="0" y="0.5" z="0"/>
                <float name="radius" value="0.5"/>
                <bsdf type="roughconductor">
                  <string name="distribution" value="ggx"/>
                  <float name="alpha" value="0.25"/>
                  <rgb name="specular_reflectance" value="0.9,0.7,0.3"/>
                </bsdf>
              </shape>
              <shape type="sphere">
                <point name="center" x="0" y="3" z="-1"/>
                <float name="radius" value="0.8"/>
                <emitter type="area"><rgb name="radiance" value="8,7,5"/></emitter>
              </shape>
            </scene>"#,
        );

        assert_eq!(scene.world.spheres().len(), 3);
        assert_eq!(scene.shaders.shaders.len(), 3);
        assert!(matches!(scene.shaders.shaders[0].base, Material::Lambert { .. }));
        assert!(matches!(scene.shaders.shaders[1].base, Material::Ggx { alpha, .. } if (alpha - 0.25).abs() < 1e-12));
        assert!(matches!(scene.shaders.shaders[2].base, Material::DiffuseLight { .. }));
        // area emitter は build_lights でライトとして登録される
        assert_eq!(scene.world.lights().len(), 1);
    }

    #[test]
    fn srgb_tag_gamma_decodes_but_rgb_is_linear() {
        let scene = load(
            r#"<scene version="3.0.0">
              <shape type="sphere">
                <bsdf type="diffuse"><srgb name="reflectance" value="0.8,0.8,0.8"/></bsdf>
              </shape>
              <shape type="sphere">
                <bsdf type="diffuse"><rgb name="reflectance" value="0.8,0.8,0.8"/></bsdf>
              </shape>
            </scene>"#,
        );
        let srgb = match scene.shaders.shaders[0].base { Material::Lambert { albedo, .. } => albedo, _ => panic!() };
        let lin = match scene.shaders.shaders[1].base { Material::Lambert { albedo, .. } => albedo, _ => panic!() };
        // srgb 0.8 はガンマ展開で約 0.603、rgb 0.8 はそのまま 0.8
        assert!((srgb.r() - Color::from_srgb(0.8, 0.8, 0.8).r()).abs() < 1e-12);
        assert!((lin.r() - 0.8).abs() < 1e-12);
        assert!(srgb.r() < lin.r());
    }

    #[test]
    fn reads_film_sampler_integrator_into_config() {
        let (_scene, settings) = load_scene_from_str(
            r#"<scene version="3.0.0">
              <integrator type="path">
                <integer name="max_depth" value="12"/>
                <integer name="rr_depth" value="5"/>
              </integrator>
              <sampler type="independent"><integer name="sample_count" value="256"/></sampler>
              <film type="hdrfilm">
                <integer name="width" value="800"/>
                <integer name="height" value="600"/>
              </film>
              <shape type="sphere"><bsdf type="diffuse"/></shape>
            </scene>"#,
            Path::new("."),
            &cfg(),
            (None, None),
        )
        .unwrap();
        let mut config = cfg();
        settings.apply(&mut config);
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
        assert_eq!(config.spp, 256);
        assert_eq!(config.max_depth, 12);
        assert_eq!(config.rr_depth, 5);
    }

    /// `<rfilter>` は `<film>` の子。既定は box（settings.filter は None のまま）。box/tent/gaussian/mitchell を読み、
    /// 未知の型・不正なパラメータは警告して box のまま（settings.filter は None）。
    #[test]
    fn rfilter_reads_each_type_and_warns_on_bad_input() {
        let film = |inner: &str| format!(
            r#"<scene version="3.0.0"><sensor type="perspective"><float name="fov" value="40"/></sensor>
                 <film type="hdrfilm">{inner}</film>
                 <shape type="sphere"><bsdf type="diffuse"/></shape></scene>"#
        );
        let load = |xml: &str| capture_warnings(|| load_scene_from_str(xml, Path::new("."), &cfg(), (None, None)).unwrap().1);

        let (settings, w) = load(&film(""));
        assert!(w.is_empty());
        assert_eq!(settings.filter, None, "no <rfilter>: box stays the default");

        let (settings, w) = load(&film(r#"<rfilter type="box"/>"#));
        assert!(w.is_empty());
        assert_eq!(settings.filter, Some(PixelFilter::Box));

        let (settings, w) = load(&film(r#"<rfilter type="tent"/>"#));
        assert!(w.is_empty());
        assert_eq!(settings.filter, Some(PixelFilter::Tent));

        let (settings, w) = load(&film(r#"<rfilter type="gaussian"/>"#));
        assert!(w.is_empty());
        assert_eq!(settings.filter, Some(PixelFilter::Gaussian { stddev: 0.5 }));

        let (settings, w) = load(&film(r#"<rfilter type="gaussian"><float name="stddev" value="0.8"/></rfilter>"#));
        assert!(w.is_empty());
        assert_eq!(settings.filter, Some(PixelFilter::Gaussian { stddev: 0.8 }));

        let (settings, w) = load(&film(r#"<rfilter type="mitchell"/>"#));
        assert!(w.is_empty());
        assert_eq!(settings.filter, Some(PixelFilter::Mitchell { b: 1.0 / 3.0, c: 1.0 / 3.0 }));

        let (settings, w) = load(&film(r#"<rfilter type="mitchell"><float name="B" value="0.5"/><float name="C" value="0.25"/></rfilter>"#));
        assert!(w.is_empty());
        assert_eq!(settings.filter, Some(PixelFilter::Mitchell { b: 0.5, c: 0.25 }));

        for bad in [
            r#"<rfilter type="lanczos"/>"#,
            r#"<rfilter type="gaussian"><float name="stddev" value="-1"/></rfilter>"#,
            r#"<rfilter type="gaussian"><float name="stddev" value="nan"/></rfilter>"#,
            r#"<rfilter type="mitchell"><float name="B" value="inf"/></rfilter>"#,
        ] {
            let (settings, w) = load(&film(bad));
            assert!(!w.is_empty(), "{bad}");
            assert_eq!(settings.filter, None, "{bad}: falls back to box");
        }
    }

    /// CLI の `--filter` はシーンファイルの `<rfilter>` より優先する（他の CLI 上書きと同じ規則）。
    #[test]
    fn cli_filter_overrides_the_scene_rfilter() {
        use crate::cli::{load_with_overrides, parse_args};
        let dir = motion_dir("filter_cli");
        let xml = format!(
            r#"<scene version="3.0.0"><sensor type="perspective"><float name="fov" value="40"/></sensor>
                 <film type="hdrfilm"><rfilter type="gaussian"/></film>{}</scene>"#,
            obj_shape("a.obj", 0.0, DIFFUSE_A, "")
        );
        std::fs::write(dir.join("s.xml"), &xml).unwrap();
        let mut config = cfg();
        config.scene_path = Some(dir.join("s.xml").to_string_lossy().into_owned());
        let (overrides, w) = parse_args(["--filter".to_string(), "mitchell".to_string()], &mut config);
        assert!(w.is_empty());
        load_with_overrides(&mut config, &overrides).unwrap();
        assert_eq!(config.filter, PixelFilter::Mitchell { b: 1.0 / 3.0, c: 1.0 / 3.0 });
        std::fs::remove_dir_all(&dir).ok();
    }

    /// max_depth = -1 は無制限、0 はそのまま（何も描かない）、それ以外の負値と rr_depth < 1 は無視。
    #[test]
    fn integrator_depth_values_follow_mitsuba() {
        let settings_for = |max_depth: &str, rr_depth: &str| {
            let xml = format!(
                r#"<scene version="3.0.0"><integrator type="path"><integer name="max_depth" value="{}"/><integer name="rr_depth" value="{}"/></integrator>
                   <shape type="sphere"><bsdf type="diffuse"/></shape></scene>"#,
                max_depth, rr_depth
            );
            load_scene_from_str(&xml, Path::new("."), &cfg(), (None, None)).unwrap().1
        };
        let s = settings_for("-1", "5");
        assert_eq!(s.max_depth, Some(usize::MAX));
        assert_eq!(s.rr_depth, Some(5));
        assert_eq!(settings_for("0", "1").max_depth, Some(0));
        assert_eq!(settings_for("1", "1").max_depth, Some(1));
        let s = settings_for("-2", "0");
        assert_eq!(s.max_depth, None);
        assert_eq!(s.rr_depth, None);
    }

    #[test]
    fn parses_parametric_shapes() {
        let scene = load(
            r#"<scene version="3.0.0">
              <shape type="rectangle"><bsdf type="diffuse"/></shape>
              <shape type="cube"><bsdf type="diffuse"/></shape>
              <shape type="disk"><bsdf type="diffuse"/></shape>
            </scene>"#,
        );
        assert_eq!(scene.world.meshes().len(), 3);
        assert_eq!(scene.world.instances().len(), 3);
        assert_eq!(scene.world.meshes()[0].tris.len(), 2); // rectangle
        assert_eq!(scene.world.meshes()[1].tris.len(), 12); // cube
        assert_eq!(scene.world.meshes()[2].tris.len(), 64); // disk
    }


    /// `face_normals` の値が `true` / `false` 以外なら警告して既定（補間）にフォールバックする。
    /// 黙って既定値にすると `value="1"` が逆の意味に解釈される。
    #[test]
    fn invalid_face_normals_value_warns_and_falls_back() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static C: AtomicUsize = AtomicUsize::new(0);
        let n = C.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let obj = dir.join(format!("tinypt_fnbad_{}_{}.obj", std::process::id(), n));
        std::fs::write(&obj, "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nvn 0 1 0\nvn 1 0 0\nf 1//1 2//2 3//3\n").unwrap();
        let objname = obj.file_name().unwrap().to_string_lossy().into_owned();
        let load = |flag: &str| {
            let xml = format!(
                r#"<scene version="3.0.0">
                  <shape type="obj">
                    <string name="filename" value="{}"/>{}
                    <bsdf type="diffuse"><rgb name="reflectance" value="0.5,0.5,0.5"/></bsdf>
                  </shape>
                </scene>"#,
                objname, flag
            );
            let (r, warnings) = capture_warnings(|| load_scene_from_str(&xml, &dir, &cfg(), (None, None)).unwrap().0);
            (r, warnings)
        };
        for bad in [r#"<boolean name="face_normals" value="1"/>"#,
                    r#"<boolean name="face_normals" value="yes"/>"#,
                    r#"<boolean name="face_normals" value="TRUE"/>"#,
                    r#"<boolean name="face_normals"/>"#] {
            let (scene, warnings) = load(bad);
            assert!(scene.world.meshes()[0].is_smooth(), "{}: 既定（補間）にフォールバックする", bad);
            assert!(
                warnings.iter().any(|w| w.contains("face_normals")),
                "{}: 警告が出ていない（warnings = {:?}）", bad, warnings
            );
        }
        // 正しい値では警告を出さない
        for good in ["", r#"<boolean name="face_normals" value="true"/>"#, r#"<boolean name="face_normals" value="false"/>"#] {
            let (_, warnings) = load(good);
            assert!(!warnings.iter().any(|w| w.contains("face_normals")), "{}: 余計な警告", good);
        }
        std::fs::remove_file(&obj).ok();
    }

    // ---- テクスチャ（T1） ----

    /// テスト用の PNG を書いて渡す。
    fn with_png<T>(w: u32, h: u32, px: &[[u8; 3]], f: impl FnOnce(&std::path::Path, &str) -> T) -> T {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static C: AtomicUsize = AtomicUsize::new(0);
        let n = C.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let name = format!("tinypt_mtex_{}_{}.png", std::process::id(), n);
        let path = dir.join(&name);
        let mut img = image::RgbImage::new(w, h);
        for (i, p) in px.iter().enumerate() {
            img.put_pixel((i as u32) % w, (i as u32) / w, image::Rgb(*p));
        }
        img.save(&path).unwrap();
        let out = f(&dir, &name);
        std::fs::remove_file(&path).ok();
        out
    }

    /// `<texture type="bitmap">` を `diffuse` の `reflectance` に指定でき、
    /// シーンのテクスチャ置き場に積まれてマテリアルが添字で参照する。
    #[test]
    fn parses_bitmap_texture_on_diffuse_reflectance() {
        let scene = with_png(1, 1, &[[255, 0, 0]], |dir, name| {
            let xml = format!(
                r#"<scene version="3.0.0">
                  <shape type="rectangle">
                    <bsdf type="diffuse">
                      <texture type="bitmap" name="reflectance">
                        <string name="filename" value="{}"/>
                      </texture>
                    </bsdf>
                  </shape>
                </scene>"#,
                name
            );
            load_scene_from_str(&xml, dir, &cfg(), (None, None)).unwrap().0
        });
        assert_eq!(scene.shaders.textures.len(), 1, "テクスチャが積まれていない");
        assert!(scene.shaders.shaders[0].albedo.is_some(), "アルベドの式（Mul(Const, Texture)）が付いていない");
        match scene.shaders.shaders[0].base {
            Material::Lambert { albedo } => {
                // テクスチャがある場合、定数側は倍率なので白（1 倍）
                assert!((albedo.r() - 1.0).abs() < 1e-12, "既定の倍率は白のはず: {}", albedo.r());
                // 赤 255 は sRGB デコードでリニア 1.0
                let c = scene.shaders.textures[0].sample((0.5, 0.5));
                assert!((c.r() - 1.0).abs() < 1e-9 && c.g().abs() < 1e-12, "{:?}", (c.r(), c.g(), c.b()));
            }
            _ => panic!("diffuse がテクスチャ付き Lambert になっていない"),
        }
    }

    /// テクスチャと定数色を両方書くと、定数色は**倍率**として掛かる。
    #[test]
    fn constant_reflectance_scales_the_texture() {
        let scene = with_png(1, 1, &[[255, 255, 255]], |dir, name| {
            let xml = format!(
                r#"<scene version="3.0.0">
                  <shape type="rectangle">
                    <bsdf type="diffuse">
                      <rgb name="reflectance" value="0.25, 0.5, 1.0"/>
                      <texture type="bitmap" name="reflectance">
                        <string name="filename" value="{}"/>
                      </texture>
                    </bsdf>
                  </shape>
                </scene>"#,
                name
            );
            load_scene_from_str(&xml, dir, &cfg(), (None, None)).unwrap().0
        });
        let resolved = scene.shaders.eval_at(0, &scene.world, Vec3::new(0.0, 0.0, 0.0), (0.5, 0.5));
        match resolved {
            Material::Lambert { albedo } => {
                assert!((albedo.r() - 0.25).abs() < 1e-9 && (albedo.b() - 1.0).abs() < 1e-9,
                        "倍率が掛かっていない: {:?}", (albedo.r(), albedo.g(), albedo.b()));
            }
            _ => panic!("resolve_textures がテクスチャを畳み込んでいない"),
        }
    }

    /// 読み込めないテクスチャは警告して定数色にフォールバックする（描画は続く）。
    #[test]
    fn missing_texture_warns_and_falls_back_to_a_constant() {
        let dir = std::env::temp_dir();
        let xml = r#"<scene version="3.0.0">
              <shape type="rectangle">
                <bsdf type="diffuse">
                  <texture type="bitmap" name="reflectance">
                    <string name="filename" value="definitely_not_here_12345.png"/>
                  </texture>
                </bsdf>
              </shape>
            </scene>"#;
        let (scene, warnings) = capture_warnings(|| load_scene_from_str(xml, &dir, &cfg(), (None, None)).unwrap().0);
        assert!(scene.shaders.textures.is_empty());
        assert!(matches!(scene.shaders.shaders[0].base, Material::Lambert { .. }));
        assert!(warnings.iter().any(|w| w.contains("failed to load texture")), "警告が出ていない: {:?}", warnings);
    }

    /// 未知のテクスチャ型・`wrap_mode` は警告する。
    #[test]
    fn unsupported_texture_type_and_wrap_mode_warn() {
        let (_, warnings) = with_png(1, 1, &[[10, 20, 30]], |dir, name| {
            let xml = format!(
                r#"<scene version="3.0.0">
                  <shape type="rectangle">
                    <bsdf type="diffuse">
                      <texture type="checkerboard" name="reflectance"/>
                    </bsdf>
                  </shape>
                  <shape type="rectangle">
                    <bsdf type="diffuse">
                      <texture type="bitmap" name="reflectance">
                        <string name="filename" value="{}"/>
                        <string name="wrap_mode" value="mirror"/>
                      </texture>
                    </bsdf>
                  </shape>
                </scene>"#,
                name
            );
            capture_warnings(|| load_scene_from_str(&xml, dir, &cfg(), (None, None)).unwrap().0)
        });
        assert!(warnings.iter().any(|w| w.contains("unsupported texture type")), "{:?}", warnings);
        assert!(warnings.iter().any(|w| w.contains("wrap_mode")), "{:?}", warnings);
    }

    /// `raw="true"` は sRGB デコードを掛けない（データテクスチャ）。
    #[test]
    fn raw_texture_is_linear() {
        let scene = with_png(1, 1, &[[128, 128, 128]], |dir, name| {
            let xml = format!(
                r#"<scene version="3.0.0">
                  <shape type="rectangle">
                    <bsdf type="diffuse">
                      <texture type="bitmap" name="reflectance">
                        <string name="filename" value="{}"/>
                        <boolean name="raw" value="true"/>
                      </texture>
                    </bsdf>
                  </shape>
                </scene>"#,
                name
            );
            load_scene_from_str(&xml, dir, &cfg(), (None, None)).unwrap().0
        });
        let c = scene.shaders.textures[0].sample((0.5, 0.5));
        assert!((c.r() - 128.0 / 255.0).abs() < 1e-12, "raw なのに sRGB デコードされている: {}", c.r());
    }

    /// パラメトリック形状には Mitsuba 準拠の UV が付く。
    #[test]
    fn parametric_shapes_get_uv() {
        let scene = load(
            r#"<scene version="3.0.0">
              <shape type="rectangle"><bsdf type="diffuse"/></shape>
              <shape type="cube"><bsdf type="diffuse"/></shape>
              <shape type="disk"><bsdf type="diffuse"/></shape>
            </scene>"#,
        );
        for (i, name) in ["rectangle", "cube", "disk"].iter().enumerate() {
            assert!(scene.world.meshes()[i].has_uv(), "{} に UV が無い", name);
        }
        // 値の確認は形状ごとに単独のシーンで行う（3 つを重ねると手前の立方体に当たってしまう）
        let only = |shape: &str| {
            load(&format!(r#"<scene version="3.0.0"><shape type="{}"><bsdf type="diffuse"/></shape></scene>"#, shape))
        };
        let shoot = |sc: &crate::scene::Scene, x: f64, y: f64| {
            sc.world
                .hit(Ray { o: Vec3::new(x, y, 3.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30)
                .expect("当たらない")
                .uv
        };

        // rectangle: uv = ((x+1)/2, (y+1)/2)
        let rect = only("rectangle");
        let uv = shoot(&rect, 0.0, 0.0);
        assert!((uv.0 - 0.5).abs() < 1e-12 && (uv.1 - 0.5).abs() < 1e-12, "rectangle 中心: {:?}", uv);
        let uv = shoot(&rect, 0.5, -0.5);
        assert!((uv.0 - 0.75).abs() < 1e-12 && (uv.1 - 0.25).abs() < 1e-12, "rectangle (0.5,-0.5): {:?}", uv);

        // cube: 面ごとに [0,1]²。+z 面の中心は (0.5, 0.5)
        let cube = only("cube");
        let uv = shoot(&cube, 0.0, 0.0);
        assert!((uv.0 - 0.5).abs() < 1e-12 && (uv.1 - 0.5).abs() < 1e-12, "cube +z 面の中心: {:?}", uv);
        let mut rng = Rng::new(3);
        for _ in 0..200 {
            let (x, y) = (rng.next_f64() * 1.8 - 0.9, rng.next_f64() * 1.8 - 0.9);
            let uv = shoot(&cube, x, y);
            assert!((0.0..=1.0).contains(&uv.0) && (0.0..=1.0).contains(&uv.1), "cube の UV が範囲外: {:?}", uv);
        }

        // disk: u = 半径 r、v = 角度 φ/2π
        let disk = only("disk");
        let uv = shoot(&disk, 0.0, 0.0);
        assert!(uv.0.abs() < 1e-9, "disk の中心は u = 0（r = 0）: {:?}", uv);
        for r in [0.25, 0.5, 0.9] {
            let uv = shoot(&disk, r, 0.0);
            assert!((uv.0 - r).abs() < 0.02, "disk の u は半径: r = {}, uv = {:?}", r, uv);
        }
    }

    /// OBJ の頂点法線は既定で使われ、`<boolean name="face_normals" value="true"/>` で捨てられる。
    /// パラメトリック形状（rectangle など）は元から頂点法線を持たないので、この指定で何も変わらない。
    #[test]
    fn face_normals_flag_controls_vertex_normal_use() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static C: AtomicUsize = AtomicUsize::new(0);
        let n = C.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let obj = dir.join(format!("tinypt_fn_{}_{}.obj", std::process::id(), n));
        std::fs::write(&obj, "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nvn 0 1 0\nvn 1 0 0\nf 1//1 2//2 3//3\n").unwrap();
        let objname = obj.file_name().unwrap().to_string_lossy().into_owned();
        let scene_of = |flag: &str| {
            let xml = format!(
                r#"<scene version="3.0.0">
                  <shape type="obj">
                    <string name="filename" value="{}"/>{}
                    <bsdf type="diffuse"><rgb name="reflectance" value="0.5,0.5,0.5"/></bsdf>
                  </shape>
                  <shape type="rectangle">{}<bsdf type="diffuse"/></shape>
                </scene>"#,
                objname, flag, flag
            );
            load_scene_from_str(&xml, &dir, &cfg(), (None, None)).unwrap().0
        };
        let smooth = scene_of("");
        let flat = scene_of(r#"<boolean name="face_normals" value="true"/>"#);
        let explicit_false = scene_of(r#"<boolean name="face_normals" value="false"/>"#);
        std::fs::remove_file(&obj).ok();

        assert!(smooth.world.meshes()[0].is_smooth(), "既定では頂点法線を使う");
        assert!(explicit_false.world.meshes()[0].is_smooth(), "false は既定と同じ");
        assert!(!flat.world.meshes()[0].is_smooth(), "face_normals=true で頂点法線を捨てる");
        // rectangle は元から頂点法線を持たないので、どちらでも面法線
        assert!(!smooth.world.meshes()[1].is_smooth());
        assert!(!flat.world.meshes()[1].is_smooth());
    }

    #[test]
    fn parses_obj_mesh_with_transform() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static C: AtomicUsize = AtomicUsize::new(0);
        let n = C.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let obj = dir.join(format!("tinypt_m2_{}_{}.obj", std::process::id(), n));
        std::fs::write(&obj, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
        let objname = obj.file_name().unwrap().to_string_lossy().into_owned();
        // XML 自体はファイル不要（load_scene_from_str）。obj だけは obj_loader が
        // ファイルパスを要求するため実ファイルとして書き出す。
        let xml = format!(
            r#"<scene version="3.0.0">
              <shape type="obj">
                <string name="filename" value="{}"/>
                <transform name="to_world"><translate x="10" y="0" z="0"/></transform>
                <bsdf type="diffuse"><rgb name="reflectance" value="0.5,0.5,0.5"/></bsdf>
              </shape>
            </scene>"#,
            objname
        );
        let scene = load_scene_from_str(&xml, &dir, &cfg(), (None, None)).unwrap().0;
        std::fs::remove_file(&obj).ok();

        assert_eq!(scene.world.meshes().len(), 1);
        assert_eq!(scene.world.instances().len(), 1);
        assert_eq!(scene.world.meshes()[0].tris.len(), 1);
        assert_eq!(scene.shaders.shaders.len(), 1);
        // 頂点 v0=(0,0,0) は translate(10,0,0) でワールド (10,0,0) になる
        let inst = scene.world.instances()[0];
        let p = inst.xform.apply_point(Vec3::new(0.0, 0.0, 0.0));
        assert!((p - Vec3::new(10.0, 0.0, 0.0)).len() < 1e-9);
    }

    #[test]
    fn constant_emitter_sets_uniform_env() {
        let scene = load(
            r#"<scene version="3.0.0">
              <emitter type="constant"><rgb name="radiance" value="0.1,0.2,0.4"/></emitter>
              <shape type="sphere"><bsdf type="diffuse"/></shape>
            </scene>"#,
        );
        let env = scene.env.expect("constant emitter should set env");
        let a = env.sample(Vec3::new(0.0, 1.0, 0.0));
        let b = env.sample(Vec3::new(1.0, 0.0, 0.0));
        // 定数なので方向によらず同じ放射輝度
        assert!((a.r() - 0.1).abs() < 1e-9 && (a.g() - 0.2).abs() < 1e-9 && (a.b() - 0.4).abs() < 1e-9);
        assert!((b.r() - 0.1).abs() < 1e-9 && (b.b() - 0.4).abs() < 1e-9);
    }

    #[test]
    fn no_env_emitter_defaults_to_black_background() {
        let scene = load(
            r#"<scene version="3.0.0">
              <shape type="sphere"><bsdf type="diffuse"/></shape>
            </scene>"#,
        );
        // Mitsuba 準拠で背景は黒（sky() フォールバックは使わない）
        let env = scene.env.expect("env should default to a black constant");
        let c = env.sample(Vec3::new(0.3, 0.8, 0.1));
        assert!(c.r() == 0.0 && c.g() == 0.0 && c.b() == 0.0);
    }

    #[test]
    fn emitter_scale_multiplies_radiance() {
        let scene = load(
            r#"<scene version="3.0.0">
              <emitter type="constant">
                <rgb name="radiance" value="1,1,1"/>
                <float name="scale" value="3"/>
              </emitter>
            </scene>"#,
        );
        let c = scene.env.unwrap().sample(Vec3::new(0.0, 1.0, 0.0));
        assert!((c.r() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn envmap_file_loads() {
        let exr = concat!(env!("CARGO_MANIFEST_DIR"), "/sample/env.exr");
        let xml = format!(
            r#"<scene version="3.0.0">
              <emitter type="envmap"><string name="filename" value="{}"/></emitter>
              <shape type="sphere"><bsdf type="diffuse"/></shape>
            </scene>"#,
            exr
        );
        let scene = load(&xml);
        let env = scene.env.expect("envmap file should load");
        assert!(env.width > 1 && env.height > 1);
    }

    #[test]
    fn unknown_bsdf_falls_back_to_diffuse() {
        let scene = load(
            r#"<scene version="3.0.0">
              <integrator type="path"><integer name="max_depth" value="8"/></integrator>
              <shape type="sphere">
                <bsdf type="plastic"><rgb name="reflectance" value="0.2,0.4,0.6"/></bsdf>
              </shape>
            </scene>"#,
        );
        assert_eq!(scene.shaders.shaders.len(), 1);
        assert!(matches!(scene.shaders.shaders[0].base, Material::Lambert { .. }));
    }

    // ---- MTL / usemtl（T2） ----

    /// 一時ディレクトリに OBJ + MTL + テクスチャ 1 枚を作る（MTL はタブ字下げ・`\` パス・実データ風）。
    /// 面は 8 枚（A×4, B×3, 未定義 C×1、先頭 1 枚は `usemtl` 前）で、警告が面ごとに出ないことも見られる。
    fn with_mtl_dir<T>(f: impl FnOnce(&Path) -> T) -> T {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static C: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!("tinypt_mtl_{}_{}", std::process::id(), C.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(dir.join("textures")).unwrap();
        let mut img = image::RgbImage::new(1, 1);
        img.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        img.save(dir.join("textures/a.png")).unwrap();
        let mut obj = String::from("mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
        for (name, n) in [("A", 4), ("B", 3), ("C", 1)] {
            obj.push_str(&format!("usemtl {}\n", name));
            for _ in 0..n {
                obj.push_str("f 1 2 3\n");
            }
        }
        std::fs::write(dir.join("m.obj"), obj).unwrap();
        std::fs::write(
            dir.join("m.mtl"),
            "newmtl A\n\tKd 1 1 1\n\tKs 0 0 0\n\tmap_Kd textures\\a.png\n\tmap_Ka textures\\a.png\n\tmap_d textures\\a.png\n\
             newmtl B\n\tKd 0.5 0.5 0.5\n\tKs 0.9 0.9 0.9\n\tNs 100\n\tmap_Kd textures\\a.png\n\td 0.5\n",
        )
        .unwrap();
        let out = f(&dir);
        std::fs::remove_dir_all(&dir).ok();
        out
    }

    fn obj_scene_xml(extra: &str) -> String {
        format!(
            r#"<scene version="3.0.0"><shape type="obj"><string name="filename" value="m.obj"/>{}</shape></scene>"#,
            extra
        )
    }

    /// `usemtl` ごとに材質が作られ、1 メッシュのまま三角形ごとの `mat_id` で引く。
    /// `usemtl` 前の面は既定、MTL に無い名前も既定。同じ画像は 1 度しか読まない。
    #[test]
    fn mtl_materials_resolve_per_triangle_and_textures_are_cached() {
        with_mtl_dir(|dir| {
            let ((scene, _), warnings) = capture_warnings(|| {
                load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap()
            });
            // 面に使われた名前だけ: 既定("")・A・B・C
            assert_eq!(scene.shaders.shaders.len(), 4);
            assert_eq!(scene.world.meshes().len(), 1, "usemtl でメッシュを割ってはいけない");
            let ids: Vec<usize> = scene.world.meshes()[0].tris.iter().map(|t| t.mat_id).collect();
            assert_eq!(ids, vec![0, 1, 1, 1, 1, 2, 2, 2, 3]);
            assert!(scene.world.instances()[0].mat_override.is_none());
            // A: テクスチャ付き Lambert。B は明るい Ks かつ Ns>1 だが map_Kd を持つので、
            // テクスチャ付き Lambert が優先される（map_Kd があれば Ks/Ns は使わない。不具合修正）。
            // C・既定: 灰色 Lambert
            assert!(matches!(scene.shaders.shaders[1].base, Material::Lambert { .. }) && scene.shaders.shaders[1].albedo.is_some());
            assert!(matches!(scene.shaders.shaders[2].base, Material::Lambert { .. }) && scene.shaders.shaders[2].albedo.is_some(), "map_Kd がある B は Lambert のはず");
            assert!(matches!(scene.shaders.shaders[3].base, Material::Lambert { .. }) && scene.shaders.shaders[3].albedo.is_none());
            // 同じ a.png を A の map_Kd / map_Ka / map_d と B の map_Kd が指しているが、読むのは 1 回（キャッシュ）
            assert_eq!(scene.shaders.textures.len(), 1);
            // 警告: map_Ka はマテリアル A に 1 回だけ。定数 d<1 はシーンで 1 回だけ。面ごとには出ない
            let n = |pat: &str| warnings.iter().filter(|w| w.contains(pat)).count();
            assert_eq!(n("map_Ka"), 1, "{:?}", warnings);
            assert_eq!(n("constant d < 1"), 1, "{:?}", warnings);
            assert_eq!(n("failed to load alpha mask"), 0, "{:?}", warnings);
            assert_eq!(n("'C' not found"), 1, "{:?}", warnings);
            // 発光マテリアルを含まない → ライト CDF は空
            assert!(scene.world.lights().is_empty());
        });
    }

    /// 2 つの材質が同じ画像を指しても 1 回しか読まない（キャッシュ）。
    #[test]
    fn same_texture_path_in_two_materials_is_loaded_once() {
        with_mtl_dir(|dir| {
            std::fs::write(
                dir.join("m.mtl"),
                "newmtl A\n\tKd 1 1 1\n\tmap_Kd textures\\a.png\nnewmtl B\n\tKd 1 1 1\n\tmap_Kd textures/a.png\n",
            )
            .unwrap();
            let scene = load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0;
            assert_eq!(scene.shaders.textures.len(), 1);
            // A / B ともテクスチャ付き Lambert で、同じ画像を指す（式 `Mul(Const, Texture(id))` の id が同じ）
            let tex_of = |m: usize| {
                let sh = &scene.shaders.shaders[m];
                assert!(matches!(sh.base, Material::Lambert { .. }));
                match scene.shaders.value(sh.albedo.expect("式がある")) {
                    crate::shader::ValueNode::Mul(_, r) => match scene.shaders.value(r) {
                        crate::shader::ValueNode::Texture(id) => id,
                        other => panic!("{other:?}"),
                    },
                    other => panic!("{other:?}"),
                }
            };
            assert_eq!(tex_of(1), tex_of(2));
        });
    }

    /// `<bsdf>` 指定があれば MTL は読まず、従来どおり 1 材質で全体を覆う。
    #[test]
    fn bsdf_child_overrides_mtl() {
        with_mtl_dir(|dir| {
            let xml = obj_scene_xml(r#"<bsdf type="diffuse"><rgb name="reflectance" value="0.2,0.3,0.4"/></bsdf>"#);
            let scene = load_scene_from_str(&xml, dir, &cfg(), (None, None)).unwrap().0;
            assert_eq!(scene.shaders.shaders.len(), 1);
            assert!(scene.shaders.textures.is_empty(), "MTL のテクスチャを読んではいけない");
            assert!(scene.world.meshes()[0].tris.iter().all(|t| t.mat_id == 0));
        });
    }

    /// `use_mtl=false` で MTL を無視し（`<bsdf>` 無しなら従来の既定拡散 + 警告）。
    #[test]
    fn use_mtl_false_ignores_mtl() {
        with_mtl_dir(|dir| {
            let xml = obj_scene_xml(r#"<boolean name="use_mtl" value="false"/>"#);
            let (scene, warnings) = capture_warnings(|| load_scene_from_str(&xml, dir, &cfg(), (None, None)).unwrap().0);
            assert_eq!(scene.shaders.shaders.len(), 1);
            assert!(scene.shaders.textures.is_empty());
            assert!(warnings.iter().any(|w| w.contains("without bsdf")), "{:?}", warnings);
        });
    }

    /// MTL ファイルが無い / 空でも落ちず、既定の灰色拡散になる。
    #[test]
    fn missing_or_empty_mtl_falls_back_to_default() {
        with_mtl_dir(|dir| {
            std::fs::remove_file(dir.join("m.mtl")).unwrap();
            let (scene, warnings) = capture_warnings(|| {
                load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0
            });
            assert_eq!(scene.shaders.shaders.len(), 4);
            assert!(scene.shaders.shaders.iter().map(|sh| &sh.base).all(|m| matches!(m, Material::Lambert { .. })));
            assert!(warnings.iter().any(|w| w.contains("failed to read mtl")), "{:?}", warnings);

            std::fs::write(dir.join("m.mtl"), "").unwrap();
            let scene = load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0;
            assert_eq!(scene.shaders.shaders.len(), 4);
        });
    }

    /// `map_d` を持つ材質の三角形だけがアルファ付きになり、`<bsdf>` 上書きや map_d 無しでは付かない。
    /// 同じマスク画像は 1 度しか読まない（Arc を共有）。
    #[test]
    fn map_d_attaches_alpha_only_to_masked_material_triangles() {
        with_mtl_dir(|dir| {
            // UV を持つ OBJ にする
            let mut obj = String::from("mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\n");
            obj.push_str("usemtl A\nf 1/1 2/2 3/3\nusemtl B\nf 1/1 2/2 3/3\n");
            std::fs::write(dir.join("m.obj"), obj).unwrap();
            std::fs::write(dir.join("m.mtl"), "newmtl A\n\tKd 1 1 1\n\tmap_d textures\\a.png\nnewmtl B\n\tKd 1 1 1\n").unwrap();
            let (scene, warnings) = capture_warnings(|| load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0);
            assert!(scene.world.meshes()[0].has_alpha());
            assert!(!warnings.iter().any(|w| w.contains("alpha")), "{:?}", warnings);
            let over = obj_scene_xml(r#"<bsdf type="diffuse"/>"#);
            let scene = load_scene_from_str(&over, dir, &cfg(), (None, None)).unwrap().0;
            assert!(!scene.world.meshes()[0].has_alpha());
            // map_d 無しの材質だけなら付かない
            std::fs::write(dir.join("m.mtl"), "newmtl A\n\tKd 1 1 1\nnewmtl B\n\tKd 1 1 1\n").unwrap();
            let scene = load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0;
            assert!(!scene.world.meshes()[0].has_alpha());
        });
    }

    // ---- 法線マップ / バンプマップのラッパー（NM S4） ----

    fn map_scene<T>(body: &str, f: impl FnOnce(Scene, Vec<String>) -> T) -> T {
        with_png(1, 1, &[[128, 128, 255]], |dir, name| {
            let xml = format!(r#"<scene version="3.0.0">{}</scene>"#, body.replace("MAP.png", name));
            let (scene, warnings) = capture_warnings(|| load_scene_from_str(&xml, dir, &cfg(), (None, None)).unwrap().0);
            f(scene, warnings)
        })
    }

    const NMAP: &str = r#"<texture type="bitmap" name="normalmap"><string name="filename" value="MAP.png"/></texture>"#;
    const DIFF: &str = r#"<bsdf type="diffuse"/>"#;

    /// 材質表とマップ表が全 shape 経路（球・メッシュ・MTL 混在）で同じ長さに保たれ、マップの添字が正しい材質に付く。
    #[test]
    fn mat_maps_stay_in_sync_with_mats_across_shapes() {
        let body = format!(
            r#"<shape type="sphere"><bsdf type="diffuse"/></shape>
               <shape type="rectangle"><bsdf type="normalmap">{n}{d}</bsdf></shape>
               <shape type="cube"><bsdf type="twosided"><bsdf type="normalmap">{n}{d}</bsdf></bsdf></shape>
               <shape type="rectangle"><bsdf type="bumpmap"><float name="scale" value="2"/><texture type="bitmap" name="bumpmap"><string name="filename" value="MAP.png"/></texture>{d}</bsdf></shape>
               <shape type="disk"><bsdf type="diffuse"/></shape>"#,
            n = NMAP, d = DIFF
        );
        map_scene(&body, |s, w| {
            assert_eq!(s.shaders.shaders.len(), 5);
            assert_eq!(s.mat_maps().len(), s.shaders.shaders.len());
            let some: Vec<bool> = s.mat_maps().iter().map(|m| m.is_some()).collect();
            assert_eq!(some, vec![false, true, true, true, false]);
            assert_eq!(s.shaders.normal_maps.len(), 3);
            assert!(matches!(s.shaders.normal_maps[0], NormalMap::Tangent { .. }));
            assert!(matches!(s.shaders.normal_maps[2], NormalMap::Height { strength, .. } if strength == 2.0));
            assert!(w.iter().all(|m| m.contains("sensor")), "{:?}", w);
        });
    }

    /// マップが 1 つも無ければ `mat_maps` は空（積分器は何も引かない）。
    #[test]
    fn scenes_without_maps_have_empty_tables() {
        let s = load(r#"<scene version="3.0.0"><shape type="sphere"><bsdf type="diffuse"/></shape></scene>"#);
        assert!(s.mat_maps().iter().all(|m| m.is_none()) && s.shaders.normal_maps.is_empty());
    }

    /// MTL 経路（マップ無し）と混在しても表がずれない。
    #[test]
    fn mat_maps_stay_in_sync_with_mtl_materials() {
        with_mtl_dir(|dir| {
            let xml = r#"<scene version="3.0.0">
                <shape type="obj"><string name="filename" value="m.obj"/></shape>
                <shape type="rectangle"><bsdf type="normalmap"><texture type="bitmap"><string name="filename" value="textures/a.png"/></texture><bsdf type="diffuse"/></bsdf></shape>
              </scene>"#;
            let s = load_scene_from_str(xml, dir, &cfg(), (None, None)).unwrap().0;
            assert_eq!(s.mat_maps().len(), s.shaders.shaders.len());
            assert_eq!(s.mat_maps().iter().filter(|m| m.is_some()).count(), 1);
            assert!(s.mat_maps().last().unwrap().is_some());
        });
    }

    /// ノーマルマップのテクスチャは raw（リニア）固定。`raw="false"` は警告して無視。
    /// 内側の `<bsdf>` が無ければ diffuse + 警告。二重ラップは警告なしで順に重ね掛け（外側が先）。
    #[test]
    fn wrapper_warnings_and_raw_reading() {
        let raw_false = format!(
            r#"<shape type="rectangle"><bsdf type="normalmap"><texture type="bitmap" name="normalmap"><string name="filename" value="MAP.png"/><boolean name="raw" value="false"/></texture>{}</bsdf></shape>"#,
            DIFF
        );
        map_scene(&raw_false, |s, w| {
            assert!(w.iter().any(|m| m.contains("raw=false")), "{:?}", w);
            // (128,128,255) をリニアで読んだ値（sRGB デコードされていない）
            match &s.shaders.normal_maps[0] {
                NormalMap::Tangent { tex, .. } => {
                    let c = tex.sample((0.5, 0.5));
                    assert!((c.r() - 128.0 / 255.0).abs() < 1e-12, "sRGB で読まれている: {}", c.r());
                }
                _ => panic!(),
            }
        });
        let no_inner = format!(r#"<shape type="rectangle"><bsdf type="normalmap">{}</bsdf></shape>"#, NMAP);
        map_scene(&no_inner, |s, w| {
            assert!(w.iter().any(|m| m.contains("without an inner")), "{:?}", w);
            assert!(matches!(s.shaders.shaders[0].base, Material::Lambert { .. }));
        });
        let nested = format!(
            r#"<shape type="rectangle"><bsdf type="normalmap">{n}<bsdf type="normalmap">{n}{d}</bsdf></bsdf></shape>"#,
            n = NMAP, d = DIFF
        );
        map_scene(&nested, |s, w| {
            // 入れ子は警告なしで重ね掛けになる: 外側（先に書いた方）が先、内側が後（登録順は内側が先なので [1, 0]）
            assert!(!w.iter().any(|m| m.contains("nested")), "{:?}", w);
            assert_eq!(s.shaders.normal_chain(0), &[1, 0]);
        });
    }

    // ---- MTL の map_bump / norm（NM S5） ----

    /// UV 付きの OBJ（2 三角形: 材質 A, B）と、指定の MTL を書く。テクスチャは textures/a.png（1x1）。
    fn write_uv_obj_and_mtl(dir: &Path, mtl: &str) {
        let obj = "mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\n\
                   usemtl A\nf 1/1 2/2 3/3\nusemtl B\nf 1/1 2/2 3/3\n";
        std::fs::write(dir.join("m.obj"), obj).unwrap();
        std::fs::write(dir.join("m.mtl"), mtl).unwrap();
    }

    /// `map_bump` を持つ材質だけがバンプ付きになり、強度は `bm · MTL_BUMP_K`。`unsupported` の警告は出ない。
    #[test]
    fn map_bump_attaches_only_to_its_material() {
        with_mtl_dir(|dir| {
            write_uv_obj_and_mtl(dir, "newmtl A\n\tKd 1 1 1\n\tmap_bump -bm 2 textures\\a.png\nnewmtl B\n\tKd 1 1 1\n");
            let (s, w) = capture_warnings(|| load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0);
            assert_eq!(s.shaders.shaders.len(), 2);
            assert_eq!(s.mat_maps().len(), s.shaders.shaders.len());
            assert_eq!((s.mat_maps()[0].is_some(), s.mat_maps()[1].is_some()), (true, false));
            match &s.shaders.normal_maps[s.mat_maps()[0].unwrap() as usize] {
                NormalMap::Height { strength, .. } => assert!((strength - 2.0 * MTL_BUMP_K).abs() < 1e-12),
                _ => panic!("ハイトマップのはず"),
            }
            assert!(!w.iter().any(|m| m.contains("map_bump")), "{:?}", w);
        });
    }

    /// `<bsdf>` 上書きと `use_mtl=false` ではマップは付かない。
    #[test]
    fn bump_maps_are_not_attached_when_mtl_is_bypassed() {
        with_mtl_dir(|dir| {
            write_uv_obj_and_mtl(dir, "newmtl A\n\tmap_bump textures\\a.png\nnewmtl B\n\tmap_bump textures\\a.png\n");
            let over = obj_scene_xml(r#"<bsdf type="diffuse"/>"#);
            let s = load_scene_from_str(&over, dir, &cfg(), (None, None)).unwrap().0;
            assert!(s.mat_maps().iter().all(|m| m.is_none()) && s.shaders.normal_maps.is_empty());
            let off = obj_scene_xml(r#"<boolean name="use_mtl" value="false"/>"#);
            let s = load_scene_from_str(&off, dir, &cfg(), (None, None)).unwrap().0;
            assert!(s.mat_maps().iter().all(|m| m.is_none()) && s.shaders.normal_maps.is_empty());
        });
    }

    /// GGX の分岐（明るい Ks かつ Ns > 1）でもマップは付く。
    #[test]
    fn ggx_materials_get_the_bump_map_too() {
        with_mtl_dir(|dir| {
            write_uv_obj_and_mtl(dir, "newmtl A\n\tKs 0.9 0.9 0.9\n\tNs 100\n\tmap_bump textures\\a.png\nnewmtl B\n\tKd 1 1 1\n");
            let s = load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0;
            assert!(matches!(s.shaders.shaders[0].base, Material::Ggx { .. }));
            assert!(s.mat_maps()[0].is_some());
        });
    }

    /// 不具合修正の回帰テスト: `map_Kd` と明るい `Ks`/`Ns` を両方持つ材質は、テクスチャ付き Lambert になる
    /// （Ggx へ早期 return して拡散テクスチャを捨ててはいけない。Sponza の floor/arch/chain/vase_hanging）。
    /// `map_Kd` を持たない明るい `Ks`/`Ns` は従来どおり Ggx。どちらの経路でもバンプマップは付く。
    #[test]
    fn map_kd_takes_priority_over_the_ggx_branch() {
        with_mtl_dir(|dir| {
            write_uv_obj_and_mtl(
                dir,
                "newmtl A\n\tKs 0.9 0.9 0.9\n\tNs 100\n\tmap_Kd textures\\a.png\n\tmap_bump textures\\a.png\n\
                 newmtl B\n\tKs 0.9 0.9 0.9\n\tNs 100\n\tmap_bump textures\\a.png\n",
            );
            let s = load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0;
            assert!(matches!(s.shaders.shaders[0].base, Material::Lambert { .. }) && s.shaders.shaders[0].albedo.is_some(), "map_Kd 付き材質は Lambert のはず");
            assert!(matches!(s.shaders.shaders[1].base, Material::Ggx { .. }), "map_Kd の無い材質は従来どおり Ggx のはず");
            assert!(s.mat_maps()[0].is_some() && s.mat_maps()[1].is_some(), "どちらの経路でもバンプマップが付くはず");
        });
    }

    /// `norm` があれば `norm`（タンジェント）を採用し、`map_bump` 併存の警告は材質ごとに 1 回。
    #[test]
    fn norm_wins_over_map_bump_with_one_warning_per_material() {
        with_mtl_dir(|dir| {
            write_uv_obj_and_mtl(
                dir,
                "newmtl A\n\tnorm textures\\a.png\n\tmap_bump textures\\a.png\nnewmtl B\n\tnorm textures\\a.png\n\tmap_bump textures\\a.png\n",
            );
            let (s, w) = capture_warnings(|| load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0);
            assert!(s.shaders.normal_maps.iter().all(|m| matches!(m, NormalMap::Tangent { .. })));
            assert_eq!(w.iter().filter(|m| m.contains("both norm and map_bump")).count(), 2, "{:?}", w);
        });
    }

    /// 同じ画像・同じ種別・同じ強度は 1 度だけ読む。強度が違えば別登録、種別が違えば別登録。
    #[test]
    fn map_cache_is_keyed_by_path_kind_and_strength() {
        with_mtl_dir(|dir| {
            write_uv_obj_and_mtl(dir, "newmtl A\n\tmap_bump textures\\a.png\nnewmtl B\n\tmap_bump textures/a.png\n");
            let s = load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0;
            assert_eq!(s.shaders.normal_maps.len(), 1);
            assert_eq!(s.mat_maps()[0], s.mat_maps()[1]);
            write_uv_obj_and_mtl(dir, "newmtl A\n\tmap_bump textures\\a.png\nnewmtl B\n\tmap_bump -bm 3 textures/a.png\n");
            let s = load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0;
            assert_eq!(s.shaders.normal_maps.len(), 2, "強度違いが同じ登録を共有した");
            write_uv_obj_and_mtl(dir, "newmtl A\n\tmap_bump textures\\a.png\nnewmtl B\n\tnorm textures/a.png\n");
            let s = load_scene_from_str(&obj_scene_xml(""), dir, &cfg(), (None, None)).unwrap().0;
            assert_eq!(s.shaders.normal_maps.len(), 2, "種別違いが同じ登録を共有した");
        });
    }

    // ---- <medium> ----

    fn medium_scene(body: &str) -> (Scene, Vec<String>) {
        let xml = format!(
            r#"<scene version="3.0.0"><sensor type="perspective"><float name="fov" value="40"/></sensor>{}</scene>"#,
            body
        );
        capture_warnings(|| load_scene_from_str(&xml, Path::new("."), &cfg(), (None, None)).unwrap().0)
    }

    #[test]
    fn medium_reads_all_properties() {
        let (scene, warnings) = medium_scene(
            r#"<medium type="homogeneous">
                 <rgb name="sigma_t" value="0.1, 0.5, 2.0"/>
                 <rgb name="albedo" value="0.9, 0.5, 0.2"/>
                 <phase type="hg"><float name="g" value="0.3"/></phase>
                 <point name="bounds_min" x="-5" y="0" z="-4"/>
                 <point name="bounds_max" x="5" y="6" z="4"/>
               </medium>"#,
        );
        assert!(warnings.is_empty(), "{:?}", warnings);
        let m = scene.medium.expect("medium");
        assert_eq!((m.sigma_t.r(), m.sigma_t.g(), m.sigma_t.b()), (0.1, 0.5, 2.0));
        assert_eq!((m.albedo.r(), m.albedo.g(), m.albedo.b()), (0.9, 0.5, 0.2));
        assert_eq!(m.g, 0.3);
        let b = m.bounds.expect("bounds");
        assert_eq!((b.min.x, b.min.y, b.min.z, b.max.x, b.max.y, b.max.z), (-5.0, 0.0, -4.0, 5.0, 6.0, 4.0));
    }

    #[test]
    fn medium_defaults_and_float_values() {
        // float 単値、albedo・phase・bounds 省略
        let (scene, warnings) = medium_scene(
            r#"<medium type="homogeneous"><float name="sigma_t" value="0.05"/></medium>"#,
        );
        assert!(warnings.is_empty(), "{:?}", warnings);
        let m = scene.medium.unwrap();
        assert_eq!((m.sigma_t.r(), m.sigma_t.g(), m.sigma_t.b()), (0.05, 0.05, 0.05));
        assert_eq!((m.albedo.r(), m.albedo.g(), m.albedo.b()), (1.0, 1.0, 1.0));
        assert_eq!(m.g, 0.0);
        assert!(m.bounds.is_none());
        // rgb 単値（"0.05" の 1 成分）と float の albedo、hg で g 省略・isotropic
        let (scene, _) = medium_scene(
            r#"<medium type="homogeneous"><rgb name="sigma_t" value="0.05"/><float name="albedo" value="0.5"/>
                 <phase type="hg"/></medium>"#,
        );
        let m = scene.medium.unwrap();
        assert_eq!(m.albedo.g(), 0.5);
        assert_eq!(m.g, 0.0);
        let (scene, warnings) = medium_scene(
            r#"<medium type="homogeneous"><phase type="isotropic"/></medium>"#,
        );
        assert!(warnings.is_empty());
        assert_eq!(scene.medium.unwrap().g, 0.0);
        // medium 無し
        assert!(medium_scene("").0.medium.is_none());
    }

    #[test]
    fn medium_invalid_input_warns_and_does_not_crash() {
        let warned = |body: &str, needle: &str| {
            let (scene, w) = medium_scene(body);
            assert!(w.iter().any(|m| m.contains(needle)), "{needle}: {:?}", w);
            scene
        };
        // type 不正: 媒質なし
        assert!(warned(r#"<medium type="heterogeneous"/>"#, "medium type").medium.is_none());
        // phase 不正: 等方扱い
        let s = warned(r#"<medium type="homogeneous"><phase type="rayleigh"/></medium>"#, "phase type");
        assert_eq!(s.medium.unwrap().g, 0.0);
        // bounds 片側だけ / min > max: 無限扱い
        let one = r#"<medium type="homogeneous"><point name="bounds_min" x="0" y="0" z="0"/></medium>"#;
        assert!(warned(one, "bounds").medium.unwrap().bounds.is_none());
        let inv = r#"<medium type="homogeneous"><point name="bounds_min" x="1" y="0" z="0"/>
                       <point name="bounds_max" x="0" y="1" z="1"/></medium>"#;
        assert!(warned(inv, "bounds_min exceeds").medium.unwrap().bounds.is_none());
        // 複数: 最初のものを使う
        let two = r#"<medium type="homogeneous"><float name="sigma_t" value="1"/></medium>
                     <medium type="homogeneous"><float name="sigma_t" value="2"/></medium>"#;
        assert_eq!(warned(two, "more than one").medium.unwrap().sigma_t.r(), 1.0);
        // 範囲外の値は補正
        let bad = r#"<medium type="homogeneous"><float name="sigma_t" value="-1"/><float name="albedo" value="2"/>
                       <phase type="hg"><float name="g" value="1.5"/></phase></medium>"#;
        let m = warned(bad, "sigma_t").medium.unwrap();
        assert_eq!(m.sigma_t.r(), 0.0);
        assert_eq!(m.albedo.r(), 1.0);
        assert_eq!(m.g, 0.99);
    }

    #[test]
    fn medium_inside_shape_warns_and_is_skipped() {
        let (scene, warnings) = medium_scene(
            r#"<shape type="sphere"><bsdf type="diffuse"/>
                 <medium name="interior" type="homogeneous"><float name="sigma_t" value="1"/></medium></shape>"#,
        );
        assert!(warnings.iter().any(|w| w.contains("inside a <shape>")), "{:?}", warnings);
        assert!(scene.medium.is_none());
    }

    // ---- 同じ OBJ のメッシュ共有 ----

    fn share_dir(tag: &str) -> PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static C: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!("tinypt_share_{}_{}_{}", tag, std::process::id(), C.fetch_add(1, Ordering::Relaxed)));
        std::fs::create_dir_all(&dir).unwrap();
        // 三角形 (0,0,0)-(1,0,0)-(0,1,0)（頂点法線付き）
        std::fs::write(dir.join("a.obj"), "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nvn 0 1 0\nvn 1 0 0\nf 1//1 2//2 3//3\n").unwrap();
        std::fs::write(dir.join("b.obj"), "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
        dir
    }

    fn obj_shape(file: &str, x: f64, bsdf: &str, extra: &str) -> String {
        format!(
            r#"<shape type="obj"><string name="filename" value="{file}"/>{extra}
                 <transform name="to_world"><translate x="{x}" y="0" z="0"/></transform>{bsdf}</shape>"#
        )
    }
    const DIFFUSE_A: &str = r#"<bsdf type="diffuse"><rgb name="reflectance" value="0.1"/></bsdf>"#;
    const DIFFUSE_B: &str = r#"<bsdf type="diffuse"><rgb name="reflectance" value="0.9"/></bsdf>"#;

    fn share_scene(dir: &Path, shapes: &str) -> Scene {
        let xml = format!(r#"<scene version="3.0.0"><sensor type="perspective"><float name="fov" value="40"/></sensor>{}</scene>"#, shapes);
        load_scene_from_str(&xml, dir, &cfg(), (None, None)).unwrap().0
    }

    /// 同じ OBJ を何度置いても、メッシュは 1 つでインスタンスが増える。材質は形状ごとに正しく効く。
    #[test]
    fn same_obj_shares_one_mesh_and_keeps_per_instance_materials() {
        let dir = share_dir("same");
        let shapes: String = (0..3)
            .map(|i| obj_shape("a.obj", 3.0 * i as f64, if i == 1 { DIFFUSE_B } else { DIFFUSE_A }, ""))
            .collect();
        let scene = share_scene(&dir, &shapes);
        assert_eq!((scene.world.mesh_count(), scene.world.instance_count()), (1, 3));
        assert_eq!(scene.shaders.shaders.len(), 3, "材質は形状ごとに 1 つ");
        for i in 0..3 {
            let ray = Ray { o: Vec3::new(0.2 + 3.0 * i as f64, 0.2, 5.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 };
            let hit = scene.world.hit(ray, 0.0, 1e30).expect("hit");
            assert_eq!(hit.mat_id, i, "インスタンス {} の材質", i);
            assert_eq!(hit.inst_id, Some(i));
        }
        // 配置（xform）はインスタンスごと: 隣のインスタンスの位置には当たらない
        let miss = Ray { o: Vec3::new(1.5, 0.2, 5.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 };
        assert!(scene.world.hit(miss, 0.0, 1e30).is_none());
        // パスの綴りが違っても（./a.obj）同じファイルなら共有する
        let scene = share_scene(&dir, &format!("{}{}", obj_shape("a.obj", 0.0, DIFFUSE_A, ""), obj_shape("./a.obj", 3.0, DIFFUSE_A, "")));
        assert_eq!(scene.world.mesh_count(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 共有してはいけないものは共有しない: 別ファイル、`face_normals` の違い。
    #[test]
    fn different_obj_or_normal_handling_is_not_shared() {
        let dir = share_dir("diff");
        let scene = share_scene(&dir, &format!("{}{}", obj_shape("a.obj", 0.0, DIFFUSE_A, ""), obj_shape("b.obj", 3.0, DIFFUSE_A, "")));
        assert_eq!((scene.world.mesh_count(), scene.world.instance_count()), (2, 2));
        let fn_true = r#"<boolean name="face_normals" value="true"/>"#;
        let scene = share_scene(&dir, &format!("{}{}", obj_shape("a.obj", 0.0, DIFFUSE_A, ""), obj_shape("a.obj", 3.0, DIFFUSE_A, fn_true)));
        assert_eq!((scene.world.mesh_count(), scene.world.instance_count()), (2, 2), "face_normals が違えば別メッシュ");
        // face_normals=true 同士は共有する
        let scene = share_scene(&dir, &format!("{}{}", obj_shape("a.obj", 0.0, DIFFUSE_A, fn_true), obj_shape("a.obj", 3.0, DIFFUSE_A, fn_true)));
        assert_eq!((scene.world.mesh_count(), scene.world.instance_count()), (1, 2));
        // 頂点法線の扱いが実際に違う: 共有されなかった 2 つで陰影法線が違う（補間法線 vs 面法線）
        let scene = share_scene(&dir, &format!("{}{}", obj_shape("a.obj", 0.0, DIFFUSE_A, ""), obj_shape("a.obj", 3.0, DIFFUSE_A, fn_true)));
        let hit_at = |x: f64| scene.world.hit(Ray { o: Vec3::new(x + 0.2, 0.2, 5.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30).unwrap();
        assert!((hit_at(0.0).ns - hit_at(0.0).ng).len() > 1e-3, "補間法線は面法線と違う");
        assert!((hit_at(3.0).ns - hit_at(3.0).ng).len() < 1e-12, "face_normals は ns = ng");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 共有したメッシュでも、面光源かどうかは形状（材質）ごとに決まる。
    #[test]
    fn shared_mesh_emitter_is_per_instance() {
        let dir = share_dir("emit");
        let emitter = r#"<emitter type="area"><rgb name="radiance" value="5"/></emitter>"#;
        // 1 つ目は通常の拡散、2 つ目（共有）だけが面光源
        let scene = share_scene(&dir, &format!("{}{}", obj_shape("a.obj", 0.0, DIFFUSE_A, ""), obj_shape("a.obj", 3.0, "", emitter)));
        assert_eq!(scene.world.mesh_count(), 1);
        let ray = |x: f64| Ray { o: Vec3::new(x + 0.2, 0.2, 5.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 };
        assert!(scene.shaders.shaders[scene.world.hit(ray(0.0), 0.0, 1e30).unwrap().mat_id].base.emitted().is_none());
        assert!(scene.shaders.shaders[scene.world.hit(ray(3.0), 0.0, 1e30).unwrap().mat_id].base.emitted().is_some());
        let mut rng = crate::rng::Rng::new(1);
        for _ in 0..50 {
            let ls = scene.world.sample_light(&mut rng, 0.0, Vec3::new(1.0, 1.0, 3.0)).expect("light");
            assert!(ls.position.x >= 3.0 - 1e-9 && ls.position.x <= 4.0 + 1e-9, "光源は 2 つ目のインスタンスだけ: {:?}", ls.position);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// MTL 経路（`<bsdf>` 無し）は形状ごとに材質を積み直して三角形に焼き込むので、共有しない（メッシュ 2 つ）。
    #[test]
    fn mtl_path_obj_is_not_shared() {
        let dir = share_dir("mtl");
        std::fs::write(dir.join("m.obj"), "mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl A\nf 1 2 3\n").unwrap();
        std::fs::write(dir.join("m.mtl"), "newmtl A\nKd 1 0 0\n").unwrap();
        let shape = |x: f64| format!(r#"<shape type="obj"><string name="filename" value="m.obj"/><transform name="to_world"><translate x="{x}" y="0" z="0"/></transform></shape>"#);
        let scene = share_scene(&dir, &format!("{}{}", shape(0.0), shape(3.0)));
        assert_eq!((scene.world.mesh_count(), scene.world.instance_count()), (2, 2));
        for i in 0..2 {
            let hit = scene.world.hit(Ray { o: Vec3::new(0.2 + 3.0 * i as f64, 0.2, 5.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 }, 0.0, 1e30).unwrap();
            assert_eq!(hit.mat_id, i);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- デルタ光源（<emitter type="point" | "directional" | "spot">）----

    #[test]
    fn delta_emitters_read_all_properties() {
        let (scene, warnings) = medium_scene(
            r#"<emitter type="point"><point name="position" x="1" y="2" z="3"/><rgb name="intensity" value="10, 20, 30"/></emitter>
               <emitter type="directional"><vector name="direction" x="0" y="-2" z="0"/><float name="irradiance" value="3"/></emitter>
               <emitter type="spot"><point name="position" x="0" y="3" z="0"/><vector name="direction" x="0" y="-1" z="0"/>
                 <rgb name="intensity" value="80"/><float name="cutoff_angle" value="25"/><float name="beam_width" value="18"/></emitter>"#,
        );
        assert!(warnings.is_empty(), "{:?}", warnings);
        let l = scene.world.delta_lights();
        assert_eq!(l.len(), 3);
        match l[0] {
            DeltaLight::Point { position, intensity } => {
                assert_eq!((position.x, position.y, position.z), (1.0, 2.0, 3.0));
                assert_eq!((intensity.r(), intensity.g(), intensity.b()), (10.0, 20.0, 30.0));
            }
            _ => panic!("{:?}", l[0]),
        }
        match l[1] {
            DeltaLight::Directional { direction, irradiance } => {
                assert_eq!((direction.x, direction.y, direction.z), (0.0, -1.0, 0.0), "正規化される");
                assert_eq!(irradiance.g(), 3.0);
            }
            _ => panic!("{:?}", l[1]),
        }
        match l[2] {
            DeltaLight::Spot { intensity, cutoff_angle, beam_width, .. } => {
                assert_eq!(intensity.b(), 80.0);
                assert!((cutoff_angle - 25f64.to_radians()).abs() < 1e-15 && (beam_width - 18f64.to_radians()).abs() < 1e-15);
            }
            _ => panic!("{:?}", l[2]),
        }
    }

    #[test]
    fn delta_emitter_defaults() {
        let (scene, warnings) = medium_scene(r#"<emitter type="spot"><rgb name="intensity" value="5"/></emitter><emitter type="point"/>"#);
        assert!(warnings.is_empty(), "{:?}", warnings);
        match scene.world.delta_lights()[0] {
            DeltaLight::Spot { position, direction, cutoff_angle, beam_width, .. } => {
                assert_eq!((position.x, position.y, position.z), (0.0, 0.0, 0.0));
                assert_eq!(direction.y, -1.0);
                assert!((cutoff_angle - 20f64.to_radians()).abs() < 1e-15, "cutoff 既定 20°");
                assert!((beam_width - 15f64.to_radians()).abs() < 1e-15, "beam 既定は cutoff × 3/4");
            }
            l => panic!("{:?}", l),
        }
    }

    /// 既存の環境 emitter の挙動は変わらない（`constant` / `envmap` は従来どおり環境、1 個は最後のもの、
    /// 未知の型は警告）。デルタ光源は環境を触らず、複数置ける。
    #[test]
    fn env_emitters_are_unchanged_and_coexist_with_delta_lights() {
        let up = Vec3::new(0.0, 1.0, 0.0);
        let (scene, w) = medium_scene(r#"<emitter type="constant"><rgb name="radiance" value="0.25"/></emitter>"#);
        assert!(w.is_empty());
        assert_eq!(scene.env.as_ref().unwrap().sample(up).r(), 0.25);
        assert!(scene.world.delta_lights().is_empty());
        // デルタ光源が混ざっても環境は変わらない。点光源 2 個
        let (scene, w) = medium_scene(
            r#"<emitter type="point"><rgb name="intensity" value="1"/></emitter>
               <emitter type="constant"><rgb name="radiance" value="0.25"/></emitter>
               <emitter type="point"><rgb name="intensity" value="2"/></emitter>"#,
        );
        assert!(w.is_empty(), "{:?}", w);
        assert_eq!(scene.env.as_ref().unwrap().sample(up).r(), 0.25);
        assert_eq!(scene.world.delta_lights().len(), 2);
        // 未知の型は従来どおり警告して無視（環境は黒のまま）
        let (scene, w) = medium_scene(r#"<emitter type="projector"/>"#);
        assert!(w.iter().any(|m| m.contains("unsupported scene emitter type")), "{:?}", w);
        assert_eq!(scene.env.as_ref().unwrap().sample(up).r(), 0.0);
        assert!(scene.world.delta_lights().is_empty());
    }

    #[test]
    fn delta_emitter_invalid_input_warns_and_does_not_crash() {
        let warned = |body: &str, needle: &str| {
            let (scene, w) = medium_scene(body);
            assert!(w.iter().any(|m| m.contains(needle)), "{needle}: {:?}", w);
            scene
        };
        // 零方向: その光源を無視
        let s = warned(r#"<emitter type="directional"><vector name="direction" x="0" y="0" z="0"/></emitter>"#, "direction");
        assert!(s.world.delta_lights().is_empty());
        let s = warned(r#"<emitter type="spot"><vector name="direction" x="0" y="0" z="0"/></emitter>"#, "direction");
        assert!(s.world.delta_lights().is_empty());
        // 角度の範囲外は既定へ、beam > cutoff は cutoff に丸める
        for bad in ["0", "-5", "120"] {
            let s = warned(&format!(r#"<emitter type="spot"><float name="cutoff_angle" value="{bad}"/></emitter>"#), "cutoff_angle");
            match s.world.delta_lights()[0] {
                DeltaLight::Spot { cutoff_angle, .. } => assert!((cutoff_angle - 20f64.to_radians()).abs() < 1e-15),
                l => panic!("{:?}", l),
            }
        }
        let s = warned(r#"<emitter type="spot"><float name="cutoff_angle" value="20"/><float name="beam_width" value="30"/></emitter>"#, "beam_width");
        match s.world.delta_lights()[0] {
            DeltaLight::Spot { cutoff_angle, beam_width, .. } => assert_eq!(cutoff_angle, beam_width),
            l => panic!("{:?}", l),
        }
        // 負の強度は 0 に
        let s = warned(r#"<emitter type="point"><rgb name="intensity" value="-1, 2, 3"/></emitter>"#, "must be >= 0");
        match s.world.delta_lights()[0] {
            DeltaLight::Point { intensity, .. } => assert_eq!((intensity.r(), intensity.g()), (0.0, 2.0)),
            l => panic!("{:?}", l),
        }
        // 形状の中: 警告して無視（形状は拡散のまま残り、光源にはならない）
        let s = warned(r#"<shape type="sphere"><bsdf type="diffuse"/><emitter type="point"><rgb name="intensity" value="5"/></emitter></shape>"#, "inside a <shape>");
        assert!(s.world.delta_lights().is_empty());
        assert!(s.shaders.shaders[0].base.emitted().is_none());
    }

    // ---- モーション（to_world_end / filename_end / shutter）----

    const END_TRANSFORM: &str = r#"<transform name="to_world_end"><translate x="3" y="0" z="0"/></transform>"#;

    fn motion_dir(tag: &str) -> PathBuf {
        let dir = share_dir(tag);
        // a.obj と同じトポロジーで、頂点が +x に 2 動いた閉側
        std::fs::write(dir.join("a_end.obj"), "v 2 0 0\nv 3 0 0\nv 2 1 0\nvn 0 0 1\nvn 0 1 0\nvn 1 0 0\nf 1//1 2//2 3//3\n").unwrap();
        // トポロジー不一致（頂点数が違う）
        std::fs::write(dir.join("bad_end.obj"), "v 2 0 0\nv 3 0 0\nv 2 1 0\nv 9 9 9\nf 1 2 3\n").unwrap();
        dir
    }

    fn hit_at(scene: &Scene, x: f64, y: f64, time: f64) -> bool {
        scene.world.hit(Ray { o: Vec3::new(x, y, 5.0), d: Vec3::new(0.0, 0.0, -1.0), time }, 0.0, 1e30).is_some()
    }

    /// `to_world_end` を読み、`time` で補間する。`to_world` と `to_world_end` の記述順は問わず、名前で区別する。
    #[test]
    fn to_world_end_animates_the_instance() {
        let dir = motion_dir("anim");
        let (scene, w) = capture_warnings(|| {
            share_scene(&dir, &obj_shape("a.obj", 0.0, DIFFUSE_A, END_TRANSFORM))
        });
        assert!(w.is_empty(), "{:?}", w);
        assert!(scene.world.instances()[0].anim.is_some());
        for (time, dx) in [(0.0, 0.0), (0.5, 1.5), (1.0, 3.0)] {
            assert!(hit_at(&scene, 0.2 + dx, 0.2, time), "time {time}");
            assert!(!hit_at(&scene, 0.2 + dx + 1.0, 0.2, time));
        }
        // to_world_end を先に書いても、to_world は名前で選ばれる
        let xml = format!(
            r#"<shape type="obj"><string name="filename" value="a.obj"/>{}<transform name="to_world"><translate x="10" y="0" z="0"/></transform>{}</shape>"#,
            END_TRANSFORM, DIFFUSE_A
        );
        let scene = share_scene(&dir, &xml);
        assert!(hit_at(&scene, 10.2, 0.2, 0.0) && hit_at(&scene, 0.2 + 3.0, 0.2, 1.0), "to_world=(10,0,0) → end=(3,0,0)");
        // 動かない形状は従来どおり
        let scene = share_scene(&dir, &obj_shape("a.obj", 0.0, DIFFUSE_A, ""));
        assert!(scene.world.instances()[0].anim.is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 頂点モーション（`filename_end`）と `to_world_end` の併用、メッシュ共有のキー、不正入力。
    #[test]
    fn filename_end_vertex_motion_and_cache_key() {
        let dir = motion_dir("vert");
        let end = r#"<string name="filename_end" value="a_end.obj"/>"#;
        // 頂点モーション単独
        let scene = share_scene(&dir, &obj_shape("a.obj", 0.0, DIFFUSE_A, end));
        assert!(hit_at(&scene, 0.2, 0.2, 0.0) && hit_at(&scene, 2.2, 0.2, 1.0) && !hit_at(&scene, 0.2, 0.2, 1.0));
        // 併用: 頂点で +2、変換で +3 → 時刻 1 で x = 5
        let both = format!("{end}{END_TRANSFORM}");
        let scene = share_scene(&dir, &obj_shape("a.obj", 0.0, DIFFUSE_A, &both));
        assert!(hit_at(&scene, 0.2, 0.2, 0.0) && hit_at(&scene, 5.2, 0.2, 1.0) && !hit_at(&scene, 2.2, 0.2, 1.0) && !hit_at(&scene, 3.2, 0.2, 1.0));
        // キー: 同じ a.obj でも filename_end の有無・違いは別メッシュ。同じ組は共有
        let scene = share_scene(&dir, &format!("{}{}", obj_shape("a.obj", 0.0, DIFFUSE_A, ""), obj_shape("a.obj", 3.0, DIFFUSE_A, end)));
        assert_eq!((scene.world.mesh_count(), scene.world.instance_count()), (2, 2), "静止版と動く版を共有しない");
        assert!(hit_at(&scene, 3.2, 0.2, 0.0) && hit_at(&scene, 5.2, 0.2, 1.0), "動く側（x=3）は時刻 1 で頂点が +2");
        assert!(hit_at(&scene, 0.2, 0.2, 1.0), "静止側は動かない");
        let scene = share_scene(&dir, &format!("{}{}", obj_shape("a.obj", 0.0, DIFFUSE_A, end), obj_shape("a.obj", 5.0, DIFFUSE_B, end)));
        assert_eq!((scene.world.mesh_count(), scene.world.instance_count()), (1, 2), "同じ開・閉の組は共有する");
        // 共有した側にもそれぞれの変換が効く（頂点モーションは共有、材質は override）
        assert!(hit_at(&scene, 5.2, 0.2, 0.0) && hit_at(&scene, 7.2, 0.2, 1.0));
        // トポロジー不一致: 警告してモーション無しで読む
        let bad = r#"<string name="filename_end" value="bad_end.obj"/>"#;
        let (scene, w) = capture_warnings(|| share_scene(&dir, &obj_shape("a.obj", 0.0, DIFFUSE_A, bad)));
        assert!(w.iter().any(|m| m.contains("vertex motion")), "{:?}", w);
        assert!(hit_at(&scene, 0.2, 0.2, 0.0) && hit_at(&scene, 0.2, 0.2, 1.0), "静止として読まれる");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 退化ケース: 特異・鏡像の `to_world_end` は警告して静止、開と閉が同一・180° 回転は動く、面光源・球は警告して静止。
    #[test]
    fn to_world_end_degenerate_inputs() {
        let dir = motion_dir("deg");
        let warned = |end: &str, needle: &str| {
            let (scene, w) = capture_warnings(|| share_scene(&dir, &obj_shape("a.obj", 0.0, DIFFUSE_A, end)));
            assert!(w.iter().any(|m| m.contains(needle)), "{needle}: {:?}", w);
            scene
        };
        let singular = r#"<transform name="to_world_end"><scale x="1" y="0" z="1"/></transform>"#;
        assert!(warned(singular, "singular").world.instances()[0].anim.is_none());
        let mirror = r#"<transform name="to_world_end"><scale x="-1" y="1" z="1"/></transform>"#;
        assert!(warned(mirror, "mirrored").world.instances()[0].anim.is_none());
        // 開と閉が同一: 動く扱いだが位置は変わらない
        let same = share_scene(&dir, &obj_shape("a.obj", 0.0, DIFFUSE_A, r#"<transform name="to_world_end"/>"#));
        assert!(same.world.instances()[0].anim.is_some());
        assert!(hit_at(&same, 0.2, 0.2, 0.0) && hit_at(&same, 0.2, 0.2, 0.5) && hit_at(&same, 0.2, 0.2, 1.0));
        // 180° 回転（z 軸まわり）: 時刻 1 で点対称の位置に三角形が来る。途中も抜け落ちない（三角形の重心付近が通る）
        let flip = share_scene(&dir, &obj_shape("a.obj", 0.0, DIFFUSE_A, r#"<transform name="to_world_end"><rotate x="0" y="0" z="1" angle="180"/></transform>"#));
        assert!(hit_at(&flip, 0.2, 0.2, 0.0) && hit_at(&flip, -0.2, -0.2, 1.0));
        // 面光源
        let emitter_xml = format!(r#"<shape type="obj"><string name="filename" value="a.obj"/>{}<emitter type="area"><rgb name="radiance" value="5"/></emitter></shape>"#, END_TRANSFORM);
        let (scene, w) = capture_warnings(|| share_scene(&dir, &emitter_xml));
        assert!(w.iter().any(|m| m.contains("area-light")), "{:?}", w);
        assert!(scene.world.instances()[0].anim.is_none());
        // 球
        let sphere = format!(r#"<shape type="sphere"><bsdf type="diffuse"/>{}</shape>"#, END_TRANSFORM);
        let (_, w) = capture_warnings(|| share_scene(&dir, &sphere));
        assert!(w.iter().any(|m| m.contains("sphere")), "{:?}", w);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// シャッター時刻: 既定は 0 と 1、指定でカメラの time 範囲と掃過ボリュームが狭まる。open > close は入れ替え、
    /// open == close は時刻固定（有効）、範囲外は [0, 1] に収める。
    #[test]
    fn sensor_shutter_times() {
        let dir = motion_dir("shut");
        let with_sensor = |props: &str| {
            let xml = format!(
                r#"<scene version="3.0.0"><sensor type="perspective"><float name="fov" value="40"/>{props}</sensor>{}</scene>"#,
                obj_shape("a.obj", 0.0, DIFFUSE_A, END_TRANSFORM)
            );
            capture_warnings(|| load_scene_from_str(&xml, &dir, &cfg(), (None, None)).unwrap().0)
        };
        let (s, w) = with_sensor("");
        assert!(w.is_empty(), "{:?}", w);
        assert_eq!(s.cam.shutter(), (0.0, 1.0));
        let full = s.world.instances()[0].world_bounds;
        let (s, _) = with_sensor(r#"<float name="shutter_open" value="0"/><float name="shutter_close" value="0.5"/>"#);
        assert_eq!(s.cam.shutter(), (0.0, 0.5));
        assert!(s.world.instances()[0].world_bounds.max.x < full.max.x - 1.0, "掃過ボリュームが狭まる");
        let mut rng = Rng::new(3);
        let max_t = (0..2000).map(|_| s.cam.ray(0.0, 0.0, &mut rng).time).fold(0.0, f64::max);
        assert!(max_t <= 0.5 && max_t > 0.45, "{max_t}");
        // 逆順: 入れ替え
        let (s, w) = with_sensor(r#"<float name="shutter_open" value="0.8"/><float name="shutter_close" value="0.2"/>"#);
        assert!(w.iter().any(|m| m.contains("swapped")));
        assert_eq!(s.cam.shutter(), (0.2, 0.8));
        // 同値: 有効（警告なし）、時刻固定
        let (s, w) = with_sensor(r#"<float name="shutter_open" value="0.3"/><float name="shutter_close" value="0.3"/>"#);
        assert!(w.is_empty(), "{:?}", w);
        assert!((0..50).all(|_| s.cam.ray(0.0, 0.0, &mut rng).time == 0.3));
        // 範囲外: 収める
        let (s, w) = with_sensor(r#"<float name="shutter_open" value="-1"/><float name="shutter_close" value="4"/>"#);
        assert!(w.iter().any(|m| m.contains("clamped")));
        assert_eq!(s.cam.shutter(), (0.0, 1.0));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `shutter_angle`（度）: close = open + angle / 360。`shutter_close` より優先、範囲外・非有限は警告して従来どおり。
    #[test]
    fn sensor_shutter_angle() {
        let dir = motion_dir("shang");
        let with_sensor = |props: &str| {
            let xml = format!(
                r#"<scene version="3.0.0"><sensor type="perspective"><float name="fov" value="40"/>{props}</sensor>{}</scene>"#,
                obj_shape("a.obj", 0.0, DIFFUSE_A, END_TRANSFORM)
            );
            capture_warnings(|| load_scene_from_str(&xml, &dir, &cfg(), (None, None)).unwrap().0)
        };
        let ang = |a: &str| format!(r#"<float name="shutter_angle" value="{a}"/>"#);
        let (s, w) = with_sensor(&ang("180"));
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(s.cam.shutter(), (0.0, 0.5));
        let (s, w) = with_sensor(&format!(r#"<float name="shutter_open" value="0.25"/>{}"#, ang("90")));
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(s.cam.shutter(), (0.25, 0.5));
        let (s, w) = with_sensor(&ang("0"));
        assert!(w.is_empty(), "{w:?}");
        let (o, c) = s.cam.shutter();
        assert_eq!(o, c);
        let (s, w) = with_sensor(&ang("360"));
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(s.cam.shutter(), (0.0, 1.0));
        // shutter_close と併記: 警告して角度が勝つ
        let (s, w) = with_sensor(&format!(r#"<float name="shutter_close" value="0.9"/>{}"#, ang("90")));
        assert!(w.iter().any(|m| m.contains("shutter_close is ignored")), "{w:?}");
        assert_eq!(s.cam.shutter(), (0.0, 0.25));
        // 負・360 超・NaN: 警告して従来どおり（shutter_close か既定）
        for bad in ["-10", "400", "nan"] {
            let (s, w) = with_sensor(&ang(bad));
            assert!(w.iter().any(|m| m.contains("shutter_angle must be")), "{bad}: {w:?}");
            assert_eq!(s.cam.shutter(), (0.0, 1.0), "{bad}");
            let (s, _) = with_sensor(&format!(r#"<float name="shutter_close" value="0.4"/>{}"#, ang(bad)));
            assert_eq!(s.cam.shutter(), (0.0, 0.4), "{bad} falls back to shutter_close");
        }
        // open + angle/360 > 1: 収めて警告
        let (s, w) = with_sensor(&format!(r#"<float name="shutter_open" value="0.8"/>{}"#, ang("180")));
        assert!(w.iter().any(|m| m.contains("clamped")), "{w:?}");
        assert_eq!(s.cam.shutter(), (0.8, 1.0));
        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- カメラのモーション（sensor の to_world_end）----

    fn sensor_scene(inner: &str) -> (Scene, Vec<String>) {
        let xml = format!(r#"<scene version="3.0.0"><sensor type="perspective"><float name="fov" value="40"/>{inner}</sensor></scene>"#);
        let (r, w) = capture_warnings(|| load_scene_from_str(&xml, Path::new("."), &cfg(), (None, None)).unwrap().0);
        (r, w)
    }
    fn eye_at(scene: &mut Scene, t: f64) -> Vec3 {
        scene.cam.set_shutter(t, t);
        scene.cam.ray(0.0, 0.0, &mut crate::rng::Rng::new(0)).o
    }
    const CAM_START: &str = r#"<transform name="to_world"><lookat origin="4, 1, 0" target="0, 0, 0" up="0, 1, 0"/></transform>"#;
    const CAM_END: &str = r#"<transform name="to_world_end"><lookat origin="0, 1, -4" target="0, 0, 0" up="0, 1, 0"/></transform>"#;

    /// `to_world_end` の `<lookat>` を読み、`to_world` との記述順は問わない。閉じ時刻でカメラは閉の位置、開で開の位置。
    #[test]
    fn sensor_to_world_end_animates_the_camera() {
        for inner in [format!("{CAM_START}{CAM_END}"), format!("{CAM_END}{CAM_START}")] {
            let (mut s, w) = sensor_scene(&inner);
            assert!(w.is_empty(), "{w:?}");
            assert!((eye_at(&mut s, 0.0) - Vec3::new(4.0, 1.0, 0.0)).len() < 1e-12);
            assert!((eye_at(&mut s, 1.0) - Vec3::new(0.0, 1.0, -4.0)).len() < 1e-12);
            // 弧の中点（半径 4 の xz、高さ 1）
            let mid = eye_at(&mut s, 0.5);
            assert!((mid - Vec3::new(4.0 * 0.5f64.sqrt(), 1.0, -4.0 * 0.5f64.sqrt())).len() < 1e-9, "{mid:?}");
        }
        // 無い場合は静止（警告なし）
        let (mut s, w) = sensor_scene(CAM_START);
        assert!(w.is_empty());
        assert!((eye_at(&mut s, 1.0) - Vec3::new(4.0, 1.0, 0.0)).len() < 1e-12);
    }

    /// `<lookat>` 以外の書き方・退化した姿勢は警告して静止（壊れない）。
    #[test]
    fn sensor_to_world_end_bad_input_warns_and_stays_static() {
        for end in [
            r#"<transform name="to_world_end"><translate x="1" y="0" z="0"/></transform>"#,
            r#"<transform name="to_world_end"/>"#,
            r#"<transform name="to_world_end"><lookat origin="0, 4, 0" target="0, 0, 0" up="0, 1, 0"/></transform>"#,
            r#"<transform name="to_world_end"><lookat origin="0, 1, 0" target="0, 1, 0" up="0, 1, 0"/></transform>"#,
        ] {
            let (mut s, w) = sensor_scene(&format!("{CAM_START}{end}"));
            assert_eq!(w.iter().filter(|m| m.contains("to_world_end")).count(), 1, "{end}: {w:?}");
            let e = eye_at(&mut s, 1.0);
            assert!((e - Vec3::new(4.0, 1.0, 0.0)).len() < 1e-12 && e.x.is_finite(), "{end}");
        }
    }

    // ---- 球のモーション（center_end）----

    fn sphere_end_scene(inner: &str) -> (Scene, Vec<String>) {
        let xml = format!(r#"<scene version="3.0.0"><sensor type="perspective"><float name="fov" value="40"/></sensor><shape type="sphere"><point name="center" x="0" y="0" z="0"/><float name="radius" value="0.5"/>{inner}</shape></scene>"#);
        capture_warnings(|| load_scene_from_str(&xml, Path::new("."), &cfg(), (None, None)).unwrap().0)
    }
    fn hits_at(scene: &Scene, x: f64, time: f64) -> bool {
        let r = Ray { o: Vec3::new(x, 0.0, 5.0), d: Vec3::new(0.0, 0.0, -1.0), time };
        scene.world.hit(r, 1e-9, 1e30).is_some()
    }

    /// `center_end` を読んで time で線形補間する。省略は静止、不正値・発光球は警告して静止。
    #[test]
    fn sphere_center_end_moves_and_bad_input_stays_static() {
        let diffuse = r#"<bsdf type="diffuse"/>"#;
        let (s, w) = sphere_end_scene(&format!(r#"{diffuse}<point name="center_end" x="4" y="0" z="0"/>"#));
        assert!(w.is_empty(), "{w:?}");
        assert!(hits_at(&s, 0.0, 0.0) && hits_at(&s, 2.0, 0.5) && hits_at(&s, 4.0, 1.0) && !hits_at(&s, 0.0, 1.0));
        let (s, w) = sphere_end_scene(diffuse);
        assert!(w.is_empty() && hits_at(&s, 0.0, 1.0) && !hits_at(&s, 4.0, 1.0));
        // 不正（成分が足りない / 数値でない）: 警告 1 回で静止
        let (s, w) = sphere_end_scene(&format!(r#"{diffuse}<point name="center_end" x="4" y="oops" z="0"/>"#));
        assert_eq!(w.iter().filter(|m| m.contains("center_end")).count(), 1, "{w:?}");
        assert!(hits_at(&s, 0.0, 1.0) && !hits_at(&s, 4.0, 1.0));
        // 発光する球は動かせない
        let (s, w) = sphere_end_scene(r#"<emitter type="area"><rgb name="radiance" value="1"/></emitter><point name="center_end" x="4" y="0" z="0"/>"#);
        assert_eq!(w.iter().filter(|m| m.contains("center_end")).count(), 1, "{w:?}");
        assert!(hits_at(&s, 0.0, 1.0) && !hits_at(&s, 4.0, 1.0));
        // to_world_end は案内付きの警告
        let (_, w) = sphere_end_scene(&format!(r#"{diffuse}<transform name="to_world_end"><translate x="1"/></transform>"#));
        assert!(w.iter().any(|m| m.contains("center_end")), "{w:?}");
    }

    // ---- 手続き的ノイズ（<texture type="noise">）----

    fn noise_scene(inner: &str) -> (Scene, Vec<String>) {
        let xml = format!(r#"<scene version="3.0.0"><sensor type="perspective"><float name="fov" value="40"/></sensor><shape type="sphere"><bsdf type="diffuse"><rgb name="reflectance" value="0.5"/><texture type="noise" name="reflectance">{inner}</texture></bsdf></shape></scene>"#);
        capture_warnings(|| load_scene_from_str(&xml, Path::new("."), &cfg(), (None, None)).unwrap().0)
    }
    fn albedo_at(s: &Scene, p: Vec3) -> f64 {
        match s.shaders.eval_at(0, &s.world, p, (0.5, 0.5)) {
            Material::Lambert { albedo } => albedo.r(),
            _ => panic!("ノイズが畳み込まれていない"),
        }
    }

    /// ノイズは位置で変わり、定数色は倍率として掛かる。不正な pattern / 範囲外の値は警告して丸め、落ちない。
    #[test]
    fn noise_texture_parses_varies_with_position_and_clamps() {
        let ok = r#"<string name="pattern" value="marble"/><float name="scale" value="3"/><rgb name="color0" value="0"/><rgb name="color1" value="1"/>"#;
        let (s, w) = noise_scene(ok);
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(s.shaders.noises.len(), 1);
        let vals: Vec<f64> = (0..20).map(|i| albedo_at(&s, Vec3::new(0.1 * i as f64, 0.3, 0.2))).collect();
        assert!(vals.iter().all(|v| (0.0..=0.5 + 1e-12).contains(v)), "0.5 倍の範囲");
        assert!(vals.iter().cloned().fold(0.0, f64::max) - vals.iter().cloned().fold(1.0, f64::min) > 0.1, "位置で変わる");
        assert_eq!(albedo_at(&s, Vec3::new(0.3, 0.3, 0.3)).to_bits(), albedo_at(&s, Vec3::new(0.3, 0.3, 0.3)).to_bits());
        let (_, w) = noise_scene(r#"<string name="pattern" value="nope"/>"#);
        assert_eq!(w.iter().filter(|m| m.contains("noise pattern")).count(), 1, "{w:?}");
        for bad in [r#"<float name="scale" value="0"/>"#, r#"<float name="scale" value="-2"/>"#, r#"<integer name="octaves" value="99"/>"#, r#"<integer name="octaves" value="0"/>"#] {
            let (s, w) = noise_scene(bad);
            assert!(w.iter().any(|m| m.contains("out of range")), "{bad}: {w:?}");
            assert!(albedo_at(&s, Vec3::new(0.3, 0.3, 0.3)).is_finite());
        }
    }

    fn placed_noise_scene(space: &str, off_b: &str) -> Scene {
        let noise = |off: &str| format!(r#"<bsdf type="diffuse"><texture type="noise" name="reflectance"><string name="pattern" value="marble"/><float name="scale" value="3"/>{space}{off}</texture></bsdf>"#);
        let xml = format!(
            r#"<scene version="3.0.0"><sensor type="perspective"><float name="fov" value="40"/></sensor>
            <shape type="rectangle"><transform name="to_world"><translate x="0" y="0" z="0"/></transform>{}</shape>
            <shape type="rectangle"><transform name="to_world"><translate x="7" y="2" z="-3"/><rotate y="1" angle="35"/></transform>{}</shape></scene>"#,
            noise(""), noise(off_b)
        );
        capture_warnings(|| load_scene_from_str(&xml, Path::new("."), &cfg(), (None, None)).unwrap().0).0
    }
    /// 矩形（ローカルの点 (0.3, 0.2, 0)）の albedo。`which` は 0 / 1 の矩形。
    fn rect_albedo(s: &Scene, which: usize) -> f64 {
        let local = Vec3::new(0.3, 0.2, 0.0);
        // 各矩形の to_world（テスト用に同じ式）でワールドの点と法線を得る
        let xf = if which == 0 {
            Transform::identity()
        } else {
            Transform::translate(Vec3::new(7.0, 2.0, -3.0)).compose(Transform::rotate(Vec3::new(0.0, 1.0, 0.0), 35.0))
        };
        let (pw, nw) = (xf.apply_point(local), xf.apply_normal(Vec3::new(0.0, 0.0, 1.0)));
        let h = s.world.hit(Ray { o: pw + nw * 2.0, d: -nw, time: 0.0 }, 1e-9, 1e30).expect("hit");
        match s.shaders.evaluate(h.mat_id, &crate::shader::ShadeCtx { world: &s.world, hit: &h, time: 0.0 }) {
            Material::Lambert { albedo, .. } => albedo.r(),
            _ => panic!(),
        }
    }

    /// ローカル座標: 違う `to_world` で置いた同じ形は、`offset` が同じなら同じ模様、違えば違う模様。
    /// `space="world"` は従来どおりワールド座標（配置が違えば別の模様）。不正な space は警告して local。
    #[test]
    fn noise_space_local_world_and_offset() {
        let s = placed_noise_scene("", "");
        assert!((rect_albedo(&s, 0) - rect_albedo(&s, 1)).abs() < 1e-9, "同じ offset は同じ模様");
        let s = placed_noise_scene("", r#"<point name="offset" x="5" y="1" z="2"/>"#);
        assert!((rect_albedo(&s, 0) - rect_albedo(&s, 1)).abs() > 1e-3, "offset が違えば別の模様");
        let s = placed_noise_scene(r#"<string name="space" value="world"/>"#, "");
        assert!((rect_albedo(&s, 0) - rect_albedo(&s, 1)).abs() > 1e-3, "world は配置で変わる");
        assert!(!s.shaders.noises[0].local);
        let xml = r#"<scene version="3.0.0"><sensor type="perspective"><float name="fov" value="40"/></sensor><shape type="sphere"><bsdf type="diffuse"><texture type="noise" name="reflectance"><string name="space" value="sideways"/></texture></bsdf></shape></scene>"#;
        let (s, w) = capture_warnings(|| load_scene_from_str(xml, Path::new("."), &cfg(), (None, None)).unwrap().0);
        assert!(s.shaders.noises[0].local && w.iter().any(|m| m.contains("noise space")), "{w:?}");
    }

    // ---- 式の記法（入れ子の <texture>、mul / add / mix、パラメータごとのテクスチャ）----

    fn expr_scene(bsdf: &str) -> (Scene, Vec<String>) {
        let xml = format!(r#"<scene version="3.0.0"><sensor type="perspective"><float name="fov" value="40"/></sensor><shape type="sphere">{bsdf}</shape></scene>"#);
        capture_warnings(|| load_scene_from_str(&xml, Path::new("."), &cfg(), (None, None)).unwrap().0)
    }
    const NOISE_A: &str = r#"<texture type="noise"><string name="pattern" value="fbm"/><float name="scale" value="3"/><rgb name="color0" value="0.1"/><rgb name="color1" value="0.9"/></texture>"#;
    const NOISE_B: &str = r#"<texture type="noise"><string name="pattern" value="turbulence"/><float name="scale" value="5"/></texture>"#;

    /// 粗さ（roughconductor の `alpha`）をノイズで変調できる: 評価した alpha が場所で変わり、[1e-3, 1] に収まる。
    #[test]
    fn roughness_can_be_a_noise_texture() {
        let xml = r#"<bsdf type="roughconductor"><rgb name="specular_reflectance" value="0.9, 0.6, 0.4"/><texture type="noise" name="alpha"><string name="pattern" value="fbm"/><float name="scale" value="6"/><rgb name="color0" value="0.05"/><rgb name="color1" value="0.6"/></texture></bsdf>"#;
        let (s, w) = expr_scene(xml);
        assert!(w.is_empty(), "{w:?}");
        assert!(s.shaders.shaders[0].alpha.is_some() && s.shaders.shaders[0].albedo.is_none());
        let mut seen = Vec::new();
        for i in 0..40 {
            match s.shaders.eval_at(0, &s.world, Vec3::new(0.13 * i as f64, 0.07 * i as f64, 0.3), (0.0, 0.0)) {
                Material::Ggx { alpha, albedo } => {
                    assert!((1e-3..=1.0).contains(&alpha) && (albedo.r() - 0.9).abs() < 1e-12);
                    seen.push(alpha);
                }
                _ => panic!(),
            }
        }
        let (lo, hi) = seen.iter().fold((1.0f64, 0.0f64), |(l, h), &a| (l.min(a), h.max(a)));
        assert!(hi - lo > 0.1, "粗さが場所で変わっていない: {lo} .. {hi}");
    }

    /// `mul` / `add` / `mix` は子の `<texture>` を組み合わせ、評価は各ノイズの値の素直な式とビット一致。定数色は倍率のまま。
    #[test]
    fn mul_add_mix_compose_child_textures() {
        let p = Vec3::new(0.4, 0.3, 0.2);
        let a = |s: &Scene| s.shaders.noises[0].eval(p);
        let b = |s: &Scene| s.shaders.noises[1].eval(p);
        let bits = |c: Color| (c.r().to_bits(), c.g().to_bits(), c.b().to_bits());
        let albedo = |s: &Scene| match s.shaders.eval_at(0, &s.world, p, (0.0, 0.0)) { Material::Lambert { albedo } => albedo, _ => panic!() };
        let white = Color::new(1.0, 1.0, 1.0);
        let mul = format!(r#"<bsdf type="diffuse"><texture type="mul" name="reflectance">{NOISE_A}{NOISE_B}</texture></bsdf>"#);
        let (s, w) = expr_scene(&mul);
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(bits(albedo(&s)), bits(white.hadamard(a(&s).hadamard(b(&s)))));
        let add = format!(r#"<bsdf type="diffuse"><texture type="add" name="reflectance">{NOISE_A}{NOISE_B}</texture></bsdf>"#);
        let (s, _) = expr_scene(&add);
        assert_eq!(bits(albedo(&s)), bits(white.hadamard(a(&s) + b(&s))));
        // mix: weight の float（既定 0.5）と、3 つ目の texture が t
        let mix = format!(r#"<bsdf type="diffuse"><texture type="mix" name="reflectance">{NOISE_A}{NOISE_B}<float name="weight" value="0.25"/></texture></bsdf>"#);
        let (s, _) = expr_scene(&mix);
        assert_eq!(bits(albedo(&s)), bits(white.hadamard(a(&s) * 0.75 + b(&s) * 0.25)));
        let mix_default = format!(r#"<bsdf type="diffuse"><texture type="mix" name="reflectance">{NOISE_A}{NOISE_B}</texture></bsdf>"#);
        let (s, _) = expr_scene(&mix_default);
        assert_eq!(bits(albedo(&s)), bits(white.hadamard(a(&s) * 0.5 + b(&s) * 0.5)));
        // 定数の反射率は倍率として掛かる
        let scaled = format!(r#"<bsdf type="diffuse"><rgb name="reflectance" value="0.5, 1.0, 2.0"/><texture type="mul" name="reflectance">{NOISE_A}{NOISE_B}</texture></bsdf>"#);
        let (s, _) = expr_scene(&scaled);
        assert_eq!(bits(albedo(&s)), bits(Color::new(0.5, 1.0, 2.0).hadamard(a(&s).hadamard(b(&s)))));
    }

    /// 不正な記法（子が足りない・型が不明・深すぎる入れ子）は警告して読めた範囲に倒す（落とさない）。
    #[test]
    fn bad_expressions_warn_and_fall_back() {
        let one_child = format!(r#"<bsdf type="diffuse"><texture type="mul" name="reflectance">{NOISE_A}</texture></bsdf>"#);
        let (s, w) = expr_scene(&one_child);
        assert!(w.iter().any(|m| m.contains("'mul' needs at least 2")), "{w:?}");
        assert!(s.shaders.shaders[0].albedo.is_some(), "1 つは読めたのでそれを使う");
        let none = r#"<bsdf type="diffuse"><texture type="add" name="reflectance"/></bsdf>"#;
        let (s, w) = expr_scene(none);
        assert!(w.iter().any(|m| m.contains("'add' needs at least 2")), "{w:?}");
        assert!(s.shaders.shaders[0].albedo.is_none());
        let unknown = r#"<bsdf type="diffuse"><texture type="checkerboard" name="reflectance"/></bsdf>"#;
        let (s, w) = expr_scene(unknown);
        assert!(w.iter().any(|m| m.contains("unsupported texture type 'checkerboard'")), "{w:?}");
        assert!(s.shaders.shaders[0].albedo.is_none());
        let mut deep = NOISE_A.to_string();
        for _ in 0..19 {
            deep = format!(r#"<texture type="mul">{deep}{NOISE_B}</texture>"#);
        }
        let deep = format!(r#"<texture type="mul" name="reflectance">{deep}{NOISE_B}</texture>"#);
        let (_, w) = expr_scene(&format!(r#"<bsdf type="diffuse">{deep}</bsdf>"#));
        assert!(w.iter().any(|m| m.contains("nested too deeply")), "{w:?}");
        let too_many = format!(r#"<bsdf type="diffuse"><texture type="mix" name="reflectance">{NOISE_A}{NOISE_B}{NOISE_A}{NOISE_B}</texture></bsdf>"#);
        let (_, w) = expr_scene(&too_many);
        assert!(w.iter().any(|m| m.contains("'mix' takes 2 textures")), "{w:?}");
    }

    /// ガラスの屈折率・吸収と、金属の反射率もテクスチャ（式）にできる。
    #[test]
    fn other_parameters_take_textures() {
        let glass = format!(r#"<bsdf type="dielectric"><float name="ext_ior" value="1.0"/><texture type="noise" name="int_ior"><string name="pattern" value="fbm"/><rgb name="color0" value="1.3"/><rgb name="color1" value="1.7"/></texture><texture type="noise" name="absorption"><string name="pattern" value="fbm"/></texture></bsdf>"#);
        let (s, w) = expr_scene(&glass);
        assert!(w.is_empty(), "{w:?}");
        assert!(s.shaders.shaders[0].ior.is_some() && s.shaders.shaders[0].absorption.is_some());
        match s.shaders.eval_at(0, &s.world, Vec3::new(0.4, 0.3, 0.2), (0.0, 0.0)) {
            Material::Dielectric { ior, absorption } => assert!((1.3..=1.7).contains(&ior) && absorption.r() >= 0.0),
            _ => panic!(),
        }
        let metal = format!(r#"<bsdf type="conductor">{}</bsdf>"#, NOISE_A.replacen("<texture type=\"noise\">", "<texture type=\"noise\" name=\"specular_reflectance\">", 1));
        let (s, w) = expr_scene(&metal);
        assert!(w.is_empty(), "{w:?}");
        assert!(matches!(s.shaders.shaders[0].base, Material::Metal { .. }) && s.shaders.shaders[0].albedo.is_some());
    }

    // ---- SDF shape ----

    fn sdf_scene(shape: &str) -> (Scene, Vec<String>) {
        let xml = format!(r#"<scene version="3.0.0"><sensor type="perspective"><float name="fov" value="40"/></sensor>{}</scene>"#, shape);
        let (r, w) = capture_warnings(|| load_scene_from_str(&xml, Path::new("."), &cfg(), (None, None)));
        (r.unwrap().0, w)
    }

    #[test]
    fn sdf_shape_parses_tree_transform_and_material() {
        let (s, w) = sdf_scene(
            r#"<shape type="sdf">
                 <sdf type="smooth_union"><float name="k" value="0.3"/>
                   <sdf type="sphere"><float name="radius" value="1"/><point name="center" x="0" y="0.5" z="0"/></sdf>
                   <sdf type="torus"><float name="major" value="1.2"/><float name="minor" value="0.3"/></sdf>
                   <sdf type="box"><vector name="half" x="0.5" y="0.5" z="0.5"/><float name="round" value="0.1"/></sdf>
                 </sdf>
                 <transform name="to_world"><translate x="0" y="0" z="-5"/></transform>
                 <bsdf type="diffuse"><rgb name="reflectance" value="0.4"/></bsdf></shape>"#,
        );
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(s.world.sdfs().len(), 1);
        let sdf = &s.world.sdfs()[0];
        // 3 つのプリミティブ + 左畳み込みの 2 演算
        assert_eq!(sdf.tree().len(), 5);
        assert_eq!(s.world.instance_count(), 0);
        // 平行移動が効いている: 原点から -Z 方向へ奥に球（半径 1、中心 (0,0.5,-5)）が見える
        let r = crate::ray::Ray { o: Vec3::new(0.0, 0.5, 0.0), d: Vec3::new(0.0, 0.0, -1.0), time: 0.0 };
        let h = s.world.hit(r, 0.0, 1e9).expect("hit");
        assert!(h.t > 3.0 && h.t < 5.0, "t = {}", h.t);
        assert_eq!(h.mat_id, sdf.mat_id);
    }

    #[test]
    fn sdf_difference_folds_left_and_smooth_ops_accept_k() {
        let (s, w) = sdf_scene(
            r#"<shape type="sdf"><sdf type="difference">
                 <sdf type="sphere"><float name="radius" value="2"/></sdf>
                 <sdf type="sphere"><float name="radius" value="0.5"/><point name="center" x="1.5" y="0" z="0"/></sdf>
                 <sdf type="cylinder"><float name="radius" value="0.3"/><float name="half_height" value="3"/></sdf>
               </sdf><bsdf type="diffuse"/></shape>"#,
        );
        assert!(w.is_empty(), "{w:?}");
        let t = s.world.sdfs()[0].tree();
        // 2 番目の球で削った場所（(1.5,0,0)）と、円柱で削った軸（原点）は外側、他は内側
        assert!(t.eval(Vec3::new(1.5, 0.0, 0.0)) > 0.0);
        assert!(t.eval(Vec3::new(0.0, 0.5, 0.0)) > 0.0);
        assert!(t.eval(Vec3::new(-1.0, 0.0, 0.0)) < 0.0);
    }

    #[test]
    fn sdf_invalid_input_warns_and_skips_the_shape() {
        for bad in [
            r#"<shape type="sdf"><bsdf type="diffuse"/></shape>"#, // 根が無い
            r#"<shape type="sdf"><sdf type="sphere"/><sdf type="sphere"/><bsdf type="diffuse"/></shape>"#, // 根が 2 個
            r#"<shape type="sdf"><sdf type="union"><sdf type="sphere"/></sdf><bsdf type="diffuse"/></shape>"#, // 子が 1 個
            r#"<shape type="sdf"><sdf type="teapot"/><bsdf type="diffuse"/></shape>"#, // 未知の型
            r#"<shape type="sdf"><sdf type="sphere"><float name="radius" value="-1"/></sdf><bsdf type="diffuse"/></shape>"#, // 負の半径
            r#"<shape type="sdf"><sdf type="smooth_union"><float name="k" value="-0.1"/><sdf type="sphere"/><sdf type="sphere"/></sdf><bsdf type="diffuse"/></shape>"#, // 負の k
        ] {
            let (s, w) = sdf_scene(bad);
            assert!(!w.is_empty(), "no warning for {bad}");
            assert_eq!(s.world.sdfs().len(), 0, "{bad} should be skipped: {w:?}");
        }
    }

    #[test]
    fn sdf_emitter_warns_but_keeps_the_shape() {
        let (s, w) = sdf_scene(
            r#"<shape type="sdf"><sdf type="sphere"/>
                 <emitter type="area"><rgb name="radiance" value="5"/></emitter>
                 <bsdf type="diffuse"/></shape>"#,
        );
        assert!(w.iter().any(|m| m.contains("never a light source")), "{w:?}");
        assert_eq!(s.world.sdfs().len(), 1);
        assert!(s.world.lights().is_empty(), "an sdf must not become a light");
    }

    #[test]
    fn sdf_to_world_end_animates_without_warning_and_bad_transforms_stay_static() {
        let sdf = |end: &str| format!(r#"<shape type="sdf"><sdf type="sphere"/>{end}<bsdf type="diffuse"/></shape>"#);
        let (s, w) = sdf_scene(&sdf(r#"<transform name="to_world_end"><translate x="3" y="0" z="0"/></transform>"#));
        assert!(w.is_empty(), "{w:?}");
        assert!(s.world.sdfs()[0].is_animated());
        // 時刻 0 は開の位置（原点）、時刻 1 は閉の位置（x = 3）に当たる
        for (time, x) in [(0.0, 0.0), (1.0, 3.0)] {
            let r = crate::ray::Ray { o: Vec3::new(x, 0.0, 10.0), d: Vec3::new(0.0, 0.0, -1.0), time };
            assert!(s.world.hit(r, 0.0, 1e9).is_some(), "time {time}");
        }
        // 特異な閉の変換（x 方向のスケール 0）と鏡像は警告して静止
        for bad in [
            r#"<transform name="to_world_end"><scale x="0" y="1" z="1"/></transform>"#,
            r#"<transform name="to_world_end"><scale x="-1" y="1" z="1"/></transform>"#,
        ] {
            let (s, w) = sdf_scene(&sdf(bad));
            assert!(w.iter().any(|m| m.contains("stays static")), "{w:?}");
            assert!(!s.world.sdfs()[0].is_animated());
        }
    }

    #[test]
    fn sdf_primitive_motion_parses_and_bad_values_warn() {
        let shape = |inner: &str| format!(r#"<shape type="sdf">{inner}<bsdf type="diffuse"/></shape>"#);
        let (s, w) = sdf_scene(&shape(
            r#"<sdf type="smooth_union"><float name="k" value="0.3"/>
                 <sdf type="sphere"><point name="center" x="0" y="0" z="0"/><point name="center_end" x="0" y="2" z="0"/></sdf>
                 <sdf type="capsule"><point name="a_end" x="1" y="0" z="0"/></sdf>
               </sdf>"#,
        ));
        assert!(w.is_empty(), "{w:?}");
        let t = s.world.sdfs()[0].tree();
        // 球（半径 1）は時刻 1 で y = 2 へ: (0, 3, 0) は表面、時刻 0 では外側
        assert!(t.eval_at(Vec3::new(0.0, 3.0, 0.0), 1.0).abs() < 1e-9);
        assert!(t.eval_at(Vec3::new(0.0, 3.0, 0.0), 0.0) > 0.5);
        // 非有限は警告して静止、演算子と種類違いの指定も警告
        for (bad, needle) in [
            (r#"<sdf type="sphere"><point name="center_end" x="nan" y="0" z="0"/></sdf>"#, "center_end is invalid"),
            (r#"<sdf type="capsule"><point name="b_end" x="inf" y="0" z="0"/></sdf>"#, "b_end is invalid"),
            (r#"<sdf type="sphere"><point name="a_end" x="1" y="0" z="0"/></sdf>"#, "does not take a_end"),
            (r#"<sdf type="union"><point name="center_end" x="1" y="0" z="0"/><sdf type="sphere"/><sdf type="sphere"/></sdf>"#, "operator"),
        ] {
            let (s, w) = sdf_scene(&shape(bad));
            assert!(w.iter().any(|m| m.contains(needle)), "{bad}: {w:?}");
            assert_eq!(s.world.sdfs().len(), 1, "the shape itself is kept");
            let t = s.world.sdfs()[0].tree();
            let p = Vec3::new(0.3, 0.2, 0.1);
            assert_eq!(t.eval_at(p, 1.0).to_bits(), t.eval_at(p, 0.0).to_bits(), "{bad} stays static");
        }
    }

    #[test]
    fn sdf_displace_parses_with_defaults_and_bad_input_warns() {
        let shape = |inner: &str| format!(r#"<shape type="sdf">{inner}<bsdf type="diffuse"/></shape>"#);
        // 既定値: fbm、amplitude 0.05、scale 1、octaves 4、lacunarity 2、gain 0.5
        let (s, w) = sdf_scene(&shape(r#"<sdf type="displace"><sdf type="sphere"/></sdf>"#));
        assert!(w.is_empty(), "{w:?}");
        let t = s.world.sdfs()[0].tree();
        assert_eq!(t.len(), 2);
        assert!(t.lipschitz() > 1.0);
        let p = Vec3::new(0.4, 0.9, 0.3);
        let expect = p.len() - 1.0 + 0.05 * crate::noise::fbm(p, 4, 2.0, 0.5);
        assert!((t.eval(p) - expect).abs() < 1e-12, "{} vs {}", t.eval(p), expect);
        // 明示した値
        let (s, w) = sdf_scene(&shape(
            r#"<sdf type="displace"><string name="pattern" value="perlin"/><float name="amplitude" value="0.2"/><float name="scale" value="4"/>
                 <vector name="offset" x="1" y="2" z="3"/><sdf type="sphere"/></sdf>"#,
        ));
        assert!(w.is_empty(), "{w:?}");
        let t = s.world.sdfs()[0].tree();
        let expect = p.len() - 1.0 + 0.2 * crate::noise::perlin((p + Vec3::new(1.0, 2.0, 3.0)) * 4.0);
        assert!((t.eval(p) - expect).abs() < 1e-12);
        // 未知のパターンは警告して fbm、範囲外のパラメータは警告して丸める
        let (s, w) = sdf_scene(&shape(r#"<sdf type="displace"><string name="pattern" value="marble"/><sdf type="sphere"/></sdf>"#));
        assert!(w.iter().any(|m| m.contains("unknown displace pattern")), "{w:?}");
        assert_eq!(s.world.sdfs().len(), 1);
        let (_, w) = sdf_scene(&shape(r#"<sdf type="displace"><float name="scale" value="-3"/><integer name="octaves" value="99"/><sdf type="sphere"/></sdf>"#));
        assert!(w.iter().any(|m| m.contains("clamped")), "{w:?}");
        // 子が 0 個・2 個以上は警告してシェープごと飛ばす
        for bad in [
            r#"<sdf type="displace"></sdf>"#,
            r#"<sdf type="displace"><sdf type="sphere"/><sdf type="sphere"/></sdf>"#,
            r#"<sdf type="displace"><float name="amplitude" value="inf"/><sdf type="sphere"/></sdf>"#,
        ] {
            let (s, w) = sdf_scene(&shape(bad));
            assert!(!w.is_empty(), "{bad}");
            assert_eq!(s.world.sdfs().len(), 0, "{bad}: {w:?}");
        }
    }
}
