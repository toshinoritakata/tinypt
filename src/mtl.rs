//! Wavefront MTL パーサ（T2）。純粋な構文解析だけを行い、マテリアルへの変換は
//! [`crate::mitsuba`] 側で行う（警告とテクスチャ読み込みがそちらにあるため）。
//!
//! 実データの癖（Sponza / Rungholt で確認済み）:
//! - 行頭が**タブ**（`\tKd 1 1 1`）。行頭の空白は無視する。
//! - テクスチャのパスが Windows 形式（`textures\foo.png`）。`/` に直す。
//! - 未知のキーは読み飛ばす。`#` 以降はコメント。
//! - `newmtl` が重複したら**最初の定義を残し**、後のブロックは捨てる（`dup_names` に記録）。

/// MTL の 1 材質（対応キーだけ）。
#[derive(Clone, Debug, PartialEq)]
pub struct MtlMaterial {
    pub name: String,
    pub kd: [f64; 3],
    pub ks: [f64; 3],
    pub ke: [f64; 3],
    pub ns: f64,
    /// `d`（不透明度）。`Tr`（= 1 − d）が指定されたらそこから換算する
    pub d: f64,
    /// `map_Kd` のパス（`/` 区切りに正規化済み、MTL からの相対）
    pub map_kd: Option<String>,
    /// `map_d`（T3 まで未対応）
    pub map_d: Option<String>,
    /// `map_bump` / `bump`（ハイトマップ）
    pub map_bump: Option<String>,
    /// `map_bump` の `-bm` 強度（既定 1.0）
    pub bm: f64,
    /// `norm`（MTL 拡張: タンジェント空間ノーマルマップ）
    pub norm: Option<String>,
    /// `map_Ka`（未対応）
    pub map_ka: Option<String>,
}

impl MtlMaterial {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            kd: [0.8, 0.8, 0.8],
            ks: [0.0; 3],
            ke: [0.0; 3],
            ns: 0.0,
            d: 1.0,
            map_kd: None,
            map_d: None,
            map_bump: None,
            bm: 1.0,
            norm: None,
            map_ka: None,
        }
    }
}

/// 解析結果。
#[derive(Default, Debug)]
pub struct MtlFile {
    pub materials: Vec<MtlMaterial>,
    /// 重複した `newmtl` の名前（後の定義は捨てた）
    pub dup_names: Vec<String>,
}

impl MtlFile {
    pub fn get(&self, name: &str) -> Option<&MtlMaterial> {
        self.materials.iter().find(|m| m.name == name)
    }
}

/// テクスチャ行のオプション（使うものだけ）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MapOptions {
    /// `-bm`（バンプ強度）。指定が無ければ 1.0
    pub bm: f64,
}

impl Default for MapOptions {
    fn default() -> Self {
        Self { bm: 1.0 }
    }
}

/// 既知のテクスチャオプションと、その引数の個数。
const MAP_FLAGS: &[(&str, usize)] = &[
    ("-bm", 1), ("-s", 3), ("-o", 3), ("-t", 3), ("-mm", 2), ("-imfchan", 1), ("-texres", 1),
    ("-type", 1), ("-clamp", 1), ("-blendu", 1), ("-blendv", 1), ("-cc", 1), ("-boost", 1),
];

/// テクスチャ行の引数を「パス + オプション」に分ける。既知のオプション（[`MAP_FLAGS`]）は引数の個数ぶん
/// 読み飛ばし、残りの行全体をパスとする（空白入りファイル名を許す）。`\` は `/` に直す。
/// 未知の `-x` に当たったら、従来どおり「最後のトークンだけをパスにする」挙動に退避する。
fn parse_map_args(rest: &str) -> (Option<String>, MapOptions) {
    let rest = rest.trim();
    let mut opts = MapOptions::default();
    let mut pos = 0usize;
    // 次のトークン（開始, 終了）を返す
    let next = |from: usize| -> Option<(usize, usize)> {
        let bytes = rest.as_bytes();
        let mut i = from;
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            return None;
        }
        let start = i;
        while i < bytes.len() && !(bytes[i] as char).is_whitespace() {
            i += 1;
        }
        Some((start, i))
    };
    loop {
        let Some((s, e)) = next(pos) else { break };
        let tok = &rest[s..e];
        if !tok.starts_with('-') {
            break;
        }
        match MAP_FLAGS.iter().find(|(f, _)| *f == tok) {
            Some(&(flag, n)) => {
                let mut p = e;
                let mut first: Option<&str> = None;
                for k in 0..n {
                    match next(p) {
                        Some((s2, e2)) => {
                            if k == 0 {
                                first = Some(&rest[s2..e2]);
                            }
                            p = e2;
                        }
                        None => return (None, opts), // 引数が足りない
                    }
                }
                if flag == "-bm" {
                    if let Some(v) = first.and_then(|t| t.parse::<f64>().ok()) {
                        opts.bm = v;
                    }
                }
                pos = p;
            }
            None => {
                // 未知のオプション: 最後のトークンだけをパスにする
                let last = rest.split_whitespace().last().unwrap_or("");
                let path = if last.is_empty() { None } else { Some(last.replace('\\', "/")) };
                return (path, MapOptions::default());
            }
        }
    }
    let path = rest[pos..].trim();
    if path.is_empty() {
        (None, opts)
    } else {
        (Some(path.replace('\\', "/")), opts)
    }
}

/// パスだけが要るとき（`-bm` 以外のオプションは無視）。
fn parse_map_path(rest: &str) -> Option<String> {
    parse_map_args(rest).0
}

fn parse3(rest: &str, fallback: [f64; 3]) -> [f64; 3] {
    let v: Vec<f64> = rest.split_whitespace().map_while(|t| t.parse().ok()).collect();
    match v.len() {
        // `Kd 0.5`（グレー 1 値）も許す
        1 => [v[0]; 3],
        n if n >= 3 => [v[0], v[1], v[2]],
        _ => fallback,
    }
}

fn first_f64(rest: &str) -> Option<f64> {
    rest.split_whitespace().next().and_then(|t| t.parse().ok())
}

/// MTL テキストを解析する。壊れた行は読み飛ばす（エラーにはしない）。
pub fn parse_mtl(text: &str) -> MtlFile {
    let mut out = MtlFile::default();
    let mut cur: Option<MtlMaterial> = None;
    // 重複ブロックの中身は捨てる
    let mut skipping = false;

    for raw in text.lines() {
        // `#` 以降はコメント。行頭のタブ・空白は無視
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let (key, rest) = match line.split_once(char::is_whitespace) {
            Some((k, r)) => (k, r.trim()),
            None => (line, ""),
        };
        if key == "newmtl" {
            if let Some(m) = cur.take() {
                out.materials.push(m);
            }
            let known = out.materials.iter().any(|m| m.name == rest);
            skipping = known;
            if known {
                out.dup_names.push(rest.to_string());
            } else {
                cur = Some(MtlMaterial::new(rest));
            }
            continue;
        }
        if skipping {
            continue;
        }
        let m = match cur.as_mut() {
            Some(m) => m,
            None => continue, // newmtl 前の行
        };
        match key {
            "Kd" => m.kd = parse3(rest, m.kd),
            "Ks" => m.ks = parse3(rest, m.ks),
            "Ke" => m.ke = parse3(rest, m.ke),
            "Ns" => m.ns = first_f64(rest).unwrap_or(m.ns),
            "d" => m.d = first_f64(rest).unwrap_or(m.d),
            "Tr" => {
                if let Some(tr) = first_f64(rest) {
                    m.d = 1.0 - tr;
                }
            }
            "map_Kd" => m.map_kd = parse_map_path(rest),
            "map_d" => m.map_d = parse_map_path(rest),
            "map_bump" | "bump" | "map_Bump" => {
                let (path, opts) = parse_map_args(rest);
                m.map_bump = path;
                m.bm = opts.bm;
            }
            "norm" => m.norm = parse_map_path(rest),
            "map_Ka" => m.map_ka = parse_map_path(rest),
            _ => {} // Ni / Tf / illum / Ka など: 使わない
        }
    }
    if let Some(m) = cur.take() {
        out.materials.push(m);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tab_indented_sponza_style_block() {
        let t = "# c\n\nnewmtl leaf\n\tNs 10.0000\n\tNi 1.5\n\td 1.0000\n\tTr 0.0000\n\tTf 1 1 1 \n\tillum 2\n\
                 \tKa 1 1 1\n\tKd 0.5 0.25 0.125\n\tKs 0 0 0\n\tKe 0.0 0.0 0.0\n\tmap_Kd textures\\sponza_thorn_diff.png\n\
                 \tmap_d textures\\a_mask.png\n\tmap_bump textures\\b.png\n\tmap_Ka textures\\sponza_thorn_diff.png\n";
        let f = parse_mtl(t);
        assert_eq!(f.materials.len(), 1);
        let m = &f.materials[0];
        assert_eq!(m.name, "leaf");
        assert_eq!(m.kd, [0.5, 0.25, 0.125]);
        assert_eq!(m.ns, 10.0);
        assert_eq!(m.d, 1.0);
        assert_eq!(m.map_kd.as_deref(), Some("textures/sponza_thorn_diff.png"));
        assert_eq!(m.map_d.as_deref(), Some("textures/a_mask.png"));
        assert_eq!(m.map_bump.as_deref(), Some("textures/b.png"));
        assert_eq!(m.map_ka.as_deref(), Some("textures/sponza_thorn_diff.png"));
    }

    #[test]
    fn unknown_keys_comments_and_pre_newmtl_lines_are_ignored() {
        let f = parse_mtl("Kd 1 0 0\nnewmtl A\nfoo bar baz\nKd 0 1 0 # trailing\n# Tr 1\n");
        assert_eq!(f.materials.len(), 1);
        assert_eq!(f.materials[0].kd, [0.0, 1.0, 0.0]);
        assert_eq!(f.materials[0].d, 1.0);
    }

    #[test]
    fn duplicate_newmtl_keeps_first_and_records_the_name() {
        let f = parse_mtl("newmtl A\nKd 1 0 0\nnewmtl B\nKd 0 0 1\nnewmtl A\nKd 0 1 0\nmap_Kd x.png\n");
        assert_eq!(f.materials.len(), 2);
        assert_eq!(f.get("A").unwrap().kd, [1.0, 0.0, 0.0]);
        assert!(f.get("A").unwrap().map_kd.is_none(), "捨てた重複ブロックの map_Kd が混ざった");
        assert_eq!(f.dup_names, vec!["A".to_string()]);
    }

    #[test]
    fn empty_input_tr_conversion_and_texture_options() {
        assert!(parse_mtl("").materials.is_empty());
        let f = parse_mtl("newmtl A\nTr 0.25\nmap_Kd -s 2 2 2 dir\\t.png\nKd 0.4\n");
        let m = &f.materials[0];
        assert!((m.d - 0.75).abs() < 1e-12);
        assert_eq!(m.map_kd.as_deref(), Some("dir/t.png"));
        assert_eq!(m.kd, [0.4; 3]);
    }

    #[test]
    fn map_options_are_consumed_and_bm_is_read() {
        let a = parse_map_args("-bm 0.5 textures/x.png");
        assert_eq!(a.0.as_deref(), Some("textures/x.png"));
        assert_eq!(a.1.bm, 0.5);
        let a = parse_map_args("-s 1 1 1 -bm 2 dir\\t.png");
        assert_eq!((a.0.as_deref(), a.1.bm), (Some("dir/t.png"), 2.0));
        // 空白入りファイル名（オプションの後ろは行の残り全体）
        let a = parse_map_args("-bm 3 my dir/a b.png");
        assert_eq!((a.0.as_deref(), a.1.bm), (Some("my dir/a b.png"), 3.0));
        // オプション無し: 既定 bm = 1、パスは行全体
        let a = parse_map_args("plain name.png");
        assert_eq!((a.0.as_deref(), a.1.bm), (Some("plain name.png"), 1.0));
        // 未知のフラグは「最後のトークン」に退避（bm は既定）
        let a = parse_map_args("-zz 1 file.png");
        assert_eq!((a.0.as_deref(), a.1.bm), (Some("file.png"), 1.0));
        // 引数が足りない
        assert_eq!(parse_map_args("-bm").0, None);
    }

    #[test]
    fn map_bump_bm_and_norm_keys_are_parsed() {
        let f = parse_mtl("newmtl A\n\tmap_Bump -bm 0.5 t\\b.png\n\tnorm n.png\nnewmtl B\n\tbump h.png\n");
        let a = f.get("A").unwrap();
        assert_eq!((a.map_bump.as_deref(), a.bm, a.norm.as_deref()), (Some("t/b.png"), 0.5, Some("n.png")));
        let b = f.get("B").unwrap();
        assert_eq!((b.map_bump.as_deref(), b.bm, b.norm.as_deref()), (Some("h.png"), 1.0, None));
    }
}
