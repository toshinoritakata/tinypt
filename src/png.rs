//! PNG 書き出し（8bit RGB、インターレースなし、外部クレートなし）。
//!
//! 画素値は呼び出し側が渡す（`output::ppm_bytes` が作る 8bit sRGB のバイト列をそのまま使う。トーンマップや露出は
//! ここでは扱わない）。エンコードは可逆なので、デコードすれば渡したバイト列と 1 ビットも違わず一致する。
//!
//! - **フィルタ**: 行ごとに None / Sub / Up / Paeth のうち、フィルタ後のバイトを符号付きとみた絶対値の和が
//!   最小のものを選ぶ（PNG 仕様が勧める簡単な基準。フラットな領域は 0 が並び、勾配は Sub / Up で小さくなる）。
//! - **deflate**: 1 ブロックの**固定ハフマン（BTYPE=01）+ 単純な LZ77**（3 バイトのハッシュに直近 1 候補、
//!   窓 32KiB、貪欲）。動的ハフマンや遅延マッチは実装しない（PPM より十分小さければよい）。
//! - IDAT は zlib（ヘッダ 0x78 0x01 + deflate + Adler-32）で 1 チャンク。CRC-32 は `const` のテーブル。

use std::io::Write;

const CRC_TABLE: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut n = 0;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        t[n] = c;
        n += 1;
    }
    t
};

/// CRC-32（PNG / zlib 共通、多項式 0xEDB88320）。
pub fn crc32(parts: &[&[u8]]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for p in parts {
        for &b in *p {
            c = CRC_TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
        }
    }
    !c
}

/// Adler-32（zlib）。
pub fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let (mut a, mut b) = (1u32, 0u32);
    // 5552 バイトごとに剰余を取れば u32 でオーバーフローしない
    for chunk in data.chunks(5552) {
        for &x in chunk {
            a += x as u32;
            b += a;
        }
        a %= MOD;
        b %= MOD;
    }
    (b << 16) | a
}

/// 行フィルタ（Sub / Up / Paeth）の予測子。`bpp` = 3。
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let (ia, ib, ic) = (a as i32, b as i32, c as i32);
    let p = ia + ib - ic;
    let (pa, pb, pc) = ((p - ia).abs(), (p - ib).abs(), (p - ic).abs());
    if pa <= pb && pa <= pc { a } else if pb <= pc { b } else { c }
}

/// 走査行をフィルタして `[filter_type, bytes..]` を返す。行ごとに符号付き絶対値和が最小のフィルタを選ぶ。
fn filter_rows(w: usize, h: usize, rgb: &[u8]) -> Vec<u8> {
    let stride = w * 3;
    let mut out = Vec::with_capacity(h * (stride + 1));
    let zero = vec![0u8; stride];
    let mut cand: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; stride]);
    for y in 0..h {
        let cur = &rgb[y * stride..(y + 1) * stride];
        let prev: &[u8] = if y == 0 { &zero } else { &rgb[(y - 1) * stride..y * stride] };
        for i in 0..stride {
            let left = if i >= 3 { cur[i - 3] } else { 0 };
            let up = prev[i];
            let ul = if i >= 3 { prev[i - 3] } else { 0 };
            cand[0][i] = cur[i];
            cand[1][i] = cur[i].wrapping_sub(left);
            cand[2][i] = cur[i].wrapping_sub(up);
            cand[3][i] = cur[i].wrapping_sub(paeth(left, up, ul));
        }
        let cost = |v: &Vec<u8>| v.iter().map(|&b| (b as i8 as i32).unsigned_abs() as u64).sum::<u64>();
        let best = (0..4).min_by_key(|&k| cost(&cand[k])).unwrap();
        out.push([0u8, 1, 2, 4][best]); // PNG のフィルタ型: 0 None / 1 Sub / 2 Up / 4 Paeth（3 は Average で使わない）
        out.extend_from_slice(&cand[best]);
    }
    out
}

/// 下位ビットから詰めるビットライタ。
struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl BitWriter {
    fn put(&mut self, value: u32, bits: u32) {
        self.acc |= (value as u64) << self.n;
        self.n += bits;
        while self.n >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }
    /// ハフマン符号は上位ビットから出す（ビット反転して `put`）。
    fn put_code(&mut self, code: u32, bits: u32) {
        self.put(code.reverse_bits() >> (32 - bits), bits);
    }
    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.out.push(self.acc as u8);
        }
        self.out
    }
}

/// 固定ハフマンのリテラル / 長さ符号（RFC 1951 3.2.6）。
fn put_lit_len(bw: &mut BitWriter, sym: u32) {
    match sym {
        0..=143 => bw.put_code(0x30 + sym, 8),
        144..=255 => bw.put_code(0x190 + (sym - 144), 9),
        256..=279 => bw.put_code(sym - 256, 7),
        _ => bw.put_code(0xC0 + (sym - 280), 8),
    }
}

const LEN_BASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE: [u16; 30] = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

/// 単純な LZ77 + 固定ハフマンの deflate（1 ブロック、最終ブロック）。
fn deflate_fixed(data: &[u8]) -> Vec<u8> {
    const WINDOW: usize = 32768;
    const HASH_BITS: u32 = 15;
    let mut bw = BitWriter { out: Vec::with_capacity(data.len() / 2 + 16), acc: 0, n: 0 };
    bw.put(1, 1); // BFINAL
    bw.put(1, 2); // BTYPE = 01（固定ハフマン）
    let mut head = vec![u32::MAX; 1 << HASH_BITS];
    let hash = |i: usize| -> usize {
        let v = (data[i] as u32) | (data[i + 1] as u32) << 8 | (data[i + 2] as u32) << 16;
        (v.wrapping_mul(0x9E37_79B1) >> (32 - HASH_BITS)) as usize
    };
    let mut i = 0;
    while i < data.len() {
        let mut best_len = 0;
        let mut best_dist = 0;
        if i + 3 <= data.len() {
            let h = hash(i);
            let cand = head[h];
            head[h] = i as u32;
            if cand != u32::MAX {
                let c = cand as usize;
                let dist = i - c;
                if dist <= WINDOW {
                    let max = (data.len() - i).min(258);
                    let mut l = 0;
                    while l < max && data[c + l] == data[i + l] {
                        l += 1;
                    }
                    if l >= 3 {
                        best_len = l;
                        best_dist = dist;
                    }
                }
            }
        }
        if best_len >= 3 {
            let li = LEN_BASE.iter().rposition(|&b| b as usize <= best_len).unwrap();
            put_lit_len(&mut bw, 257 + li as u32);
            bw.put((best_len - LEN_BASE[li] as usize) as u32, LEN_EXTRA[li] as u32);
            let di = DIST_BASE.iter().rposition(|&b| b as usize <= best_dist).unwrap();
            bw.put_code(di as u32, 5);
            bw.put((best_dist - DIST_BASE[di] as usize) as u32, DIST_EXTRA[di] as u32);
            // 一致区間の位置もハッシュに登録する（次の一致を見つけやすくする）
            for k in 1..best_len {
                if i + k + 3 <= data.len() {
                    head[hash(i + k)] = (i + k) as u32;
                }
            }
            i += best_len;
        } else {
            put_lit_len(&mut bw, data[i] as u32);
            i += 1;
        }
    }
    put_lit_len(&mut bw, 256); // ブロック終端
    bw.finish()
}

fn write_chunk(out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(tag);
    out.extend_from_slice(data);
    out.extend_from_slice(&crc32(&[tag, data]).to_be_bytes());
}

/// 8bit RGB（行優先、`w*h*3` バイト）を PNG のバイト列にする。`w`, `h` は 1 以上。
pub fn encode_rgb8(w: usize, h: usize, rgb: &[u8]) -> Vec<u8> {
    assert_eq!(rgb.len(), w * h * 3, "rgb length must be w*h*3");
    assert!(w >= 1 && h >= 1 && w <= u32::MAX as usize && h <= u32::MAX as usize);
    let filtered = filter_rows(w, h, rgb);
    let mut z = vec![0x78, 0x01];
    z.extend_from_slice(&deflate_fixed(&filtered));
    z.extend_from_slice(&adler32(&filtered).to_be_bytes());
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8bit、カラータイプ 2（RGB）、deflate、フィルタ 0、インターレースなし
    write_chunk(&mut out, b"IHDR", &ihdr);
    write_chunk(&mut out, b"IDAT", &z);
    write_chunk(&mut out, b"IEND", &[]);
    out
}

/// PNG ファイルを書く。
pub fn write_png(path: &str, w: usize, h: usize, rgb: &[u8]) -> std::io::Result<()> {
    let bytes = encode_rgb8(w, h, rgb);
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&bytes)?;
    f.flush()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// 固定ハフマン / 非圧縮ブロックだけの inflate（テスト用デコーダ。このエンコーダの出力を検証できれば足りる）。
    fn inflate(data: &[u8]) -> Vec<u8> {
        let mut pos = 0usize; // ビット位置
        let bit = |pos: &mut usize| -> u32 {
            let b = (data[*pos / 8] >> (*pos % 8)) & 1;
            *pos += 1;
            b as u32
        };
        let bits = |pos: &mut usize, n: u32| -> u32 { (0..n).fold(0, |acc, k| acc | bit(pos) << k) };
        let mut out: Vec<u8> = Vec::new();
        loop {
            let last = bit(&mut pos);
            let btype = bits(&mut pos, 2);
            assert_eq!(btype, 1, "固定ハフマンのみ");
            loop {
                // 固定ハフマンのリテラル / 長さ符号を 1 ビットずつ読む
                let mut code = 0u32;
                let mut len = 0;
                let sym = loop {
                    code = (code << 1) | bit(&mut pos);
                    len += 1;
                    match len {
                        7 if code <= 0x17 => break 256 + code,
                        8 if (0x30..=0xBF).contains(&code) => break code - 0x30,
                        8 if (0xC0..=0xC7).contains(&code) => break 280 + (code - 0xC0),
                        9 if code >= 0x190 => break 144 + (code - 0x190),
                        _ => {}
                    }
                    assert!(len < 9, "不正な符号");
                };
                if sym < 256 {
                    out.push(sym as u8);
                } else if sym == 256 {
                    break;
                } else {
                    let li = (sym - 257) as usize;
                    let l = LEN_BASE[li] as usize + bits(&mut pos, LEN_EXTRA[li] as u32) as usize;
                    let mut dcode = 0u32;
                    for _ in 0..5 {
                        dcode = (dcode << 1) | bit(&mut pos);
                    }
                    let di = dcode as usize;
                    let d = DIST_BASE[di] as usize + bits(&mut pos, DIST_EXTRA[di] as u32) as usize;
                    assert!(d <= out.len());
                    for _ in 0..l {
                        out.push(out[out.len() - d]);
                    }
                }
            }
            if last == 1 {
                return out;
            }
        }
    }

    /// PNG をデコードして (w, h, rgb) を返す。チャンクの CRC・zlib ヘッダ・Adler-32・IHDR も検証する。
    pub(crate) fn decode(png: &[u8]) -> (usize, usize, Vec<u8>) {
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        let mut pos = 8;
        let (mut w, mut h) = (0, 0);
        let mut idat = Vec::new();
        let mut seen_end = false;
        while pos < png.len() {
            let len = u32::from_be_bytes(png[pos..pos + 4].try_into().unwrap()) as usize;
            let tag = &png[pos + 4..pos + 8];
            let data = &png[pos + 8..pos + 8 + len];
            let crc = u32::from_be_bytes(png[pos + 8 + len..pos + 12 + len].try_into().unwrap());
            assert_eq!(crc, crc32(&[tag, data]), "CRC mismatch in {:?}", std::str::from_utf8(tag));
            match tag {
                b"IHDR" => {
                    w = u32::from_be_bytes(data[0..4].try_into().unwrap()) as usize;
                    h = u32::from_be_bytes(data[4..8].try_into().unwrap()) as usize;
                    assert_eq!(&data[8..], &[8, 2, 0, 0, 0]);
                }
                b"IDAT" => idat.extend_from_slice(data),
                b"IEND" => seen_end = true,
                _ => panic!("unexpected chunk"),
            }
            pos += 12 + len;
        }
        assert!(seen_end && pos == png.len());
        assert_eq!((idat[0] & 0x0F, (idat[0] as u32 * 256 + idat[1] as u32) % 31), (8, 0), "zlib header");
        let raw = inflate(&idat[2..idat.len() - 4]);
        assert_eq!(u32::from_be_bytes(idat[idat.len() - 4..].try_into().unwrap()), adler32(&raw), "Adler-32");
        let stride = w * 3;
        assert_eq!(raw.len(), h * (stride + 1));
        let mut rgb = vec![0u8; h * stride];
        for y in 0..h {
            let f = raw[y * (stride + 1)];
            for i in 0..stride {
                let x = raw[y * (stride + 1) + 1 + i];
                let left = if i >= 3 { rgb[y * stride + i - 3] } else { 0 };
                let up = if y > 0 { rgb[(y - 1) * stride + i] } else { 0 };
                let ul = if y > 0 && i >= 3 { rgb[(y - 1) * stride + i - 3] } else { 0 };
                rgb[y * stride + i] = x.wrapping_add(match f {
                    0 => 0,
                    1 => left,
                    2 => up,
                    4 => paeth(left, up, ul),
                    _ => panic!("filter {f}"),
                });
            }
        }
        (w, h, rgb)
    }

    #[test]
    fn crc_and_adler_known_values() {
        assert_eq!(crc32(&[b"123456789"]), 0xCBF4_3926);
        assert_eq!(crc32(&[b"IEND"]), 0xAE42_6082); // 空の IEND チャンクの CRC
        assert_eq!(crc32(&[b"1234", b"56789"]), 0xCBF4_3926, "分割しても同じ");
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        assert_eq!(adler32(b""), 1);
        assert_eq!(adler32(&vec![0xFFu8; 100_000]), { let (mut a, mut b) = (1u64, 0u64); for _ in 0..100_000 { a = (a + 255) % 65521; b = (b + a) % 65521; } ((b << 16) | a) as u32 });
    }

    fn gradient_noise(w: usize, h: usize) -> Vec<u8> {
        let mut s = 12345u64;
        let mut v = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                s = crate::rng::splitmix64(s);
                let n = (s >> 60) as usize; // 0..15 の小さなノイズ
                v.extend_from_slice(&[((x * 255 / w.max(1)) + n) as u8, (y * 255 / h.max(1)) as u8, if (x / 8 + y / 8) % 2 == 0 { 30 } else { 200 }]);
            }
        }
        v
    }

    /// 1x1、幅 1、高さ 1、一様、大きな画像で、デコードすると入力と 1 バイトも違わない。
    #[test]
    fn roundtrip_is_lossless_for_edge_sizes() {
        for (w, h) in [(1, 1), (1, 7), (9, 1), (2, 2), (17, 13), (640, 3), (300, 200)] {
            let rgb = gradient_noise(w, h);
            let png = encode_rgb8(w, h, &rgb);
            assert_eq!(decode(&png), (w, h, rgb), "{w}x{h}");
        }
        // 一様（長い一致・距離 3 の 258 バイト一致の連続）と全 0 / 全 255
        for v in [0u8, 255, 77] {
            let rgb = vec![v; 500 * 400 * 3];
            let png = encode_rgb8(500, 400, &rgb);
            assert_eq!(decode(&png), (500, 400, rgb));
            assert!(png.len() < 5000, "一様な画像はごく小さい: {}", png.len());
        }
        // 全バイト値（リテラルの 8 / 9 ビット符号の全域）
        let rgb: Vec<u8> = (0..256 * 3).map(|i| (i * 7 + i / 3) as u8).collect();
        assert_eq!(decode(&encode_rgb8(256, 1, &rgb)).2, rgb);
    }

    /// ノイズだけの画像でも PPM（w*h*3 + ヘッダ）より大きくならない範囲で（リテラルの 9 ビット符号が最悪で 1.125 倍）。
    #[test]
    fn smooth_image_is_smaller_than_ppm() {
        let (w, h) = (512, 288);
        let rgb: Vec<u8> = (0..w * h).flat_map(|i| { let (x, y) = (i % w, i / w); [(x / 3) as u8, (y / 2) as u8, 128] }).collect();
        let png = encode_rgb8(w, h, &rgb);
        assert!(png.len() * 4 < rgb.len(), "{} vs {}", png.len(), rgb.len());
        assert_eq!(decode(&png).2, rgb);
    }
}
