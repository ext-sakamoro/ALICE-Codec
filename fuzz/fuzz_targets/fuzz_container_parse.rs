//! Fuzz target: MP4 (ISO BMFF) + Matroska/WebM container parser の panic-freedom
//!
//! ALICE-Codec は独自 codec bitstream を標準 container に格納/読み取り可能に
//! するため MP4 / MKV parser を露出している 任意 bytes (攻撃者制御 file /
//! network stream) を食わせても panic せず Option/Vec で返ることを保証する
//!
//! canonical CI template [[reference_alice_ci_canonical_template]] 準拠

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // 入力サイズ上限 (fuzz corpus 巨大化予防、1 MB 相当)
    if data.len() > 1024 * 1024 {
        return;
    }

    // 1. Format detection は panic すべきでない
    let _ = alice_codec::container::detect_format(data);
    let _ = alice_codec::container::is_mp4(data);
    let _ = alice_codec::container::is_matroska(data);

    // 2. MP4 box 走査 (parse_mp4_box は Option、list_mp4_boxes は Vec)
    let _ = alice_codec::container::list_mp4_boxes(data);
    if let Some(mp4box) = alice_codec::container::parse_mp4_box(data, 0) {
        let _ = alice_codec::container::box_type_str(&mp4box.box_type);
    }
    let _ = alice_codec::container::parse_ftyp(data);

    // 3. EBML (Matroska/WebM) element 走査
    let _ = alice_codec::container::list_ebml_elements(data);
    let _ = alice_codec::container::parse_ebml_element(data, 0);

    // 4. VINT decode (Matroska variable-length integer)
    let _ = alice_codec::container::decode_vint(data, 0);
    let _ = alice_codec::container::decode_vint_size(data, 0);
});
