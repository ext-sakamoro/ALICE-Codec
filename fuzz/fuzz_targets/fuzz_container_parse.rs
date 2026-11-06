//! Fuzz target: MP4 (ISO BMFF) + Matroska/WebM container parser の panic-freedom
//!
//! ALICE-Codec は独自 codec bitstream を標準 container に格納/読み取り可能に
//! するため MP4 / MKV parser を露出している 任意 bytes (攻撃者制御 file /
//! network stream) を食わせても panic せず Result で返ることを保証する
//!

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

    // 2. MP4 box 走査: 成功した walk は入力の中で隙間なく終わる
    if let Ok(boxes) = alice_codec::container::list_mp4_boxes(data) {
        for b in &boxes {
            assert!(b.data_offset as u64 + b.payload_size <= data.len() as u64);
        }
        if let Some(last) = boxes.last() {
            assert_eq!(last.data_offset as u64 + last.payload_size, data.len() as u64);
        }
    }
    if let Ok(Some(mp4box)) = alice_codec::container::parse_mp4_box(data, 0) {
        let _ = alice_codec::container::box_type_str(&mp4box.box_type);
    }
    let _ = alice_codec::container::parse_ftyp(data);

    // 3. EBML (Matroska/WebM) element 走査
    if let Ok(elements) = alice_codec::container::list_ebml_elements(data) {
        if let Some(last) = elements.last() {
            assert_eq!(last.data_offset as u64 + last.data_size, data.len() as u64);
        }
    }
    let _ = alice_codec::container::parse_ebml_element(data, 0);

    // 4. VINT decode (Matroska variable-length integer)
    let _ = alice_codec::container::decode_vint(data, 0);
    let _ = alice_codec::container::decode_vint_size(data, 0);
});
