//! Fuzz target: `EncodedChunk::from_bytes` の panic-freedom
//!
//! ALICE-Codec の `.alc` bitstream は attacker-controlled data になり得る
//! (ネットワーク配信 / disk load) 任意 bytes を食わせても panic せず
//! Result で返ることを保証する
//!
//! canonical CI template [[reference_alice_ci_canonical_template]] 準拠

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // 入力サイズ上限 (fuzz corpus 巨大化予防、1 MB 相当)
    if data.len() > 1024 * 1024 {
        return;
    }

    // from_bytes は Result を返し、panic すべきではない
    let _ = alice_codec::pipeline::EncodedChunk::from_bytes(data);
});
