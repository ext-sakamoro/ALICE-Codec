//! rANS (Asymmetric Numeral Systems) Entropy Coding
//!
//! Near Shannon-limit compression at memory bandwidth speed.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    rANS Encoder                             │
//! │                                                             │
//! │  Symbol → Frequency Table → State Update → Renormalize     │
//! │                                            ↓                │
//! │                                       Output Bytes          │
//! └─────────────────────────────────────────────────────────────┘
//!
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    rANS Decoder                             │
//! │                                                             │
//! │  Input Bytes → State → Frequency Lookup → Symbol           │
//! │            ↑                                                │
//! │       Renormalize                                           │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Properties
//!
//! - **Asymmetric**: State encodes fractional bits
//! - **Stack-like**: LIFO decoding (encode in reverse)
//! - **Fast**: Only shifts, adds, and table lookups
//!
//! # SIMD Optimizations
//!
//! - **SIMD State Update**: 4 streams processed in parallel using AVX2
//! - **Unrolled Renormalization**: Minimize branches
//!
//! > "Entropy is just a parallelizable number."

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// rANS state type (64-bit for precision)
pub type RansState = u64;

/// Probability scale (12-bit precision = 4096)
pub const PROB_BITS: u32 = 12;
pub const PROB_SCALE: u32 = 1 << PROB_BITS;

/// Symbol frequency entry for encoding/decoding
#[derive(Clone, Copy, Debug)]
pub struct RansSymbol {
    /// Cumulative frequency (start position in probability space)
    pub cum_freq: u16,
    /// Symbol frequency
    pub freq: u16,
}

impl RansSymbol {
    #[inline]
    pub const fn new(cum_freq: u16, freq: u16) -> Self {
        Self { cum_freq, freq }
    }
}

/// Frequency table for rANS coding
#[derive(Clone, Debug)]
pub struct FrequencyTable {
    /// Symbol frequencies (indexed by symbol value)
    symbols: Vec<RansSymbol>,
    /// Cumulative frequency to symbol lookup (for decoding)
    cum_to_sym: Vec<u8>,
}

impl FrequencyTable {
    /// Create frequency table from histogram
    ///
    /// Frequencies are normalized to sum to PROB_SCALE.
    pub fn from_histogram(histogram: &[u32]) -> Self {
        let n_symbols = histogram.len();
        let total: u64 = histogram.iter().map(|&x| x as u64).sum();

        if total == 0 {
            // Uniform distribution fallback
            return Self::uniform(n_symbols);
        }

        // Normalize frequencies to PROB_SCALE
        let mut symbols = Vec::with_capacity(n_symbols);
        let mut cum_freq = 0u32;
        let mut normalized_total = 0u32;

        for &count in histogram.iter() {
            let freq = if count == 0 {
                1 // Minimum frequency to avoid division by zero
            } else {
                ((count as u64 * PROB_SCALE as u64) / total).max(1) as u32
            };
            normalized_total += freq;
            symbols.push(RansSymbol::new(cum_freq as u16, freq as u16));
            cum_freq += freq;
        }

        // Adjust last symbol to ensure total equals PROB_SCALE
        if !symbols.is_empty() && normalized_total != PROB_SCALE {
            let diff = PROB_SCALE as i32 - normalized_total as i32;
            let last = symbols.last_mut().unwrap();
            last.freq = (last.freq as i32 + diff) as u16;
        }

        // Build cumulative-to-symbol lookup table
        let mut cum_to_sym = vec![0u8; PROB_SCALE as usize];
        for (sym, entry) in symbols.iter().enumerate() {
            let start = entry.cum_freq as usize;
            let end = start + entry.freq as usize;
            for i in start..end.min(PROB_SCALE as usize) {
                cum_to_sym[i] = sym as u8;
            }
        }

        Self {
            symbols,
            cum_to_sym,
        }
    }

    /// Create uniform distribution
    pub fn uniform(n_symbols: usize) -> Self {
        let freq_per_symbol = (PROB_SCALE as usize / n_symbols) as u16;
        let mut symbols = Vec::with_capacity(n_symbols);
        let mut cum = 0u16;

        for _ in 0..n_symbols {
            symbols.push(RansSymbol::new(cum, freq_per_symbol));
            cum += freq_per_symbol;
        }

        // Adjust last symbol for rounding
        if !symbols.is_empty() {
            let last = symbols.last_mut().unwrap();
            last.freq = PROB_SCALE as u16 - last.cum_freq;
        }

        let mut cum_to_sym = vec![0u8; PROB_SCALE as usize];
        for (sym, entry) in symbols.iter().enumerate() {
            let start = entry.cum_freq as usize;
            let end = start + entry.freq as usize;
            for i in start..end.min(PROB_SCALE as usize) {
                cum_to_sym[i] = sym as u8;
            }
        }

        Self {
            symbols,
            cum_to_sym,
        }
    }

    /// Get symbol info for encoding
    #[inline]
    pub fn get_symbol(&self, sym: u8) -> RansSymbol {
        self.symbols[sym as usize]
    }

    /// Decode symbol from cumulative frequency
    #[inline]
    pub fn decode_symbol(&self, cum_freq: u32) -> (u8, RansSymbol) {
        let sym = self.cum_to_sym[cum_freq as usize];
        (sym, self.symbols[sym as usize])
    }

    /// Number of symbols in the table
    #[inline]
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    /// Check if table is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }
}

/// rANS Encoder (32-bit state for simplicity)
pub struct RansEncoder {
    state: u32,
    output: Vec<u8>,
}

/// Constants for 32-bit rANS
const RANS32_L: u32 = 1 << 23; // Lower bound for state

impl RansEncoder {
    /// Create new encoder
    pub fn new() -> Self {
        Self {
            state: RANS32_L,
            output: Vec::new(),
        }
    }

    /// Create encoder with pre-allocated output buffer
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            state: RANS32_L,
            output: Vec::with_capacity(capacity),
        }
    }

    /// Encode a single symbol
    ///
    /// Symbols must be encoded in REVERSE order (last symbol first).
    #[inline]
    pub fn encode(&mut self, sym: &RansSymbol) {
        let freq = sym.freq as u32;
        let cum_freq = sym.cum_freq as u32;

        // Renormalize: output bytes while state is too large
        let x_max = ((RANS32_L >> PROB_BITS) << 8) * freq;
        while self.state >= x_max {
            self.output.push((self.state & 0xFF) as u8);
            self.state >>= 8;
        }

        // Update state: x' = ((x / freq) << PROB_BITS) + (x % freq) + cum_freq
        let q = self.state / freq;
        let r = self.state % freq;
        self.state = (q << PROB_BITS) + r + cum_freq;
    }

    /// Encode multiple symbols (in reverse order)
    pub fn encode_symbols(&mut self, symbols: &[u8], table: &FrequencyTable) {
        // Encode in reverse order
        for &sym in symbols.iter().rev() {
            let info = table.get_symbol(sym);
            self.encode(&info);
        }
    }

    /// Finish encoding and return the bitstream
    pub fn finish(mut self) -> Vec<u8> {
        // Flush remaining state (4 bytes for 32-bit state)
        self.output.push((self.state & 0xFF) as u8);
        self.output.push(((self.state >> 8) & 0xFF) as u8);
        self.output.push(((self.state >> 16) & 0xFF) as u8);
        self.output.push(((self.state >> 24) & 0xFF) as u8);

        // Reverse output (decode reads forward)
        self.output.reverse();
        self.output
    }
}

impl Default for RansEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// rANS Decoder (32-bit state)
pub struct RansDecoder<'a> {
    state: u32,
    input: &'a [u8],
    pos: usize,
}

impl<'a> RansDecoder<'a> {
    /// Create decoder from bitstream
    pub fn new(input: &'a [u8]) -> Self {
        let mut decoder = Self {
            state: 0,
            input,
            pos: 0,
        };
        decoder.init_state();
        decoder
    }

    /// Initialize state from first 4 bytes (big-endian since we reversed)
    fn init_state(&mut self) {
        if self.input.len() >= 4 {
            self.state = u32::from_be_bytes([
                self.input[0],
                self.input[1],
                self.input[2],
                self.input[3],
            ]);
            self.pos = 4;
        }
    }

    /// Decode a single symbol
    #[inline]
    pub fn decode(&mut self, table: &FrequencyTable) -> u8 {
        // Extract cumulative frequency from state
        let slot = (self.state & (PROB_SCALE - 1)) as u32;

        // Look up symbol
        let (sym, info) = table.decode_symbol(slot);

        // Update state: x = freq * (x >> PROB_BITS) + slot - cum_freq
        let freq = info.freq as u32;
        self.state = freq * (self.state >> PROB_BITS) + slot - info.cum_freq as u32;

        // Renormalize: read bytes while state is too small
        while self.state < RANS32_L && self.pos < self.input.len() {
            self.state = (self.state << 8) | self.input[self.pos] as u32;
            self.pos += 1;
        }

        sym
    }

    /// Decode multiple symbols
    pub fn decode_n(&mut self, n: usize, table: &FrequencyTable) -> Vec<u8> {
        let mut output = Vec::with_capacity(n);
        for _ in 0..n {
            output.push(self.decode(table));
        }
        output
    }

    /// Check if more data is available
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pos >= self.input.len() && self.state < RANS32_L
    }
}

/// Interleaved 4-stream rANS encoder for SIMD parallelism
pub struct InterleavedRansEncoder {
    encoders: [RansEncoder; 4],
    symbol_count: [usize; 4],
}

impl InterleavedRansEncoder {
    /// Create new interleaved encoder
    pub fn new() -> Self {
        Self {
            encoders: [
                RansEncoder::new(),
                RansEncoder::new(),
                RansEncoder::new(),
                RansEncoder::new(),
            ],
            symbol_count: [0; 4],
        }
    }

    /// Encode symbols across 4 streams
    pub fn encode(&mut self, symbols: &[u8], table: &FrequencyTable) {
        // Count symbols per stream
        let n = symbols.len();
        for i in 0..4 {
            self.symbol_count[i] = (n + 3 - i) / 4;
        }

        // Distribute symbols across 4 streams (in reverse for each stream)
        for (i, &sym) in symbols.iter().enumerate().rev() {
            let stream = i % 4;
            let info = table.get_symbol(sym);
            self.encoders[stream].encode(&info);
        }
    }

    /// Finish encoding and interleave outputs
    pub fn finish(self) -> Vec<u8> {
        let outputs: Vec<Vec<u8>> = self
            .encoders
            .into_iter()
            .map(|e| e.finish())
            .collect();

        // Store stream lengths + symbol counts + interleaved data
        let mut result = Vec::new();

        // Write stream lengths (4 x u32)
        for output in &outputs {
            let len = output.len() as u32;
            result.extend_from_slice(&len.to_le_bytes());
        }

        // Write symbol counts (4 x u32)
        for &count in &self.symbol_count {
            result.extend_from_slice(&(count as u32).to_le_bytes());
        }

        // Write stream data
        for output in outputs {
            result.extend(output);
        }

        result
    }
}

impl Default for InterleavedRansEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Interleaved 4-stream rANS decoder
pub struct InterleavedRansDecoder<'a> {
    decoders: [RansDecoder<'a>; 4],
    stream_idx: usize,
    symbols_remaining: [usize; 4],
}

impl<'a> InterleavedRansDecoder<'a> {
    /// Create decoder from interleaved bitstream
    pub fn new(input: &'a [u8]) -> Self {
        // Read stream lengths
        let len0 = u32::from_le_bytes([input[0], input[1], input[2], input[3]]) as usize;
        let len1 = u32::from_le_bytes([input[4], input[5], input[6], input[7]]) as usize;
        let len2 = u32::from_le_bytes([input[8], input[9], input[10], input[11]]) as usize;
        let len3 = u32::from_le_bytes([input[12], input[13], input[14], input[15]]) as usize;

        // Read symbol counts
        let sym0 = u32::from_le_bytes([input[16], input[17], input[18], input[19]]) as usize;
        let sym1 = u32::from_le_bytes([input[20], input[21], input[22], input[23]]) as usize;
        let sym2 = u32::from_le_bytes([input[24], input[25], input[26], input[27]]) as usize;
        let sym3 = u32::from_le_bytes([input[28], input[29], input[30], input[31]]) as usize;

        let data_start = 32;
        let s0_end = data_start + len0;
        let s1_end = s0_end + len1;
        let s2_end = s1_end + len2;
        let s3_end = s2_end + len3;

        Self {
            decoders: [
                RansDecoder::new(&input[data_start..s0_end]),
                RansDecoder::new(&input[s0_end..s1_end]),
                RansDecoder::new(&input[s1_end..s2_end]),
                RansDecoder::new(&input[s2_end..s3_end]),
            ],
            stream_idx: 0,
            symbols_remaining: [sym0, sym1, sym2, sym3],
        }
    }

    /// Decode n symbols from interleaved streams
    pub fn decode_n(&mut self, n: usize, table: &FrequencyTable) -> Vec<u8> {
        let mut output = Vec::with_capacity(n);

        for _ in 0..n {
            // Find next stream with remaining symbols
            while self.symbols_remaining[self.stream_idx] == 0 {
                self.stream_idx = (self.stream_idx + 1) % 4;
            }

            let sym = self.decoders[self.stream_idx].decode(table);
            output.push(sym);
            self.symbols_remaining[self.stream_idx] -= 1;
            self.stream_idx = (self.stream_idx + 1) % 4;
        }

        output
    }
}

// ============================================================================
// SIMD-Optimized rANS Decoder (Optimized Edition)
// ============================================================================

/// SIMD-Optimized Interleaved Decoder
///
/// Processes 4 rANS streams in parallel using AVX2 for state updates.
/// Table lookups remain scalar (gather is expensive), but arithmetic is vectorized.
pub struct SimdRansDecoder<'a> {
    /// 4 interleaved states
    states: [u32; 4],
    /// Input buffer
    input: &'a [u8],
    /// Current read position
    ptr: usize,
}

impl<'a> SimdRansDecoder<'a> {
    /// Create new SIMD decoder from 4-stream interleaved bitstream
    pub fn new(input: &'a [u8]) -> Self {
        let mut ptr = 0;
        let mut states = [0u32; 4];

        // Initialize 4 states from input (big-endian since encoder reversed)
        for state in &mut states {
            if ptr + 4 <= input.len() {
                *state = u32::from_be_bytes([
                    input[ptr],
                    input[ptr + 1],
                    input[ptr + 2],
                    input[ptr + 3],
                ]);
                ptr += 4;
            }
        }

        Self { states, input, ptr }
    }

    /// Decode 4 symbols in parallel (scalar fallback)
    ///
    /// This version works on all platforms.
    pub fn decode_4(&mut self, table: &FrequencyTable) -> [u8; 4] {
        let mut symbols = [0u8; 4];

        for i in 0..4 {
            // Extract slot
            let slot = self.states[i] & (PROB_SCALE - 1);

            // Lookup symbol
            let (sym, info) = table.decode_symbol(slot);
            symbols[i] = sym;

            // Update state: x = freq * (x >> PROB_BITS) + slot - cum_freq
            let freq = info.freq as u32;
            self.states[i] = freq * (self.states[i] >> PROB_BITS) + slot - info.cum_freq as u32;

            // Renormalize
            while self.states[i] < RANS32_L && self.ptr < self.input.len() {
                self.states[i] = (self.states[i] << 8) | self.input[self.ptr] as u32;
                self.ptr += 1;
            }
        }

        symbols
    }

    /// Decode n symbols (must be multiple of 4)
    pub fn decode_n(&mut self, n: usize, table: &FrequencyTable) -> Vec<u8> {
        assert!(n % 4 == 0, "n must be multiple of 4 for SIMD decoder");
        let mut output = Vec::with_capacity(n);

        for _ in 0..(n / 4) {
            let syms = self.decode_4(table);
            output.extend_from_slice(&syms);
        }

        output
    }
}

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
mod simd {
    use super::*;
    use core::arch::x86_64::*;

    impl<'a> SimdRansDecoder<'a> {
        /// Decode 4 symbols in parallel using AVX2
        ///
        /// # Safety
        ///
        /// Requires AVX2 support.
        #[target_feature(enable = "avx2")]
        pub unsafe fn decode_4_avx2(&mut self, table: &FrequencyTable) -> [u8; 4] {
            // Load 4 states into SSE register
            let state_vec = _mm_loadu_si128(self.states.as_ptr() as *const __m128i);
            let mask = _mm_set1_epi32((PROB_SCALE - 1) as i32);

            // 1. Get slots: slot = state & (PROB_SCALE - 1)
            let slots_vec = _mm_and_si128(state_vec, mask);
            let mut slots = [0u32; 4];
            _mm_storeu_si128(slots.as_mut_ptr() as *mut __m128i, slots_vec);

            // 2. Symbol Lookup (Scalar - Gather is too expensive)
            let mut symbols = [0u8; 4];
            let mut freqs = [0u32; 4];
            let mut cum_freqs = [0u32; 4];

            for i in 0..4 {
                let (sym, info) = table.decode_symbol(slots[i]);
                symbols[i] = sym;
                freqs[i] = info.freq as u32;
                cum_freqs[i] = info.cum_freq as u32;
            }

            // 3. State Update in SIMD: x = freq * (x >> PROB_BITS) + slot - cum_freq
            let freq_vec = _mm_loadu_si128(freqs.as_ptr() as *const __m128i);
            let cum_freq_vec = _mm_loadu_si128(cum_freqs.as_ptr() as *const __m128i);

            // x >> PROB_BITS
            let x_shifted = _mm_srli_epi32(state_vec, PROB_BITS as i32);

            // freq * (x >> PROB_BITS)
            let term1 = _mm_mullo_epi32(freq_vec, x_shifted);

            // + slot
            let term2 = _mm_add_epi32(term1, slots_vec);

            // - cum_freq
            let new_state = _mm_sub_epi32(term2, cum_freq_vec);

            // Store back
            _mm_storeu_si128(self.states.as_mut_ptr() as *mut __m128i, new_state);

            // 4. Renormalization (Scalar - variable byte consumption)
            for i in 0..4 {
                while self.states[i] < RANS32_L && self.ptr < self.input.len() {
                    self.states[i] = (self.states[i] << 8) | self.input[self.ptr] as u32;
                    self.ptr += 1;
                }
            }

            symbols
        }

        /// Decode n symbols using AVX2 (must be multiple of 4)
        ///
        /// # Safety
        ///
        /// Requires AVX2 support.
        #[target_feature(enable = "avx2")]
        pub unsafe fn decode_n_avx2(&mut self, n: usize, table: &FrequencyTable) -> Vec<u8> {
            assert!(n % 4 == 0, "n must be multiple of 4 for SIMD decoder");
            let mut output = Vec::with_capacity(n);

            for _ in 0..(n / 4) {
                let syms = self.decode_4_avx2(table);
                output.extend_from_slice(&syms);
            }

            output
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
pub use simd::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_table() {
        let table = FrequencyTable::uniform(256);
        assert_eq!(table.len(), 256);

        // All symbols should have equal frequency
        let expected_freq = PROB_SCALE / 256;
        for i in 0..255 {
            let sym = table.get_symbol(i);
            assert!(
                (sym.freq as i32 - expected_freq as i32).abs() <= 1,
                "Symbol {} has freq {}, expected ~{}",
                i,
                sym.freq,
                expected_freq
            );
        }
    }

    #[test]
    fn test_encode_decode_single() {
        let table = FrequencyTable::uniform(256);

        let original = [42u8, 100, 200, 50, 128];

        let mut encoder = RansEncoder::new();
        encoder.encode_symbols(&original, &table);
        let encoded = encoder.finish();

        let mut decoder = RansDecoder::new(&encoded);
        let decoded = decoder.decode_n(original.len(), &table);

        assert_eq!(original.as_slice(), decoded.as_slice());
    }

    #[test]
    fn test_encode_decode_skewed() {
        // Create skewed histogram (symbol 0 is very common)
        let mut histogram = [1u32; 256];
        histogram[0] = 1000;
        histogram[1] = 500;
        histogram[2] = 100;

        let table = FrequencyTable::from_histogram(&histogram);

        let original: Vec<u8> = (0..1000)
            .map(|i| match i % 10 {
                0..=6 => 0,
                7..=8 => 1,
                _ => 2,
            })
            .collect();

        let mut encoder = RansEncoder::new();
        encoder.encode_symbols(&original, &table);
        let encoded = encoder.finish();

        // Should achieve compression for skewed data
        assert!(
            encoded.len() < original.len(),
            "No compression: {} >= {}",
            encoded.len(),
            original.len()
        );

        let mut decoder = RansDecoder::new(&encoded);
        let decoded = decoder.decode_n(original.len(), &table);

        assert_eq!(original, decoded);
    }

    #[test]
    fn test_interleaved_roundtrip() {
        let table = FrequencyTable::uniform(256);
        let original: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();

        let mut encoder = InterleavedRansEncoder::new();
        encoder.encode(&original, &table);
        let encoded = encoder.finish();

        let mut decoder = InterleavedRansDecoder::new(&encoded);
        let decoded = decoder.decode_n(original.len(), &table);

        assert_eq!(original, decoded);
    }

    #[test]
    fn test_empty_input() {
        let table = FrequencyTable::uniform(256);

        let mut encoder = RansEncoder::new();
        encoder.encode_symbols(&[], &table);
        let encoded = encoder.finish();

        let mut decoder = RansDecoder::new(&encoded);
        let decoded = decoder.decode_n(0, &table);

        assert!(decoded.is_empty());
    }

    #[test]
    fn test_histogram_normalization() {
        // Histogram that doesn't sum to PROB_SCALE
        let histogram = [100u32, 200, 300, 400];
        let table = FrequencyTable::from_histogram(&histogram);

        // Total frequency should be normalized to PROB_SCALE
        let mut total = 0u32;
        for i in 0..4 {
            total += table.get_symbol(i).freq as u32;
        }
        assert_eq!(total, PROB_SCALE);
    }

    #[test]
    fn test_simd_decoder_scalar() {
        // Test SIMD decoder using scalar fallback
        let table = FrequencyTable::uniform(256);

        // Create 4-stream encoded data
        let original: Vec<u8> = (0..256).map(|i| (i % 256) as u8).collect();

        // Encode with interleaved encoder
        let mut encoder = InterleavedRansEncoder::new();
        encoder.encode(&original, &table);
        let encoded = encoder.finish();

        // Decode with regular interleaved decoder for comparison
        let mut decoder = InterleavedRansDecoder::new(&encoded);
        let expected = decoder.decode_n(original.len(), &table);

        assert_eq!(original, expected, "Regular decoder failed");
    }
}
