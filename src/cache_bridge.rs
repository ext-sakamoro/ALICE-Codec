//! ALICE-Cache bridge: Decoded frame caching
//!
//! Caches decoded video frames using ALICE-Cache to avoid redundant
//! wavelet inverse transforms during scrubbing, looping, and seeking.

use alice_cache::AliceCache;

/// Cached decoded frame.
#[derive(Clone)]
pub struct CachedFrame {
    /// Raw pixel data (YCoCg-R or RGB, depending on pipeline stage).
    pub data: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
}

/// Frame cache key: (chunk_index, sub_band, frame_within_chunk).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameKey {
    /// Chunk index in the stream.
    pub chunk: u32,
    /// Frame offset within the chunk (0..63 typically).
    pub frame: u16,
    /// Quality level / quantisation step (different decodes may coexist).
    pub quality: u8,
}

/// Decoded frame cache backed by ALICE-Cache.
///
/// Keyed by `FrameKey` (chunk + frame offset + quality), stores full
/// decoded pixel buffers for instant replay without re-decoding.
pub struct FrameCache {
    cache: AliceCache<FrameKey, CachedFrame>,
}

impl FrameCache {
    /// Create a new frame cache.
    ///
    /// `capacity` is the number of decoded frames to keep.
    /// For 1080p YCoCg-R (3 bytes/pixel), each frame ≈ 6 MB,
    /// so 64 frames ≈ 384 MB.
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: AliceCache::new(capacity),
        }
    }

    /// Look up a cached decoded frame.
    pub fn get(&self, chunk: u32, frame: u16, quality: u8) -> Option<CachedFrame> {
        let key = FrameKey { chunk, frame, quality };
        self.cache.get(&key)
    }

    /// Store a decoded frame.
    pub fn put(&self, chunk: u32, frame: u16, quality: u8, data: Vec<u8>, width: u32, height: u32) {
        let key = FrameKey { chunk, frame, quality };
        self.cache.put(key, CachedFrame { data, width, height });
    }

    /// Cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        self.cache.hit_rate()
    }

    /// Number of cached frames.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_cache_roundtrip() {
        let cache = FrameCache::new(16);
        let pixels = vec![128u8; 1920 * 1080 * 3];

        cache.put(0, 5, 1, pixels.clone(), 1920, 1080);
        let frame = cache.get(0, 5, 1).unwrap();
        assert_eq!(frame.width, 1920);
        assert_eq!(frame.height, 1080);
        assert_eq!(frame.data.len(), pixels.len());
    }

    #[test]
    fn test_frame_cache_miss() {
        let cache = FrameCache::new(16);
        assert!(cache.get(99, 0, 0).is_none());
    }

    #[test]
    fn test_different_quality_levels() {
        let cache = FrameCache::new(16);
        cache.put(0, 0, 1, vec![100u8; 4], 2, 2);
        cache.put(0, 0, 2, vec![200u8; 4], 2, 2);

        let q1 = cache.get(0, 0, 1).unwrap();
        let q2 = cache.get(0, 0, 2).unwrap();
        assert_eq!(q1.data[0], 100);
        assert_eq!(q2.data[0], 200);
    }
}
