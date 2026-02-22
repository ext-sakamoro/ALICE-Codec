//! ALICE-DB bridge: Store codec encoding metrics as time-series
//!
//! Records bitrate, PSNR, encode timing, and frame statistics
//! into ALICE-DB for monitoring dashboards and quality analytics.
//!
//! # Example
//!
//! ```ignore
//! use alice_codec::db_bridge::CodecMetricsSink;
//!
//! let sink = CodecMetricsSink::open("/tmp/codec_metrics")?;
//! sink.record(&CodecMetrics {
//!     timestamp_ms: 1700000000000,
//!     bitrate_bps: 2_500_000.0,
//!     psnr_db: 38.5,
//!     encode_time_us: 1200.0,
//!     frame_type: FrameType::Intra,
//! })?;
//! ```

use alice_db::{Aggregation, AliceDB};
use std::io;
use std::path::Path;

/// Frame type indicator for metrics recording
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameType {
    /// Intra (full 3D wavelet chunk)
    Intra = 0,
    /// Predicted (delta from previous chunk)
    Predicted = 1,
}

/// Codec encoding metrics for a single frame or chunk
#[derive(Debug, Clone, Copy)]
pub struct CodecMetrics {
    /// Timestamp in milliseconds (epoch)
    pub timestamp_ms: i64,
    /// Encoded bitrate in bits per second
    pub bitrate_bps: f32,
    /// Peak Signal-to-Noise Ratio in dB
    pub psnr_db: f32,
    /// Encoding time in microseconds
    pub encode_time_us: f32,
    /// Frame type
    pub frame_type: FrameType,
}

/// Persistent sink for codec metrics backed by ALICE-DB.
///
/// Each metric dimension is stored in a separate DB instance
/// for efficient per-dimension queries and downsampling.
pub struct CodecMetricsSink {
    db_bitrate: AliceDB,
    db_psnr: AliceDB,
    db_encode_time: AliceDB,
}

impl std::fmt::Debug for CodecMetricsSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodecMetricsSink").finish_non_exhaustive()
    }
}

impl CodecMetricsSink {
    /// Open (or create) metric databases at the given directory.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if the directory cannot be created or
    /// the underlying databases fail to open.
    pub fn open<P: AsRef<Path>>(dir: P) -> io::Result<Self> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        Ok(Self {
            db_bitrate: AliceDB::open(dir.join("bitrate"))?,
            db_psnr: AliceDB::open(dir.join("psnr"))?,
            db_encode_time: AliceDB::open(dir.join("encode_time"))?,
        })
    }

    /// Record a single metric sample.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if writing to any metric database fails.
    pub fn record(&self, m: &CodecMetrics) -> io::Result<()> {
        self.db_bitrate.put(m.timestamp_ms, m.bitrate_bps)?;
        self.db_psnr.put(m.timestamp_ms, m.psnr_db)?;
        self.db_encode_time.put(m.timestamp_ms, m.encode_time_us)?;
        Ok(())
    }

    /// Record a batch of metric samples (more efficient than individual puts).
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if writing to any metric database fails.
    pub fn record_batch(&self, metrics: &[CodecMetrics]) -> io::Result<()> {
        let bitrate: Vec<(i64, f32)> = metrics
            .iter()
            .map(|m| (m.timestamp_ms, m.bitrate_bps))
            .collect();
        let psnr: Vec<(i64, f32)> = metrics
            .iter()
            .map(|m| (m.timestamp_ms, m.psnr_db))
            .collect();
        let encode_time: Vec<(i64, f32)> = metrics
            .iter()
            .map(|m| (m.timestamp_ms, m.encode_time_us))
            .collect();

        self.db_bitrate.put_batch(&bitrate)?;
        self.db_psnr.put_batch(&psnr)?;
        self.db_encode_time.put_batch(&encode_time)?;
        Ok(())
    }

    /// Query raw bitrate samples in a time range.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if the database read fails.
    pub fn query_bitrate(&self, start: i64, end: i64) -> io::Result<Vec<(i64, f32)>> {
        self.db_bitrate.scan(start, end)
    }

    /// Query raw PSNR samples in a time range.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if the database read fails.
    pub fn query_psnr(&self, start: i64, end: i64) -> io::Result<Vec<(i64, f32)>> {
        self.db_psnr.scan(start, end)
    }

    /// Query raw encode-time samples in a time range.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if the database read fails.
    pub fn query_encode_time(&self, start: i64, end: i64) -> io::Result<Vec<(i64, f32)>> {
        self.db_encode_time.scan(start, end)
    }

    /// Average bitrate over a time range.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if the aggregation query fails.
    pub fn average_bitrate(&self, start: i64, end: i64) -> io::Result<f64> {
        self.db_bitrate.aggregate(start, end, Aggregation::Avg)
    }

    /// Average PSNR over a time range.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if the aggregation query fails.
    pub fn average_psnr(&self, start: i64, end: i64) -> io::Result<f64> {
        self.db_psnr.aggregate(start, end, Aggregation::Avg)
    }

    /// P99 encode time (approximated via Max over small intervals).
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if the aggregation query fails.
    pub fn max_encode_time(&self, start: i64, end: i64) -> io::Result<f64> {
        self.db_encode_time.aggregate(start, end, Aggregation::Max)
    }

    /// Downsample bitrate for dashboard display.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if the downsample query fails.
    pub fn downsample_bitrate(
        &self,
        start: i64,
        end: i64,
        interval_ms: i64,
    ) -> io::Result<Vec<(i64, f64)>> {
        self.db_bitrate
            .downsample(start, end, interval_ms, Aggregation::Avg)
    }

    /// Downsample PSNR for dashboard display.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if the downsample query fails.
    pub fn downsample_psnr(
        &self,
        start: i64,
        end: i64,
        interval_ms: i64,
    ) -> io::Result<Vec<(i64, f64)>> {
        self.db_psnr
            .downsample(start, end, interval_ms, Aggregation::Avg)
    }

    /// Flush all metric databases to disk.
    ///
    /// # Errors
    ///
    /// Returns [`io::Error`] if flushing any database fails.
    pub fn flush(&self) -> io::Result<()> {
        self.db_bitrate.flush()?;
        self.db_psnr.flush()?;
        self.db_encode_time.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_metrics_sink() {
        let dir = std::env::temp_dir().join("alice_codec_db_test");
        let _ = std::fs::remove_dir_all(&dir);

        let sink = CodecMetricsSink::open(&dir).unwrap();
        sink.record(&CodecMetrics {
            timestamp_ms: 1000,
            bitrate_bps: 2_500_000.0,
            psnr_db: 38.5,
            encode_time_us: 1200.0,
            frame_type: FrameType::Intra,
        })
        .unwrap();
        sink.flush().unwrap();

        let bitrate = sink.query_bitrate(0, 2000).unwrap();
        assert_eq!(bitrate.len(), 1);
        assert!((bitrate[0].1 - 2_500_000.0).abs() < 1.0);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
