//! Python bindings for ALICE-Codec
//!
//! # Optimization Layers
//!
//! | Layer | Technique | Effect |
//! |-------|-----------|--------|
//! | L1 | GIL Release (`py.allow_threads`) | Parallel computation |
//! | L2 | Zero-Copy NumPy (as_slice/into_pyarray) | No memcpy |
//! | L3 | Batch API (whole-frame ops) | FFI amortization |
//! | L4 | Rust backend (segment, wavelet, rANS) | Hardware-speed |

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::color::{rgb_bytes_to_ycocg_r, ycocg_r_to_rgb_bytes};
use crate::pipeline::{CodecError, EncodedChunk, FrameDecoder, FrameEncoder, WaveletType};
use crate::segment::{
    crop_to_bbox, paste_from_bbox, segment_by_chroma, segment_by_motion, SegmentConfig,
    SegmentResult,
};

/// Wrapper for raw pointer that implements Send (for GIL release)
struct SendPtr(*const u8, usize);
// SAFETY: The raw pointer is derived from a NumPy array that is kept alive by Python's GC.
// The pointer is only used within py.allow_threads() while the GIL is released, and the
// NumPy array reference guarantees validity for the duration.
unsafe impl Send for SendPtr {}
impl SendPtr {
    #[inline]
    fn new(ptr: *const u8, len: usize) -> Self {
        Self(ptr, len)
    }
    #[inline]
    // SAFETY: Caller guarantees pointer is valid and the len matches the original slice.
    // Used only inside py.allow_threads() with data from a live NumPy array.
    unsafe fn as_slice(&self) -> &[u8] {
        std::slice::from_raw_parts(self.0, self.1)
    }
}

struct SendPtrI16(*const i16, usize);
// SAFETY: Same rationale as SendPtr — derived from NumPy i16 array kept alive by Python GC.
unsafe impl Send for SendPtrI16 {}
impl SendPtrI16 {
    #[inline]
    fn new(ptr: *const i16, len: usize) -> Self {
        Self(ptr, len)
    }
    #[inline]
    // SAFETY: Caller guarantees pointer is valid and len matches the original i16 slice.
    unsafe fn as_slice(&self) -> &[i16] {
        std::slice::from_raw_parts(self.0, self.1)
    }
}

/// Convert [`CodecError`] to a Python exception.
fn codec_err_to_py(e: CodecError) -> pyo3::PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

// ═══════════════════════════════════════════════════════════════
// Person Segmentation (NumPy Zero-Copy + GIL Release)
// ═══════════════════════════════════════════════════════════════

/// Motion-based person segmentation with NumPy zero-copy I/O.
///
/// Args:
///     current: Current frame (H, W) as uint8 NumPy array
///     reference: Reference/background frame (H, W) as uint8 NumPy array
///     motion_threshold: Pixel difference threshold (default: 25)
///     dilate_radius: Morphological dilation radius (default: 2)
///     erode_radius: Morphological erosion radius (default: 1)
///
/// Returns:
///     Tuple of (mask: ndarray[H,W] uint8, bbox: [x,y,w,h], fg_count: int)
#[pyfunction]
#[pyo3(signature = (current, reference, motion_threshold=25, dilate_radius=2, erode_radius=1))]
fn segment_motion_numpy<'py>(
    py: Python<'py>,
    current: PyReadonlyArray2<'py, u8>,
    reference: PyReadonlyArray2<'py, u8>,
    motion_threshold: u8,
    dilate_radius: u32,
    erode_radius: u32,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Vec<u32>, u32)> {
    let curr_array = current.as_array();
    let ref_array = reference.as_array();
    let (h, w) = (curr_array.shape()[0], curr_array.shape()[1]);

    let curr_slice = curr_array.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Current array must be C-contiguous")
    })?;
    let ref_slice = ref_array.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Reference array must be C-contiguous")
    })?;

    // GIL release: heavy computation in Rust threads
    let curr_send = SendPtr::new(curr_slice.as_ptr(), curr_slice.len());
    let ref_send = SendPtr::new(ref_slice.as_ptr(), ref_slice.len());

    let result = py.allow_threads(move || {
        // SAFETY: Pointers derived from C-contiguous NumPy arrays verified above.
        // Arrays are kept alive by Python GC; GIL release does not deallocate them.
        let curr = unsafe { curr_send.as_slice() };
        let ref_ = unsafe { ref_send.as_slice() };
        let config = SegmentConfig {
            motion_threshold,
            dilate_radius,
            erode_radius,
            ..Default::default()
        };
        segment_by_motion(curr, ref_, w as u32, h as u32, &config)
            .expect("pre-validated buffer sizes")
    });

    // Zero-copy output: mask → NumPy array
    let mask_array = result
        .mask
        .into_pyarray(py)
        .reshape([h, w])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
    let bbox = result.bbox.to_vec();

    Ok((mask_array, bbox, result.foreground_count))
}

/// Chroma-key segmentation with NumPy zero-copy I/O.
///
/// Args:
///     y_channel: Y luma (H, W) as int16 NumPy array
///     co_channel: Co orange chrominance (H, W) as int16
///     cg_channel: Cg green chrominance (H, W) as int16
///     green_threshold: Cg threshold for green screen (default: 30)
///
/// Returns:
///     Tuple of (mask: ndarray[H,W] uint8, bbox: [x,y,w,h], fg_count: int)
#[pyfunction]
#[pyo3(signature = (y_channel, co_channel, cg_channel, green_threshold=30))]
fn segment_chroma_numpy<'py>(
    py: Python<'py>,
    y_channel: PyReadonlyArray2<'py, i16>,
    co_channel: PyReadonlyArray2<'py, i16>,
    cg_channel: PyReadonlyArray2<'py, i16>,
    green_threshold: i16,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Vec<u32>, u32)> {
    let y_arr = y_channel.as_array();
    let co_arr = co_channel.as_array();
    let cg_arr = cg_channel.as_array();
    let (h, w) = (y_arr.shape()[0], y_arr.shape()[1]);

    let y_slice = y_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Y array must be C-contiguous"))?;
    let co_slice = co_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Co array must be C-contiguous"))?;
    let cg_slice = cg_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Cg array must be C-contiguous"))?;

    let y_send = SendPtrI16::new(y_slice.as_ptr(), y_slice.len());
    let co_send = SendPtrI16::new(co_slice.as_ptr(), co_slice.len());
    let cg_send = SendPtrI16::new(cg_slice.as_ptr(), cg_slice.len());

    let result = py.allow_threads(move || {
        // SAFETY: Pointers derived from C-contiguous NumPy i16 arrays verified above.
        // Python GC keeps arrays alive during GIL release.
        let y = unsafe { y_send.as_slice() };
        let co = unsafe { co_send.as_slice() };
        let cg = unsafe { cg_send.as_slice() };
        segment_by_chroma(y, co, cg, w as u32, h as u32, green_threshold)
    });

    let mask_array = result
        .mask
        .into_pyarray(py)
        .reshape([h, w])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))?;
    let bbox = result.bbox.to_vec();
    Ok((mask_array, bbox, result.foreground_count))
}

/// Crop frame to person bounding box (NumPy zero-copy).
///
/// Args:
///     frame: Full frame (H, W) as uint8 NumPy array
///     bbox: [x, y, width, height] bounding box
///
/// Returns:
///     Cropped region as NumPy array (bh, bw)
#[pyfunction]
fn crop_bbox_numpy<'py>(
    py: Python<'py>,
    frame: PyReadonlyArray2<'py, u8>,
    bbox: Vec<u32>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    if bbox.len() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bbox must have 4 elements",
        ));
    }
    let arr = frame.as_array();
    let w = arr.shape()[1] as u32;
    let slice = arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Frame must be C-contiguous"))?;
    let bbox_arr = [bbox[0], bbox[1], bbox[2], bbox[3]];
    let cropped = crop_to_bbox(slice, w, &bbox_arr);
    let bw = bbox[2] as usize;
    let bh = bbox[3] as usize;
    cropped
        .into_pyarray(py)
        .reshape([bh, bw])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))
}

/// Paste person data back into full frame (in-place NumPy).
///
/// Args:
///     frame: Full frame (H, W) as uint8 NumPy array (modified in-place)
///     person_data: Cropped person data as 1D uint8 array
///     bbox: [x, y, width, height] bounding box
#[pyfunction]
fn paste_bbox_numpy(
    frame: &Bound<'_, PyArray2<u8>>,
    person_data: &[u8],
    bbox: Vec<u32>,
) -> PyResult<()> {
    if bbox.len() != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bbox must have 4 elements",
        ));
    }
    // SAFETY: frame is a valid PyArray2 bound to the Python interpreter.
    // dims() returns the array shape which is always valid for a 2D array.
    let w = unsafe { frame.dims()[1] } as u32;
    // SAFETY: frame is a valid, C-contiguous PyArray2<u8>. We have exclusive access
    // since this is the only mutable reference and Python GIL is held.
    let slice = unsafe { frame.as_slice_mut() }
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{:?}", e)))?;
    let bbox_arr = [bbox[0], bbox[1], bbox[2], bbox[3]];
    paste_from_bbox(slice, w, person_data, &bbox_arr);
    Ok(())
}

/// RLE encode a binary mask.
///
/// Args:
///     mask: Binary mask (H, W) as uint8 NumPy array
///
/// Returns:
///     RLE-encoded bytes
#[pyfunction]
fn rle_encode_numpy(mask: PyReadonlyArray2<u8>) -> PyResult<Vec<u8>> {
    let arr = mask.as_array();
    let slice = arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Mask must be C-contiguous"))?;
    let result = SegmentResult {
        mask: slice.to_vec(),
        bbox: [0, 0, 0, 0],
        foreground_count: 0,
        width: arr.shape()[1] as u32,
        height: arr.shape()[0] as u32,
    };
    Ok(result.rle_encode_mask())
}

/// Get library version
#[pyfunction]
fn version() -> &'static str {
    crate::VERSION
}

// ═══════════════════════════════════════════════════════════════
// Pipeline (Opaque Pointer + GIL Release + Zero-Copy NumPy)
// ═══════════════════════════════════════════════════════════════

/// Opaque wrapper around [`EncodedChunk`] for Python.
///
/// Holds the compressed bitstream produced by [`PyFrameEncoder`] and consumed
/// by [`PyFrameDecoder`].  Exposes metadata but not internal buffers.
#[pyclass(name = "EncodedChunk")]
struct PyEncodedChunk {
    inner: EncodedChunk,
}

#[pymethods]
impl PyEncodedChunk {
    /// Total compressed payload size in bytes.
    #[getter]
    fn compressed_size(&self) -> usize {
        self.inner.compressed_size()
    }

    /// Frame width in pixels.
    #[getter]
    fn width(&self) -> u32 {
        self.inner.width
    }

    /// Frame height in pixels.
    #[getter]
    fn height(&self) -> u32 {
        self.inner.height
    }

    /// Number of frames in this chunk.
    #[getter]
    fn frames(&self) -> u32 {
        self.inner.frames
    }

    /// Wavelet type used during encoding ("cdf53", "cdf97", or "haar").
    #[getter]
    fn wavelet(&self) -> &'static str {
        match self.inner.wavelet_type {
            WaveletType::Cdf53 => "cdf53",
            WaveletType::Cdf97 => "cdf97",
            WaveletType::Haar => "haar",
        }
    }

    /// Serialize to bytes (for file I/O or network transfer).
    fn to_bytes(&self) -> Vec<u8> {
        self.inner.to_bytes()
    }

    /// Reconstruct from bytes previously produced by `to_bytes()`.
    ///
    /// Raises:
    ///     ValueError: if the data is truncated or has wrong magic.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        let inner = EncodedChunk::from_bytes(data).map_err(codec_err_to_py)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "EncodedChunk({}x{}x{}, {} bytes, {})",
            self.inner.width,
            self.inner.height,
            self.inner.frames,
            self.inner.compressed_size(),
            match self.inner.wavelet_type {
                WaveletType::Cdf53 => "cdf53",
                WaveletType::Cdf97 => "cdf97",
                WaveletType::Haar => "haar",
            }
        )
    }
}

/// Video frame encoder (GIL-released, SIMD-accelerated).
///
/// Example:
///     >>> encoder = alice_codec.FrameEncoder(quality=90)
///     >>> chunk = encoder.encode(rgb_array, 640, 480, 2)
///     >>> print(chunk.compressed_size)
#[pyclass(name = "FrameEncoder")]
struct PyFrameEncoder {
    inner: FrameEncoder,
}

#[pymethods]
impl PyFrameEncoder {
    /// Create an encoder with quality 0-100 and optional wavelet type.
    ///
    /// Args:
    ///     quality: 0 = max compression, 100 = near-lossless (default: 90)
    ///     wavelet: "cdf53" (default), "cdf97", or "haar"
    #[new]
    #[pyo3(signature = (quality=90, wavelet="cdf53"))]
    fn new(quality: u8, wavelet: &str) -> PyResult<Self> {
        let wt = match wavelet {
            "cdf53" => WaveletType::Cdf53,
            "cdf97" => WaveletType::Cdf97,
            "haar" => WaveletType::Haar,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown wavelet type '{wavelet}'; expected 'cdf53', 'cdf97', or 'haar'"
                )));
            }
        };
        Ok(Self {
            inner: FrameEncoder::with_wavelet(quality, wt),
        })
    }

    /// Encode interleaved RGB frames into a compressed chunk.
    ///
    /// Args:
    ///     rgb_frames: 1-D uint8 NumPy array [R0,G0,B0, R1,G1,B1, ...]
    ///     width: Frame width in pixels
    ///     height: Frame height in pixels
    ///     frames: Number of frames
    ///
    /// Returns:
    ///     EncodedChunk containing the compressed bitstream.
    ///
    /// Raises:
    ///     ValueError: if buffer size is wrong or dimensions are invalid.
    fn encode(
        &self,
        py: Python<'_>,
        rgb_frames: PyReadonlyArray1<'_, u8>,
        width: u32,
        height: u32,
        frames: u32,
    ) -> PyResult<PyEncodedChunk> {
        let arr = rgb_frames.as_array();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("rgb_frames must be C-contiguous")
        })?;

        let send = SendPtr::new(slice.as_ptr(), slice.len());
        let quality = self.inner.quality;
        let wavelet_type = self.inner.wavelet_type;

        let result = py.allow_threads(move || {
            // SAFETY: Pointer from C-contiguous NumPy array, kept alive by Python GC.
            let data = unsafe { send.as_slice() };
            let enc = FrameEncoder::with_wavelet(quality, wavelet_type);
            enc.encode(data, width, height, frames)
        });

        Ok(PyEncodedChunk {
            inner: result.map_err(codec_err_to_py)?,
        })
    }
}

/// Video frame decoder (GIL-released).
///
/// Example:
///     >>> decoder = alice_codec.FrameDecoder()
///     >>> rgb = decoder.decode(chunk)  # numpy uint8 array
#[pyclass(name = "FrameDecoder")]
struct PyFrameDecoder {
    inner: FrameDecoder,
}

#[pymethods]
impl PyFrameDecoder {
    #[new]
    fn new() -> Self {
        Self {
            inner: FrameDecoder::new(),
        }
    }

    /// Decode a compressed chunk back to interleaved RGB bytes.
    ///
    /// Args:
    ///     chunk: EncodedChunk from FrameEncoder.encode()
    ///
    /// Returns:
    ///     1-D uint8 NumPy array [R0,G0,B0, R1,G1,B1, ...]
    ///
    /// Raises:
    ///     ValueError: if the bitstream is malformed.
    fn decode<'py>(
        &self,
        py: Python<'py>,
        chunk: &PyEncodedChunk,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        // Clone the chunk so we can move it into the GIL-released closure.
        let chunk_clone = chunk.inner.clone();

        let result = py.allow_threads(move || {
            let dec = FrameDecoder::new();
            dec.decode(&chunk_clone)
        });

        let rgb = result.map_err(codec_err_to_py)?;
        Ok(rgb.into_pyarray(py))
    }
}

// ═══════════════════════════════════════════════════════════════
// Color Conversion (NumPy Zero-Copy + GIL Release)
// ═══════════════════════════════════════════════════════════════

/// Convert interleaved RGB bytes to planar YCoCg-R channels.
///
/// Args:
///     rgb_bytes: 1-D uint8 NumPy array [R0,G0,B0, R1,G1,B1, ...]
///
/// Returns:
///     Tuple of (y, co, cg) each as 1-D int16 NumPy array.
#[pyfunction]
fn rgb_to_ycocg_r_numpy<'py>(
    py: Python<'py>,
    rgb_bytes: PyReadonlyArray1<'py, u8>,
) -> PyResult<(
    Bound<'py, PyArray1<i16>>,
    Bound<'py, PyArray1<i16>>,
    Bound<'py, PyArray1<i16>>,
)> {
    let arr = rgb_bytes.as_array();
    let slice = arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("rgb_bytes must be C-contiguous"))?;
    if slice.len() % 3 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rgb_bytes length must be a multiple of 3",
        ));
    }
    let n_pixels = slice.len() / 3;
    let send = SendPtr::new(slice.as_ptr(), slice.len());

    let (y, co, cg) = py.allow_threads(move || {
        // SAFETY: Pointer from C-contiguous NumPy array, kept alive by Python GC.
        let data = unsafe { send.as_slice() };
        let mut y_out = vec![0i16; n_pixels];
        let mut co_out = vec![0i16; n_pixels];
        let mut cg_out = vec![0i16; n_pixels];
        rgb_bytes_to_ycocg_r(data, &mut y_out, &mut co_out, &mut cg_out)
            .expect("pre-validated buffer sizes");
        (y_out, co_out, cg_out)
    });

    Ok((y.into_pyarray(py), co.into_pyarray(py), cg.into_pyarray(py)))
}

/// Convert planar YCoCg-R channels back to interleaved RGB bytes.
///
/// Args:
///     y: 1-D int16 NumPy array (Y channel)
///     co: 1-D int16 NumPy array (Co channel)
///     cg: 1-D int16 NumPy array (Cg channel)
///
/// Returns:
///     1-D uint8 NumPy array [R0,G0,B0, R1,G1,B1, ...]
#[pyfunction]
fn ycocg_r_to_rgb_numpy<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, i16>,
    co: PyReadonlyArray1<'py, i16>,
    cg: PyReadonlyArray1<'py, i16>,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let y_arr = y.as_array();
    let co_arr = co.as_array();
    let cg_arr = cg.as_array();

    if y_arr.len() != co_arr.len() || y_arr.len() != cg_arr.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "y, co, cg arrays must have the same length",
        ));
    }

    let y_slice = y_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("y must be C-contiguous"))?;
    let co_slice = co_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("co must be C-contiguous"))?;
    let cg_slice = cg_arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("cg must be C-contiguous"))?;

    let y_send = SendPtrI16::new(y_slice.as_ptr(), y_slice.len());
    let co_send = SendPtrI16::new(co_slice.as_ptr(), co_slice.len());
    let cg_send = SendPtrI16::new(cg_slice.as_ptr(), cg_slice.len());

    let rgb = py.allow_threads(move || {
        // SAFETY: Pointers from C-contiguous NumPy i16 arrays, kept alive by Python GC.
        let y_data = unsafe { y_send.as_slice() };
        let co_data = unsafe { co_send.as_slice() };
        let cg_data = unsafe { cg_send.as_slice() };
        let mut out = vec![0u8; y_data.len() * 3];
        ycocg_r_to_rgb_bytes(y_data, co_data, cg_data, &mut out)
            .expect("pre-validated buffer sizes");
        out
    });

    Ok(rgb.into_pyarray(py))
}

// ═══════════════════════════════════════════════════════════════
// Python Module
// ═══════════════════════════════════════════════════════════════

/// ALICE Codec - Hyper-Fast 3D Wavelet Video/Audio Codec
#[pymodule]
fn alice_codec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Pipeline (opaque pointer + GIL release)
    m.add_class::<PyFrameEncoder>()?;
    m.add_class::<PyFrameDecoder>()?;
    m.add_class::<PyEncodedChunk>()?;

    // Color conversion (NumPy zero-copy + GIL release)
    m.add_function(wrap_pyfunction!(rgb_to_ycocg_r_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(ycocg_r_to_rgb_numpy, m)?)?;

    // Segmentation (NumPy zero-copy + GIL release)
    m.add_function(wrap_pyfunction!(segment_motion_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(segment_chroma_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(crop_bbox_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(paste_bbox_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(rle_encode_numpy, m)?)?;

    // Utility
    m.add_function(wrap_pyfunction!(version, m)?)?;

    Ok(())
}
