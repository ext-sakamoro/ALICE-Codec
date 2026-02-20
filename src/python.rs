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

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, IntoPyArray, PyArrayMethods};

use crate::segment::{SegmentConfig, SegmentResult, segment_by_motion, segment_by_chroma, crop_to_bbox, paste_from_bbox};

/// Wrapper for raw pointer that implements Send (for GIL release)
struct SendPtr(*const u8, usize);
unsafe impl Send for SendPtr {}
impl SendPtr {
    #[inline]
    fn new(ptr: *const u8, len: usize) -> Self { Self(ptr, len) }
    #[inline]
    unsafe fn as_slice(&self) -> &[u8] { std::slice::from_raw_parts(self.0, self.1) }
}

struct SendPtrI16(*const i16, usize);
unsafe impl Send for SendPtrI16 {}
impl SendPtrI16 {
    #[inline]
    fn new(ptr: *const i16, len: usize) -> Self { Self(ptr, len) }
    #[inline]
    unsafe fn as_slice(&self) -> &[i16] { std::slice::from_raw_parts(self.0, self.1) }
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
        let curr = unsafe { curr_send.as_slice() };
        let ref_ = unsafe { ref_send.as_slice() };
        let config = SegmentConfig {
            motion_threshold,
            dilate_radius,
            erode_radius,
            ..Default::default()
        };
        segment_by_motion(curr, ref_, w as u32, h as u32, &config)
    });

    // Zero-copy output: mask → NumPy array
    let mask_array = result.mask.into_pyarray(py).reshape([h, w])
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

    let y_slice = y_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Y array must be C-contiguous")
    })?;
    let co_slice = co_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Co array must be C-contiguous")
    })?;
    let cg_slice = cg_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Cg array must be C-contiguous")
    })?;

    let y_send = SendPtrI16::new(y_slice.as_ptr(), y_slice.len());
    let co_send = SendPtrI16::new(co_slice.as_ptr(), co_slice.len());
    let cg_send = SendPtrI16::new(cg_slice.as_ptr(), cg_slice.len());

    let result = py.allow_threads(move || {
        let y = unsafe { y_send.as_slice() };
        let co = unsafe { co_send.as_slice() };
        let cg = unsafe { cg_send.as_slice() };
        segment_by_chroma(y, co, cg, w as u32, h as u32, green_threshold)
    });

    let mask_array = result.mask.into_pyarray(py).reshape([h, w])
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
        return Err(pyo3::exceptions::PyValueError::new_err("bbox must have 4 elements"));
    }
    let arr = frame.as_array();
    let w = arr.shape()[1] as u32;
    let slice = arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Frame must be C-contiguous")
    })?;
    let bbox_arr = [bbox[0], bbox[1], bbox[2], bbox[3]];
    let cropped = crop_to_bbox(slice, w, &bbox_arr);
    let bw = bbox[2] as usize;
    let bh = bbox[3] as usize;
    cropped.into_pyarray(py).reshape([bh, bw])
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
        return Err(pyo3::exceptions::PyValueError::new_err("bbox must have 4 elements"));
    }
    let w = unsafe { frame.dims()[1] } as u32;
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
    let slice = arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Mask must be C-contiguous")
    })?;
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
// Python Module
// ═══════════════════════════════════════════════════════════════

/// ALICE Codec - Hyper-Fast 3D Wavelet Video/Audio Codec
#[pymodule]
fn alice_codec(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
