//! C FFI for ALICE-Codec
//!
//! 20 `extern "C"` functions exposing wavelet, pipeline, metrics, and utilities.

use std::ffi::{c_char, CString};
use std::ptr;
use std::slice;

use crate::metrics;
use crate::pipeline::{EncodedChunk, FrameDecoder, FrameEncoder};
use crate::wavelet::Wavelet1D;

// ── Wavelet1D (6 functions) ─────────────────────────────────────

/// 1. Create Haar wavelet.
#[no_mangle]
pub extern "C" fn alice_codec_wavelet1d_haar() -> *mut Wavelet1D {
    ffi_guard(ptr::null_mut(), || {
        Box::into_raw(Box::new(Wavelet1D::haar()))
    })
}

/// 2. Create CDF 5/3 wavelet (lossless).
#[no_mangle]
pub extern "C" fn alice_codec_wavelet1d_cdf53() -> *mut Wavelet1D {
    ffi_guard(ptr::null_mut(), || {
        Box::into_raw(Box::new(Wavelet1D::cdf53()))
    })
}

/// 3. Create CDF 9/7 wavelet (lossy).
#[no_mangle]
pub extern "C" fn alice_codec_wavelet1d_cdf97() -> *mut Wavelet1D {
    ffi_guard(ptr::null_mut(), || {
        Box::into_raw(Box::new(Wavelet1D::cdf97()))
    })
}

/// 4. Destroy wavelet.
///
/// # Safety
///
/// `ptr` must be a valid pointer from `alice_codec_wavelet1d_*` or null.
#[no_mangle]
pub unsafe extern "C" fn alice_codec_wavelet1d_destroy(ptr: *mut Wavelet1D) {
    ffi_guard((), || {
        if !ptr.is_null() {
            // SAFETY: ptr was created by Box::into_raw in alice_codec_wavelet1d_*
            // and the caller guarantees it has not been freed yet.
            drop(Box::from_raw(ptr));
        }
    })
}

/// 5. Forward wavelet transform (in-place).
///
/// # Safety
///
/// `wavelet` must be valid. `data`/`len` must point to a valid `i32` buffer.
#[no_mangle]
pub unsafe extern "C" fn alice_codec_wavelet1d_forward(
    wavelet: *const Wavelet1D,
    data: *mut i32,
    len: u32,
) {
    ffi_guard((), || {
        if wavelet.is_null() || data.is_null() || len < 2 {
            return;
        }
        // SAFETY: wavelet is non-null and valid per caller contract.
        let w = &*wavelet;
        // SAFETY: data is non-null and points to `len` contiguous i32 values per caller contract.
        let signal = slice::from_raw_parts_mut(data, len as usize);
        w.forward(signal);
    })
}

/// 6. Inverse wavelet transform (in-place).
///
/// # Safety
///
/// `wavelet` must be valid. `data`/`len` must point to a valid `i32` buffer.
#[no_mangle]
pub unsafe extern "C" fn alice_codec_wavelet1d_inverse(
    wavelet: *const Wavelet1D,
    data: *mut i32,
    len: u32,
) {
    ffi_guard((), || {
        if wavelet.is_null() || data.is_null() || len < 2 {
            return;
        }
        // SAFETY: wavelet is non-null and valid per caller contract.
        let w = &*wavelet;
        // SAFETY: data is non-null and points to `len` contiguous i32 values per caller contract.
        let signal = slice::from_raw_parts_mut(data, len as usize);
        w.inverse(signal);
    })
}

// ── FrameEncoder (3 functions) ──────────────────────────────────

/// 7. Create frame encoder with quality (0-100).
#[no_mangle]
pub extern "C" fn alice_codec_encoder_create(quality: u8) -> *mut FrameEncoder {
    ffi_guard(ptr::null_mut(), || {
        Box::into_raw(Box::new(FrameEncoder::new(quality)))
    })
}

/// 8. Destroy encoder.
///
/// # Safety
///
/// `ptr` must be a valid pointer from `alice_codec_encoder_create` or null.
#[no_mangle]
pub unsafe extern "C" fn alice_codec_encoder_destroy(ptr: *mut FrameEncoder) {
    ffi_guard((), || {
        if !ptr.is_null() {
            // SAFETY: ptr was created by Box::into_raw in alice_codec_encoder_create.
            drop(Box::from_raw(ptr));
        }
    })
}

/// 9. Encode RGB frames. Returns an opaque `EncodedChunk` handle (null on error).
///
/// # Safety
///
/// `encoder` must be valid. `rgb_data`/`rgb_len` must be a valid RGB buffer
/// of size `width * height * frames * 3`.
#[no_mangle]
pub unsafe extern "C" fn alice_codec_encode(
    encoder: *const FrameEncoder,
    rgb_data: *const u8,
    rgb_len: u32,
    width: u32,
    height: u32,
    frames: u32,
) -> *mut EncodedChunk {
    ffi_guard(std::ptr::null_mut(), || {
        if encoder.is_null() || rgb_data.is_null() {
            return std::ptr::null_mut();
        }
        // SAFETY: encoder is non-null and valid per caller contract.
        let enc = &*encoder;
        // SAFETY: rgb_data is non-null and points to rgb_len contiguous bytes per caller contract.
        let rgb = slice::from_raw_parts(rgb_data, rgb_len as usize);
        enc.encode(rgb, width, height, frames)
            .map_or(std::ptr::null_mut(), |chunk| Box::into_raw(Box::new(chunk)))
    })
}

// ── Decode (1 function) ─────────────────────────────────────────

/// 10. Decode chunk to RGB bytes. Caller must free with `alice_codec_data_free`.
///
/// Returns null on error. Writes output byte count to `out_len`.
///
/// # Safety
///
/// `chunk` must be a valid `EncodedChunk` pointer. `out_len` must be valid.
#[no_mangle]
pub unsafe extern "C" fn alice_codec_decode(
    chunk: *const EncodedChunk,
    out_len: *mut u32,
) -> *mut u8 {
    ffi_guard(std::ptr::null_mut(), || {
        if chunk.is_null() || out_len.is_null() {
            return std::ptr::null_mut();
        }
        // SAFETY: chunk is non-null and valid per caller contract.
        let c = &*chunk;
        let decoder = FrameDecoder::new();
        decoder.decode(c).map_or(std::ptr::null_mut(), |rgb| {
            // SAFETY: out_len is non-null and writable per caller contract.
            *out_len = rgb.len() as u32;
            let boxed = rgb.into_boxed_slice();
            Box::into_raw(boxed).cast::<u8>()
        })
    })
}

// ── EncodedChunk (6 functions) ──────────────────────────────────

/// 11. Destroy encoded chunk.
///
/// # Safety
///
/// `ptr` must be a valid `EncodedChunk` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn alice_codec_chunk_destroy(ptr: *mut EncodedChunk) {
    ffi_guard((), || {
        if !ptr.is_null() {
            // SAFETY: ptr was created by Box::into_raw in alice_codec_encode or
            // alice_codec_chunk_from_bytes.
            drop(Box::from_raw(ptr));
        }
    })
}

/// 12. Serialize chunk to bytes. Caller must free with `alice_codec_data_free`.
///
/// # Safety
///
/// `chunk` must be valid. `out_len` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn alice_codec_chunk_to_bytes(
    chunk: *const EncodedChunk,
    out_len: *mut u32,
) -> *mut u8 {
    ffi_guard(std::ptr::null_mut(), || {
        if chunk.is_null() || out_len.is_null() {
            return std::ptr::null_mut();
        }
        // SAFETY: chunk is non-null and valid per caller contract.
        let c = &*chunk;
        let bytes = c.to_bytes();
        // SAFETY: out_len is non-null and writable per caller contract.
        *out_len = bytes.len() as u32;
        let boxed = bytes.into_boxed_slice();
        Box::into_raw(boxed).cast::<u8>()
    })
}

/// 13. Deserialize chunk from bytes. Returns null on error.
///
/// # Safety
///
/// `data`/`len` must point to a valid byte buffer.
#[no_mangle]
pub unsafe extern "C" fn alice_codec_chunk_from_bytes(
    data: *const u8,
    len: u32,
) -> *mut EncodedChunk {
    ffi_guard(std::ptr::null_mut(), || {
        if data.is_null() {
            return std::ptr::null_mut();
        }
        // SAFETY: data is non-null and points to `len` contiguous bytes per caller contract.
        let bytes = slice::from_raw_parts(data, len as usize);
        EncodedChunk::from_bytes(bytes)
            .map_or(std::ptr::null_mut(), |chunk| Box::into_raw(Box::new(chunk)))
    })
}

/// 14. Get chunk width.
///
/// # Safety
///
/// `chunk` must be a valid pointer.
#[no_mangle]
pub const unsafe extern "C" fn alice_codec_chunk_width(chunk: *const EncodedChunk) -> u32 {
    if chunk.is_null() {
        return 0;
    }
    // SAFETY: chunk is non-null and valid per caller contract.
    (*chunk).width
}

/// 15. Get chunk height.
///
/// # Safety
///
/// `chunk` must be a valid pointer.
#[no_mangle]
pub const unsafe extern "C" fn alice_codec_chunk_height(chunk: *const EncodedChunk) -> u32 {
    if chunk.is_null() {
        return 0;
    }
    // SAFETY: chunk is non-null and valid per caller contract.
    (*chunk).height
}

/// 16. Get chunk frame count.
///
/// # Safety
///
/// `chunk` must be a valid pointer.
#[no_mangle]
pub const unsafe extern "C" fn alice_codec_chunk_frames(chunk: *const EncodedChunk) -> u32 {
    if chunk.is_null() {
        return 0;
    }
    // SAFETY: chunk is non-null and valid per caller contract.
    (*chunk).frames
}

// ── Metrics (1 function) ────────────────────────────────────────

/// 17. Compute PSNR between two byte buffers. Returns -1.0 on error.
///
/// # Safety
///
/// `a` and `b` must point to valid buffers of `len` bytes each.
#[no_mangle]
pub unsafe extern "C" fn alice_codec_psnr(a: *const u8, b: *const u8, len: u32) -> f64 {
    ffi_guard(-1.0, || {
        if a.is_null() || b.is_null() {
            return -1.0;
        }
        // SAFETY: a and b are non-null and each point to `len` contiguous bytes per caller contract.
        let sa = slice::from_raw_parts(a, len as usize);
        let sb = slice::from_raw_parts(b, len as usize);
        metrics::psnr(sa, sb).unwrap_or(-1.0)
    })
}

// ── Utility (3 functions) ───────────────────────────────────────

/// 18. Free a byte buffer returned by `alice_codec_decode`, `alice_codec_chunk_to_bytes`, etc.
///
/// # Safety
///
/// `ptr`/`len` must describe a buffer previously returned by this library.
#[no_mangle]
pub unsafe extern "C" fn alice_codec_data_free(ptr: *mut u8, len: u32) {
    ffi_guard((), || {
        if !ptr.is_null() && len > 0 {
            // SAFETY: ptr/len describe a buffer previously returned by this library
            // via Box::into_raw(into_boxed_slice()). Caller guarantees single free.
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len as usize));
        }
    })
}

/// 19. Free a C string returned by `alice_codec_version`.
///
/// # Safety
///
/// `s` must be a pointer returned by `alice_codec_version` or null.
#[no_mangle]
pub unsafe extern "C" fn alice_codec_string_free(s: *mut c_char) {
    ffi_guard((), || {
        if !s.is_null() {
            // SAFETY: s was created by CString::into_raw in alice_codec_version.
            drop(CString::from_raw(s));
        }
    })
}

/// 20. Get library version string. Caller must free with `alice_codec_string_free`.
#[no_mangle]
pub extern "C" fn alice_codec_version() -> *mut c_char {
    ffi_guard(ptr::null_mut(), || {
        CString::new(crate::VERSION)
            .map(CString::into_raw)
            .unwrap_or(std::ptr::null_mut())
    })
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn test_wavelet_roundtrip() {
        unsafe {
            let w = alice_codec_wavelet1d_cdf53();
            assert!(!w.is_null());

            let mut signal = [10i32, 20, 30, 40, 50, 60, 70, 80];
            let original = signal;

            alice_codec_wavelet1d_forward(w, signal.as_mut_ptr(), 8);
            alice_codec_wavelet1d_inverse(w, signal.as_mut_ptr(), 8);

            for (i, (&o, &r)) in original.iter().zip(signal.iter()).enumerate() {
                assert!((o - r).abs() <= 1, "Mismatch at {i}: {o} vs {r}");
            }

            alice_codec_wavelet1d_destroy(w);
        }
    }

    #[test]
    fn test_wavelet_haar_and_cdf97() {
        unsafe {
            let h = alice_codec_wavelet1d_haar();
            let c = alice_codec_wavelet1d_cdf97();
            assert!(!h.is_null());
            assert!(!c.is_null());
            alice_codec_wavelet1d_destroy(h);
            alice_codec_wavelet1d_destroy(c);
        }
    }

    #[test]
    fn test_wavelet_null_safety() {
        unsafe {
            alice_codec_wavelet1d_forward(std::ptr::null(), std::ptr::null_mut(), 0);
            alice_codec_wavelet1d_inverse(std::ptr::null(), std::ptr::null_mut(), 0);
            alice_codec_wavelet1d_destroy(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_encode_decode_pipeline() {
        unsafe {
            let enc = alice_codec_encoder_create(80);
            assert!(!enc.is_null());

            // 4x4, 2 frames, RGB
            let rgb = [128u8; 4 * 4 * 2 * 3];
            let chunk = alice_codec_encode(enc, rgb.as_ptr(), rgb.len() as u32, 4, 4, 2);
            assert!(!chunk.is_null());

            // Check dimensions
            assert_eq!(alice_codec_chunk_width(chunk), 4);
            assert_eq!(alice_codec_chunk_height(chunk), 4);
            assert_eq!(alice_codec_chunk_frames(chunk), 2);

            // Decode
            let mut out_len = 0u32;
            let decoded = alice_codec_decode(chunk, &mut out_len);
            assert!(!decoded.is_null());
            assert_eq!(out_len as usize, rgb.len());

            alice_codec_data_free(decoded, out_len);
            alice_codec_chunk_destroy(chunk);
            alice_codec_encoder_destroy(enc);
        }
    }

    #[test]
    fn test_chunk_serialization() {
        unsafe {
            let enc = alice_codec_encoder_create(90);
            let rgb = [100u8; 4 * 4 * 2 * 3];
            let chunk = alice_codec_encode(enc, rgb.as_ptr(), rgb.len() as u32, 4, 4, 2);
            assert!(!chunk.is_null());

            // Serialize
            let mut bytes_len = 0u32;
            let bytes = alice_codec_chunk_to_bytes(chunk, &mut bytes_len);
            assert!(!bytes.is_null());
            assert!(bytes_len > 0);

            // Deserialize
            let restored = alice_codec_chunk_from_bytes(bytes, bytes_len);
            assert!(!restored.is_null());
            assert_eq!(alice_codec_chunk_width(restored), 4);
            assert_eq!(alice_codec_chunk_height(restored), 4);

            alice_codec_chunk_destroy(restored);
            alice_codec_data_free(bytes, bytes_len);
            alice_codec_chunk_destroy(chunk);
            alice_codec_encoder_destroy(enc);
        }
    }

    #[test]
    fn test_chunk_from_bytes_invalid() {
        unsafe {
            let bad = alice_codec_chunk_from_bytes(b"BAD".as_ptr(), 3);
            assert!(bad.is_null());

            let null = alice_codec_chunk_from_bytes(std::ptr::null(), 0);
            assert!(null.is_null());
        }
    }

    #[test]
    fn test_encode_null_safety() {
        unsafe {
            let result = alice_codec_encode(std::ptr::null(), std::ptr::null(), 0, 0, 0, 0);
            assert!(result.is_null());
        }
    }

    #[test]
    fn test_decode_null_safety() {
        unsafe {
            let mut out_len = 0u32;
            let result = alice_codec_decode(std::ptr::null(), &mut out_len);
            assert!(result.is_null());
        }
    }

    #[test]
    fn test_psnr_ffi() {
        unsafe {
            let a = [100u8, 150, 200];
            let b = [101u8, 149, 198];
            let db = alice_codec_psnr(a.as_ptr(), b.as_ptr(), 3);
            assert!(db > 30.0, "PSNR too low: {db}");

            // Identical buffers
            let db_same = alice_codec_psnr(a.as_ptr(), a.as_ptr(), 3);
            assert!(db_same.is_infinite());

            // Null safety
            let db_null = alice_codec_psnr(std::ptr::null(), a.as_ptr(), 3);
            assert_eq!(db_null, -1.0);
        }
    }

    #[test]
    fn test_version() {
        unsafe {
            let v = alice_codec_version();
            assert!(!v.is_null());
            let s = CStr::from_ptr(v).to_str().unwrap();
            assert!(!s.is_empty());
            alice_codec_string_free(v);
        }
    }

    #[test]
    fn test_chunk_null_getters() {
        unsafe {
            assert_eq!(alice_codec_chunk_width(std::ptr::null()), 0);
            assert_eq!(alice_codec_chunk_height(std::ptr::null()), 0);
            assert_eq!(alice_codec_chunk_frames(std::ptr::null()), 0);
        }
    }
}

// ============================================================================
// Panic isolation (see `guard`)
// ============================================================================

/// Message of the most recent panic caught at the FFI boundary on this thread
/// (NUL-terminated, owned by the callee — release with
/// `alice_codec_free_error_string`), or null if none.
#[no_mangle]
pub extern "C" fn alice_codec_last_error() -> *mut std::os::raw::c_char {
    ffi_guard(ptr::null_mut(), || match guard::take_last_error() {
        Some(msg) => {
            std::ffi::CString::new(msg).map_or(ptr::null_mut(), std::ffi::CString::into_raw)
        }
        None => ptr::null_mut(),
    })
}

/// Clear the most recent FFI error message.
#[no_mangle]
pub extern "C" fn alice_codec_clear_last_error() {
    ffi_guard((), guard::clear_last_error);
}

/// Release a string returned by `alice_codec_last_error`.
///
/// # Safety
/// `s` must be null or a pointer returned by `alice_codec_last_error` (freed once).
#[no_mangle]
pub unsafe extern "C" fn alice_codec_free_error_string(s: *mut std::os::raw::c_char) {
    ffi_guard((), || {
        if !s.is_null() {
            drop(std::ffi::CString::from_raw(s));
        }
    });
}

/// Panic isolation for the C ABI.
///
/// A panic that reaches an `extern "C"` boundary aborts the whole process
/// (Rust 1.81+), taking the host (Unity, Unreal, a Python interpreter) down
/// with it. Every exported function therefore runs its body through
/// [`guard::ffi_guard`]: a panic is caught inside the function, its message is
/// stored in a thread-local slot the host reads with `alice_codec_last_error`, and
/// the function returns the caller's sentinel (null handle, 0, -1, NaN, ()).
/// Requires the crate to be built with `panic = "unwind"` (the default);
/// `panic = "abort"` makes `catch_unwind` a no-op.
pub(crate) mod guard {
    use std::cell::RefCell;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    thread_local! {
        static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
    }

    /// Record an error message for `alice_codec_last_error`.
    pub fn set_last_error(msg: impl Into<String>) {
        LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(msg.into()));
    }

    /// Take the most recent error message (leaves the slot empty).
    pub fn take_last_error() -> Option<String> {
        LAST_ERROR.with(|slot| slot.borrow_mut().take())
    }

    /// Clear the most recent error message.
    pub fn clear_last_error() {
        LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
    }

    /// Run `body`, converting a panic into `default` plus a recorded message.
    ///
    /// The closure is treated as unwind-safe: every FFI body only touches its
    /// arguments and heap handles owned by the caller, so no partially-updated
    /// shared state is observable afterwards.
    #[inline]
    pub fn ffi_guard<T>(default: T, body: impl FnOnce() -> T) -> T {
        match catch_unwind(AssertUnwindSafe(body)) {
            Ok(v) => v,
            Err(payload) => {
                let msg = payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "panic with non-string payload".to_string());
                set_last_error(format!("alice_codec FFI panic: {msg}"));
                default
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn panic_becomes_default_and_message() {
            clear_last_error();
            let v = ffi_guard(-1i32, || -> i32 { panic!("boom {}", 42) });
            assert_eq!(v, -1);
            let msg = take_last_error().expect("message recorded");
            assert!(msg.contains("boom 42"), "{msg}");
            assert!(take_last_error().is_none(), "take clears the slot");
        }

        #[test]
        fn success_leaves_slot_untouched() {
            clear_last_error();
            assert_eq!(ffi_guard(0, || 5), 5);
            assert!(take_last_error().is_none());
        }
    }
}

use guard::ffi_guard;

#[cfg(test)]
mod guard_ffi_tests {
    use super::*;

    /// panic が sentinel + `alice_codec_last_error` の message に変換され、host が
    /// 文字列を取得 / 解放できること (extern "C" 3 本の end-to-end)
    #[test]
    fn last_error_roundtrip_through_c_abi() {
        alice_codec_clear_last_error();
        assert!(alice_codec_last_error().is_null(), "初期状態は null");
        let v = ffi_guard(-1i32, || -> i32 { panic!("ffi test panic") });
        assert_eq!(v, -1);
        let s = alice_codec_last_error();
        assert!(!s.is_null());
        let msg = unsafe { std::ffi::CStr::from_ptr(s) }
            .to_string_lossy()
            .into_owned();
        assert!(msg.contains("ffi test panic"), "{msg}");
        unsafe { alice_codec_free_error_string(s) };
        assert!(
            alice_codec_last_error().is_null(),
            "take で slot は空になる"
        );
    }
}
