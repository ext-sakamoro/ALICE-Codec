//! ALICE-Crypto bridge: AEAD encryption for encoded bitstreams
//!
//! Wraps compressed data with XChaCha20-Poly1305 authenticated encryption
//! for secure storage or transmission (DRM, secure delivery).
//!
//! # Pipeline
//!
//! ```text
//! Raw Data → ALICE-Codec (compress) → crypto_bridge::seal_bitstream → Encrypted
//! Encrypted → crypto_bridge::open_bitstream → ALICE-Codec (decompress) → Raw Data
//! ```

use alice_crypto::{self as crypto, CipherError, Key};

/// Encrypted bitstream with embedded nonce and auth tag.
///
/// Format: `[nonce 24B][ciphertext][tag 16B]`
#[derive(Debug, Clone)]
pub struct SealedBitstream {
    /// The sealed data (nonce + ciphertext + tag)
    pub data: Vec<u8>,
    /// Original plaintext length (for pre-allocation on decrypt)
    pub plaintext_len: usize,
}

/// Encrypt a compressed bitstream using XChaCha20-Poly1305.
///
/// The returned [`SealedBitstream`] contains the nonce, ciphertext, and
/// authentication tag. The key must be 32 bytes.
///
/// # Errors
///
/// Returns [`CipherError`] if the underlying AEAD operation fails.
pub fn seal_bitstream(plaintext: &[u8], key: &Key) -> Result<SealedBitstream, CipherError> {
    let sealed = crypto::seal(key, plaintext)?;
    Ok(SealedBitstream {
        plaintext_len: plaintext.len(),
        data: sealed,
    })
}

/// Decrypt a sealed bitstream back to the original compressed data.
///
/// # Errors
///
/// Returns [`CipherError`] if decryption fails (wrong key, tampered data, etc.).
pub fn open_bitstream(sealed: &SealedBitstream, key: &Key) -> Result<Vec<u8>, CipherError> {
    crypto::open(key, &sealed.data)
}

/// Compute a BLAKE3 content hash for a bitstream.
///
/// Useful for content-addressed storage or deduplication
/// without needing to decrypt.
#[must_use]
pub fn content_hash(data: &[u8]) -> crypto::Hash {
    crypto::hash(data)
}

/// Derive an encryption key from a passphrase and context string.
///
/// Uses BLAKE3 key derivation (not suitable for weak passwords;
/// use a proper KDF like Argon2 for user passwords).
#[must_use]
pub fn derive_key(context: &str, passphrase: &[u8]) -> Key {
    let raw = crypto::derive_key(context, passphrase);
    Key::from_bytes(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seal_open_roundtrip() {
        let key = Key::generate().unwrap();
        let plaintext = b"compressed wavelet data here";

        let sealed = seal_bitstream(plaintext, &key).unwrap();
        assert_ne!(&sealed.data, plaintext);
        assert_eq!(sealed.plaintext_len, plaintext.len());

        let recovered = open_bitstream(&sealed, &key).unwrap();
        assert_eq!(&recovered, plaintext);
    }

    #[test]
    fn test_wrong_key_fails() {
        let key1 = Key::generate().unwrap();
        let key2 = Key::generate().unwrap();
        let plaintext = b"secret data";

        let sealed = seal_bitstream(plaintext, &key1).unwrap();
        assert!(open_bitstream(&sealed, &key2).is_err());
    }

    #[test]
    fn test_content_hash_deterministic() {
        let data = b"some bitstream";
        let h1 = content_hash(data);
        let h2 = content_hash(data);
        assert_eq!(h1.as_bytes(), h2.as_bytes());
    }

    #[test]
    fn test_derive_key_deterministic() {
        let k1 = derive_key("alice-codec-v1", b"my-passphrase");
        let k2 = derive_key("alice-codec-v1", b"my-passphrase");
        assert_eq!(k1.as_bytes(), k2.as_bytes());
    }
}
