// ALICE-Codec UE5 C++ Header
// Auto-generated — 20 extern C + RAII wrappers
// Author: Moroya Sakamoto

#pragma once

#include <cstdint>
#include <utility>

// ============================================
// C API (20 functions)
// ============================================

extern "C"
{
    // Opaque handles
    typedef struct Wavelet1D Wavelet1D;
    typedef struct FrameEncoder FrameEncoder;
    typedef struct EncodedChunk EncodedChunk;

    // 1. Create Haar wavelet
    Wavelet1D* alice_codec_wavelet1d_haar();
    // 2. Create CDF 5/3 wavelet
    Wavelet1D* alice_codec_wavelet1d_cdf53();
    // 3. Create CDF 9/7 wavelet
    Wavelet1D* alice_codec_wavelet1d_cdf97();
    // 4. Destroy wavelet
    void alice_codec_wavelet1d_destroy(Wavelet1D* ptr);
    // 5. Forward transform (in-place)
    void alice_codec_wavelet1d_forward(const Wavelet1D* wavelet, int32_t* data, uint32_t len);
    // 6. Inverse transform (in-place)
    void alice_codec_wavelet1d_inverse(const Wavelet1D* wavelet, int32_t* data, uint32_t len);

    // 7. Create encoder
    FrameEncoder* alice_codec_encoder_create(uint8_t quality);
    // 8. Destroy encoder
    void alice_codec_encoder_destroy(FrameEncoder* ptr);
    // 9. Encode RGB frames
    EncodedChunk* alice_codec_encode(
        const FrameEncoder* encoder, const uint8_t* rgb, uint32_t rgbLen,
        uint32_t width, uint32_t height, uint32_t frames);

    // 10. Decode chunk to RGB
    uint8_t* alice_codec_decode(const EncodedChunk* chunk, uint32_t* outLen);

    // 11. Destroy chunk
    void alice_codec_chunk_destroy(EncodedChunk* ptr);
    // 12. Serialize chunk
    uint8_t* alice_codec_chunk_to_bytes(const EncodedChunk* chunk, uint32_t* outLen);
    // 13. Deserialize chunk
    EncodedChunk* alice_codec_chunk_from_bytes(const uint8_t* data, uint32_t len);
    // 14. Get width
    uint32_t alice_codec_chunk_width(const EncodedChunk* chunk);
    // 15. Get height
    uint32_t alice_codec_chunk_height(const EncodedChunk* chunk);
    // 16. Get frames
    uint32_t alice_codec_chunk_frames(const EncodedChunk* chunk);

    // 17. PSNR
    double alice_codec_psnr(const uint8_t* a, const uint8_t* b, uint32_t len);

    // 18. Free data buffer
    void alice_codec_data_free(uint8_t* ptr, uint32_t len);
    // 19. Free string
    void alice_codec_string_free(char* s);
    // 20. Version
    char* alice_codec_version();
}

// ============================================
// RAII Wrappers
// ============================================

namespace Alice
{

/// RAII wrapper for Wavelet1D
class FWavelet1D
{
    Wavelet1D* Handle;

public:
    static FWavelet1D Haar() { return FWavelet1D(alice_codec_wavelet1d_haar()); }
    static FWavelet1D Cdf53() { return FWavelet1D(alice_codec_wavelet1d_cdf53()); }
    static FWavelet1D Cdf97() { return FWavelet1D(alice_codec_wavelet1d_cdf97()); }

    ~FWavelet1D()
    {
        if (Handle) alice_codec_wavelet1d_destroy(Handle);
    }

    FWavelet1D(FWavelet1D&& Other) noexcept : Handle(Other.Handle) { Other.Handle = nullptr; }
    FWavelet1D& operator=(FWavelet1D&& Other) noexcept
    {
        if (this != &Other) { if (Handle) alice_codec_wavelet1d_destroy(Handle); Handle = Other.Handle; Other.Handle = nullptr; }
        return *this;
    }
    FWavelet1D(const FWavelet1D&) = delete;
    FWavelet1D& operator=(const FWavelet1D&) = delete;

    void Forward(int32_t* Data, uint32_t Len) const { alice_codec_wavelet1d_forward(Handle, Data, Len); }
    void Inverse(int32_t* Data, uint32_t Len) const { alice_codec_wavelet1d_inverse(Handle, Data, Len); }

private:
    explicit FWavelet1D(Wavelet1D* H) : Handle(H) {}
};

/// RAII wrapper for FrameEncoder
class FFrameEncoder
{
    FrameEncoder* Handle;

public:
    explicit FFrameEncoder(uint8_t Quality) : Handle(alice_codec_encoder_create(Quality)) {}

    ~FFrameEncoder()
    {
        if (Handle) alice_codec_encoder_destroy(Handle);
    }

    FFrameEncoder(FFrameEncoder&& Other) noexcept : Handle(Other.Handle) { Other.Handle = nullptr; }
    FFrameEncoder& operator=(FFrameEncoder&& Other) noexcept
    {
        if (this != &Other) { if (Handle) alice_codec_encoder_destroy(Handle); Handle = Other.Handle; Other.Handle = nullptr; }
        return *this;
    }
    FFrameEncoder(const FFrameEncoder&) = delete;
    FFrameEncoder& operator=(const FFrameEncoder&) = delete;

    EncodedChunk* Encode(const uint8_t* Rgb, uint32_t RgbLen,
        uint32_t Width, uint32_t Height, uint32_t Frames) const
    {
        return alice_codec_encode(Handle, Rgb, RgbLen, Width, Height, Frames);
    }
};

/// RAII wrapper for EncodedChunk
class FEncodedChunk
{
    EncodedChunk* Handle;

public:
    explicit FEncodedChunk(EncodedChunk* H) : Handle(H) {}

    ~FEncodedChunk()
    {
        if (Handle) alice_codec_chunk_destroy(Handle);
    }

    FEncodedChunk(FEncodedChunk&& Other) noexcept : Handle(Other.Handle) { Other.Handle = nullptr; }
    FEncodedChunk& operator=(FEncodedChunk&& Other) noexcept
    {
        if (this != &Other) { if (Handle) alice_codec_chunk_destroy(Handle); Handle = Other.Handle; Other.Handle = nullptr; }
        return *this;
    }
    FEncodedChunk(const FEncodedChunk&) = delete;
    FEncodedChunk& operator=(const FEncodedChunk&) = delete;

    uint32_t Width() const { return alice_codec_chunk_width(Handle); }
    uint32_t Height() const { return alice_codec_chunk_height(Handle); }
    uint32_t Frames() const { return alice_codec_chunk_frames(Handle); }

    const EncodedChunk* Get() const { return Handle; }

    static FEncodedChunk FromBytes(const uint8_t* Data, uint32_t Len)
    {
        return FEncodedChunk(alice_codec_chunk_from_bytes(Data, Len));
    }
};

} // namespace Alice
