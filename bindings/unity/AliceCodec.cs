// ALICE-Codec Unity C# Bindings
// Auto-generated — 20 DllImport functions
// Author: Moroya Sakamoto

using System;
using System.Runtime.InteropServices;

namespace Alice.Codec
{
    // ========================================
    // Wavelet1D — Integer Wavelet Transform
    // ========================================

    public sealed class Wavelet1D : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        private Wavelet1D(IntPtr handle) { _handle = handle; }

        /// <summary>Create Haar wavelet.</summary>
        public static Wavelet1D Haar() => new Wavelet1D(NativeMethods.alice_codec_wavelet1d_haar());

        /// <summary>Create CDF 5/3 wavelet (lossless).</summary>
        public static Wavelet1D Cdf53() => new Wavelet1D(NativeMethods.alice_codec_wavelet1d_cdf53());

        /// <summary>Create CDF 9/7 wavelet (lossy).</summary>
        public static Wavelet1D Cdf97() => new Wavelet1D(NativeMethods.alice_codec_wavelet1d_cdf97());

        /// <summary>Forward wavelet transform (in-place).</summary>
        public void Forward(int[] signal)
        {
            NativeMethods.alice_codec_wavelet1d_forward(_handle, signal, (uint)signal.Length);
        }

        /// <summary>Inverse wavelet transform (in-place).</summary>
        public void Inverse(int[] signal)
        {
            NativeMethods.alice_codec_wavelet1d_inverse(_handle, signal, (uint)signal.Length);
        }

        public void Dispose()
        {
            if (!_disposed && _handle != IntPtr.Zero)
            {
                NativeMethods.alice_codec_wavelet1d_destroy(_handle);
                _handle = IntPtr.Zero;
                _disposed = true;
            }
        }

        ~Wavelet1D() { Dispose(); }
    }

    // ========================================
    // FrameEncoder — Video Frame Encoder
    // ========================================

    public sealed class FrameEncoder : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        /// <summary>Create encoder with quality (0-100).</summary>
        public FrameEncoder(byte quality)
        {
            _handle = NativeMethods.alice_codec_encoder_create(quality);
        }

        /// <summary>Encode RGB frames. Returns an EncodedChunk handle.</summary>
        public IntPtr Encode(byte[] rgb, uint width, uint height, uint frames)
        {
            return NativeMethods.alice_codec_encode(
                _handle, rgb, (uint)rgb.Length, width, height, frames);
        }

        public void Dispose()
        {
            if (!_disposed && _handle != IntPtr.Zero)
            {
                NativeMethods.alice_codec_encoder_destroy(_handle);
                _handle = IntPtr.Zero;
                _disposed = true;
            }
        }

        ~FrameEncoder() { Dispose(); }
    }

    // ========================================
    // EncodedChunk — Compressed Data Handle
    // ========================================

    public static class EncodedChunk
    {
        /// <summary>Get chunk width.</summary>
        public static uint Width(IntPtr chunk) => NativeMethods.alice_codec_chunk_width(chunk);

        /// <summary>Get chunk height.</summary>
        public static uint Height(IntPtr chunk) => NativeMethods.alice_codec_chunk_height(chunk);

        /// <summary>Get chunk frame count.</summary>
        public static uint Frames(IntPtr chunk) => NativeMethods.alice_codec_chunk_frames(chunk);

        /// <summary>Destroy chunk handle.</summary>
        public static void Destroy(IntPtr chunk) => NativeMethods.alice_codec_chunk_destroy(chunk);

        /// <summary>Serialize chunk to bytes.</summary>
        public static byte[] ToBytes(IntPtr chunk)
        {
            uint len = 0;
            var ptr = NativeMethods.alice_codec_chunk_to_bytes(chunk, ref len);
            if (ptr == IntPtr.Zero) return Array.Empty<byte>();
            var bytes = new byte[len];
            Marshal.Copy(ptr, bytes, 0, (int)len);
            NativeMethods.alice_codec_data_free(ptr, len);
            return bytes;
        }

        /// <summary>Deserialize chunk from bytes.</summary>
        public static IntPtr FromBytes(byte[] data)
        {
            return NativeMethods.alice_codec_chunk_from_bytes(data, (uint)data.Length);
        }
    }

    // ========================================
    // Decoder — Decode compressed chunks
    // ========================================

    public static class FrameDecoder
    {
        /// <summary>Decode chunk to RGB bytes.</summary>
        public static byte[] Decode(IntPtr chunk)
        {
            uint len = 0;
            var ptr = NativeMethods.alice_codec_decode(chunk, ref len);
            if (ptr == IntPtr.Zero) return Array.Empty<byte>();
            var rgb = new byte[len];
            Marshal.Copy(ptr, rgb, 0, (int)len);
            NativeMethods.alice_codec_data_free(ptr, len);
            return rgb;
        }
    }

    // ========================================
    // Metrics — Quality Measurement
    // ========================================

    public static class Metrics
    {
        /// <summary>Compute PSNR between two byte arrays. Returns -1 on error.</summary>
        public static double Psnr(byte[] a, byte[] b)
        {
            return NativeMethods.alice_codec_psnr(a, b, (uint)a.Length);
        }
    }

    // ========================================
    // Version
    // ========================================

    public static class Version
    {
        public static string Get()
        {
            var ptr = NativeMethods.alice_codec_version();
            if (ptr == IntPtr.Zero) return "";
            var str = Marshal.PtrToStringAnsi(ptr);
            NativeMethods.alice_codec_string_free(ptr);
            return str ?? "";
        }
    }

    // ========================================
    // P/Invoke declarations (20 functions)
    // ========================================

    internal static class NativeMethods
    {
        private const string Lib = "alice_codec";

        // 1
        [DllImport(Lib)] internal static extern IntPtr alice_codec_wavelet1d_haar();
        // 2
        [DllImport(Lib)] internal static extern IntPtr alice_codec_wavelet1d_cdf53();
        // 3
        [DllImport(Lib)] internal static extern IntPtr alice_codec_wavelet1d_cdf97();
        // 4
        [DllImport(Lib)] internal static extern void alice_codec_wavelet1d_destroy(IntPtr ptr);
        // 5
        [DllImport(Lib)] internal static extern void alice_codec_wavelet1d_forward(
            IntPtr wavelet, [In, Out] int[] data, uint len);
        // 6
        [DllImport(Lib)] internal static extern void alice_codec_wavelet1d_inverse(
            IntPtr wavelet, [In, Out] int[] data, uint len);
        // 7
        [DllImport(Lib)] internal static extern IntPtr alice_codec_encoder_create(byte quality);
        // 8
        [DllImport(Lib)] internal static extern void alice_codec_encoder_destroy(IntPtr ptr);
        // 9
        [DllImport(Lib)] internal static extern IntPtr alice_codec_encode(
            IntPtr encoder, byte[] rgb, uint rgbLen, uint width, uint height, uint frames);
        // 10
        [DllImport(Lib)] internal static extern IntPtr alice_codec_decode(
            IntPtr chunk, ref uint outLen);
        // 11
        [DllImport(Lib)] internal static extern void alice_codec_chunk_destroy(IntPtr ptr);
        // 12
        [DllImport(Lib)] internal static extern IntPtr alice_codec_chunk_to_bytes(
            IntPtr chunk, ref uint outLen);
        // 13
        [DllImport(Lib)] internal static extern IntPtr alice_codec_chunk_from_bytes(
            byte[] data, uint len);
        // 14
        [DllImport(Lib)] internal static extern uint alice_codec_chunk_width(IntPtr chunk);
        // 15
        [DllImport(Lib)] internal static extern uint alice_codec_chunk_height(IntPtr chunk);
        // 16
        [DllImport(Lib)] internal static extern uint alice_codec_chunk_frames(IntPtr chunk);
        // 17
        [DllImport(Lib)] internal static extern double alice_codec_psnr(
            byte[] a, byte[] b, uint len);
        // 18
        [DllImport(Lib)] internal static extern void alice_codec_data_free(IntPtr ptr, uint len);
        // 19
        [DllImport(Lib)] internal static extern void alice_codec_string_free(IntPtr s);
        // 20
        [DllImport(Lib)] internal static extern IntPtr alice_codec_version();
    }
}
