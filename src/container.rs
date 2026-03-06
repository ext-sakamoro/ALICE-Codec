//! コンテナ形式パーサー — MP4 (ISO BMFF) / MKV (Matroska/WebM) ヘッダー解析。
//!
//! ビデオコンテナのボックス/エレメント構造を解析し、
//! トラック情報・コーデックパラメータ・サンプルテーブルを抽出する。
//! ALICE-Codec 独自ストリームを標準コンテナに格納・読み取り可能にする基盤。
//!
//! Author: Moroya Sakamoto

use std::collections::HashMap;

// ============================================================================
// 共通型
// ============================================================================

/// コンテナ形式の種別。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerFormat {
    /// ISO Base Media File Format (MP4/MOV/3GP)。
    IsoBaseMedia,
    /// Matroska / `WebM`。
    Matroska,
    /// 不明。
    Unknown,
}

/// トラック種別。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackKind {
    Video,
    Audio,
    Subtitle,
    Other,
}

/// トラック情報。
#[derive(Debug, Clone)]
pub struct TrackInfo {
    /// トラック ID。
    pub id: u32,
    /// トラック種別。
    pub kind: TrackKind,
    /// コーデック識別子 (例: "avc1", "hev1", "`V_VP9`", "`A_OPUS`")。
    pub codec_id: String,
    /// 幅 (ビデオのみ)。
    pub width: Option<u32>,
    /// 高さ (ビデオのみ)。
    pub height: Option<u32>,
    /// サンプルレート (オーディオのみ)。
    pub sample_rate: Option<u32>,
    /// チャンネル数 (オーディオのみ)。
    pub channels: Option<u16>,
    /// 時間スケール (timescale)。
    pub timescale: u32,
    /// 長さ (timescale 単位)。
    pub duration: u64,
}

/// コンテナ解析結果。
#[derive(Debug, Clone)]
pub struct ContainerInfo {
    /// コンテナ形式。
    pub format: ContainerFormat,
    /// 全体の長さ (秒)。
    pub duration_secs: f64,
    /// トラック一覧。
    pub tracks: Vec<TrackInfo>,
    /// メタデータ (key-value)。
    pub metadata: HashMap<String, String>,
}

// ============================================================================
// ISO Base Media File Format (MP4) パーサー
// ============================================================================

/// MP4 ボックスヘッダー。
#[derive(Debug, Clone)]
pub struct Mp4Box {
    /// ボックスタイプ (4 文字コード)。
    pub box_type: [u8; 4],
    /// ペイロードサイズ (ヘッダー除く)。
    pub payload_size: u64,
    /// データ内のオフセット。
    pub data_offset: usize,
}

/// MP4 ボックスヘッダーをパース。
///
/// # Errors
///
/// データが不足している場合は `None` を返す。
#[must_use]
pub fn parse_mp4_box(data: &[u8], offset: usize) -> Option<Mp4Box> {
    if offset + 8 > data.len() {
        return None;
    }

    let size = u32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]) as u64;

    let box_type = [
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ];

    let (payload_size, data_offset) = if size == 1 {
        // 64-bit extended size
        if offset + 16 > data.len() {
            return None;
        }
        let ext_size = u64::from_be_bytes([
            data[offset + 8],
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
            data[offset + 12],
            data[offset + 13],
            data[offset + 14],
            data[offset + 15],
        ]);
        (ext_size.saturating_sub(16), offset + 16)
    } else if size == 0 {
        // ファイル末尾までのボックス
        let remaining = (data.len() - offset - 8) as u64;
        (remaining, offset + 8)
    } else {
        (size.saturating_sub(8), offset + 8)
    };

    Some(Mp4Box {
        box_type,
        payload_size,
        data_offset,
    })
}

/// MP4 のトップレベルボックスを列挙。
#[must_use]
pub fn list_mp4_boxes(data: &[u8]) -> Vec<Mp4Box> {
    let mut boxes = Vec::new();
    let mut offset = 0;

    while let Some(b) = parse_mp4_box(data, offset) {
        let total_size = b.payload_size as usize + (b.data_offset - offset);
        boxes.push(b);
        offset += total_size;
        if offset >= data.len() || total_size == 0 {
            break;
        }
    }

    boxes
}

/// ボックスタイプを文字列に変換。
#[must_use]
pub fn box_type_str(bt: &[u8; 4]) -> String {
    String::from_utf8_lossy(bt).to_string()
}

/// MP4 `ftyp` ボックスからブランドを抽出。
#[must_use]
pub fn parse_ftyp(data: &[u8]) -> Option<String> {
    if data.len() < 4 {
        return None;
    }
    Some(String::from_utf8_lossy(&data[..4]).to_string())
}

/// データが MP4 形式かどうか判定。
#[must_use]
pub fn is_mp4(data: &[u8]) -> bool {
    parse_mp4_box(data, 0).is_some_and(|b| b.box_type == *b"ftyp")
}

// ============================================================================
// Matroska (MKV/WebM) パーサー
// ============================================================================

/// EBML エレメントヘッダー。
#[derive(Debug, Clone)]
pub struct EbmlElement {
    /// エレメント ID。
    pub id: u64,
    /// データサイズ。
    pub data_size: u64,
    /// データ開始オフセット。
    pub data_offset: usize,
    /// ヘッダーサイズ (ID + サイズフィールド)。
    pub header_size: usize,
}

/// EBML 可変長整数をデコード。
///
/// 返り値: (値, 読み取りバイト数)。
#[must_use]
pub fn decode_vint(data: &[u8], offset: usize) -> Option<(u64, usize)> {
    if offset >= data.len() {
        return None;
    }

    let first = data[offset];
    if first == 0 {
        return None;
    }

    let len = first.leading_zeros() as usize + 1;
    if offset + len > data.len() || len > 8 {
        return None;
    }

    let mut value = u64::from(first);
    for i in 1..len {
        value = (value << 8) | u64::from(data[offset + i]);
    }

    Some((value, len))
}

/// EBML 可変長整数をデコード (サイズフィールド用、マスクビット除去)。
#[must_use]
pub fn decode_vint_size(data: &[u8], offset: usize) -> Option<(u64, usize)> {
    let (raw, len) = decode_vint(data, offset)?;
    // 先頭ビットを除去
    let mask = 1u64 << (7 * len);
    let value = raw ^ mask;
    Some((value, len))
}

/// EBML エレメントをパース。
#[must_use]
pub fn parse_ebml_element(data: &[u8], offset: usize) -> Option<EbmlElement> {
    let (id, id_len) = decode_vint(data, offset)?;
    let (data_size, size_len) = decode_vint_size(data, offset + id_len)?;

    Some(EbmlElement {
        id,
        data_size,
        data_offset: offset + id_len + size_len,
        header_size: id_len + size_len,
    })
}

/// データが Matroska/WebM 形式かどうか判定。
///
/// EBML ヘッダー (ID = 0x1A45DFA3) で開始するかチェック。
#[must_use]
pub fn is_matroska(data: &[u8]) -> bool {
    if data.len() < 4 {
        return false;
    }
    // EBML header magic
    data[0] == 0x1A && data[1] == 0x45 && data[2] == 0xDF && data[3] == 0xA3
}

/// コンテナ形式を自動検出。
#[must_use]
pub fn detect_format(data: &[u8]) -> ContainerFormat {
    if is_mp4(data) {
        ContainerFormat::IsoBaseMedia
    } else if is_matroska(data) {
        ContainerFormat::Matroska
    } else {
        ContainerFormat::Unknown
    }
}

/// EBML のトップレベルエレメントを列挙。
#[must_use]
pub fn list_ebml_elements(data: &[u8]) -> Vec<EbmlElement> {
    let mut elements = Vec::new();
    let mut offset = 0;

    while let Some(elem) = parse_ebml_element(data, offset) {
        let total = elem.header_size + elem.data_size as usize;
        elements.push(elem);
        offset += total;
        if offset >= data.len() || total == 0 {
            break;
        }
    }

    elements
}

// ============================================================================
// テスト
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- MP4 テスト ---

    #[test]
    fn parse_mp4_box_basic() {
        // size=20 (0x14), type="ftyp"
        let mut data = vec![0x00, 0x00, 0x00, 0x14, b'f', b't', b'y', b'p'];
        data.extend_from_slice(&[0; 12]); // ペイロード
        let b = parse_mp4_box(&data, 0).unwrap();
        assert_eq!(&b.box_type, b"ftyp");
        assert_eq!(b.payload_size, 12);
        assert_eq!(b.data_offset, 8);
    }

    #[test]
    fn parse_mp4_box_extended_size() {
        // size=1 → 64-bit extended
        let mut data = vec![0x00, 0x00, 0x00, 0x01, b'm', b'd', b'a', b't'];
        // 64-bit extended size = 24
        data.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 24]);
        data.extend_from_slice(&[0; 8]); // ペイロード
        let b = parse_mp4_box(&data, 0).unwrap();
        assert_eq!(&b.box_type, b"mdat");
        assert_eq!(b.payload_size, 8); // 24 - 16
        assert_eq!(b.data_offset, 16);
    }

    #[test]
    fn parse_mp4_box_too_short() {
        let data = vec![0x00, 0x00, 0x00];
        assert!(parse_mp4_box(&data, 0).is_none());
    }

    #[test]
    fn list_mp4_boxes_multiple() {
        let mut data = Vec::new();
        // Box 1: ftyp, size=12
        data.extend_from_slice(&[0, 0, 0, 12, b'f', b't', b'y', b'p', 0, 0, 0, 0]);
        // Box 2: moov, size=16
        data.extend_from_slice(&[0, 0, 0, 16, b'm', b'o', b'o', b'v', 0, 0, 0, 0, 0, 0, 0, 0]);

        let boxes = list_mp4_boxes(&data);
        assert_eq!(boxes.len(), 2);
        assert_eq!(box_type_str(&boxes[0].box_type), "ftyp");
        assert_eq!(box_type_str(&boxes[1].box_type), "moov");
    }

    #[test]
    fn box_type_str_ascii() {
        assert_eq!(box_type_str(b"ftyp"), "ftyp");
        assert_eq!(box_type_str(b"mdat"), "mdat");
    }

    #[test]
    fn parse_ftyp_brand() {
        let data = b"isom\x00\x00\x00\x00";
        let brand = parse_ftyp(data).unwrap();
        assert_eq!(brand, "isom");
    }

    #[test]
    fn parse_ftyp_too_short() {
        assert!(parse_ftyp(&[0, 1, 2]).is_none());
    }

    #[test]
    fn is_mp4_valid() {
        let mut data = vec![0, 0, 0, 20, b'f', b't', b'y', b'p'];
        data.extend_from_slice(&[0; 12]);
        assert!(is_mp4(&data));
    }

    #[test]
    fn is_mp4_invalid() {
        let data = vec![0, 0, 0, 20, b'm', b'o', b'o', b'v', 0, 0, 0, 0];
        assert!(!is_mp4(&data));
    }

    // --- Matroska テスト ---

    #[test]
    fn decode_vint_1byte() {
        // 0x82 = 1000_0010 → len=1, value=0x82
        let (val, len) = decode_vint(&[0x82], 0).unwrap();
        assert_eq!(len, 1);
        assert_eq!(val, 0x82);
    }

    #[test]
    fn decode_vint_2byte() {
        // 0x40, 0x01 → len=2, value=0x4001
        let (val, len) = decode_vint(&[0x40, 0x01], 0).unwrap();
        assert_eq!(len, 2);
        assert_eq!(val, 0x4001);
    }

    #[test]
    fn decode_vint_zero() {
        assert!(decode_vint(&[0x00], 0).is_none());
    }

    #[test]
    fn decode_vint_empty() {
        assert!(decode_vint(&[], 0).is_none());
    }

    #[test]
    fn decode_vint_size_1byte() {
        // 0x85 = 1000_0101 → len=1, raw=0x85, mask=0x80 → size=5
        let (val, len) = decode_vint_size(&[0x85], 0).unwrap();
        assert_eq!(len, 1);
        assert_eq!(val, 5);
    }

    #[test]
    fn decode_vint_size_2byte() {
        // 0x40, 0x10 → len=2, raw=0x4010, mask=0x4000 → size=0x10=16
        let (val, len) = decode_vint_size(&[0x40, 0x10], 0).unwrap();
        assert_eq!(len, 2);
        assert_eq!(val, 16);
    }

    #[test]
    fn parse_ebml_element_basic() {
        // ID: 0x1A45DFA3 (4 bytes), Size: 0x85 (1 byte, value=5)
        let data = [0x1A, 0x45, 0xDF, 0xA3, 0x85, 0, 0, 0, 0, 0];
        let elem = parse_ebml_element(&data, 0).unwrap();
        assert_eq!(elem.id, 0x1A45_DFA3);
        assert_eq!(elem.data_size, 5);
        assert_eq!(elem.data_offset, 5);
    }

    #[test]
    fn is_matroska_valid() {
        let data = [0x1A, 0x45, 0xDF, 0xA3, 0x85, 0, 0, 0, 0, 0];
        assert!(is_matroska(&data));
    }

    #[test]
    fn is_matroska_invalid() {
        let data = [0x00, 0x00, 0x00, 0x14];
        assert!(!is_matroska(&data));
    }

    #[test]
    fn is_matroska_too_short() {
        assert!(!is_matroska(&[0x1A, 0x45]));
    }

    #[test]
    fn detect_format_mp4() {
        let mut data = vec![0, 0, 0, 20, b'f', b't', b'y', b'p'];
        data.extend_from_slice(&[0; 12]);
        assert_eq!(detect_format(&data), ContainerFormat::IsoBaseMedia);
    }

    #[test]
    fn detect_format_matroska() {
        let data = [0x1A, 0x45, 0xDF, 0xA3, 0x85, 0, 0, 0, 0, 0];
        assert_eq!(detect_format(&data), ContainerFormat::Matroska);
    }

    #[test]
    fn detect_format_unknown() {
        let data = [0xFF, 0xFF, 0xFF, 0xFF];
        assert_eq!(detect_format(&data), ContainerFormat::Unknown);
    }

    #[test]
    fn list_ebml_elements_basic() {
        // Element 1: ID=0x82(1byte), Size=0x82(1byte, val=2), data=2bytes
        // Element 2: ID=0x83(1byte), Size=0x81(1byte, val=1), data=1byte
        let data = [0x82, 0x82, 0xAA, 0xBB, 0x83, 0x81, 0xCC];
        let elems = list_ebml_elements(&data);
        assert_eq!(elems.len(), 2);
        assert_eq!(elems[0].id, 0x82);
        assert_eq!(elems[0].data_size, 2);
        assert_eq!(elems[1].id, 0x83);
        assert_eq!(elems[1].data_size, 1);
    }

    #[test]
    fn container_info_default() {
        let info = ContainerInfo {
            format: ContainerFormat::Unknown,
            duration_secs: 0.0,
            tracks: Vec::new(),
            metadata: HashMap::new(),
        };
        assert!(info.tracks.is_empty());
        assert_eq!(info.format, ContainerFormat::Unknown);
    }

    #[test]
    fn track_info_video() {
        let track = TrackInfo {
            id: 1,
            kind: TrackKind::Video,
            codec_id: "avc1".to_string(),
            width: Some(1920),
            height: Some(1080),
            sample_rate: None,
            channels: None,
            timescale: 90000,
            duration: 900_000,
        };
        assert_eq!(track.kind, TrackKind::Video);
        assert_eq!(track.width, Some(1920));
    }

    #[test]
    fn track_info_audio() {
        let track = TrackInfo {
            id: 2,
            kind: TrackKind::Audio,
            codec_id: "A_OPUS".to_string(),
            width: None,
            height: None,
            sample_rate: Some(48000),
            channels: Some(2),
            timescale: 48000,
            duration: 480_000,
        };
        assert_eq!(track.kind, TrackKind::Audio);
        assert_eq!(track.sample_rate, Some(48000));
    }

    #[test]
    fn mp4_box_size_zero() {
        // size=0 → ファイル末尾までのボックス
        let data = vec![0, 0, 0, 0, b'm', b'd', b'a', b't', 1, 2, 3, 4, 5];
        let b = parse_mp4_box(&data, 0).unwrap();
        assert_eq!(&b.box_type, b"mdat");
        assert_eq!(b.payload_size, 5); // 13 - 8
    }
}
