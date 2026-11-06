//! `alice_codec::container`: box and element sizes against their specifications.
//!
//! oracle: the expected column is written from the specifications, not from the code.
//! - ISO/IEC 14496-12 §4.2.2: `size` counts the header; `size == 1` means a 64-bit `largesize` follows;
//!   `size == 0` means the box extends to the end of the file. A header is at least 8 bytes (16 with
//!   `largesize`), so a `size` of 2 to 7 and a `largesize` under 16 cannot be a box. A `uuid` box has a
//!   16-byte `usertype` in its header, which `payload_size` and `data_offset` leave out.
//! - RFC 8794 §4: a VINT has its width in the leading zeros of the first byte. §5 and §6.1 allow widths
//!   over 8 through `EBMLMaxIDLength` / `EBMLMaxSizeLength`; a width over 8 is refused as the limit of
//!   this parser, not of the RFC. §5: the data of an Element ID must not be all zeros or all ones, and
//!   an ID uses the shortest encoding. §6.2: a size with all data bits set means an unknown size,
//!   legal for a master element in a stream.
//!
//! `Truncated` is a header the data ends inside. `PastEnd` is a whole header whose box or element
//! ends after the data: a file cut short and a corrupt size look the same, and the specifications
//! give no way to tell them apart.
//!
//! The bug this pins differed by profile (a panic with overflow checks, a wrong offset without
//! them), so CI runs this file in debug and in release.

use alice_codec::container::{
    detect_format, is_mp4, list_ebml_elements, list_mp4_boxes, parse_ebml_element, parse_mp4_box,
    ContainerError, ContainerFormat,
};
use ContainerError::{InvalidSize, PastEnd, Truncated, UnknownSize};

/// A box with a 32-bit size field holding `size`, followed by `payload`.
fn mp4(size: u32, kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = size.to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    out
}

/// A box with `size == 1` and the given `largesize`, followed by `payload`.
fn mp4_large(largesize: u64, kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = mp4(1, kind, &largesize.to_be_bytes());
    out.extend_from_slice(payload);
    out
}

/// (type, `payload_size`, `data_offset`) of each box.
type Boxes = Result<Vec<(&'static [u8; 4], u64, usize)>, ContainerError>;

fn mp4_rows() -> Vec<(&'static str, Vec<u8>, Boxes)> {
    let two = [mp4(10, b"aaaa", b"xy"), mp4(8, b"bbbb", b"")].concat();
    vec![
        ("no bytes", vec![], Ok(vec![])),
        (
            "one box",
            mp4(10, b"aaaa", b"xy"),
            Ok(vec![(b"aaaa", 2, 8)]),
        ),
        (
            "no payload",
            mp4(8, b"aaaa", b""),
            Ok(vec![(b"aaaa", 0, 8)]),
        ),
        (
            "two boxes",
            two.clone(),
            Ok(vec![(b"aaaa", 2, 8), (b"bbbb", 0, 18)]),
        ),
        // size 0: to the end of the file, and the walk ends there
        (
            "size 0 with bytes after the header",
            mp4(0, b"mdat", b"abc"),
            Ok(vec![(b"mdat", 3, 8)]),
        ),
        (
            "size 0 at the very end",
            mp4(0, b"mdat", b""),
            Ok(vec![(b"mdat", 0, 8)]),
        ),
        (
            "size 0 after another box",
            [mp4(8, b"aaaa", b""), mp4(0, b"mdat", b"abcdefghijkl")].concat(),
            Ok(vec![(b"aaaa", 0, 8), (b"mdat", 12, 16)]),
        ),
        (
            "bytes of a box after size 0 are payload",
            [mp4(0, b"mdat", b""), mp4(8, b"aaaa", b"")].concat(),
            Ok(vec![(b"mdat", 8, 8)]),
        ),
        // size 1: largesize
        (
            "largesize",
            mp4_large(18, b"aaaa", b"xy"),
            Ok(vec![(b"aaaa", 2, 16)]),
        ),
        (
            "largesize, no payload",
            mp4_large(16, b"aaaa", b""),
            Ok(vec![(b"aaaa", 0, 16)]),
        ),
        (
            "largesize then a box",
            [mp4_large(16, b"aaaa", b""), mp4(8, b"bbbb", b"")].concat(),
            Ok(vec![(b"aaaa", 0, 16), (b"bbbb", 0, 24)]),
        ),
        (
            "largesize 15",
            mp4_large(15, b"aaaa", b""),
            Err(InvalidSize { offset: 0 }),
        ),
        (
            "largesize 8",
            mp4_large(8, b"aaaa", b""),
            Err(InvalidSize { offset: 0 }),
        ),
        (
            "largesize 1",
            mp4_large(1, b"aaaa", b""),
            Err(InvalidSize { offset: 0 }),
        ),
        (
            "largesize 0",
            mp4_large(0, b"aaaa", b""),
            Err(InvalidSize { offset: 0 }),
        ),
        // uuid: 16 more bytes of header, left out of the payload
        (
            "uuid",
            mp4(26, b"uuid", &[7; 18]),
            Ok(vec![(b"uuid", 2, 24)]),
        ),
        (
            "uuid, no payload",
            mp4(24, b"uuid", &[7; 16]),
            Ok(vec![(b"uuid", 0, 24)]),
        ),
        (
            "uuid then a box",
            [mp4(24, b"uuid", &[7; 16]), mp4(8, b"bbbb", b"")].concat(),
            Ok(vec![(b"uuid", 0, 24), (b"bbbb", 0, 32)]),
        ),
        (
            "uuid, size 0",
            mp4(0, b"uuid", &[7; 19]),
            Ok(vec![(b"uuid", 3, 24)]),
        ),
        (
            "uuid, size 0, no payload",
            mp4(0, b"uuid", &[7; 16]),
            Ok(vec![(b"uuid", 0, 24)]),
        ),
        (
            "uuid, largesize",
            mp4_large(34, b"uuid", &[7; 18]),
            Ok(vec![(b"uuid", 2, 32)]),
        ),
        (
            "uuid, largesize 32",
            mp4_large(32, b"uuid", &[7; 16]),
            Ok(vec![(b"uuid", 0, 32)]),
        ),
        (
            "uuid, size 23",
            mp4(23, b"uuid", &[7; 16]),
            Err(InvalidSize { offset: 0 }),
        ),
        (
            "uuid, size 23, cut",
            mp4(23, b"uuid", &[7; 3]),
            Err(InvalidSize { offset: 0 }),
        ),
        (
            "uuid, size 8",
            mp4(8, b"uuid", &[7; 16]),
            Err(InvalidSize { offset: 0 }),
        ),
        (
            "uuid, largesize 31",
            mp4_large(31, b"uuid", &[7; 16]),
            Err(InvalidSize { offset: 0 }),
        ),
        (
            "uuid, largesize 16",
            mp4_large(16, b"uuid", &[7; 16]),
            Err(InvalidSize { offset: 0 }),
        ),
        (
            "uuid, usertype cut",
            mp4(24, b"uuid", &[7; 15]),
            Err(Truncated { offset: 0 }),
        ),
        (
            "uuid, size 0, usertype cut",
            mp4(0, b"uuid", &[7; 15]),
            Err(Truncated { offset: 0 }),
        ),
        (
            "uuid, largesize, usertype cut",
            mp4_large(32, b"uuid", &[7; 15]),
            Err(Truncated { offset: 0 }),
        ),
        (
            "uuid, payload past the end",
            mp4(25, b"uuid", &[7; 16]),
            Err(PastEnd { offset: 0 }),
        ),
        // sizes that cannot hold the header
        (
            "size 2",
            mp4(2, b"aaaa", b""),
            Err(InvalidSize { offset: 0 }),
        ),
        (
            "size 4",
            mp4(4, b"aaaa", b""),
            Err(InvalidSize { offset: 0 }),
        ),
        (
            "size 7",
            mp4(7, b"aaaa", b""),
            Err(InvalidSize { offset: 0 }),
        ),
        (
            "size 7 after a box",
            [mp4(8, b"aaaa", b""), mp4(7, b"bbbb", b"")].concat(),
            Err(InvalidSize { offset: 8 }),
        ),
        // the data ends inside a header
        ("1 byte", vec![0], Err(Truncated { offset: 0 })),
        (
            "7 bytes",
            vec![0, 0, 0, 8, b'a', b'b', b'c'],
            Err(Truncated { offset: 0 }),
        ),
        (
            "size 1, no largesize",
            mp4(1, b"aaaa", b""),
            Err(Truncated { offset: 0 }),
        ),
        (
            "size 1, part of largesize",
            mp4(1, b"aaaa", &[0, 0, 0, 0, 0, 0, 0]),
            Err(Truncated { offset: 0 }),
        ),
        (
            "bytes after a box that are not a header",
            [mp4(8, b"aaaa", b""), vec![0, 0, 0]].concat(),
            Err(Truncated { offset: 8 }),
        ),
        // a whole header, the box ends after the data
        (
            "one byte short",
            mp4(9, b"aaaa", b""),
            Err(PastEnd { offset: 0 }),
        ),
        (
            "size u32::MAX",
            mp4(u32::MAX, b"aaaa", b""),
            Err(PastEnd { offset: 0 }),
        ),
        (
            "second box past the end",
            [mp4(8, b"aaaa", b""), mp4(100, b"bbbb", b"")].concat(),
            Err(PastEnd { offset: 8 }),
        ),
        (
            "largesize one byte short",
            mp4_large(17, b"aaaa", b""),
            Err(PastEnd { offset: 0 }),
        ),
        // adds to the offset without overflow only if the sum is checked
        (
            "largesize u64::MAX",
            mp4_large(u64::MAX, b"aaaa", b""),
            Err(PastEnd { offset: 0 }),
        ),
        (
            "largesize u64::MAX after a box",
            [mp4(8, b"aaaa", b""), mp4_large(u64::MAX, b"bbbb", b"")].concat(),
            Err(PastEnd { offset: 8 }),
        ),
    ]
}

#[test]
fn walking_boxes() {
    let rows = mp4_rows();
    assert!(rows.len() >= 30, "compared {}", rows.len());
    let mut wrong = 0;
    for (name, data, expected) in &rows {
        let got = list_mp4_boxes(data).map(|boxes| {
            boxes
                .iter()
                .map(|b| (b.box_type, b.payload_size, b.data_offset))
                .collect::<Vec<_>>()
        });
        let expected = expected
            .as_ref()
            .map(|v| v.iter().map(|(k, p, o)| (**k, *p, *o)).collect::<Vec<_>>())
            .map_err(|e| *e);
        if got != expected {
            wrong += 1;
            eprintln!("{name}: got {got:?}, expected {expected:?}");
        }
    }
    assert_eq!(wrong, 0, "{wrong} of {} rows differ", rows.len());
}

#[test]
fn the_input_of_the_fuzz_crash() {
    // the scheduled fuzz run's input (fuzz_container_parse): a box of size 4, then at offset 8 a box
    // with size 1 and largesize all ones
    let mut input = vec![0, 0, 0, 4, 5, 0x29, 0, 0, 0, 0, 0, 1];
    input.extend_from_slice(&[0xff; 85]);
    input.extend_from_slice(&[0; 11]);
    input.extend_from_slice(&[1, 0x40, 0x90, 0x0d, 0, 1, 0xf3]);
    assert_eq!(input.len(), 115);
    assert_eq!(
        list_mp4_boxes(&input).map(|b| b.len()),
        Err(InvalidSize { offset: 0 })
    );
    // past the first box, as the old walk reached it
    assert_eq!(
        list_mp4_boxes(&input[8..]).map(|b| b.len()),
        Err(PastEnd { offset: 0 })
    );
}

#[test]
fn parsing_one_box_reads_its_header_only() {
    // a box past the end of the data is a header that can be read; only a walk needs its end
    let b = parse_mp4_box(&mp4(100, b"aaaa", b""), 0).unwrap().unwrap();
    assert_eq!((b.payload_size, b.data_offset), (92, 8));
    assert_eq!(
        parse_mp4_box(&mp4(4, b"aaaa", b""), 0).map(|b| b.is_some()),
        Err(InvalidSize { offset: 0 })
    );
    let data = mp4(8, b"aaaa", b"");
    assert_eq!(parse_mp4_box(&data, 8).map(|b| b.is_some()), Ok(false));
    assert_eq!(parse_mp4_box(&data, 9).map(|b| b.is_some()), Ok(false));
    assert_eq!(
        parse_mp4_box(&data, usize::MAX).map(|b| b.is_some()),
        Ok(false)
    );
    assert_eq!(
        parse_mp4_box(&data, 4).map(|b| b.is_some()),
        Err(Truncated { offset: 4 })
    );
}

#[test]
fn recognising_mp4() {
    assert!(is_mp4(&mp4(8, b"ftyp", b"")));
    assert!(
        is_mp4(&mp4(100, b"ftyp", b"")),
        "a cut file is still recognised"
    );
    assert!(!is_mp4(&mp4(8, b"moov", b"")));
    assert!(!is_mp4(&mp4(4, b"ftyp", b"")));
    assert!(!is_mp4(&mp4_large(15, b"ftyp", b"")));
    assert!(!is_mp4(b"ftyp"));
    assert!(!is_mp4(b""));
    assert_eq!(
        detect_format(&mp4(4, b"ftyp", b"")),
        ContainerFormat::Unknown
    );
    assert_eq!(
        detect_format(&mp4(8, b"ftyp", b"")),
        ContainerFormat::IsoBaseMedia
    );
}

/// (id, `data_size`, `data_offset`) of each element.
type Elements = Result<Vec<(u64, u64, usize)>, ContainerError>;

fn ebml_rows() -> Vec<(&'static str, Vec<u8>, Elements)> {
    let one = vec![0x81, 0x82, b'x', b'y'];
    vec![
        ("no bytes", vec![], Ok(vec![])),
        ("one element", one.clone(), Ok(vec![(0x81, 2, 2)])),
        ("no data", vec![0x81, 0x80], Ok(vec![(0x81, 0, 2)])),
        (
            "two elements",
            [one.clone(), vec![0x82, 0x80]].concat(),
            Ok(vec![(0x81, 2, 2), (0x82, 0, 6)]),
        ),
        (
            "a 4-byte id",
            vec![0x1a, 0x45, 0xdf, 0xa3, 0x80],
            Ok(vec![(0x1a45_dfa3, 0, 5)]),
        ),
        (
            "a 2-byte size",
            vec![0x81, 0x40, 0x01, b'x'],
            Ok(vec![(0x81, 1, 3)]),
        ),
        (
            "an 8-byte size",
            vec![0x81, 1, 0, 0, 0, 0, 0, 0, 1, b'x'],
            Ok(vec![(0x81, 1, 9)]),
        ),
        // unknown size: all data bits set, at every width (RFC 8794 §6.2)
        (
            "unknown, 1 byte",
            vec![0x81, 0xff],
            Err(UnknownSize { offset: 0 }),
        ),
        (
            "unknown, 2 bytes",
            vec![0x81, 0x7f, 0xff],
            Err(UnknownSize { offset: 0 }),
        ),
        (
            "unknown, 8 bytes",
            vec![0x81, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            Err(UnknownSize { offset: 0 }),
        ),
        (
            "unknown after an element",
            [one.clone(), vec![0x82, 0xff]].concat(),
            Err(UnknownSize { offset: 4 }),
        ),
        // one below unknown is an ordinary size
        (
            "largest 8-byte size",
            vec![0x81, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe],
            Err(PastEnd { offset: 0 }),
        ),
        // the shortest encoding of an ID (RFC 8794 §5)
        (
            "id 1 in 2 bytes",
            vec![0x40, 0x01, 0x80],
            Err(InvalidSize { offset: 0 }),
        ),
        (
            "id 126 in 2 bytes",
            vec![0x40, 0x7e, 0x80],
            Err(InvalidSize { offset: 0 }),
        ),
        (
            "id 1 in 4 bytes",
            vec![0x10, 0, 0, 0x01, 0x80],
            Err(InvalidSize { offset: 0 }),
        ),
        // 127 is all ones in one byte, which an ID must not be, so two bytes is its shortest
        (
            "id 127 in 2 bytes",
            vec![0x40, 0x7f, 0x80],
            Ok(vec![(0x407f, 0, 3)]),
        ),
        (
            "id 128 in 2 bytes",
            vec![0x40, 0x80, 0x80],
            Ok(vec![(0x4080, 0, 3)]),
        ),
        // the width of a size has no such rule
        (
            "size 1 in 2 bytes",
            vec![0x81, 0x40, 0x01, b'x'],
            Ok(vec![(0x81, 1, 3)]),
        ),
        // invalid
        // over 8 is the limit of this parser, not of RFC 8794 (§5, §6.1)
        (
            "id width over 8",
            vec![0x00, 0x81, 0x80],
            Err(InvalidSize { offset: 0 }),
        ),
        (
            "size width over 8",
            vec![0x81, 0x00, 0x80],
            Err(InvalidSize { offset: 0 }),
        ),
        (
            "id all zero data",
            vec![0x80, 0x80],
            Err(InvalidSize { offset: 0 }),
        ),
        (
            "id all one data",
            vec![0xff, 0x80],
            Err(InvalidSize { offset: 0 }),
        ),
        (
            "2-byte id all one data",
            vec![0x7f, 0xff, 0x80],
            Err(InvalidSize { offset: 0 }),
        ),
        // the data ends inside a header
        ("id only", vec![0x81], Err(Truncated { offset: 0 })),
        ("part of an id", vec![0x40], Err(Truncated { offset: 0 })),
        (
            "part of a size",
            vec![0x81, 0x40],
            Err(Truncated { offset: 0 }),
        ),
        (
            "part of a header after an element",
            [one.clone(), vec![0x82]].concat(),
            Err(Truncated { offset: 4 }),
        ),
        // a whole header, the element ends after the data
        (
            "one byte short",
            vec![0x81, 0x81],
            Err(PastEnd { offset: 0 }),
        ),
        (
            "second element past the end",
            [one, vec![0x82, 0x85]].concat(),
            Err(PastEnd { offset: 4 }),
        ),
    ]
}

#[test]
fn walking_elements() {
    let rows = ebml_rows();
    assert!(rows.len() >= 20, "compared {}", rows.len());
    let mut wrong = 0;
    for (name, data, expected) in &rows {
        let got = list_ebml_elements(data).map(|elements| {
            elements
                .iter()
                .map(|e| (e.id, e.data_size, e.data_offset))
                .collect::<Vec<_>>()
        });
        if got != *expected {
            wrong += 1;
            eprintln!("{name}: got {got:?}, expected {expected:?}");
        }
    }
    assert_eq!(wrong, 0, "{wrong} of {} rows differ", rows.len());
}

#[test]
fn parsing_one_element_reads_its_header_only() {
    let e = parse_ebml_element(&[0x81, 0x85], 0).unwrap().unwrap();
    assert_eq!((e.id, e.data_size, e.header_size), (0x81, 5, 2));
    assert_eq!(
        parse_ebml_element(&[0x81, 0xff], 0).map(|e| e.is_some()),
        Err(UnknownSize { offset: 0 })
    );
    assert_eq!(
        parse_ebml_element(&[0x81, 0x80], 2).map(|e| e.is_some()),
        Ok(false)
    );
    assert_eq!(
        parse_ebml_element(&[0x81, 0x80], usize::MAX).map(|e| e.is_some()),
        Ok(false)
    );
}
