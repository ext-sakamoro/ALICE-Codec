#!/usr/bin/env python3
"""Breaks the box and element size handling one way at a time; tests/container_sizes.rs must go red.

usage: python3 scripts/container_mutants.py [--release]
exit 0 if every mutant applied, compiled and made a test report FAILED; exit 1 otherwise.

Run by hand when src/container.rs changes: each mutant rebuilds the crate. A mutant that does not
compile or no longer applies counts against the result, since it tested nothing. The file is restored
after each mutant, and at the end even after an interrupt.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src" / "container.rs"
COMMAND = ["cargo", "test", "--test", "container_sizes"]
FAILED = re.compile(r"^test .* \.\.\. .*FAILED", re.M)

END = """        .and_then(|size| data_offset.checked_add(size))
        .filter(|&end| end <= data.len())"""
# (what breaks, old text, new text)
MUTANTS = [
    ("offset sum wraps", END, "        .map(|size| data_offset.wrapping_add(size))\n        .filter(|&end| end <= data.len())"),
    ("a box past the end allowed", END, "        .and_then(|size| data_offset.checked_add(size))"),
    ("size 0 reads no payload", "0 => (after_header.len() as u64, 8),", "0 => (0, 8),"),
    ("size 2 to 7 allowed", "2..=7 => return Err(invalid),", "2..=7 => (0, 8),"),
    ("size 7 allowed", "2..=7 => return Err(invalid),", "2..=6 => return Err(invalid),"),
    ("largesize under 16 allowed", "largesize.checked_sub(16).ok_or(invalid)?", "largesize.saturating_sub(16)"),
    ("largesize header counted as 8", "(largesize.checked_sub(16).ok_or(invalid)?, 16)", "(largesize.checked_sub(16).ok_or(invalid)?, 8)"),
    ("a cut header is no box", "        return Err(truncated);\n    };\n    let box_type", "        return Ok(None);\n    };\n    let box_type"),
    ("a cut largesize is no box", "                return Err(truncated);", "                return Ok(None);"),
    ("offset past the data panics", "data.get(offset..).filter(|rest| !rest.is_empty())", "Some(&data[offset..]).filter(|rest| !rest.is_empty())"),
    ("is_mp4 ignores the error", 'matches!(parse_mp4_box(data, 0), Ok(Some(b)) if b.box_type == *b"ftyp")', 'data.get(4..8) == Some(b"ftyp")'),
    ("width over 8 is cut short", "Some(0) => Err(ContainerError::InvalidSize { offset }),", "Some(0) => Err(ContainerError::Truncated { offset }),"),
    ("a cut vint is invalid", "decode_vint(data, at).ok_or(ContainerError::Truncated { offset })", "decode_vint(data, at).ok_or(ContainerError::InvalidSize { offset })"),
    ("id all zero allowed", "if id == id_marker || id == (id_marker << 1) - 1 {", "if id == (id_marker << 1) - 1 {"),
    ("id all one allowed", "if id == id_marker || id == (id_marker << 1) - 1 {", "if id == id_marker {"),
    ("unknown size read as a length", "if data_size == size_marker - 1 {", "if false {"),
    ("unknown size is invalid", "return Err(ContainerError::UnknownSize { offset });", "return Err(ContainerError::InvalidSize { offset });"),
    ("one below unknown is unknown", "if data_size == size_marker - 1 {", "if data_size >= size_marker - 2 {"),
    ("uuid has the common header", 'if box_type == *b"uuid" {', 'if false {'),
    ("uuid usertype counted as payload", """        let payload = payload_size
            .checked_sub(UUID_USERTYPE_LEN as u64)
            .ok_or(invalid)?;
""", """        let payload = payload_size;
"""),
    ("uuid too short is cut, not invalid", """            return Err(if declares_too_little {
                invalid
            } else {
                truncated
            });
""", "            return Err(truncated);\n"),
    ("uuid cut is invalid", """            return Err(if declares_too_little {
                invalid
            } else {
                truncated
            });
""", "            return Err(invalid);\n"),
    ("uuid size 0 is too short", "let declares_too_little = size_field != 0 && payload_size", "let declares_too_little = payload_size"),
    ("an id need not be shortest", "if id_len > 1 && (id ^ id_marker) < (1u64 << (7 * (id_len - 1))) - 1 {", "if false {"),
    ("all ones of the shorter width is not shortest", "(id ^ id_marker) < (1u64 << (7 * (id_len - 1))) - 1", "(id ^ id_marker) < (1u64 << (7 * (id_len - 1)))"),
    ("a value one lower is shortest", "(id ^ id_marker) < (1u64 << (7 * (id_len - 1))) - 1", "(id ^ id_marker) < (1u64 << (7 * (id_len - 1))) - 2"),
    ("offset of past-end is 0", ".ok_or(ContainerError::PastEnd { offset })", ".ok_or(ContainerError::PastEnd { offset: 0 })"),
]


def main() -> int:
    extra = sys.argv[1:]
    if extra not in ([], ["--release"]):
        print(__doc__, file=sys.stderr)
        return 2
    original = SOURCE.read_text(encoding="utf-8")
    bad = []
    try:
        for name, old, new in MUTANTS:
            if original.count(old) != 1:
                bad.append(name)
                print(f"NOT APPLIED  {name}", flush=True)
                continue
            SOURCE.write_text(original.replace(old, new), encoding="utf-8")
            done = subprocess.run(
                COMMAND + extra, cwd=ROOT, capture_output=True, text=True, check=False
            )
            output = done.stdout + done.stderr
            if "could not compile" in output:
                bad.append(name)
                print(f"BROKEN       {name}", flush=True)
            elif done.returncode != 0 and FAILED.search(output):
                print(f"red          {name}", flush=True)
            else:
                bad.append(name)
                print(f"SURVIVED     {name} (rc {done.returncode})", flush=True)
    finally:
        SOURCE.write_text(original, encoding="utf-8")
    print(f"{len(MUTANTS) - len(bad)} of {len(MUTANTS)} red")
    return 1 if bad or not MUTANTS else 0


if __name__ == "__main__":
    sys.exit(main())
