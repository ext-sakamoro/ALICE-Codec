#!/usr/bin/env python3
"""A consumer that takes alice-codec by git gets a checkout with no sibling repositories next to it, so the
dependencies it uses must resolve, and build, from the manifests alone.

- No manifest names a sibling by path (`path = "../…"`).
- A consumer crate next to a copy of the tracked tree placed alone builds alice-codec with the default features,
  with `ml` (the optional alice-ml) and with `db` (the optional alice-db), both taken by git rev. Each build must
  compile alice-codec itself and at least one crate, counted from cargo's JSON messages: a build that compiled nothing fails rather than passing.

This replaces the packaged-crate check on push while alice-codec's alice-* dependencies are taken by git rev
(`.github/workflows/package-integrity.yml` says when that check comes back).

Usage: python3 scripts/test_git_consumer.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIBLING_PATH = re.compile(r'''^\s*([A-Za-z0-9_-]+)\s*=\s*\{[^}]*\bpath\s*=\s*["']\.\./''')


def tracked(pattern: str | None = None) -> list[str]:
    args = ["git", "-C", str(ROOT), "ls-files"] + ([pattern] if pattern else [])
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.splitlines()


class GitConsumer(unittest.TestCase):
    def test_no_manifest_names_a_sibling_by_path(self):
        manifests = tracked("*Cargo.toml")
        self.assertTrue(manifests, "no Cargo.toml was read (compared nothing)")
        found = []
        for m in manifests:
            for n, line in enumerate((ROOT / m).read_text(encoding="utf-8").splitlines(), 1):
                hit = SIBLING_PATH.search(line.split("#", 1)[0])
                if hit:
                    found.append(f"{m}:{n}: {line.strip()}")
        self.assertEqual(found, [], "sibling path dependencies a git consumer cannot resolve")

    def build(self, features: list[str]) -> tuple[subprocess.CompletedProcess, list[str]]:
        """Builds a consumer of a copy of the tracked tree placed alone; returns the run and the names of the
        crates cargo compiled (`compiler-artifact` messages)"""
        with tempfile.TemporaryDirectory() as d:
            alone = Path(d) / "only" / "ALICE-Codec"
            copied = 0
            for f in tracked():
                src = ROOT / f
                if src.is_file():
                    dst = alone / f
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    copied += 1
            self.assertGreater(copied, 0, "no tracked file was copied")
            consumer = Path(d) / "consumer"
            (consumer / "src").mkdir(parents=True)
            feats = ", ".join(f'"{f}"' for f in features)
            (consumer / "Cargo.toml").write_text(
                '[package]\nname = "consumer"\nversion = "0.0.0"\nedition = "2021"\n\n[dependencies]\n'
                f'alice-codec = {{ path = "../only/ALICE-Codec", features = [{feats}] }}\n\n[workspace]\n',
                encoding="utf-8")
            (consumer / "src" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
            env = dict(os.environ, CARGO_TARGET_DIR=str(Path(d) / "target"))
            p = subprocess.run(["cargo", "build", "--message-format=json"], cwd=consumer, env=env,
                               capture_output=True, text=True, timeout=1800)
            built = []
            for line in p.stdout.splitlines():
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("reason") == "compiler-artifact":
                    built.append(msg["target"]["name"])
            return p, built

    def check(self, features: list[str], also: str | None = None):
        p, built = self.build(features)
        self.assertEqual(p.returncode, 0, p.stderr[-3000:])
        self.assertGreater(len(built), 0, "the consumer build compiled no crate")
        self.assertIn("alice_codec", built, f"alice-codec was not compiled: {sorted(set(built))}")
        if also:
            self.assertIn(also, built, f"{also} was not compiled: {sorted(set(built))}")

    def test_a_consumer_with_the_default_features_builds_alone(self):
        self.check([], also="alice_det_math")

    def test_a_consumer_with_ml_builds_alone(self):
        self.check(["ml"], also="alice_ml")

    def test_a_consumer_with_db_builds_alone(self):
        self.check(["db"], also="alice_db")


if __name__ == "__main__":
    unittest.main()
