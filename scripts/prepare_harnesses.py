#!/usr/bin/env python3
"""Prepare the exact grading sources used by the supported HERO datasets."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
HARNESSES = (
    ("SWE-bench", "SWE-bench/SWE-bench", "f7bbbb2ccdf479001d6467c9e34af59e44a840f9", "swebench_network.patch", "swebench/harness/test_spec/test_spec.py"),
    ("SWE-Bench-Package", "SWE-Gym/SWE-Bench-Package", "16dd480cce9b27bf111a362d280881c6def5d2a7", "swegym_grading_compat.patch", "swegym/harness/grading.py"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT.parent / "harnesses")
    parser.add_argument("--check", action="store_true", help="Check prepared sources without downloading")
    args = parser.parse_args()
    for name, repo, revision, patch, required in HARNESSES:
        target = args.output_root / name
        marker = target / ".hero-revision"
        if target.exists():
            if not marker.is_file() or marker.read_text().strip() != revision or not (target / required).is_file():
                raise SystemExit(f"Unrecognized/incomplete harness directory: {target}; select a clean --output-root")
            subprocess.run(["git", "apply", "--reverse", "--check", str(ROOT / "scripts/patch" / patch)], cwd=target, check=True)
            print(f"Ready: {target} ({revision})")
            continue
        if args.check:
            raise SystemExit(f"Missing harness: {target}; run this script without --check")
        args.output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".hero-harness-", dir=args.output_root) as temp:
            archive = Path(temp) / "source.tar.gz"
            url = f"https://codeload.github.com/{repo}/tar.gz/{revision}"
            print(f"Downloading {name} at {revision}", flush=True)
            urllib.request.urlretrieve(url, archive)
            extract = Path(temp) / "source"
            with tarfile.open(archive, "r:gz") as source:
                source.extractall(extract, filter="data")
            directory = next(extract.iterdir())
            subprocess.run(["git", "apply", str(ROOT / "scripts/patch" / patch)], cwd=directory, check=True)
            (directory / ".hero-revision").write_text(revision + "\n")
            if not (directory / required).is_file():
                raise SystemExit(f"Downloaded harness is incomplete: {name}")
            shutil.move(str(directory), target)
            print(f"Prepared: {target}")


if __name__ == "__main__":
    main()
