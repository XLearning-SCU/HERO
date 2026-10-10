#!/usr/bin/env python3
"""Check or download the Docker archives referenced by the release datasets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "train": "swe_train_640.jsonl",
    "verified": "swe_bench_verified.jsonl",
    "multilingual": "swe_bench_multilingual.jsonl",
}


def image_manifest(dataset: str) -> dict[Path, str]:
    images = {}
    names = DATASETS.values() if dataset == "all" else [DATASETS[dataset]]
    for name in names:
        for line in (ROOT.parent / "data" / name).read_text().splitlines():
            if not line.strip():
                continue
            metadata = json.loads(line)["metadata"]
            relative = Path(metadata["image_tar"])
            if relative.parts[:2] != ("..", "docker") or len(relative.parts) != 4:
                raise ValueError(f"Invalid release archive path: {relative}")
            path = ROOT.parent / "docker" / relative.parts[2] / relative.name
            image = metadata.get("sandbox_image") or metadata.get("image")
            if not image or image.startswith("-"):
                raise ValueError(f"Missing or invalid registry image in {name}")
            if path in images and images[path] != image:
                raise ValueError(f"Conflicting image references for {path}")
            images[path] = image
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--download", action="store_true", help="Pull missing images and save tar archives")
    parser.add_argument("--limit", type=int, default=0, help="Limit checked/downloaded images; 0 means all")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    images = list(image_manifest(args.dataset).items())
    if args.limit:
        images = images[:args.limit]
    missing = 0
    for path, image in images:
        if path.is_file() and path.stat().st_size > 0:
            continue
        if not args.download:
            missing += 1
            print(f"Missing: {path}")
            continue
        # A migrated release can contain broken local symlinks. Replace only
        # the broken link, never a live archive or its external target.
        if path.is_symlink():
            if path.exists():
                raise RuntimeError(f"Invalid existing archive link: {path}")
            path.unlink()
        elif path.exists():
            raise RuntimeError(f"Invalid existing archive: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Pulling {image}", flush=True)
        subprocess.run(["docker", "image", "pull", "--platform", "linux/amd64", image], check=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            subprocess.run(["docker", "image", "save", "--output", str(temporary), image], check=True)
            if temporary.stat().st_size == 0:
                raise RuntimeError(f"Docker produced an empty archive for {image}")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Checked {len(images)} archives; missing: {missing}")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
