#!/usr/bin/env python3
"""Validate the per-row routing metadata used by training and evaluation."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


SOURCES = {"swe_gym", "multi_swe", "swe_rebench"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--check-images", action="store_true", help="Check local Docker archives without loading them")
    args = parser.parse_args()

    counts: collections.Counter[str] = collections.Counter()
    with args.input.open(encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, 1):
            if not line.strip():
                continue
            metadata = json.loads(line).get("metadata") or {}
            source = str(metadata.get("source_dataset") or "")
            if source not in SOURCES or not str(metadata.get("repo_dir") or "").startswith("/"):
                raise SystemExit(
                    f"{args.input}:{line_number}: invalid routing metadata for "
                    f"{metadata.get('instance_id', '<unknown>')}"
                )
            if args.check_images:
                raw_path = metadata.get("image_tar")
                if not raw_path:
                    raise SystemExit(f"{args.input}:{line_number}: missing image_tar")
                image_path = Path(str(raw_path))
                if not image_path.is_absolute():
                    image_path = Path(__file__).resolve().parents[2] / image_path
                if not image_path.is_file():
                    raise SystemExit(f"Missing Docker archive: {image_path}")
            counts[source] += 1
    if not counts:
        raise SystemExit(f"{args.input}: prompt file is empty")
    print(
        f"validated {sum(counts.values())} routed rows: "
        + ", ".join(f"{name}={counts[name]}" for name in sorted(counts))
    )


if __name__ == "__main__":
    main()
