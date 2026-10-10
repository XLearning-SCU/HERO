#!/usr/bin/env python3
"""Install the Claude Code version used by HERO into a shared mount."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

NODE_VERSION = "22.11.0"
CLAUDE_CODE_VERSION = "2.1.111"
NODE_URL = (
    f"https://nodejs.org/dist/v{NODE_VERSION}/"
    f"node-v{NODE_VERSION}-linux-x64.tar.xz"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[4] / "cli/claude_code"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    claude = args.output / "bin/claude"
    if claude.is_file() and not args.force:
        print(f"Claude Code mount already exists: {args.output}")
        return
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix="hero-node-") as temp_dir:
        archive = Path(temp_dir) / "node.tar.xz"
        print(f"downloading Node.js {NODE_VERSION} from {NODE_URL}")
        urllib.request.urlretrieve(NODE_URL, archive)
        with tarfile.open(archive, "r:xz") as source:
            prefix = f"node-v{NODE_VERSION}-linux-x64/"
            for member in source.getmembers():
                if not member.name.startswith(prefix):
                    continue
                member.name = member.name[len(prefix):]
                if member.name:
                    source.extract(member, args.output, filter="data")

    env = {**os.environ, "PATH": f"{args.output / 'bin'}:{os.environ.get('PATH', '')}"}
    subprocess.run(
        [
            str(args.output / "bin/npm"), "install", "-g", "--no-audit",
            "--no-fund", f"--prefix={args.output}",
            f"@anthropic-ai/claude-code@{CLAUDE_CODE_VERSION}",
        ],
        check=True,
        env=env,
    )
    if not claude.is_file():
        raise RuntimeError(f"npm did not create {claude}")
    print(f"prepared HERO Claude Code {CLAUDE_CODE_VERSION} at {args.output}")


if __name__ == "__main__":
    main()
