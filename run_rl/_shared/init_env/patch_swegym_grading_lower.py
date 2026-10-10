#!/usr/bin/env python3
"""Idempotent patch: fix swegym get_logs_eval repo-case KeyError.

Run via the bash wrapper patch_swegym_grading_lower.sh (mirrors the other
run_rl/_shared/init_env/patch_*.sh scripts). Re-apply after every venv reinstall.

Background
----------
swegym 2.0.13 ``harness/grading.py::get_logs_eval`` derives the repo key from
the test-log path and indexes ``MAP_REPO_TO_PARSER`` without lowercasing:

    sample_id = str(Path(log_fp).parent.stem)        # e.g. Project-MONAI__MONAI-6924
    repo = "-".join(sample_id.replace("__", "/").split("-")[:-1])  # -> Project-MONAI/MONAI
    log_parser = MAP_REPO_TO_PARSER[repo]             # key is 'project-monai/monai' -> KeyError

``MAP_REPO_TO_PARSER`` ships all-lowercase keys, so any instance whose owner
or repo is uppercase (Project-MONAI/MONAI is the one in swe-gym-293, 53/293 =
18% of prompts) raises ``KeyError`` and the whole episode is graded as
failed instead of resolved. Both training-time rollout grading and the
eval-side ``run_harness_on_applied_patch`` share this code path, so the fix
benefits both.

Fix
---
Lowercase ``repo`` and use ``.get`` with a clear error so an unknown repo
still fails loudly instead of silently returning an empty status map.
"""
from __future__ import annotations

import pathlib
import sys

GRADING_MARKER = "log_parser = MAP_REPO_TO_PARSER[repo]"
LOWER_LINE = '    repo = repo.lower()  # ProRL patch: MAP_REPO_TO_PARSER keys are lowercase\n'


def main(grading_py: str) -> int:
    path = pathlib.Path(grading_py)
    if not path.is_file():
        print(f"ERROR: swegym grading.py not found: {path}", file=sys.stderr)
        return 1

    src = path.read_text()

    if "MAP_REPO_TO_PARSER.get(repo.lower())" in src:
        print("[patch_swegym_grading] repo.lower() + .get: already applied")
        return 0

    if GRADING_MARKER not in src:
        if "repo = repo.lower()" in src:
            print("[patch_swegym_grading] partial state (lower present, .get missing); "
                  "manual review needed", file=sys.stderr)
            return 1
        print("[patch_swegym_grading] WARNING: target line not found; swegym may have "
              "drifted, skipping", file=sys.stderr)
        return 1

    # 1) lowercase repo right after it is derived.
    sample_line = '    repo = "-".join(sample_id.replace("__", "/").split("-")[:-1])'
    if sample_line not in src:
        print("[patch_swegym_grading] WARNING: repo-derivation line not found; "
              "skipping", file=sys.stderr)
        return 1
    src = src.replace(sample_line, sample_line + "\n" + LOWER_LINE, 1)

    # 2) tolerate unknown repos with a loud error instead of bare KeyError.
    new_lookup = (
        '    log_parser = MAP_REPO_TO_PARSER.get(repo.lower())\n'
        '    if log_parser is None:\n'
        '        raise KeyError(\n'
        '            f"swegym MAP_REPO_TO_PARSER has no parser for repo={repo!r} '
        '(derived from {sample_id!r}); install/upgrade swe-gym harness or add the parser"\n'
        '        )\n'
    )
    src = src.replace(GRADING_MARKER + "\n", new_lookup, 1)

    path.write_text(src)
    print(f"[patch_swegym_grading] repo.lower() + .get: applied -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))