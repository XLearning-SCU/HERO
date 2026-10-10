#!/usr/bin/env bash
# Reapply the slime_bridge adapter training-id migration for latest Slime.
# Idempotent: re-running is a no-op once applied. Run after every hero reinstall
# / re-clone (called from run_rl/_shared/init_env/install_env.sh and the training wrapper).
#
# Slime v0.3 used group_id for both DP-step scheduling and loss aggregation.
# Current Slime splits those roles into rollout_id and trajectory_id, so HERO
# siblings must receive the same value in both fields.
#
# NOTE: this only touches the Sample(...) constructor kwargs. The dict key
# "group_id" read from hero trajectory metadata in _scheduler_metadata is a
# different thing (hero's own metadata field name) and must stay.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"

ADAPTER_PY="${ADAPTER_PY:-${PROJECT_ROOT}/src/slime_bridge/adapter.py}"
[[ -f "${ADAPTER_PY}" ]] || { echo "ERROR: slime_bridge adapter.py not found: ${ADAPTER_PY}" >&2; exit 1; }

"${PYTHON_BIN}" - "${ADAPTER_PY}" <<'PY'
import sys, pathlib

adapter = pathlib.Path(sys.argv[1])
src = adapter.read_text()

group_line = "        group_id=index,\n"
rollout_line = "        rollout_id=index,\n"
trajectory_line = "        trajectory_id=index,\n"

# Normalize upstream v0.3, the earlier trajectory-only migration, and any
# partially migrated file to the same two-field block at both Sample sites.
lines = src.splitlines(keepends=True)
new_lines = []
changed = False
for i, line in enumerate(lines):
    previous = lines[i - 1] if i else None
    following = lines[i + 1] if i + 1 < len(lines) else None
    if line == group_line:
        new_lines.extend((rollout_line, trajectory_line))
        changed = True
    elif line == rollout_line:
        new_lines.append(line)
        if following != trajectory_line:
            new_lines.append(trajectory_line)
            changed = True
    elif line == trajectory_line:
        if previous != rollout_line:
            new_lines.append(rollout_line)
            changed = True
        new_lines.append(line)
    else:
        new_lines.append(line)

new_src = "".join(new_lines)
pair = rollout_line + trajectory_line
if new_src.count(pair) != 2 or group_line in new_src:
    print(
        "[patch_adapter] group_id -> rollout_id+trajectory_id: WARNING expected "
        "two Sample sites; adapter.py may have drifted",
        file=sys.stderr,
    )
    sys.exit(1)
if changed:
    adapter.write_text(new_src)
    print(f"[patch_adapter] training ids aligned at two Sample sites -> {adapter}")
else:
    print("[patch_adapter] rollout_id + trajectory_id: already aligned")
PY
