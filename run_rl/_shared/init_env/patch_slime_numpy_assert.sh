#!/usr/bin/env bash
# Reapply Slime compatibility patches so actor init survives the cu130 venv.
# Idempotent: re-running is a no-op once each fix is in place. Run after every
# Slime reinstall/re-clone (called from run_rl/_shared/init_env/install_env.sh and the
# training wrapper) so the editable checkout keeps these fixes.
#
# Fixes (all stem from Slime referencing megatron-internal args that the cu130
# stack neither registers nor satisfies):
#   1. numpy 1.x hard assert → warning. The cu130 stack pins numpy 2.x (SGLang +
#      SciPy 1.18 depend on it; 1.x breaks SGLang via scipy np.long). numpy 2.x
#      runs Megatron fine (official ProRL uv.lock pins numpy 2.4.4).
#   2. enable_gloo_process_groups default True. Slime references it but never
#      registers it (was injected by the old external mbridge). CP rollout-log
#      gathering requires the CP+DP Gloo group after every training step.
#   3. async_strategy default "mcore". Megatron defaults to "nvrx" which needs
#      nvidia-resiliency-ext (not installed); "mcore" uses megatron.core only.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/framework/slime}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"

INIT_PY="${SLIME_DIR}/slime/backends/megatron_utils/initialize.py"
ARGS_PY="${SLIME_DIR}/slime/backends/megatron_utils/arguments.py"
[[ -f "${INIT_PY}" ]] || { echo "ERROR: slime initialize.py not found: ${INIT_PY}" >&2; exit 1; }
[[ -f "${ARGS_PY}" ]] || { echo "ERROR: slime arguments.py not found: ${ARGS_PY}" >&2; exit 1; }

"${PYTHON_BIN}" - "${INIT_PY}" "${ARGS_PY}" <<'PY'
import sys, pathlib

init_py, args_py = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])

# --- Fix 1: numpy assert → warning (initialize.py) ---
init_src = init_py.read_text()
numpy_marker = 'if not np.__version__.startswith("1."):'
numpy_old = (
    '    # https://github.com/NVIDIA/Megatron-LM/issues/1563\n'
    '    assert np.__version__.startswith("1."), "Megatron does not support numpy 2.x"\n'
)
numpy_new = (
    '    # https://github.com/NVIDIA/Megatron-LM/issues/1563\n'
    '    # Originally a hard assert requiring numpy 1.x. The ProRL/SGLang cu130 stack\n'
    '    # pins numpy 2.x (SGLang/SciPy 1.18 depend on it; downgrading to 1.x breaks\n'
    '    # SGLang import via scipy\'s np.long). numpy 2.x runs Megatron fine in\n'
    '    # practice (the official ProRL uv.lock itself pins numpy 2.4.4), so demote\n'
    '    # to a warning instead of aborting actor init.\n'
    '    if not np.__version__.startswith("1."):\n'
    '        logger.warning(\n'
    '            "Running Megatron with numpy %s (2.x). Megatron issue #1563 "\n'
    '            "historically warned about numpy 2.x, but the ProRL cu130 stack "\n'
    '            "requires it; proceeding.", np.__version__\n'
    '        )\n'
)
if numpy_marker in init_src:
    print("[patch_slime] fix1 numpy-assert: already applied")
elif numpy_old in init_src:
    init_py.write_text(init_src.replace(numpy_old, numpy_new, 1))
    print("[patch_slime] fix1 numpy-assert: applied ->", init_py)
else:
    print("[patch_slime] fix1 numpy-assert: ERROR original not found, skipping", file=sys.stderr)

# --- Fix 2 & 3: arg defaults in validate_args (arguments.py) ---
args_src = args_py.read_text()
anchor = '    _megatron_validate_args(args)\n'
if anchor not in args_src:
    print("[patch_slime] fix2/3 arg-defaults: ERROR anchor not found", file=sys.stderr)
else:
    # Slime has historically injected this block itself, and an earlier local
    # patch rewrote it separately.  Remove every existing variant first, then
    # install exactly one canonical True block.  This makes repeated launches
    # converge rather than alternating False/True duplicate blocks.
    import re
    gloo_pattern = re.compile(
        r'(?:    #[^\n]*\n)*'
        r'    if not hasattr\(args, "enable_gloo_process_groups"\):\n'
        r'        args\.enable_gloo_process_groups = (?:False|True)\n'
    )
    args_src, removed_gloo_blocks = gloo_pattern.subn('', args_src)
    gloo_block = (
        '\n    # ProRL local-Docker CP training collects rollout logs through the\n'
        '    # CP+DP Gloo group.  Slime does not register this CLI argument, so\n'
        '    # provide one deterministic backfill for CP>1.\n'
        '    if not hasattr(args, "enable_gloo_process_groups"):\n'
        '        args.enable_gloo_process_groups = True\n'
    )
    needs_async = 'args.async_strategy = "mcore"' not in args_src
    if removed_gloo_blocks == 1 and not needs_async:
        print("[patch_slime] fix2/3 arg-defaults: already applied")
    else:
        block = gloo_block
        if needs_async:
            block += (
                '\n    # async_strategy defaults to "nvrx" in megatron\'s get_async_strategy, which\n'
                '    # requires nvidia-resiliency-ext (not installed in the cu130 venv). Megatron\'s\n'
                '    # "mcore" async strategy uses only megatron.core modules, so when nvrx is\n'
                '    # selected but the package is missing, fall back to mcore to avoid\n'
                '    # ModuleNotFoundError during load_checkpoint / async save.\n'
                '    if getattr(args, "async_strategy", "nvrx") == "nvrx":\n'
                '        try:\n'
                '            import nvidia_resiliency_ext  # noqa: F401\n'
                '        except ImportError:\n'
                '            args.async_strategy = "mcore"\n'
            )
        args_py.write_text(args_src.replace(anchor, anchor + block, 1))
        print("[patch_slime] fix2/3 arg-defaults: applied ->", args_py)
PY
