"""Compatibility helpers scoped to the Qwen3.5 MoE entrypoints."""

from __future__ import annotations


def normalize_megatron_args(args):
    """Bridge a Megatron argument rename without changing dense entrypoints."""

    if not hasattr(args, "enable_gloo_process_groups"):
        args.enable_gloo_process_groups = getattr(args, "use_gloo_process_groups", True)
    return args


def patch_converter_defaults() -> None:
    """Apply the argument bridge before the upstream converter imports it."""

    from slime.backends.megatron_utils import arguments as megatron_arguments

    original = megatron_arguments.set_default_megatron_args

    def set_default_megatron_args(args):
        return normalize_megatron_args(original(args))

    megatron_arguments.set_default_megatron_args = set_default_megatron_args


def main():
    """Run Slime training, or its converter with --convert, using the same bridge."""
    import os
    import runpy
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    slime_dir = project_root / "framework" / "slime"
    if len(sys.argv) > 1 and sys.argv[1] == "--convert":
        sys.argv.pop(1)
        upstream = Path(
            os.environ.get(
                "SLIME_UPSTREAM_CONVERTER", slime_dir / "tools/convert_hf_to_torch_dist.py"
            )
        )
        if not upstream.is_file():
            raise SystemExit(f"Missing upstream Slime converter: {upstream}")
        patch_converter_defaults()
        runpy.run_path(str(upstream), run_name="__main__")
    else:
        sys.path.insert(0, str(slime_dir))
        from slime.utils.arguments import parse_args
        from train import train

        train(normalize_megatron_args(parse_args()))


if __name__ == "__main__":
    main()
