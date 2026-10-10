"""CLI argument registration for `hero dashboard`."""

from __future__ import annotations

import argparse

from hero.platform.server import serve


def add_subcommand(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "dashboard",
        help="Start the HERO observability dashboard.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="topology.yaml",
        help="Path to topology.yaml.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host. Defaults to 127.0.0.1 (local-only).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8090,
        help="Bind port. Defaults to 8090.",
    )
    parser.add_argument(
        "--rollout-url",
        default=None,
        help="Override the rollout server URL discovered from topology.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Override the rollout results directory (default: topology rollout.save_dir).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    return parser


def handle(args: argparse.Namespace) -> int:
    serve(
        args.config,
        host=args.host,
        port=args.port,
        rollout_url=args.rollout_url,
        save_dir=args.save_dir,
        log_level=args.log_level,
    )
    return 0


__all__ = ["add_subcommand", "handle"]
