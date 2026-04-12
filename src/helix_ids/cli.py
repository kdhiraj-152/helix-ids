"""HELIX-IDS command-line entrypoint.

This CLI provides a stable package entrypoint and delegates execution to
existing project scripts.
"""

from __future__ import annotations

import argparse
import runpy
from collections.abc import Sequence


def _run_script_module(module_name: str) -> int:
    """Execute a script module through its __main__ entrypoint."""
    runpy.run_module(module_name, run_name="__main__")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="helix-ids",
        description="HELIX-IDS command-line interface",
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("train", help="Run V2 training (multidataset)")
    subparsers.add_parser("adversarial", help="Run adversarial training (V2)")
    subparsers.add_parser("holdout_eval", help="Run V2 holdout evaluation")
    subparsers.add_parser("benchmark", help="Run V2 benchmarking")
    subparsers.add_parser("deploy", help="Run deployment utility")
    subparsers.add_parser("download_data", help="Download datasets")
    subparsers.add_parser("train_edge", help="Train edge models")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        return _run_script_module("scripts.train_multidataset_v2_fixed")

    if args.command == "adversarial":
        return _run_script_module("scripts.adversarial_training_v2")

    if args.command == "holdout_eval":
        return _run_script_module("scripts.holdout_evaluation_v2")

    if args.command == "benchmark":
        return _run_script_module("scripts.benchmark_e2e_v2_fixed")

    if args.command == "deploy":
        return _run_script_module("scripts.deploy")

    if args.command == "download_data":
        return _run_script_module("scripts.download_datasets")

    if args.command == "train_edge":
        return _run_script_module("scripts.train_edge_models")

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
