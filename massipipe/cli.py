import argparse
from typing import Optional

from massipipe.pipeline import run, setup


def parse_arguments(args: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MassiPipe library CLI tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to dataset directory",
    )

    parser.add_argument(
        "-config",
        type=str,
        default=None,
        help="Path to config file",
    )

    parser.add_argument(
        "-calibration-dir",
        type=str,
        default=None,
        help="Path to calibration directory",
    )

    return parser.parse_args(args)

def main(args=None):
    parsed_args = parse_arguments(args)

    print("here!")
    _pack_args = {}

    if parsed_args.dataset_dir:
        _pack_args["dataset_dir"] = parsed_args.dataset_dir

    if parsed_args.config:
        _pack_args["config"] = parsed_args.config

    if parsed_args.calibration_dir:
        _pack_args["calibration_dir"] = parsed_args.calibration_dir

    runtime_context = setup(**_pack_args)

    run(runtime_context)

    return 0