#!/usr/bin/env python3
"""Run the MassiPipe pipeline with event profiling enabled.

Every processing stage is switched on (and allowed to overwrite existing
products) so that a single profile run covers all product types, including
reflectance and glint correction. Stages that need extra configuration - glint
correction reference images, for instance - still run only when that
configuration is present in the config file.

Examples:
    python scripts/profile_pipeline.py --profile-output run.jsonl DATASET
    python scripts/profile_pipeline.py --profile-output second-run.jsonl DATASET -config config.yaml
    python scripts/visualise_benchmark.py run.jsonl second-run.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from massipipe.pipeline import run, setup
from massipipe.timing import profile_events, profiled
from massipipe.utils import parse_arguments


def _validate_profile_output(
    parser: argparse.ArgumentParser, profile_output: Path, cli_args: list[str]
) -> None:
    """Keep the event log separate from inputs and prevent accidental overwrites.

    MassiPipe's parser extracts the forwarded dataset, config, and calibration paths
    using the CLI's existing option spellings. Relative paths and symlinks are
    resolved before comparing them with the profile output. The output must be
    outside the dataset and calibration directories, must not replace the config
    file, and must not already exist (the later ``open('x')`` also enforces that).
    """
    cli_options = parse_arguments(cli_args)
    output_path = profile_output.resolve()

    # Directory inputs are protected as trees, not just as exact paths.
    protected_directories = [Path(cli_options.dataset_dir).resolve()]
    if cli_options.calibration_dir is not None:
        protected_directories.append(Path(cli_options.calibration_dir).resolve())
    if any(
        output_path == directory or directory in output_path.parents
        for directory in protected_directories
    ):
        parser.error("--profile-output must not be inside the dataset or calibration directory")

    # The config is a file input: forbid replacing that path, but not sibling outputs.
    if cli_options.config is not None and output_path == Path(cli_options.config).resolve():
        parser.error("--profile-output must not overwrite the config file")

    if output_path.exists():
        parser.error(f"profile output already exists: {profile_output}")


def parse_args(argv: list[str] | None = None) -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile-output",
        type=Path,
        required=True,
        help=(
            "new JSONL file for profile events; keep it outside the dataset/calibration "
            "directories and do not overwrite the config or an existing file"
        ),
    )
    profile_args, cli_args = parser.parse_known_args(argv)
    if not cli_args:
        parser.error("MassiPipe CLI arguments are required after --profile-output")
    _validate_profile_output(parser, profile_args.profile_output, cli_args)
    return profile_args.profile_output, cli_args


def enable_all_products(options: BaseModel, *, overwrite: bool = True) -> None:
    """Switch on every processing stage, optionally forcing products to be rebuilt.

    Walks the nested configuration models (for example ``mosaic.radiance_gc_rgb``),
    so a stage added to the configuration is enabled here without further changes.
    """

    for name in type(options).model_fields:
        value = getattr(options, name)
        if isinstance(value, BaseModel):
            enable_all_products(value, overwrite=overwrite)
        elif name == "create" or (name == "overwrite" and overwrite):
            setattr(options, name, True)


@profiled("pipeline")
def run_profiled_cli(cli_args: list[str]) -> int:
    parsed_args = parse_arguments(cli_args)
    setup_args: dict[str, Any] = {"dataset_dir": parsed_args.dataset_dir}
    if parsed_args.config:
        setup_args["config_file_path"] = parsed_args.config
    if parsed_args.calibration_dir:
        setup_args["calibration_dir"] = parsed_args.calibration_dir

    runtime_context = setup(**setup_args)
    enable_all_products(runtime_context.config.massipipe_options)
    run(runtime_context)
    return 0


def main(argv: list[str] | None = None) -> int:
    profile_output, cli_args = parse_args(argv)
    profile_output.parent.mkdir(parents=True, exist_ok=True)

    with profile_output.open("x", encoding="utf-8") as output:
        with profile_events(output):
            result = run_profiled_cli(cli_args)

    print(f"Wrote profile events to {profile_output}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
