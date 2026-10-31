from massipipe.pipeline import run, setup
from massipipe.utils import parse_arguments


def main(args=None):
    parsed_args = parse_arguments(args)

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