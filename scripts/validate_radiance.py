import shutil
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from massipipe.pipeline import convert_raw_images_to_radiance, setup
from massipipe.radiance import (
    NumbaOptimisedParallelRadianceConverter,
    NumbaOptimisedRadianceConverter,
    NumpyOptimisedRadianceConverter,
    RadianceConverter,
)
from massipipe.utils import parse_arguments, read_envi


def main(args=None):
    parsed_args = parse_arguments(args)

    setup_args = {"dataset_dir": parsed_args.dataset_dir}
    if parsed_args.config:
        setup_args["config_file_path"] = parsed_args.config
    if parsed_args.calibration_dir:
        setup_args["calibration_dir"] = parsed_args.calibration_dir

    runtime_context = setup(**setup_args)
    if runtime_context.data_starting_point != "raw":
        raise ValueError("Radiance validation requires a dataset with raw images")

    runtime_context.config.massipipe_options.radiance.overwrite = True
    converters = (
        RadianceConverter,
        NumpyOptimisedRadianceConverter,
        NumbaOptimisedRadianceConverter,
        NumbaOptimisedParallelRadianceConverter,
    )

    # Keep benchmark outputs isolated so the source dataset's products are untouched.
    with TemporaryDirectory(prefix="massipipe-radiance-validation-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        reference_dir = None

        for index, converter in enumerate(converters):
            output_dir = temporary_path / converter.__name__
            output_dir.mkdir()
            runtime_context.base_product_paths.radiance = output_dir
            runtime_context.product_paths.radiance = [
                output_dir / path.name for path in runtime_context.product_paths.radiance
            ]
            runtime_context.radiance_converter = converter
            start = time.perf_counter()
            convert_raw_images_to_radiance(runtime_context)
            elapsed = time.perf_counter() - start

            outputs = runtime_context.product_paths.radiance
            missing = [path for path in outputs if not path.exists()]
            if missing:
                raise RuntimeError(
                    f"{converter.__name__} failed to create radiance outputs: {missing}"
                )

            if index == 0:
                reference_dir = temporary_path / "reference"
                shutil.copytree(output_dir, reference_dir)
            else:
                for output in outputs:
                    expected, _, _ = read_envi(reference_dir / output.name)
                    actual, _, _ = read_envi(output)
                    if not np.array_equal(actual, expected):
                        raise AssertionError(
                            f"{converter.__name__} differs from RadianceConverter "
                            f"for {output.name}"
                        )

            print(f"{converter.__name__}: {elapsed:.2f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
