import sys
import time

from massipipe.pipeline import delete_existing_products, run, setup
from massipipe.radiance import (
    NumbaOptimisedParallelRadianceConverter,
    NumbaOptimisedRadianceConverter,
    NumpyOptimisedRadianceConverter,
    RadianceConverter,
)
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

    m_opt = runtime_context.config.massipipe_options

    m_opt.quicklook.create = False
    m_opt.irradiance.create = False
    m_opt.radiance_rgb.create = False
    m_opt.radiance_gc.create = False
    m_opt.radiance_gc_rgb.create = False
    m_opt.reflectance_gc.create = False
    m_opt.reflectance.create = False
    m_opt.reflectance_gc_rgb.create = False

    sweep = (
        NumbaOptimisedRadianceConverter,
        NumbaOptimisedParallelRadianceConverter,
        RadianceConverter,
        NumpyOptimisedRadianceConverter
    )
    
    for rad_converter in sweep:
        runtime_context.radiance_converter = rad_converter

        run(runtime_context)

        delete_existing_products(runtime_context, delete_radiance=True)

        time.sleep(10)


    return 0


if __name__ == "__main__":
    sys.exit(main())