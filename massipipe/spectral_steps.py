import logging
from pathlib import Path
from typing import Type

from pipefunc import Pipeline, pipefunc

from .config import RuntimeContext
from .hsi.irradiance import IrradianceConverter, WavelengthCalibrator
from .hsi.radiance import RadianceConverter, IRadianceConverter, Radiance
from .utils import time_scope

logger = logging.getLogger(__name__)

@pipefunc(output_name="raw_path")
def get_raw_paths(runtime_context: RuntimeContext) -> list[Path]:
    return runtime_context.product_paths.raw

@pipefunc(output_name="rad_path")
def get_rad_paths(runtime_context: RuntimeContext) -> list[Path]:
    return runtime_context.product_paths.radiance

@pipefunc(output_name="irrad_path")
def get_irrad_paths(runtime_context: RuntimeContext) -> list[Path]:
    return runtime_context.product_paths.irradiance

@pipefunc(output_name="raw_spec_path")
def get_raw_spec_paths(runtime_context: RuntimeContext) -> list[Path]:
    return runtime_context.product_paths.raw_spec

def create_rad_converter(runtime_context: RuntimeContext) -> IRadianceConverter:
    return create_custom_rad_converter(runtime_context, RadianceConverter)

def create_custom_rad_converter(runtime_context: RuntimeContext,
                                constructor: Type[RadianceConverter]) -> IRadianceConverter:
    return constructor(
        runtime_context.calibration_paths.radiance,
        set_saturated_pixels_to_zero=runtime_context.config.massipipe_options.radiance.set_saturated_pixels_to_zero,
    )

@pipefunc(output_name="radiance_converter")
def assemble_rad_converter(runtime_context: RuntimeContext) -> Radiance:
    assert runtime_context.radiance_converter is not None

    return Radiance(radiance_converter = runtime_context.radiance_converter)

@pipefunc(output_name="rad_dir_gate")
def ensure_rad_dir(runtime_context: RuntimeContext) -> None:
    runtime_context.base_product_paths.radiance.mkdir(exist_ok=True)

@pipefunc(output_name="radiance_gate", mapspec="raw_path[i], rad_path[i] -> radiance_gate[i]")
def raw_spectra_to_radiance(
        runtime_context: RuntimeContext,
        radiance_converter: Radiance,
        raw_path: Path,
        rad_path: Path,
        *,
        rad_dir_gate: None = None
):
    if rad_path.exists() and not runtime_context.config.massipipe_options.radiance.overwrite:
        logger.info(f"Image {rad_path.name} exists - skipping.")

    logger.info(f"Converting {raw_path.name} to radiance")

    try:
        radiance_converter.convert_raw_file(raw_path, rad_path)
    except Exception as _:
        logger.exception(f"Error converting {raw_path.name} to radiance.")
        raise

@pipefunc(output_name="irradiance_converter")
def assemble_irrad_converter(runtime_context: RuntimeContext) -> IrradianceConverter:
    return IrradianceConverter(runtime_context.calibration_paths.irradiance)

@pipefunc(output_name="irrad_dir_gate")
def ensure_irrad_dir(runtime_context: RuntimeContext) -> None:
    runtime_context.base_product_paths.radiance.mkdir(exist_ok=True)

@pipefunc(output_name="irradiance_gate", mapspec="raw_spec_path[i], irrad_path[i] -> irradiance_gate[i]")
def raw_spectra_to_irradiance(
    runtime_context: RuntimeContext,
    irradiance_converter: IrradianceConverter,
    raw_spec_path: Path,
    irrad_path: Path,
    *,
    radiance_gate: None = None,
    irrad_dir_gate: None = None
):
    if irrad_path.exists() and not runtime_context.config.massipipe_options.irradiance.overwrite:
        logger.info(f"Image {irrad_path.name} exists - skipping.")

    logger.info(f"Converting {raw_spec_path.name} to irradiance")

    try:
        irradiance_converter.convert_raw_file_to_irradiance(raw_spec_path, irrad_path)
    except Exception as _:
        logger.exception(f"Error converting {raw_spec_path.name} to irradiance.")
        raise

@pipefunc(output_name="wavelength_calibrator")
def assemble_wavelength_calibrator() -> WavelengthCalibrator:
    return WavelengthCalibrator()

@pipefunc(output_name="cal_irrad_gate")
def calibrate_irrad_wavelengths(
        runtime_context: RuntimeContext,
        wavelength_calibrator: WavelengthCalibrator,
        *,
        irradiance_gate: None = None
):
    logger.info("---- IRRADIANCE WAVELENGTH CALIBRATION ----")

    if not (runtime_context.base_product_paths.radiance.exists()):
        raise FileNotFoundError("Radiance folder with irradiance spectra does not exist")

    irradiance_spec_paths = sorted(runtime_context.base_product_paths.radiance.glob("*.spec.hdr"))

    if not irradiance_spec_paths:
        return

    wavelength_calibrator.fit_batch(irradiance_spec_paths)

@pipefunc(output_name="header_wl_gate", mapspec="irrad_path[i] -> header_wl_gate[i]")
def update_irrad_header_wavelengths(wavelength_calibrator: WavelengthCalibrator, irrad_path: Path):
    logger.info(f"Calibrating wavelengths for {irrad_path.name}")

    try:
        wavelength_calibrator.update_header_wavelengths(irrad_path)
    except Exception as e:
        logger.exception(f"Wavelength calibration failed for {irrad_path.name}")
        raise

@pipefunc(output_name="raw_processed_gate")
def raw_processing_complete(header_wl_gate: None) -> None:
    return

def build_raw_data_cube_pipeline(runtime_context: RuntimeContext) -> Pipeline:
    _config = runtime_context.config.massipipe_options

    fns = []

    includes_raw_paths = False

    if _config.radiance.create:
        fns.extend([
            get_raw_paths,
            get_rad_paths,
            assemble_rad_converter,
            ensure_rad_dir,
            raw_spectra_to_radiance
        ])

        includes_raw_paths = True

    if _config.irradiance.create:
        if includes_raw_paths is False:
            fns.extend([
                get_raw_paths,
            ])

        fns.extend([
            get_irrad_paths,
            assemble_irrad_converter,
            get_raw_spec_paths,
            ensure_irrad_dir,
            raw_spectra_to_irradiance,
            assemble_wavelength_calibrator,
            calibrate_irrad_wavelengths,
            update_irrad_header_wavelengths,
        ])

        includes_raw_paths = True

    pipeline = Pipeline(fns)

    return pipeline
