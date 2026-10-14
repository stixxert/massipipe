import logging

from massipipe.config import RuntimeContext
from massipipe.georeferencing import ImuDataParser
from massipipe.irradiance import IrradianceConverter, WavelengthCalibrator
from massipipe.radiance import RadianceConverter

logger = logging.getLogger(__name__)

def parse_and_save_imu_data(runtime_context: RuntimeContext):
    """Parse *.lcf and *.times files with IMU data and save as JSON"""

    logger.info("---- IMU DATA PROCESSING ----")

    runtime_context.base_product_paths.imu_data.mkdir(exist_ok=True)

    imu_data_parser = ImuDataParser()

    lcf_paths, times_paths, imu_data_paths = (
        runtime_context.product_paths.lcf,
        runtime_context.product_paths.times,
        runtime_context.product_paths.imu_data
    )

    for lcf_path, times_path, imu_data_path in zip(lcf_paths, times_paths, imu_data_paths):
        if (imu_data_path.exists() and
                not runtime_context.config.massipipe_options.imu_data.overwrite):
            logger.info(f"Image {imu_data_path.name} exists - skipping.")
            continue

        logger.info(f"Processing IMU data from {lcf_path.name}")
        try:
            imu_data_parser.read_and_save_imu_data(lcf_path, times_path, imu_data_path)
        except Exception as e:
            logger.exception(f"Error processing IMU data from {lcf_path.name}")
            continue


def convert_raw_images_to_radiance(runtime_context: RuntimeContext) -> None:
    """Convert raw hyperspectral images (DN) to radiance (microflicks)"""

    logger.info("---- RADIANCE CONVERSION ----")

    runtime_context.base_product_paths.radiance.mkdir(exist_ok=True)

    radiance_converter = RadianceConverter(
        runtime_context.calibration_paths.radiance,
        set_saturated_pixels_to_zero=runtime_context.config.massipipe_options.radiance.set_saturated_pixels_to_zero,
    )

    raw_paths, rad_paths = runtime_context.product_paths.raw, runtime_context.product_paths.radiance

    for raw_path, rad_path in zip(raw_paths, rad_paths):
        if rad_path.exists() and not runtime_context.config.massipipe_options.radiance.overwrite:
            logger.info(f"Image {rad_path.name} exists - skipping.")
            continue

        logger.info(f"Converting {raw_path.name} to radiance")
        try:
            radiance_converter.convert_raw_file_to_radiance(raw_path, rad_path)
        except Exception as e:
            logger.exception(f"Error converting {raw_path.name} to radiance.")
            continue


def convert_raw_spectra_to_irradiance(runtime_context: RuntimeContext) -> None:
    """Convert raw spectra (DN) to irradiance (W/(m2*nm))"""

    logger.info("---- IRRADIANCE CONVERSION ----")

    runtime_context.base_product_paths.radiance.mkdir(exist_ok=True)

    irradiance_converter = IrradianceConverter(runtime_context.calibration_paths.irradiance)

    for raw_spec_path, irrad_spec_path in zip(
            runtime_context.product_paths.raw_spec, runtime_context.product_paths.irradiance):

        if (irrad_spec_path.exists() and
                not runtime_context.config.massipipe_options.irradiance.overwrite):
            logger.info(f"Image {irrad_spec_path.name} exists - skipping.")
            continue

        if raw_spec_path is not None and raw_spec_path.exists():
            logger.info(f"Converting {raw_spec_path.name} to downwelling irradiance")
            try:
                irradiance_converter.convert_raw_file_to_irradiance(
                    raw_spec_path, irrad_spec_path
                )
            except Exception as e:
                logger.exception(f"Irradiance conversion failed for {raw_spec_path.name}")
                continue


def calibrate_irradiance_wavelengths(runtime_context: RuntimeContext) -> None:
    """Calibrate irradiance wavelengths using Fraunhofer absorption lines"""

    logger.info("---- IRRADIANCE WAVELENGTH CALIBRATION ----")

    if not (runtime_context.base_product_paths.radiance.exists()):
        raise FileNotFoundError("Radiance folder with irradiance spectra does not exist")

    wavelength_calibrator = WavelengthCalibrator()

    irradiance_spec_paths = sorted(runtime_context.base_product_paths.radiance.glob("*.spec.hdr"))

    if irradiance_spec_paths:
        wavelength_calibrator.fit_batch(irradiance_spec_paths)

        for irradiance_spec_path in irradiance_spec_paths:
            logger.info(f"Calibrating wavelengths for {irradiance_spec_path.name}")

            try:
                wavelength_calibrator.update_header_wavelengths(irradiance_spec_path)
            except Exception as e:
                logger.exception(
                    f"Wavelength calibration failed for {irradiance_spec_path.name}"
                )
                continue
