"""Massipipe pipeline module.

This module provides the Pipeline class for processing hyperspectral datasets.
It offers configuration loading, image conversion, georeferencing, glint correction,
mosaicing and export functionality for raw and radiance data.
"""

# Imports
import logging
import os
import pprint
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence, Type, Union

import yaml
from pydantic import ValidationError

from massipipe.config import (
    BaseProductPaths,
    BootstrapOptions,
    CalibrationPaths,
    Config,
    DataStartingPoint,
    ProductPaths,
    RuntimeContext,
    export_template_yaml,
)
from massipipe.export import (
    copy_visualization_mosaic,
    export_dataset_zip,
)
from massipipe.georeferencing import (
    ImuDataParser,
    ImuGeoTransformer,
    SimpleGeoreferencer,
    georeferenced_hyspec_to_rgb_geotiff,
)
from massipipe.glint import FlatSpecGlintCorrector, HedleyGlintCorrector
from massipipe.irradiance import IrradianceConverter, WavelengthCalibrator
from massipipe.mosaic import add_geotiff_overviews, convert_geotiff_to_8bit, mosaic_geotiffs
from massipipe.quicklook import QuickLookProcessor
from massipipe.radiance import IRadianceConverter, Radiance, RadianceConverter
from massipipe.reflectance import ReflectanceConverter
from massipipe.timing import profiled
from massipipe.utils import add_header_mapinfo, header_contains_mapinfo

# Get logger
logger = logging.getLogger(__name__)

def setup_bootstrap_options(
        dataset_dir: str | os.PathLike[str],
        config_file_path: str | os.PathLike[str] | None = None,
        calibration_dir: str | os.PathLike[str] | None = None
) -> BootstrapOptions:
    _dataset_dir = Path(dataset_dir)

    return BootstrapOptions(
        config_file_path=Path(config_file_path) if config_file_path
        else _dataset_dir / "config.seabee.yaml",

        calibration_dir=Path(calibration_dir) if calibration_dir
        else _dataset_dir / "calibration",

        dataset_dir=_dataset_dir,
        dataset_name=_dataset_dir.name,
        logs=_dataset_dir / "logs"
    )

def setup_config(bootstrap_options: BootstrapOptions):
    if not bootstrap_options.config_file_path.exists():
        logger.info(f"No config file found - exporting template file "
                    f"{bootstrap_options.config_file_path}")
        export_template_yaml(bootstrap_options.config_file_path)

    return load_config_from_file(bootstrap_options)

def setup_base_product_paths(bootstrap_options: BootstrapOptions) -> BaseProductPaths:
    dataset_dir = bootstrap_options.dataset_dir

    radiance_dir = dataset_dir / "1a_radiance"
    radiance_gc_dir = dataset_dir / "1b_radiance_gc"
    reflectance_gc_dir = dataset_dir / "2b_reflectance_gc"

    return BaseProductPaths(
        raw = dataset_dir / "0_raw",
        quicklook = dataset_dir / "quicklook",
        radiance = radiance_dir,
        radiance_rgb = radiance_dir / "rgb",
        radiance_gc = radiance_gc_dir,
        radiance_gc_rgb = radiance_gc_dir / "rgb",
        reflectance = dataset_dir / "2a_reflectance",
        reflectance_gc  = reflectance_gc_dir,
        reflectance_gc_rgb  = reflectance_gc_dir / "rgb",
        imu_data = dataset_dir / "imudata",
        geo_transform = dataset_dir / "geotransform",
        mosaic = dataset_dir / "mosaics",
        mosaic_orthophoto = dataset_dir / "orthophoto",
    )

def load_config_from_file(boostrap_options: BootstrapOptions) -> Config | None:
    """Load or reload configuration from a YAML file.

    Reads the configuration from `self.config_file_path`, validates it using Pydantic,
    and assigns the loaded options to `self.config`. Logs warnings on validation errors.
    """
    try:
        with open(boostrap_options.config_file_path, "r") as stream:
            config = yaml.safe_load(stream)
    except IOError:
        logger.exception(f"Error parsing config file {boostrap_options.config_file_path}")
        raise

    try:
        full_config = Config(**config)
    except ValidationError as e:
        logger.warning(f"Validation error while processing {boostrap_options.config_file_path}")
        logger.warning(str(e))
        logger.warning(f"No configuration loaded for {boostrap_options.dataset_dir}.")
        return None

    return full_config

def setup_logging(
        bootstrap_options: BootstrapOptions,
        *,
        log_level: int = logging.INFO):
    file_handler = None

    if bootstrap_options.logs is not None:
        bootstrap_options.logs.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        log_filename = f"{timestamp}_{bootstrap_options.dataset_name}.log"
        log_filepath = bootstrap_options.logs / log_filename

        file_handler = logging.FileHandler(log_filepath, encoding="utf-8")

    log_format = "[%(asctime)s] %(levelname)s: [%(name)s:%(lineno)d] %(message)s"

    stdout_handler = logging.StreamHandler(sys.stdout)

    log_handlers = [stdout_handler]

    if file_handler is not None:
        log_handlers.append(file_handler)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=log_handlers,
        force=True # force makes this function reentrant - dropping previous log handlers (stdout, file)
    )

    logger.info("-" * 64)
    logger.info(f"File logging for {bootstrap_options.dataset_name} initialized.")

def get_data_starting_point(base_product_paths: BaseProductPaths) -> DataStartingPoint:
    """Determine if processing should start from raw or radiance data.

    Returns
    -------
    str
        "raw" if raw data exists and "radiance" if radiance data exists.

    Raises
    ------
    FileNotFoundError
        If neither raw nor radiance data is found in the dataset directory.
    """
    if base_product_paths.raw.exists():
        logger.info(f"Found directory {base_product_paths.raw} - processing from raw files.")
        data_starting_point = "raw"
    else:
        if base_product_paths.radiance.exists():
            logger.info(
                f"No raw files found, but radiance directory {base_product_paths.radiance.name} found"
                + " - processing based on radiance data."
            )
            data_starting_point = "radiance"
        else:
            raise FileNotFoundError(f"Found neither raw or radiance files in dataset dir.")
    return data_starting_point

def validate_raw_files(raw_paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """Check that all expected raw files exist

    Returns:
    --------
    times_paths, lcf_paths: list[Path]
        Lists of paths to *.times and *.lcf files for every valid raw file
    """
    times_paths = []
    lcf_paths = []
    for raw_image_path in raw_paths:  # Use list() to copy
        file_base_name = raw_image_path.name.split(".")[0]
        binary_im_path = raw_image_path.parent / (file_base_name + ".bil")
        times_path = raw_image_path.parent / (file_base_name + ".bil.times")
        lcf_path = raw_image_path.parent / (file_base_name + ".lcf")
        if (
                not (binary_im_path.exists())
                or not (times_path.exists())
                or not (lcf_path.exists())
        ):
            logger.warning(f"Set of raw files for image {raw_image_path} is incomplete.")
            raw_paths.remove(raw_image_path)
        else:
            times_paths.append(times_path)
            lcf_paths.append(lcf_path)

    return times_paths, lcf_paths

def get_image_number(path: str | os.PathLike[str]) -> int:
    """Get image number from raw image

    Parameters
    ----------
    path : Union[Path, str]
        Path to raw image (ENVI header file)

    Returns
    -------
    int
        Image number

    Examples
    --------
    >>> get_image_number("ExampleLocation_Pika_L_5.bil.hdr")
    5

    Notes
    -----
    Raw files are numbered sequentially, but the numbers are not
    zero-padded. This can lead to incorrect sorting of the images, e.g.
    ['im_1','im_2','im_10'] (simplified names for example) are sorted
    ['im_1','im_10','im_2']. By extracting the numbers from filenames
    of raw files and sorting explicitly on these (as integers),
    correct ordering can be achieved.
    """

    return int(
        Path(path)
        .name           # filename_type.ext
        .split(".")[0]  # filename_type
        .split("_")[-1] # filename
    )

def create_base_file_names(runtime_context: RuntimeContext) -> list[str]:
    """Create numbered base names for processed files"""

    if runtime_context.data_starting_point == "raw":
        return [
            f"{runtime_context.bootstrap_options.dataset_name}_{i:03d}"
            for i in range(len(runtime_context.product_paths.raw))
        ]
    else:
        return sorted([
            file.name.split("_radiance")[0]
            for file in runtime_context.base_product_paths.radiance.glob("*_radiance*.hdr")]
        )

def setup_processed_product_paths(runtime_context: RuntimeContext):
    product_paths = runtime_context.product_paths
    base_paths = runtime_context.base_product_paths

    dataset_name = runtime_context.bootstrap_options.dataset_name

    product_paths.mosaic_rad = base_paths.mosaic / f"{dataset_name}_rad_rgb.tiff"
    product_paths.mosaic_rad_gc = base_paths.mosaic / f"{dataset_name}_rad_gc_rgb.tiff"
    product_paths.mosaic_refl_gc = base_paths.mosaic / f"{dataset_name}_refl_gc_rgb.tiff"

    for base_filename in runtime_context.base_filenames:
        # Quicklook
        product_paths.quicklook.append(
            base_paths.quicklook / f"{base_filename}_quicklook.png"
        )

        # Radiance
        product_paths.radiance.append(
            base_paths.radiance / f"{base_filename}_radiance.bip.hdr"
        )

        # Radiance, RGB version
        product_paths.radiance_rgb.append(
            base_paths.radiance_rgb / f"{base_filename}_radiance_rgb.tiff"
        )

        # Radiance, glint corrected
        product_paths.radiance_gc.append(
            base_paths.radiance_gc / f"{base_filename}_radiance_gc.bip.hdr"
        )

        # Radiance, glint corrected, RGB version
        product_paths.radiance_gc_rgb.append(
            base_paths.radiance_gc_rgb / f"{base_filename}_radiance_gc_rgb.tiff"
        )

        # Irradiance
        product_paths.irradiance.append(
            base_paths.radiance / f"{base_filename}_irradiance.spec.hdr"
        )

        # IMU data
        product_paths.imu_data.append(
            base_paths.imu_data / f"{base_filename}_imudata.json"
        )

        # Geotransform
        product_paths.geo_transform.append(
            base_paths.geo_transform / f"{base_filename}_geotransform.json"
        )

        # Reflectance
        product_paths.reflectance.append(
            base_paths.reflectance / f"{base_filename}_reflectance.bip.hdr"
        )

        # Reflectance, glint corrected
        product_paths.reflectance_gc.append(
            base_paths.reflectance_gc / f"{base_filename}_reflectance_gc.bip.hdr"
        )

        # Reflectance, glint corrected, RGB version
        product_paths.reflectance_gc_rgb.append(
            base_paths.reflectance_gc_rgb / f"{base_filename}_reflectance_gc_rgb.tiff"
        )

def get_raw_spectrum_paths(raw_paths: list[Path]) -> list[Path]:
    """Search for raw files matching Resonon default naming"""
    spec_paths = []
    for raw_image_path in raw_paths:
        spec_base_name = raw_image_path.name.split("_Pika_L")[0]
        image_number = get_image_number(raw_image_path)
        spec_binary = (
                raw_image_path.parent / f"{spec_base_name}_downwelling_{image_number}_pre.spec"
        )
        spec_header = raw_image_path.parent / (spec_binary.name + ".hdr")
        if spec_binary.exists() and spec_header.exists():
            spec_paths.append(spec_header)
        else:
            spec_paths.append(None)
    return spec_paths

def get_radiance_calibration_path(calibration_dir: Path) -> Path:
    """Search for radiance calibration file (*.icp)"""
    icp_files = list(calibration_dir.glob("*.icp"))
    if len(icp_files) == 1:
        return icp_files[0]
    elif len(icp_files) == 0:
        raise FileNotFoundError(
            f"No radiance calibration file (*.icp) found in {calibration_dir}"
        )
    else:
        raise ValueError(
            "More than one radiance calibration file (*.icp) found in "
            + f"{calibration_dir}"
        )

def get_irradiance_calibration_path(calibration_dir: Path) -> Path:
    """Search for irradiance calibration file (*.dcp)"""
    dcp_files = list(calibration_dir.glob("*.dcp"))
    if len(dcp_files) == 1:
        return dcp_files[0]
    elif len(dcp_files) == 0:
        raise FileNotFoundError(
            "No irradiance calibration file (*.dcp) found in " + f"{calibration_dir}"
        )
    else:
        raise ValueError(
            "More than one irradiance calibration file (*.dcp) found in "
            + f"{calibration_dir}"
        )

def create_quicklook_images(runtime_context: RuntimeContext) -> None:
    """Create quicklook versions of hyperspectral images"""

    logger.info("---- QUICKLOOK IMAGE GENERATION ----")

    runtime_context.base_product_paths.quicklook.mkdir(exist_ok=True)

    _config = runtime_context.config.massipipe_options

    quicklook_processor = QuickLookProcessor(
        rgb_wl=_config.general.rgb_wl,
        percentiles=_config.quicklook.percentiles
    )

    # Determine whether raw or radiance images are used for quicklook
    if runtime_context.base_product_paths.raw.exists():
        hyspec_image_paths = runtime_context.product_paths.raw
        logger.info(f"Creating quicklook images from raw images")
    else:
        hyspec_image_paths = runtime_context.product_paths.radiance
        logger.info(f"Creating quicklook images from radiance images")

    for hyspec_image_path, quicklook_image_path in (
            zip(hyspec_image_paths, runtime_context.product_paths.quicklook)):
        if quicklook_image_path.exists() and not _config.quicklook.overwrite:
            logger.info(f"Image {quicklook_image_path.name} exists - skipping.")
            continue

        logger.info(f"Creating quicklook version of {hyspec_image_path.name}")
        try:
            quicklook_processor.create_quicklook_image_file(
                hyspec_image_path, quicklook_image_path
            )
        except Exception as e:
            logger.exception(f"Error creating quicklook for {hyspec_image_path.name}")
            continue

def parse_and_save_imu_data(runtime_context: RuntimeContext) -> None:
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

def create_and_save_geotransform(runtime_context: RuntimeContext) -> None:
    logger.info("---- GEOTRANSFORM CALCULATION ----")

    runtime_context.base_product_paths.geo_transform.mkdir(exist_ok=True)

    _config = runtime_context.config.massipipe_options

    if not any([imu_file.exists() for imu_file in runtime_context.product_paths.imu_data]):
        raise FileNotFoundError("No IMU data files found.")

    # Determine whether raw or radiance images are used
    if runtime_context.base_product_paths.raw.exists():
        hyspec_image_paths = runtime_context.product_paths.raw
    else:
        hyspec_image_paths = runtime_context.product_paths.radiance

    imu_data_paths, geo_transform_paths = (
        runtime_context.product_paths.imu_data,
        runtime_context.product_paths.geo_transform,
    )

    # Create geotransform for each image
    for hyspec_im_path, imu_data_path, geotrans_path in zip(
            hyspec_image_paths, imu_data_paths, geo_transform_paths):

        if geotrans_path.exists() and not _config.geotransform.overwrite:
            logger.info(f"Image {geotrans_path.name} exists - skipping.")
            continue

        logger.info(f"Creating and saving geotransform based on {imu_data_path.name}")
        try:
            _geo_config = _config.geotransform
            if imu_data_path.exists() and hyspec_im_path.exists():
                geotransformer = ImuGeoTransformer(
                    imu_data_path,
                    hyspec_im_path,
                    camera_opening_angle=_geo_config.camera_opening_angle_deg,
                    pitch_offset=_geo_config.pitch_offset_deg,
                    roll_offset=_geo_config.roll_offset_deg,
                    altitude_offset=_geo_config.altitude_offset_m,
                    utm_x_offset=_geo_config.utm_x_offset_m,
                    utm_y_offset=_geo_config.utm_y_offset_m,
                    assume_square_pixels=_geo_config.assume_square_pixels,
                )
                geotransformer.save_image_geotransform(geotrans_path)
        except Exception as e:
            logger.exception(f"Error creating geotransform based on {imu_data_path.name}")
            continue

def create_rad_converter(runtime_context: RuntimeContext) -> IRadianceConverter:
    return create_custom_rad_converter(runtime_context, RadianceConverter)


def create_custom_rad_converter(
    runtime_context: RuntimeContext,
    constructor: Type[RadianceConverter],
) -> IRadianceConverter:
    return constructor(
        runtime_context.calibration_paths.radiance,
        set_saturated_pixels_to_zero=(
            runtime_context.config.massipipe_options.radiance.set_saturated_pixels_to_zero
        ),
    )


def convert_raw_images_to_radiance(runtime_context: RuntimeContext) -> None:
    """Convert raw hyperspectral images (DN) to radiance (microflicks)"""

    logger.info("---- RADIANCE CONVERSION ----")

    runtime_context.base_product_paths.radiance.mkdir(exist_ok=True)

    radiance_converter = Radiance(
        radiance_converter=create_custom_rad_converter(
            runtime_context,
            runtime_context.radiance_converter,
        ),
    )

    raw_paths, rad_paths = runtime_context.product_paths.raw, runtime_context.product_paths.radiance

    for raw_path, rad_path in zip(raw_paths, rad_paths):
        if rad_path.exists() and not runtime_context.config.massipipe_options.radiance.overwrite:
            logger.info(f"Image {rad_path.name} exists - skipping.")
            continue

        logger.info(f"Converting {raw_path.name} to radiance")
        try:
            radiance_converter.convert_raw_file(raw_path, rad_path)
        except Exception as e:
            logger.exception(f"Error converting {raw_path.name} to radiance.")
            continue

def create_rgb_geotiff(
        hyspec_paths: list[Path],
        geotransform_paths: list[Path],
        geotiff_paths: list[Path],
        geotiff_overwrite: bool,
        rgb_wl: tuple[float, float, float] | None,
) -> None:
    """Create georeferenced RGB GeoTIFF versions of hyperspectral image

    Parameters
    ----------
    hyspec_paths : list[Path]
        Paths to hyperspectral files from which to create RGB images.
        If the file does not exist, the corresponding image is skipped.
    geotransform_paths : list[Path]
        Paths to geotransform files with affine transform for images.
        If the file does not exist, the corresponding image is skipped.
    geotiff_paths : list[Path]
        Paths for output ((GeoTIFF) files
    geotiff_overwrite : bool
        Boolean indicating if existing GeoTIFF files should be overwritten.
    rgb_wl : tuple[float, float, float]
        Wavelengths (in nm) to use for red, green and blue.
    """
    # TODO: Clean up the two possible ways to generate geotiffs
    # (via ENVI header map info or via geotransform file)

    georeferencer = SimpleGeoreferencer(rgb_only=True, rgb_wl=rgb_wl)

    if not any([image_path.exists() for image_path in hyspec_paths]):
        logger.warning(f"None of the listed hyperspectral files exist")
        return
    if not any([geotrans_path.exists() for geotrans_path in geotransform_paths]):
        logger.info(f"No geotransform files exist for the hyperspectral images.")

    for hyspec_path, geotrans_path, geotiff_path in zip(
            hyspec_paths, geotransform_paths, geotiff_paths
    ):
        if geotiff_path.exists() and not geotiff_overwrite:
            logger.info(f"Image {geotiff_path.name} exists - skipping.")
            continue

        if hyspec_path.exists():
            logger.info(f"Exporting RGB GeoTIFF from {hyspec_path.name}.")

            if geotrans_path.exists():
                logger.info(f"Using geotransform in file {geotrans_path.name}")
                try:
                    georeferencer.georeference_hyspec_save_geotiff(
                        hyspec_path,
                        geotrans_path,
                        geotiff_path,
                    )
                except Exception:
                    logger.exception(
                        f"Error occured while georeferencing RGB version of {hyspec_path}"
                    )

            elif header_contains_mapinfo(hyspec_path):
                logger.info(f"Generating GeoTiff using ENVI header map info and rasterio")
                try:
                    georeferenced_hyspec_to_rgb_geotiff(
                        hyspec_path,
                        geotiff_path,
                        rgb_wl=rgb_wl,
                    )
                except Exception:
                    logger.exception(
                        f"Error occured while creating RGB version of {hyspec_path}"
                    )

            else:
                logger.error("No ENVI header map info or geotransform file - skipping.")
                continue

@profiled("radiance_rgb")
def create_radiance_rgb_geotiff(runtime_context: RuntimeContext) -> None:
        """Create georeferenced RGB GeoTIFF versions of radiance"""

        logger.info("---- EXPORTING RGB GEOTIFF FOR RADIANCE ----")

        runtime_context.base_product_paths.radiance_rgb.mkdir(exist_ok=True)

        product_paths = runtime_context.product_paths

        config = runtime_context.config.massipipe_options

        create_rgb_geotiff(
            hyspec_paths=product_paths.radiance,
            geotransform_paths=product_paths.geo_transform,
            geotiff_paths=product_paths.radiance_rgb,
            geotiff_overwrite=config.radiance_rgb.overwrite,
            rgb_wl=config.general.rgb_wl,
        )

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

@profiled("irradiance")
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

@profiled("radiance_gc")
def glint_correct_radiance_images(runtime_context: RuntimeContext) -> None:
    """Remove water surface reflections of sun and sky light"""

    logger.info("---- RADIANCE GLINT CORRECTION ----")

    runtime_context.base_product_paths.radiance_gc.mkdir(exist_ok=True)

    _rad_gc_config = runtime_context.config.massipipe_options.radiance_gc

    # Read glint correction reference information from config
    ref_im_nums = _rad_gc_config.reference_image_numbers
    ref_im_ranges = _rad_gc_config.reference_image_ranges

    if not ref_im_nums:
        logger.error("No reference images for sun glint correction specified - aborting.")
        return

    if (ref_im_ranges is not None) and (len(ref_im_nums) != len(ref_im_ranges)):
        raise ValueError(
            "The number of reference image numbers and reference image ranges do not match."
        )

    if (
            all([rp.exists() for rp in runtime_context.product_paths.radiance_gc])
            and not _rad_gc_config.overwrite
    ):
        logger.info("Glint corrected radiance images already exist - skipping")
        return

    # Fit glint corrector
    ref_im_paths = [runtime_context.product_paths.radiance[im_num] for im_num in ref_im_nums]
    glint_corrector = HedleyGlintCorrector(
        smooth_spectra=_rad_gc_config.smooth_spectra,
        subtract_dark_spec=_rad_gc_config.subtract_dark_spec,
        set_negative_values_to_zero=_rad_gc_config.set_negative_values_to_zero,
    )

    logger.info(f"Fitting glint correction model based on image numbers {ref_im_nums}")
    glint_corrector.fit_to_reference_images(ref_im_paths, ref_im_ranges)

    rad_paths, rad_gc_paths = (
        runtime_context.product_paths.radiance,
        runtime_context.product_paths.radiance_gc
    )

    # Run glint correction
    for rad_image, rad_gc_image in zip(rad_paths, rad_gc_paths):
        if rad_gc_image.exists() and not _rad_gc_config.overwrite:
            logger.info(f"Image {rad_gc_image.name} exists - skipping.")
            continue
        if rad_image.exists():
            logger.info(f"Running glint correction for {rad_image.name}")

            try:
                glint_corrector.glint_correct_image_file(rad_image, rad_gc_image)
            except Exception as e:
                logger.exception(f"Glint correction failed for {rad_image.name}")
                continue

@profiled("radiance_gc_rgb")
def create_glint_corrected_radiance_rgb_geotiff(runtime_context: RuntimeContext) -> None:
    """Create georeferenced GeoTIFF versions of glint corrected radiance"""

    logger.info("---- EXPORTING RGB GEOTIFF FOR GLINT CORRECTED RADIANCE ----")

    runtime_context.base_product_paths.radiance_gc_rgb.mkdir(exist_ok=True)

    _paths = runtime_context.product_paths
    _config = runtime_context.config.massipipe_options

    create_rgb_geotiff(
        hyspec_paths=_paths.radiance_gc,
        geotransform_paths=_paths.geo_transform,
        geotiff_paths=_paths.radiance_gc_rgb,
        geotiff_overwrite=_config.radiance_gc_rgb.overwrite,
        rgb_wl=_config.general.rgb_wl,
    )

def convert_radiance_images_to_reflectance(runtime_context: RuntimeContext) -> None:
    """Convert radiance images (microflicks) to reflectance (unitless)"""

    logger.info("---- REFLECTANCE CONVERSION ----")

    runtime_context.base_product_paths.reflectance.mkdir(exist_ok=True)

    _config = runtime_context.config.massipipe_options

    reflectance_converter = ReflectanceConverter(
        wl_min=_config.reflectance.wl_min,
        wl_max=_config.reflectance.wl_max,
        conv_irrad_with_gauss=_config.reflectance.conv_irrad_with_gauss,
        smooth_spectra=_config.reflectance.smooth_spectra,
        refl_from_mean_irrad=_config.reflectance.refl_from_mean_irrad,
        irrad_spec_paths=runtime_context.product_paths.irradiance
    )

    if not any([rp.exists() for rp in runtime_context.product_paths.radiance]):
        raise FileNotFoundError(f"No radiance images found in "
                                f"{runtime_context.base_product_paths.radiance}")

    if not any([irp.exists() for irp in runtime_context.product_paths.irradiance]):
        raise FileNotFoundError(f"No irradiance spectra found in "
                                f"{runtime_context.base_product_paths.radiance}")

    rad_paths, irrad_paths, refl_paths = (
        runtime_context.product_paths.radiance,
        runtime_context.product_paths.irradiance,
        runtime_context.product_paths.reflectance
    )

    for rad_path, irrad_path, refl_path in zip(rad_paths, irrad_paths, refl_paths):
        if refl_path.exists() and not _config.reflectance.overwrite:
            logger.info(f"Image {refl_path.name} exists - skipping.")
            continue

        if rad_path.exists() and irrad_path.exists():
            logger.info(f"Converting {rad_path.name} to reflectance.")
            try:
                reflectance_converter.convert_radiance_file_to_reflectance(
                    rad_path, irrad_path, refl_path
                )
            except Exception as e:
                logger.exception(
                    f"Error occurred while converting {rad_path.name} to reflectance"
                )

def add_irradiance_to_radiance_header(runtime_context: RuntimeContext) -> None:
    """Pre-process irradiance for reflectance calc. and save to radiance header"""

    logger.info("---- WRITING IRRADIANCE TO RADIANCE HEADER ----")

    _config = runtime_context.config.massipipe_options

    reflectance_converter = ReflectanceConverter(
        wl_min=None,  # Not used, but set to None to show that ...
        wl_max=None,  # ... original radiance wavelengths are not modified.
        conv_irrad_with_gauss=_config.reflectance.conv_irrad_with_gauss,
        smooth_spectra=_config.reflectance.smooth_spectra,
        refl_from_mean_irrad=_config.reflectance.refl_from_mean_irrad,
        irrad_spec_paths=runtime_context.product_paths.irradiance,
    )

    if not any([rp.exists() for rp in runtime_context.product_paths.radiance]):
        raise FileNotFoundError(f"No radiance images found in "
                                f"{runtime_context.base_product_paths.radiance}")

    if not any([irp.exists() for irp in runtime_context.product_paths.irradiance]):
        raise FileNotFoundError(f"No irradiance spectra found in "
                                f"{runtime_context.base_product_paths.radiance}")

    rad_paths, irrad_paths = (
        runtime_context.product_paths.radiance,
        runtime_context.product_paths.irradiance,
    )

    for rad_path, irrad_path in zip(rad_paths, irrad_paths):
        if rad_path.exists() and irrad_path.exists():
            logger.info(f"Writing irradiance to radiance header {rad_path.name}.")

            try:
                reflectance_converter.add_irradiance_spectrum_to_header(rad_path, irrad_path)
            except Exception as e:
                logger.exception(
                    f"Error occured while adding irradiance spectrum to {rad_path.name}"
                )

def add_mapinfo_to_radiance_header(runtime_context: RuntimeContext) -> None:
    """Add ENVI mapinfo (geotransform) to radiance header"""

    logger.info("---- WRITING MAP INFO TO RADIANCE HEADER ----")

    if not any([rp.exists() for rp in runtime_context.product_paths.radiance]):
        raise FileNotFoundError(f"No radiance images found in "
                                f"{runtime_context.base_product_paths.radiance}")

    if not any([gtp.exists() for gtp in runtime_context.product_paths.geo_transform]):
        raise FileNotFoundError(f"No geotransform JSON files found in "
                                f"{runtime_context.base_product_paths.geo_transform}")

    rad_paths, geo_transform_paths = (
        runtime_context.product_paths.radiance,
        runtime_context.product_paths.geo_transform,
    )

    for rad_path, geotrans_path in zip(rad_paths, geo_transform_paths):
        logger.info(f"Adding map info to {rad_path.name}")

        try:
            add_header_mapinfo(rad_path, geotrans_path)
        except Exception as e:
            logger.exception(f"Error adding map info to {rad_path.name}")
            continue

@profiled("reflectance_gc")
def glint_correct_reflectance_images(runtime_context: RuntimeContext) -> None:
    """Correct for sun and sky glint in reflectance images"""

    logger.info("---- REFLECTANCE GLINT CORRECTION ----")

    runtime_context.base_product_paths.reflectance_gc.mkdir(exist_ok=True)

    _config = runtime_context.config.massipipe_options

    rad_gc_paths, refl_gc_paths = (
        runtime_context.product_paths.radiance_gc,
        runtime_context.product_paths.reflectance_gc,
    )

    # Calculate reflectance based on glint corrected radiance
    if _config.reflectance_gc.method == "from_rad_gc":
        logger.info("Calculating glint corrected reflectance from glint corrected radiance")
        reflectance_converter = ReflectanceConverter(
            wl_min=_config.reflectance.wl_min,
            wl_max=_config.reflectance.wl_max,
            smooth_spectra=_config.reflectance_gc.smooth_spectra,
        )

        if not any([rp.exists() for rp in runtime_context.product_paths.radiance_gc]):
            logger.warning(
                f"No glint corrected radiance images found in "
                f"{runtime_context.base_product_paths.radiance_gc}"
            )

        for rad_gc_path, refl_gc_path in zip(rad_gc_paths, refl_gc_paths):
            if refl_gc_path.exists() and not _config.reflectance_gc.overwrite:
                logger.info(f"Image {refl_gc_path.name} exists - skipping.")
                continue

            if rad_gc_path.exists():
                logger.info(f"Calculating reflectance based on {rad_gc_path.name}.")

                try:
                    reflectance_converter.convert_radiance_file_with_irradiance_to_reflectance(
                        rad_gc_path, refl_gc_path
                    )
                except KeyError:
                    logger.error(f"Irradiance spectrum missing in {rad_gc_path}")
                    continue
                except Exception as e:
                    logger.exception(
                        f"Glint corrected reflectance from rad_gc failed for {rad_gc_path.name}"
                    )
                    continue

    # Calculate glint corrected reflectance based on assumption of flat glint spectrum
    elif _config.reflectance_gc.method == "flat_spec":
        logger.info("Using flat spectrum method for reflectance glint correction")
        glint_corrector = FlatSpecGlintCorrector(
            smooth_with_savitsky_golay=_config.reflectance_gc.smooth_spectra
        )

        if not any([rp.exists() for rp in runtime_context.product_paths.reflectance]):
            raise FileNotFoundError(f"No reflectance images found in "
                                    f"{runtime_context.base_product_paths.reflectance}")

        for refl_path, refl_gc_path in zip(rad_gc_paths, refl_gc_paths):
            if refl_gc_path.exists() and not _config.reflectance_gc.overwrite:
                logger.info(f"Image {refl_gc_path.name} exists - skipping.")
                continue

            if refl_path.exists():
                logger.info(f"Applying glint correction to {refl_path.name}.")

                try:
                    glint_corrector.glint_correct_image_file(refl_path, refl_gc_path)
                except Exception as e:
                    logger.exception(f"Flat spec glint correction failed for {refl_path.name}")
                    continue
    else:
        logger.error("Unrecognized glint correction method specified in configuration.")

@profiled("reflectance_gc_rgb")
def create_glint_corrected_reflectance_rgb_geotiff(runtime_context: RuntimeContext) -> None:
    """Create georeferenced GeoTIFF versions of glint corrected reflectance"""

    logger.info("---- EXPORTING RGB GEOTIFF FOR GLINT CORRECTED REFLECTANCE ----")

    runtime_context.base_product_paths.reflectance_gc_rgb.mkdir(exist_ok=True)

    _config = runtime_context.config.massipipe_options
    _paths = runtime_context.product_paths

    create_rgb_geotiff(
        hyspec_paths=_paths.reflectance_gc,
        geotransform_paths=_paths.geo_transform,
        geotiff_paths=_paths.reflectance_gc_rgb,
        geotiff_overwrite=_config.reflectance_gc.overwrite,
        rgb_wl=_config.general.rgb_wl,
    )

@profiled("mosaic")
def mosaic_geotiffs_managed(
        geotiff_paths: list[Path],
        mosaic_path: Path,
        mosaic_overwrite: bool,
        overview_factors: Sequence[int],
        convert_to_8bit: bool = True,
) -> None:
    """Mosaic GeoTIFF images into single GeoTIFF with overviews

    Parameters
    ----------
    geotiff_paths : list[Path]
        List of GeoTIFFs to mosaic
    mosaic_path : Path
        Path to output GeoTIFF
    mosaic_overwrite : bool
        Whether to overwrite existing output GeoTIFF.
    overview_factors : Sequence[int]
        GeoTIFF overview factors, typically powers of 2
        Used to add "pyramid" of lower-resolution images for faster image browsing.
    convert_to_8bit : bool, optional
        If true, the mosaic is percentile stretched and converted to 8-bit.
        This decreases file size and enhances contrast, at the cost of losing the
        (physical) units of the original image.
    """
    if (mosaic_path.exists()) and (not mosaic_overwrite):
        logger.info(f"Mosaic {mosaic_path.name} already exists - skipping.")
        return

    if not any([gtp.exists() for gtp in geotiff_paths]):
        logger.warning(f"None of the listed GeoTIFFs exist.")
        return

    mosaic_geotiffs(geotiff_paths, mosaic_path)

    if convert_to_8bit:
        convert_geotiff_to_8bit(input_image_path=mosaic_path, output_image_path=mosaic_path)

    add_geotiff_overviews(mosaic_path, overview_factors)

def mosaic_radiance_geotiffs(runtime_context: RuntimeContext) -> None:
    """Merge radiance RGB images into mosaic with overviews"""

    logger.info("---- MOSAICING RADIANCE ----")

    runtime_context.base_product_paths.mosaic.mkdir(exist_ok=True)

    _mosaic_config = runtime_context.config.massipipe_options.mosaic

    rad_rgb_paths, mosaic_rad_paths =(
        runtime_context.product_paths.radiance_rgb,
        runtime_context.product_paths.mosaic_rad
    )

    mosaic_geotiffs_managed(
        geotiff_paths=rad_rgb_paths,
        mosaic_path=mosaic_rad_paths,
        mosaic_overwrite=_mosaic_config.radiance_rgb.overwrite,
        overview_factors=_mosaic_config.overview_factors,
    )

def mosaic_radiance_gc_geotiffs(runtime_context: RuntimeContext) -> None:
    """Merge radiance_gc RGB images into mosaic with overviews"""

    logger.info("---- MOSAICING GLINT CORRECTED RADIANCE ----")

    runtime_context.base_product_paths.mosaic.mkdir(exist_ok=True)

    _mosaic_config = runtime_context.config.massipipe_options.mosaic

    rad_gc_rgb_paths, mosaic_rad_gc_paths =(
        runtime_context.product_paths.radiance_rgb,
        runtime_context.product_paths.mosaic_rad
    )

    mosaic_geotiffs_managed(
        geotiff_paths=rad_gc_rgb_paths,
        mosaic_path=mosaic_rad_gc_paths,
        mosaic_overwrite=_mosaic_config.radiance_gc_rgb.overwrite,
        overview_factors=_mosaic_config.overview_factors,
    )

def mosaic_reflectance_gc_geotiffs(runtime_context: RuntimeContext) -> None:
    """Merge reflectance_gc RGB images into mosaic with overviews"""

    logger.info("---- MOSAICING GLINT CORRECTED REFLECTANCE ----")

    runtime_context.base_product_paths.mosaic.mkdir(exist_ok=True)

    _mosaic_config = runtime_context.config.massipipe_options.mosaic

    refl_gc_rgb_paths, mosaic_refl_gc_path =(
        runtime_context.product_paths.reflectance_gc_rgb,
        runtime_context.product_paths.mosaic_refl_gc
    )

    mosaic_geotiffs_managed(
        geotiff_paths=refl_gc_rgb_paths,
        mosaic_path=mosaic_refl_gc_path,
        mosaic_overwrite=_mosaic_config.reflectance_gc_rgb.overwrite,
        overview_factors=_mosaic_config.overview_factors,
    )

def delete_existing_products(
        runtime_context: RuntimeContext,
        *,
        delete_quicklook: bool = True,
        delete_radiance: bool = True,
        delete_radiance_gc: bool = True,
        delete_reflectance: bool = True,
        delete_reflectance_gc: bool = True,
        delete_geotransform: bool = True,
        delete_imudata: bool = True,
        delete_mosaics: bool = True,
) -> None:
    """Delete existing image products ("reset" after previous processing)

    Parameters
    ----------
    delete_quicklook : bool, optional
        If true, delete 0b_quicklook folder (if it exists)
    delete_radiance : bool, optional
        If true, delete 1a_radiance folder (if it exists)
        If data "starting point" is radiance and not raw files,
        radiance will not be deleted (delete manually if needed).
    delete_radiance_gc : bool, optional
        If true, delete 1b_radiance folder (if it exists)
    delete_reflectance : bool, optional
        If true, delete 2a_reflectance folder (if it exists)
    delete_reflectance_gc : bool, optional
        If true, delete 2b_reflectance folder (if it exists)
    delete_geotransform : bool, optional
        If true, delete geotransform folder (if it exists)
    delete_imudata : bool, optional
        If true, delete imudata folder (if it exists)
        If data "starting point" is radiance and not raw files,
        imu data will not be deleted (delete manually if needed).
    delete_mosaics : bool, optional
        If true, delete mosaics folder (if it exists)
    """
    _base_paths = runtime_context.base_product_paths

    if _base_paths.quicklook.exists() and delete_quicklook:
        logger.info(f"Deleting {_base_paths.quicklook}")
        shutil.rmtree(_base_paths.quicklook)

    if (
            _base_paths.radiance.exists()
            and delete_radiance
            and not (runtime_context.data_starting_point == "radiance")
    ):
        logger.info(f"Deleting {_base_paths.radiance}")
        shutil.rmtree(_base_paths.radiance)

    if _base_paths.radiance_gc.exists() and delete_radiance_gc:
        logger.info(f"Deleting {_base_paths.radiance_gc}")
        shutil.rmtree(_base_paths.radiance_gc)

    if _base_paths.reflectance.exists() and delete_reflectance:
        logger.info(f"Deleting {_base_paths.reflectance}")
        shutil.rmtree(_base_paths.reflectance)

    if _base_paths.reflectance_gc.exists() and delete_reflectance_gc:
        logger.info(f"Deleting {_base_paths.reflectance_gc}")
        shutil.rmtree(_base_paths.reflectance_gc)

    if _base_paths.geo_transform.exists() and delete_geotransform:
        logger.info(f"Deleting {_base_paths.geo_transform}")
        shutil.rmtree(_base_paths.geo_transform)

    if (
            _base_paths.imu_data.exists()
            and delete_imudata
            and not (runtime_context.data_starting_point == "radiance")
    ):
        logger.info(f"Deleting {_base_paths.imu_data}")
        shutil.rmtree(_base_paths.imu_data)

    if _base_paths.mosaic.exists() and delete_mosaics:
        logger.info(f"Deleting {_base_paths.mosaic}")
        shutil.rmtree(_base_paths.mosaic)

def run_quicklook(runtime_context: RuntimeContext) -> None:
    """Create quicklook versions of images (percentile stretched)"""

    if runtime_context.config.massipipe_options.quicklook.create:
        try:
            create_quicklook_images(runtime_context)
        except Exception:
            logger.exception("Error while creating quicklook images")

def run_raw_data_processing(runtime_context: RuntimeContext) -> None:
    """Run all data processing steps based on raw data"""

    _config = runtime_context.config.massipipe_options

    if _config.imu_data.create:
        try:
            parse_and_save_imu_data(runtime_context)
        except Exception:
            logger.exception("Error while parsing and saving IMU data")

    if _config.radiance.create:
        try:
            convert_raw_images_to_radiance(runtime_context)
        except Exception:
            logger.exception("Error while converting raw images to radiance")

    if _config.irradiance.create:
        try:
            convert_raw_spectra_to_irradiance(runtime_context)
        except Exception:
            logger.exception("Error while converting raw spectra to irradiance")

        try:
            calibrate_irradiance_wavelengths(runtime_context)
        except Exception:
            logger.exception("Error while calibrating irradiance wavelengths")

def run_secondary_processing(runtime_context: RuntimeContext) -> None:

    _config = runtime_context.config.massipipe_options

    if _config.geotransform.create:
        try:
            create_and_save_geotransform(runtime_context)
        except Exception:
            logger.exception("Error while parsing and saving IMU data")

    if _config.radiance.add_irradiance_to_header:
        try:
            add_irradiance_to_radiance_header(runtime_context)
        except Exception:
            logger.exception("Error while adding irradiance to radiance header")

    if _config.radiance.add_envi_mapinfo_to_header:
        try:
            add_mapinfo_to_radiance_header(runtime_context)
        except Exception:
            logger.exception("Error while adding map info to radiance header")

    if _config.radiance_rgb.create:
        try:
            create_radiance_rgb_geotiff(runtime_context)
        except Exception:
            logger.exception("Error while creating RGB GeoTOFFs from radiance")

    if _config.reflectance.create:
        try:
            convert_radiance_images_to_reflectance(runtime_context)
        except FileNotFoundError:
            logger.exception(
                "Missing input radiance / irradiance files, skipping reflectance conversion."
            )
        except Exception:
            logger.exception("Error while converting from radiance to reflectance")

def run_glint_correction(runtime_context: RuntimeContext) -> None:
    """Run glint correction using parameters defined in YAML file

    The processing steps include:
        - Fitting a glint correction model to radiance images
        - Running glint correction and saving glint corrected images
        - Creating RGB GeoTiff versions of glint corrected images

    See massipipe.config.Config and template YAML file for all options.

    """

    _config = runtime_context.config.massipipe_options

    if _config.radiance_gc.create:
        try:
            glint_correct_radiance_images(runtime_context)
        except Exception:
            logger.exception("Error while glint correcting radiance images")

    if _config.radiance_gc_rgb.create:
        try:
            create_glint_corrected_radiance_rgb_geotiff(runtime_context)
        except Exception:
            logger.exception("Error while creating RGB GeoTIFFs from glint corrected radiance")

    if _config.reflectance_gc.create:
        try:
            glint_correct_reflectance_images(runtime_context)
        except Exception:
            logger.exception("Error while glint correcting radiance images")

    if _config.reflectance_gc_rgb.create:
        try:
            create_glint_corrected_reflectance_rgb_geotiff(runtime_context)
        except Exception:
            logger.exception("Error while creating RGB GeoTIFFs from glint corrected radiance")

def run_mosaics(runtime_context: RuntimeContext) -> None:
    """Run all mosaicing operations"""

    _config = runtime_context.config.massipipe_options

    if _config.mosaic.radiance_rgb.create:
        try:
            mosaic_radiance_geotiffs(runtime_context)
        except Exception:
            logger.exception(f"Error occured while mosaicing radiance", exc_info=True)

    if _config.mosaic.radiance_gc_rgb.create:
        try:
            mosaic_radiance_gc_geotiffs(runtime_context)
        except Exception:
            logger.error(f"Error occured while mosaicing glint corrected radiance")

    if _config.mosaic.reflectance_gc_rgb.create:
        try:
            mosaic_reflectance_gc_geotiffs(runtime_context)
        except Exception:
            logger.exception(f"Error occured while mosaicing glint corrected reflectance")

def get_calibration_paths(bootstrap_options: BootstrapOptions) -> CalibrationPaths:
    calibration_dir = bootstrap_options.calibration_dir

    if not calibration_dir.exists():
        raise FileNotFoundError(f"calibration_dir: {calibration_dir} not found")

    return CalibrationPaths(
        radiance=get_radiance_calibration_path(calibration_dir),
        irradiance=get_irradiance_calibration_path(calibration_dir),
    )

def process_raw_data_paths(runtime_context: RuntimeContext):
    product_paths = runtime_context.product_paths

    product_paths.raw = list(
        runtime_context.base_product_paths.raw.rglob("*.bil.hdr")
    )

    product_paths.raw = sorted(product_paths.raw, key=get_image_number)

    product_paths.times, product_paths.lcf = validate_raw_files(product_paths.raw)

    # Search for raw irradiance spectrum files (not always present)
    product_paths.raw_spec = get_raw_spectrum_paths(product_paths.raw)

    # Create "base" file names numbered from 0
    runtime_context.base_filenames = create_base_file_names(runtime_context)

def setup(dataset_dir: str | os.PathLike[str],
          *,
          config_file_path: str | os.PathLike[str] | None = None,
          calibration_dir: str | os.PathLike[str] | None = None) -> RuntimeContext:
    bootstrap_options = setup_bootstrap_options(dataset_dir, config_file_path, calibration_dir)

    base_product_paths = setup_base_product_paths(bootstrap_options)

    setup_logging(bootstrap_options)

    config = setup_config(bootstrap_options)
    if config is None:
        raise RuntimeError("Failed to setup config")

    data_starting_point = get_data_starting_point(base_product_paths)

    runtime_context = RuntimeContext(
        bootstrap_options=bootstrap_options,
        base_product_paths=base_product_paths,
        data_starting_point=data_starting_point,
        calibration_paths=get_calibration_paths(bootstrap_options),
        product_paths=ProductPaths(),
        config=config,
        base_filenames=[],
        radiance_converter=RadianceConverter,
    )

    if data_starting_point == "raw":
        process_raw_data_paths(runtime_context)

    base_filenames = create_base_file_names(runtime_context)

    runtime_context.base_filenames = base_filenames

    setup_processed_product_paths(runtime_context)

    return runtime_context

def run(runtime_context: RuntimeContext) -> None:
    """Run all processing steps"""

    logger.info(
        f"Running pipeline using configuration from "
        f"{runtime_context.bootstrap_options.config_file_path}: \n"
        f"{pprint.pformat(runtime_context.config.massipipe_options.model_dump())}"
    )

    run_quicklook(runtime_context)
    if runtime_context.data_starting_point == "raw":
        run_raw_data_processing(runtime_context)

    run_secondary_processing(runtime_context)

    run_glint_correction(runtime_context)

    run_mosaics(runtime_context)

def export(runtime_context: RuntimeContext) -> None:
    """Export dataset to ZIP file for archival / publishing"""

    # Configure export module logger to use pipeline file log handler
    # configure_export_log_file_handler(self.log_file_handler)
    # TODO: I do not see exactly why this is needed,
    #  when the original Pipeline class had log init in class definition

    # Copy "best" mosaic to separate directory
    if runtime_context.config.massipipe_options.mosaic.visualization_mosaic == "radiance":
        mosaic_source_path = runtime_context.base_product_paths.mosaic
    else:
        mosaic_source_path = runtime_context.product_paths.mosaic_rad_gc

    mosaic_dest_path = (
            runtime_context.base_product_paths.mosaic_orthophoto /
            f"{mosaic_source_path.stem}_orthophoto.original.tif"
    )

    copy_visualization_mosaic(mosaic_source_path, mosaic_dest_path)

    # Package selected processed data as ZIP file
    export_dataset_zip(
        runtime_context.bootstrap_options.dataset_dir,
        runtime_context.base_product_paths.quicklook,
        runtime_context.base_product_paths.radiance,
        runtime_context.base_product_paths.imu_data,
        runtime_context.base_product_paths.mosaic_orthophoto,
        runtime_context.bootstrap_options.config_file_path,
    )

def find_datasets(
        base_dir: Union[Path, str], subdir_search_strings: list[str] = ["0_raw", "1a_radiance"]
) -> list[Path]:
    """Find dataset paths based on expected subdirectories in dataset

    Parameters
    ----------
    base_dir : Path
        Filesystem starting point (searching tree below this point)
    subdir_search_strings : list[str], default ["0_raw", "1a_radiance"]
        List with names of subdirectories that are expected to be within a dataset
        directory. If any of the names match, the dataset directory is included.

    Returns
    -------
    dataset_dirs
        List of dataset dirctories mathcing search criteria.
    """
    base_dir = Path(base_dir)
    dataset_dirs = set()  # Use set to avoid duplicates
    for subdir_search_str in subdir_search_strings:
        dataset_dirs.update(p.parent for p in base_dir.rglob(subdir_search_str))
    return sorted(dataset_dirs)
