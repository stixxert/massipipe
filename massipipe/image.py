import logging
from enum import StrEnum
from pathlib import Path
from typing import Any, Sequence

from pipefunc import pipefunc

from massipipe import (
    add_geotiff_overviews,
    convert_geotiff_to_8bit,
    georeferenced_hyspec_to_rgb_geotiff,
    mosaic_geotiffs,
)
from massipipe.config import RuntimeContext
from massipipe.georeferencing import SimpleGeoreferencer
from massipipe.utils import header_contains_mapinfo

logger = logging.getLogger(__name__)

class ExportTarget(StrEnum):
    RADIANCE = "radiance"
    RADIANCE_GLINT_CORRECTED = "radiance_gc"
    REFLECTANCE_GLINT_CORRECTED = "reflectance_gc"


@pipefunc(output_name="nodata")
def export_geotiffs(runtime_context: RuntimeContext, export_target: ExportTarget):
    logger.info(f"---- EXPORTING RGB GEOTIFF FOR {export_target.name} ----")

    _base_paths = runtime_context.base_product_paths
    _paths = runtime_context.product_paths
    _config = runtime_context.config.massipipe_options

    match export_target:
        case ExportTarget.RADIANCE:
            _base_paths.radiance_rgb.mkdir(exist_ok=True)
            props = {
                "hyspec_paths": _paths.radiance,
                "geotransform_paths": _paths.geo_transform,
                "geotiff_paths": _paths.radiance_rgb,
                "geotiff_overwrite": _config.radiance_rgb.overwrite,
                "rgb_wl": _config.general.rgb_wl,
            }

        case ExportTarget.RADIANCE_GLINT_CORRECTED:
            _base_paths.radiance_gc_rgb.mkdir(exist_ok=True)
            props = {
                "hyspec_paths": _paths.radiance_gc,
                "geotransform_paths": _paths.geo_transform,
                "geotiff_paths": _paths.radiance_gc_rgb,
                "geotiff_overwrite": _config.radiance_gc_rgb.overwrite,
                "rgb_wl": _config.general.rgb_wl,
            }

        case ExportTarget.REFLECTANCE_GLINT_CORRECTED:
            _base_paths.reflectance_gc_rgb.mkdir(exist_ok=True)
            props = {
                "hyspec_paths" :_paths.radiance_gc,
                "geotransform_paths": _paths.geo_transform,
                "geotiff_paths": _paths.radiance_gc_rgb,
                "geotiff_overwrite": _config.radiance_gc_rgb.overwrite,
                "rgb_wl": _config.general.rgb_wl,
            }

        case _:
            return

    try:
        create_rgb_geotiff(**props)
    except Exception:
        logger.exception(f"Error occurred while exporting geotiffs for type: {export_target.name}")
        raise

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

def export_mosaic(runtime_context: RuntimeContext, export_target: ExportTarget):
    logger.info(f"---- MOSAICING {export_target.name} ----")

    _base_paths = runtime_context.base_product_paths
    _paths = runtime_context.product_paths
    _config = runtime_context.config.massipipe_options

    _mosaic_config = runtime_context.config.massipipe_options.mosaic

    props: dict[str, Any]

    match export_target:
        case ExportTarget.RADIANCE:
            props = {
                "geotiff_paths": _paths.radiance_rgb,
                "mosaic_path": _paths.mosaic_rad,
                "mosaic_overwrite": _mosaic_config.radiance_rgb.overwrite,
            }

        case ExportTarget.RADIANCE_GLINT_CORRECTED:
            props = {
                "geotiff_paths": _paths.radiance_gc_rgb,
                "mosaic_path": _paths.mosaic_rad_gc,
                "mosaic_overwrite": _mosaic_config.radiance_gc_rgb.overwrite,
            }

        case ExportTarget.REFLECTANCE_GLINT_CORRECTED:
            props = {
                "geotiff_paths": _paths.reflectance_gc_rgb,
                "mosaic_path": _paths.mosaic_refl_gc,
                "mosaic_overwrite": _mosaic_config.reflectance_gc_rgb.overwrite,
            }

        case _:
            return

    _base_paths.mosaic.mkdir(exist_ok=True)

    props.update({"overview_factors": _mosaic_config.overview_factors})

    try:
        mosaic_geotiffs_managed(**props)
    except Exception:
        logger.exception(f"Error occurred while exporting mosaic for type: {export_target.name}")
        raise

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