"""Massipipe configuration module.

Provides functions for reading, writing, and exporting YAML configuration files,
and defines Pydantic models for Massipipe processing options.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import yaml
from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    field_validator,
)

logger = logging.getLogger(__name__)

DataStartingPoint = Literal["raw", "radiance"]

class RuntimeContext(BaseModel):
    bootstrap_options: BootstrapOptions
    calibration_paths: CalibrationPaths
    base_product_paths: BaseProductPaths
    product_paths: ProductPaths

    config: Config
    data_starting_point: DataStartingPoint
    base_filenames: list[str]
    radiance_converter: Any  # TODO: IRadianceConverter - pydantic not cooperating

class BootstrapOptions(BaseModel):
    dataset_dir: Path
    config_file_path: Path
    calibration_dir: Path
    logs: Path
    dataset_name: str


class CalibrationPaths(BaseModel):
    radiance: Path
    irradiance: Path


class BaseProductPaths(BaseModel):
    raw: Path
    imu_data: Path
    quicklook: Path
    radiance: Path
    radiance_rgb: Path
    radiance_gc: Path
    radiance_gc_rgb: Path
    # irradiance: Path # where does this come from?
    reflectance: Path
    reflectance_gc: Path
    reflectance_gc_rgb: Path
    geo_transform: Path
    mosaic: Path
    mosaic_orthophoto: Path


class ProductPaths(BaseModel):
    raw: list[Path] = Field(default_factory=list)
    raw_spec: list[Path] = Field(default_factory=list)
    imu_data: list[Path] = Field(default_factory=list)
    quicklook: list[Path] = Field(default_factory=list)
    radiance: list[Path] = Field(default_factory=list)
    radiance_rgb: list[Path] = Field(default_factory=list)
    radiance_gc: list[Path] = Field(default_factory=list)
    radiance_gc_rgb: list[Path] = Field(default_factory=list)
    irradiance: list[Path] = Field(default_factory=list)
    reflectance: list[Path] = Field(default_factory=list)
    reflectance_gc: list[Path] = Field(default_factory=list)
    reflectance_gc_rgb: list[Path] = Field(default_factory=list)
    geo_transform: list[Path] = Field(default_factory=list)
    times: list[Path] = Field(default_factory=list)
    lcf: list[Path] = Field(default_factory=list)
    mosaic_rad: Path = Field(default_factory=Path)
    mosaic_rad_gc: Path = Field(default_factory=Path)
    mosaic_refl_gc: Path = Field(default_factory=Path)


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


def read_config(yaml_path: Union[Path, str]) -> Any:
    """Parse YAML config file, accepting only basic YAML tags"""
    with open(yaml_path, "r") as stream:
        config = yaml.safe_load(stream)
    return config


def write_config(data: dict, yaml_path: Union[Path, str]) -> None:
    """Write config data formatted as dictionary to YAML file"""
    with open(yaml_path, "w") as yaml_file:
        yaml.safe_dump(data, yaml_file, default_flow_style=False, sort_keys=False)


def update_dict(dictionary: dict, keys: list[str], value: Any, key_must_exist: bool) -> dict:
    """
    Updates or adds a value in a nested dictionary based on a list of keys.

    Parameters
    ----------
    dictionary : dict
        The original dictionary to be updated.
    keys : list[str]
        A list of keys representing the path to the value in the nested dictionary.
    value : Any
        The value to set at the specified key path.
    key_must_exist : bool
        If True, raises a warning if the final key does not exist in the dictionary.
        If False, adds the key-value pair to the dictionary if the key does not exist.

    Returns
    -------
    dict
        A new dictionary with the updated or added key-value pair.

    Raises
    ------
    KeyError
        If any of the intermediate keys in `keys` do not exist in the dictionary.

    Warns
    -----
    UserWarning
        If `key_must_exist` is True and the final key does not exist in the dictionary.

    Examples
    --------
    >>> original_dict = {"a": {"b": {"c": 1}}}
    >>> updated = update_dict(original_dict, ["a", "b", "d"], 2, key_must_exist=False)
    >>> print(updated)
    {'a': {'b': {'c': 1, 'd': 2}}}
    """

    updated_dict = dictionary.copy()
    dict_subtree = updated_dict

    # Get to dict level above last key
    for key in keys[0:-1]:
        if key in dict_subtree:
            dict_subtree = dict_subtree[key]
        else:
            if key_must_exist:
                raise KeyError(f"Key {key} does not exist in dictionary")
            else:
                dict_subtree[key] = {}  # Create before descending
                dict_subtree = dict_subtree[key]

    # Update or add last key
    if key_must_exist:
        if keys[-1] in dict_subtree:
            dict_subtree[keys[-1]] = value
        else:
            raise KeyError(f"Key {keys[-1]} does not exist in dictionary.")
    else:
        dict_subtree[keys[-1]] = value

    return updated_dict


def update_yaml(yaml_file: Path, keys: list[str], value: Any, key_must_exist: bool = False) -> None:
    """
    Updates a YAML configuration file by modifying or adding a value at the specified
    key path.

    Parameters
    ----------
    yaml_file : Path
        The path to the YAML file to be updated.
    keys : list[str]
        A list of keys representing the hierarchical path in the YAML structure where
        the value should be updated.
    value : Any
        The new value to set at the specified key path.
    key_must_exist : bool, default False
        If True, raises an error if the specified key path does not exist in the YAML
        structure. If False, creates the key path if it does not exist.

    """

    yaml_dict = read_config(yaml_file)
    updated_dict = update_dict(yaml_dict, keys, value, key_must_exist)
    write_config(updated_dict, yaml_file)


#### DEFINE PYDANTIC CONFIG STRUCTURE ####


class MpGeneral(BaseModel):
    """General options for Massipipe processing"""

    rgb_wl: Optional[Tuple[PositiveFloat, PositiveFloat, PositiveFloat]] = (640.0, 550.0, 460.0)


class MpQuickLook(BaseModel):
    """Configuration for creating quicklook image"""

    create: bool = True
    overwrite: bool = False
    percentiles: Optional[Tuple[NonNegativeFloat, PositiveFloat]] = None


class MpImuData(BaseModel):
    """Configuration for processing IMU data"""

    create: bool = True
    overwrite: bool = False


class MpGeoTransform(BaseModel):
    """Configuration for creating geotransform from IMU data"""

    create: bool = True
    overwrite: bool = False
    camera_opening_angle_deg: PositiveFloat = 36.5
    pitch_offset_deg: float = 0.0
    roll_offset_deg: float = 0.0
    altitude_offset_m: float = 0.0
    utm_x_offset_m: float = 0.0
    utm_y_offset_m: float = 0.0
    assume_square_pixels: bool = True


class MpRadiance(BaseModel):
    """Configuration for converting raw data to radiance"""

    create: bool = True
    overwrite: bool = False
    set_saturated_pixels_to_zero: bool = True
    add_envi_mapinfo_to_header: bool = True
    add_irradiance_to_header: bool = True


class MpRadianceRgb(BaseModel):
    """Comfiguration for creating RGB image from radiance data"""

    create: bool = True
    overwrite: bool = False


class MpRadianceGc(BaseModel):
    """Configuration for converting radiance to glint corrected radiance"""

    create: bool = False
    overwrite: bool = False
    smooth_spectra: bool = True
    subtract_dark_spec: bool = False
    set_negative_values_to_zero: bool = False
    reference_image_numbers: Optional[List[NonNegativeInt]] = None
    reference_image_ranges: Optional[
        List[Tuple[NonNegativeInt, PositiveInt, NonNegativeInt, PositiveInt]]
    ] = None


class MpRadianceGcRgb(BaseModel):
    """Configuration for creating RGB image from glint corrected radiance data"""

    create: bool = False
    overwrite: bool = False


class MpIrradiance(BaseModel):
    """Configuration for converting raw spectrum to irradiance"""

    create: bool = True
    overwrite: bool = False


class MpReflectance(BaseModel):
    """Configuration for converting radiance to reflectance"""

    create: bool = False
    overwrite: bool = False
    wl_min: float = 400
    wl_max: float = 930
    conv_irrad_with_gauss: bool = True
    fwhm_irrad_smooth: float = 3.5
    smooth_spectra: bool = False
    refl_from_mean_irrad: bool = False


class MpReflectanceGc(BaseModel):
    """Configuration for creating glint corrected reflectance"""

    create: bool = False
    overwrite: bool = False
    smooth_spectra: bool = True
    method: Literal["from_rad_gc", "flat_spec"] = "from_rad_gc"


class MpReflectanceGcRgb(BaseModel):
    """Configuration for creating RGB image from glint corrected reflectance data"""

    create: bool = False
    overwrite: bool = False


class MpMosaicRadiance(BaseModel):
    """Configuration for creating radiance mosaic image"""

    create: bool = True
    overwrite: bool = False


class MpMosaicRadianceGc(BaseModel):
    """Configuration for creating glint corrected radiance mosaic image"""

    create: bool = False
    overwrite: bool = False


class MpMosaicReflectanceGc(BaseModel):
    """Configuration for creating glint corrected reflectance mosaic image"""

    create: bool = False
    overwrite: bool = False


class MpMosaic(BaseModel):
    """General configuration for creating mosaic images"""

    overview_factors: Sequence[PositiveInt] = [2, 4, 8, 16, 32]
    visualization_mosaic: Literal["radiance", "radiance_gc"] = "radiance"
    radiance_rgb: MpMosaicRadiance = MpMosaicRadiance()
    radiance_gc_rgb: MpMosaicRadianceGc = MpMosaicRadianceGc()
    reflectance_gc_rgb: MpMosaicReflectanceGc = MpMosaicReflectanceGc()


class MassipipeOptions(BaseModel):
    """Configuration for Massipipe processing"""

    general: MpGeneral = MpGeneral()
    quicklook: MpQuickLook = MpQuickLook()
    imu_data: MpImuData = MpImuData()
    geotransform: MpGeoTransform = MpGeoTransform()
    radiance: MpRadiance = MpRadiance()
    radiance_rgb: MpRadianceRgb = MpRadianceRgb()
    radiance_gc: MpRadianceGc = MpRadianceGc()
    radiance_gc_rgb: MpRadianceGcRgb = MpRadianceGcRgb()
    irradiance: MpIrradiance = MpIrradiance()
    reflectance: MpReflectance = MpReflectance()
    reflectance_gc: MpReflectanceGc = MpReflectanceGc()
    reflectance_gc_rgb: MpReflectanceGcRgb = MpReflectanceGcRgb()
    mosaic: MpMosaic = MpMosaic(
        radiance_rgb=MpMosaicRadiance(),
        radiance_gc_rgb=MpMosaicRadianceGc(),
        reflectance_gc_rgb=MpMosaicReflectanceGc(),
    )


class Config(BaseModel):
    """Top-level configuration including standard SeaBee fields"""

    grouping: str = "grouping_name"
    area: str = "area_name"
    datetime: str = "197001010000"
    nfiles: PositiveInt = 1
    organisation: str = "organization_name"
    project: str = "project_name"
    creator_name: str = "creator_name"
    mosaic: bool = False
    classify: bool = False
    theme: str = "Habitat"
    spectrum_type: Literal["RGB", "MSI", "HSI"] = "HSI"
    massipipe_options: MassipipeOptions = MassipipeOptions()

    # Validation of datetime string
    # Note that string is not converted to datetime object,
    # this is to keep original string for template generation.
    @field_validator("datetime", mode="before")
    @classmethod
    def datetime_validate(cls, datetime_str: str):
        if len(datetime_str) == 8:  # Expect YYYYmmdd
            try:
                _ = datetime.strptime(datetime_str, "%Y%m%d")
            except ValueError:
                raise
        elif len(datetime_str) == 12:  # Expect YYYYmmddHHMM
            try:
                _ = datetime.strptime(datetime_str, "%Y%m%d%H%M")
            except ValueError:
                raise
        else:
            raise ValueError(f"Invalid datetime string, use YYYYmmdd or YYYYmmddHHMM")
        return datetime_str


def get_config_template() -> Config:
    """Generate template configuration based on Pydantic schema"""
    template_config = Config()
    return template_config


def nested_config_to_dict(config: BaseModel) -> Dict[str, Any]:
    """Convert nested Pydantic configuration to nested dictionary (recursively)"""
    config_dict = dict(config)
    for key, value in config_dict.items():
        if isinstance(value, BaseModel):
            config_dict[key] = nested_config_to_dict(value)
    return config_dict


def export_template_yaml(yaml_path: Union[Path, str]) -> None:
    """Export YAML template based on Pydantic schema"""
    template_config = get_config_template()
    template_dict = nested_config_to_dict(template_config)
    write_config(template_dict, yaml_path)
