import numpy as np
import pytest

import massipipe
from massipipe.radiance import (
    RadianceCalibrationDataset,
    _njit_noparallel,
    _njit_parallel,
    _np_common,
)


def test_radiance_calibration_dataset(example_dataset_dir, rad_cal_file_path):
    rad_cal_dataset = RadianceCalibrationDataset(rad_cal_file_path)
    dark_frame, _, _, dark_gain, dark_shutter = rad_cal_dataset.get_closest_dark_frame(
        gain=15, shutter=9.274
    )
    assert (dark_gain == 15) and (dark_shutter == 5)
    assert dark_frame.shape == (1, 900, 600)
    assert dark_frame[0, 400, 300] == 1
    rad_conv_frame, _, _ = rad_cal_dataset.get_rad_conv_frame()
    assert rad_conv_frame.shape == (1, 900, 600)
    assert rad_conv_frame[0, 400, 300] == pytest.approx(10.337038)


def test_radiance_converter(rad_cal_file_path, example_raw_image):
    radiance_converter = massipipe.RadianceConverter(rad_cal_file_path)
    raw_image, raw_wl, raw_meta = example_raw_image
    rad_image = radiance_converter.convert_raw_image_to_radiance(raw_image, raw_meta)
    assert rad_image[10, 300, 100] == 441


@pytest.mark.parametrize(
    ("converter", "raw_dtype"),
    [
        (_np_common, np.uint16),
        (_np_common, np.float32),
        (_njit_noparallel, np.float32),
        (_njit_parallel, np.float32),
    ],
)
def test_optimized_radiance_kernels_match_reference(converter, raw_dtype):
    raw_image = np.array([[[1, 4, 5], [8, 5, 2], [7, 10, 1]]], dtype=raw_dtype)
    original_raw_image = raw_image.copy()
    dark_frame = np.array(
        [[[2.0, 1.0, 3.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]],
        dtype=np.float32,
    )
    rad_conv_frame = np.array(
        [[[2.0, 3.0, 4.0], [0.5, 1.5, 2.0], [2.0, 2.0, 2.0]]],
        dtype=np.float32,
    )

    expected = (raw_image - dark_frame) * rad_conv_frame
    expected[expected < 0] = 0
    expected[np.any(raw_image >= 10, axis=2)] = 0
    expected = expected.astype(np.uint16)

    actual = converter(
        raw_image,
        dark_frame,
        rad_conv_frame,
        set_saturated_pixels_to_zero=True,
        saturation_value=10,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(raw_image, original_raw_image)


def test_radiance_converter_file_api(example_dataset_dir, rad_cal_file_path, tmp_path):
    raw_path = (
        example_dataset_dir
        / "0_raw"
        / "OlbergholmenS1-5"
        / "OlbergholmenS1_Pika_L_5.bil.hdr"
    )
    radiance_path = tmp_path / "radiance.bip.hdr"

    radiance_converter = massipipe.RadianceConverter(rad_cal_file_path)
    radiance_converter.convert_raw_file_to_radiance(raw_path, radiance_path)

    radiance_image, _, _ = massipipe.read_envi(radiance_path)
    assert radiance_image[10, 300, 100] == 441


def test_imu_data_parser(example_times_lcf_path):
    pass


def test_image_flight_metadata(example_times_lcf_path):
    pass


def test_irradiance_converter(irrad_cal_file_path, example_raw_spec):
    irrad_converter = massipipe.IrradianceConverter(irrad_cal_file_path)
    raw_spec, _, raw_spec_meta = example_raw_spec
    irrad_spec = irrad_converter.convert_raw_spectrum_to_irradiance(raw_spec, raw_spec_meta)
    # assert np.all(irrad_spec < 1.0) # Not true if including all bands (wl < 360 nm)
    np.testing.assert_allclose(irrad_spec.squeeze()[500], 0.74095922)
