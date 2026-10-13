#! /usr/bin/python3.11
# Imports
from pathlib import Path
import sys
import numpy as np
import massipipe.utils as mpu
from massipipe import FlatTerrainOrthorectifier, Pipeline

# ====================================================================================================
# ===================      USER SETTINGS      ========================================================
# ====================================================================================================

run_pipeline = False # run the main pipeline processing?
run_orthorect = True # output georectified geotiffs?
flip_process = False # usually false: only affects orthorect...
# ...loop thru reflectance files in REVERSE order (so you can run two processes at once)

# Paths
data_dir = Path('/mnt/projects/osat/hyperspectral/data/erie/2026/09-08/')
#data_dir = Path('/mnt/projects/osat/hyperspectral/data/erie/2026/lauren_process/08-18/')
#data_dir = Path('/mnt/projects/osat/hyperspectral/data/muskegon_duck_white_lake/2026/Muskegon/08-20')
data_dir = Path('/mnt/projects/osat/hyperspectral/data/muskegon_duck_white_lake/2026/09-08')
#data_dir = Path('/mnt/projects/osat/hyperspectral/data/Drone_Work/2026/SeaTrac/08-11')

# calibration and config file
cfg_file = './config.yaml' 
#caldir = '/mnt/projects/osat/hyperspectral/calibration/2026/pikaL_uncrewed/'
caldir = '/mnt/projects/osat/hyperspectral/calibration/2026/pikaL_crewed/'
#caldir = '/mnt/projects/osat/hyperspectral/calibration/2022'

# Parameters
n_pix = 900
camera_fov = np.radians(24.8)
#camera_fov = np.radians(12.0)
#altitude_correction = -227  # Subtract Lake Winnebago elevation to get true altitude
altitude_correction = -176  # Subtract Lake Michigan
#altitude_correction = -274  # Subtract Manchesters elevation
#altitude_correction = -174  # Subtract Lake Erie
euler_imu_from_camera = np.array([0.0, 0.0, 0.0])  # Assume zero offset between IMU and camera

# ====================================================================================================
# ====================================================================================================
# ====================================================================================================

pipeline = Pipeline(data_dir, calibration_dir=caldir, config_file_path=cfg_file)


if run_pipeline:
    print('running main pipeline...')
    pipeline.run()


if run_orthorect:
    tif_dir = data_dir / '2c_reflectance_geotiff'
    tif_dir.mkdir(exist_ok=True)
    print('running orthorectifier...')
    # Create orthorectifier
    orthorectifier = FlatTerrainOrthorectifier(
        camera_cross_track_fov=camera_fov,
        camera_cross_track_n_pixels=n_pix,
        euler_imu_from_camera=euler_imu_from_camera,
        camera_altitude_correction=altitude_correction,
        imu_roll_direction_is_right_wing_down=False,  # LCF file convention (non-standard)
    )

    # Process each image - RGB or all Bands
    #for hyspec_im_path, imu_data_path in zip(pipeline.rad_gc_im_paths, pipeline.imu_data_paths):

    if(flip_process): # process BAAAACKWARDZ
        pipeline.refl_gc_im_paths = reversed(pipeline.refl_gc_im_paths)
        pipeline.imu_data_paths = reversed(pipeline.imu_data_paths)
    
    for hyspec_im_path, imu_data_path in zip(pipeline.refl_gc_im_paths, pipeline.imu_data_paths): 
    #for hyspec_im_path, imu_data_path in zip(pipeline.refl_im_paths, pipeline.imu_data_paths): # non-glint corrected
        geotiff_path = tif_dir / (hyspec_im_path.name.split(".")[0] + "_orthorect.tif")
        if geotiff_path.exists():
            print(f'{geotiff_path} already processed; skipping...')
            continue
        
        if not ( imu_data_path.exists() and hyspec_im_path.exists() ):
            print('file not found; skipping...')
            continue

        print(f"Processing image {hyspec_im_path.name}...", end='')
        #rgb_geotiff_path = output_dir / (hyspec_im_path.name.split(".")[0] + "_orthorect_rgb.tif")
        #orthorectifier.orthorectify_image_file(hyspec_im_path, imu_data_path, rgb_geotiff_path, rgb_only=True)
        try: 
            orthorectifier.orthorectify_image_file(hyspec_im_path, imu_data_path, geotiff_path, rgb_only=False)
            print('✓')

        except:
            print('failed')
       # except: 
            #print('failed. (trying RGB only)')
            #orthorectifier.orthorectify_image_file( hyspec_im_path, imu_data_path, rgb_geotiff_path, rgb_only=True)
