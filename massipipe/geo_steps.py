import logging
from pathlib import Path
from typing import Annotated, Tuple, Any

from pipefunc import pipefunc

from .config import RuntimeContext
from .georeferencing import ImuDataParser

logger = logging.getLogger(__name__)

@pipefunc(output_name="imu_data_parser")
def assemble_rad_converter() -> ImuDataParser:
    return ImuDataParser()

"""(lcf, times, imu_data)"""
ImuDataPacket = Tuple[Path, Path, Path]

@pipefunc(output_name="imu_data_packet")
def pack_imu_data_packets(runtime_context: RuntimeContext) -> Tuple[ImuDataPacket]:
    return tuple(zip (
        runtime_context.product_paths.lcf,
        runtime_context.product_paths.times,
        runtime_context.product_paths.imu_data
    ))

@pipefunc(output_name="imu_processed", mapspec="imu_data_packet[i] -> imu_processed[i]")
def process_imu_data_pack(
        runtime_context: RuntimeContext,
        imu_data_parser: ImuDataParser,
        imu_data_packet: ImuDataPacket
) -> None:
    lcf_path, times_path, imu_data_path = imu_data_packet

    if imu_data_path.exists() and not runtime_context.config.massipipe_options.imu_data.overwrite:
        logger.info(f"Image {imu_data_path.name} exists - skipping.")

    logger.info(f"Processing IMU data from {lcf_path.name}")
    try:
        imu_data_parser.read_and_save_imu_data(lcf_path, times_path, imu_data_path)
    except Exception as _:
        logger.exception(f"Error processing IMU data from {lcf_path.name}")
        raise

@pipefunc(output_name="imu_processed_gate")
def imu_processed_gate(imu_processed: list[Any]) -> None:
    return None

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
