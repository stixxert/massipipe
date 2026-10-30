import logging
from pathlib import Path

from pipefunc import Pipeline, pipefunc

from massipipe import QuickLookProcessor
from massipipe.config import RuntimeContext

logger = logging.getLogger(__name__)

@pipefunc(output_name="quicklook_dir_gate")
def ensure_quicklook_dir(runtime_context: RuntimeContext) -> None:
    runtime_context.base_product_paths.quicklook.mkdir(exist_ok=True)

@pipefunc(output_name="quicklook_processor")
def assemble_quicklook_processor(runtime_context: RuntimeContext) -> QuickLookProcessor:
    _config = runtime_context.config.massipipe_options

    return QuickLookProcessor(
        rgb_wl=_config.general.rgb_wl,
        percentiles=_config.quicklook.percentiles
    )

@pipefunc(output_name="hsi_path", mapspec="... -> hsi_path[i]")
def get_hsi_paths(runtime_context: RuntimeContext) -> list[Path]:
    """Determine whether raw or radiance images are used for quicklook"""

    if runtime_context.base_product_paths.raw.exists():
        logger.info(f"Creating quicklook images from raw images")

        return runtime_context.product_paths.raw
    else:
        logger.info(f"Creating quicklook images from radiance images")

        return runtime_context.product_paths.radiance

@pipefunc(output_name="quicklook_path", mapspec="... -> quicklook_path[i]")
def get_quicklook_paths(runtime_context: RuntimeContext) -> list[Path]:
    return runtime_context.product_paths.quicklook

@pipefunc(output_name="should_process_img",
          mapspec="quicklook_path[i], overwrite -> should_process_img[i]")
def should_process_quicklook_img(quicklook_image_path: Path, overwrite: bool) -> bool:
    return quicklook_image_path.exists() and not overwrite

@pipefunc(output_name="quicklook_img_gate",
          mapspec="hsi_path[i], quicklook_path[i], should_process_img[i], quicklook_processor"
                  " -> "
                  "quicklook_img_gate[i]")
def process_quicklook_img(hsi_path: Path, quicklook_path: Path, should_process_img: bool,
                          quicklook_processor: QuickLookProcessor) -> None:
    if not should_process_img:
        logger.info(f"Image {quicklook_path.name} exists - skipping.")
        return

    logger.info(f"Creating quicklook version of {hsi_path.name}")

    try:
        quicklook_processor.create_quicklook_image_file(hsi_path, quicklook_path)
    except Exception as _:
        logger.exception(f"Error creating quicklook for {hsi_path.name}")
        raise



def build_quicklook_export_pipeline(runtime_context: RuntimeContext) -> Pipeline:
    return Pipeline([
        ensure_quicklook_dir,
        assemble_quicklook_processor,
        get_quicklook_paths,
        get_hsi_paths,
        should_process_quicklook_img,
        process_quicklook_img,
    ])