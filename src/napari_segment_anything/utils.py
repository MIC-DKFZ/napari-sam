import urllib.request
from pathlib import Path
import os
import os.path
from os.path import join

SAM_WEIGHTS_URL = {
    "default": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


def _report_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    percent = downloaded * 100 / total_size
    downloaded_mb = downloaded / 1024 / 1024
    total_size_mb = total_size / 1024 / 1024
    print(
        f"Download progress: {percent:.1f}% ({downloaded_mb:.1f}/{total_size_mb:.1f} MB)",
        end="\r",
    )


def get_weights_path(model_type: str) -> Path:
    """Returns the path to the weight of a given model architecture."""
    weight_url = SAM_WEIGHTS_URL[model_type]

    cache_dir = Path.home() / ".cache/napari-segment-anything"
    cache_dir.mkdir(parents=True, exist_ok=True)

    weight_path = cache_dir / weight_url.split("/")[-1]

    # Download the weights if they don't exist
    if not weight_path.exists():
        print(f"Downloading {weight_url} to {weight_path} ...")
        urllib.request.urlretrieve(
            weight_url, str(weight_path), reporthook=_report_hook
        )
        print("\rDownload complete.                           ")

    return weight_path


def get_cached_weight_types(model_types):
    cached_weight_types = {}
    cache_dir = str(Path.home() / ".cache/napari-segment-anything")

    for model_type in model_types:
        model_type_name = os.path.basename(SAM_WEIGHTS_URL[model_type])
        if os.path.isfile(join(cache_dir, model_type_name)):
            cached_weight_types[model_type] = True
        else:
            cached_weight_types[model_type] = False

    return cached_weight_types