import urllib.request
from pathlib import Path
import os
import os.path
from os.path import join
from tqdm import tqdm

SAM_WEIGHTS_URL = {
    "default": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


def download_with_progress(url, output_file):
    """
    Download a file from a URL to a local file, showing a progress bar.

    Args:
        url (str): The URL to download from.
        output_file (str): The path to save the downloaded file.

    Returns:
        None
    """
    # Open the URL and get the content length
    req = urllib.request.urlopen(url)
    content_length = int(req.headers.get('Content-Length'))

    # Set up the progress bar
    progress_bar = tqdm(total=content_length, unit='B', unit_scale=True)

    # Download the file and update the progress bar
    with open(output_file, 'wb') as f:
        downloaded_bytes = 0
        while True:
            buffer = req.read(8192)
            if not buffer:
                break
            downloaded_bytes += len(buffer)
            f.write(buffer)
            progress_bar.update(len(buffer))

    # Close the progress bar and the URL
    progress_bar.close()
    req.close()


def get_weights_path(model_type):
    """
    Get the path to the cached weight file for a given model type.
    If the file is not cached, download it from the corresponding URL.

    Args:
        model_type (str): The type of the model to get weights for.

    Returns:
        Path: The path to the cached or downloaded weight file.
    """
    weight_url = SAM_WEIGHTS_URL[model_type]

    cache_dir = Path.home() / ".cache/napari-segment-anything"
    cache_dir.mkdir(parents=True, exist_ok=True)

    weight_path = cache_dir / weight_url.split("/")[-1]

    if not weight_path.exists():
        print("Downloading {} to {} ...".format(weight_url, weight_path))
        download_with_progress(weight_url, weight_path)

    return weight_path


def get_cached_weight_types(model_types):
    """
    Check if the cached weight files for the given model types exist.

    Args:
        model_types (List[str]): A list of model types to check.

    Returns:
        Dict[str, bool]: A dictionary mapping model types to a boolean indicating
        whether their cached weight file exists.
    """
    cached_weight_types = {}
    cache_dir = str(Path.home() / ".cache/napari-segment-anything")

    for model_type in model_types:
        model_type_name = os.path.basename(SAM_WEIGHTS_URL[model_type])
        if os.path.isfile(join(cache_dir, model_type_name)):
            cached_weight_types[model_type] = True
        else:
            cached_weight_types[model_type] = False

    return cached_weight_types
