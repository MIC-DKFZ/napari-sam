from napari import Viewer
import napari
from src.napari_sam._widget import SamWidget
import SimpleITK as sitk
import numpy as np
from PIL import Image

viewer = Viewer()
viewer.window.add_dock_widget(SamWidget(viewer))

# image = np.array(Image.open("/home/k539i/Documents/syncthing-DKFZ/SAM/cats_raw.jpg"))
image = sitk.GetArrayFromImage(sitk.ReadImage("/home/k539i/Downloads/A2E3W4_0000_0000.nii.gz"))
# image = image[:10, :, :]
image = np.array(image)
layer_1 = viewer.add_image(image)
# layer_1.contrast_limits = (0, 0.075)
viewer.add_labels(np.zeros(image.shape, dtype=np.int), name="labels")

napari.run()