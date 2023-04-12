from napari import Viewer
import napari
from src.napari_sam._widget import SamWidget
import SimpleITK as sitk
import numpy as np

viewer = Viewer()
viewer.window.add_dock_widget(SamWidget(viewer))

# image = sitk.GetArrayFromImage(sitk.ReadImage("D:/Datasets/DKFZ/AFK_M1f_cropped.nii.gz"))
# image = image[:10, :, :]
# image = np.array(image)
# layer_1 = viewer.add_image(image)
# layer_1.contrast_limits = (0, 0.075)
# viewer.add_labels(np.zeros(image.shape, dtype=np.int), name="labels")

napari.run()