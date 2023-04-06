from napari import Viewer
import napari
from src.napari_sam._widget import SamWidget
from PIL import Image
import numpy as np

viewer = Viewer()
viewer.window.add_dock_widget(SamWidget(viewer))

image = Image.open("C:/Users/Cookie/Downloads/Unbenannt.PNG")
image = np.array(image)
layer_1 = viewer.add_image(image)
viewer.add_labels(np.zeros_like(image), name="labels")

napari.run()