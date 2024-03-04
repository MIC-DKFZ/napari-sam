from napari._vispy.overlays.base import (
    LayerOverlayMixin,
    VispyBaseOverlay,
)
from napari._vispy.overlays.labels_polygon import _only_when_enabled
from napari._vispy.utils.visual import overlay_to_visual
from napari._vispy.visuals.image import Image as ImageNode
from vispy.visuals.transforms import TransformSystem
from vispy.visuals.transforms import (
    STTransform,
)

from napari.components.overlays import SceneOverlay
from napari.layers import Layer
from napari.utils.events import Event

import napari
import numpy as np
from time import time
from copy import copy
import warnings

from typing import Tuple

MIN_TIME_S: float = 0.08


class ImageOverlay(SceneOverlay):
    enabled: bool = False


class VispyImageOverlay(LayerOverlayMixin, VispyBaseOverlay):
    def __init__(
        self, *, layer: Layer, overlay: SceneOverlay, parent=None
    ) -> None:
        self.node = ImageNode((None), method="auto", texture_format=None)
        super().__init__(
            node=self.node, layer=layer, overlay=overlay, parent=None
        )
        self.node.visible = True
        self.widget = None
        self.prev_t = time()

        #self.layer.mouse_move_callbacks.append(self._on_mouse_move)
        self.reset()

    def _add_widget(self, widget) -> None:
        self.widget = widget

    def _get_cropped_mask(
        self, whole_mask: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Draw bbox round whole mask and crop to it. Return offset as well to translate node later."""
        y_nonzero, x_nonzero = np.nonzero(whole_mask)
        # in case of no response from SAM mask, draw nothing
        if len(y_nonzero) == 0 or len(x_nonzero) == 0:
            return np.zeros((10, 10)), (0, 0)

        x_min, x_max = np.amin(x_nonzero), np.amax(x_nonzero)
        y_min, y_max = np.amin(y_nonzero), np.amax(y_nonzero)
        return whole_mask[y_min:y_max, x_min:x_max], (x_min, y_min)  # type: ignore

    def _update_img_from_mask(
        self, mask: np.ndarray, offset: Tuple[int, int], color: Tuple[int, int, int, int] = (255, 0, 0, 100)
    ) -> None:
        mask = np.expand_dims(mask, -1)
        cmapped = np.where(mask == 1, color, (0, 0, 0, 0)).astype(
            np.uint8
        )
        self.node.set_data(cmapped)
        x, y = offset
        self.node.transform = STTransform(translate=[x, y])
        self.node.update()
    
    def remove_current(self) -> None:
        self.draw_mask(np.zeros((4, 4)), (0, 0, 0, 0))
    
    def draw_mask(self, mask: np.ndarray, color: Tuple[int, int, int, int] = (255, 0, 0, 100)) -> None:
        cropped_mask, offset = self._get_cropped_mask(mask)
        self._update_img_from_mask(cropped_mask, offset, color)

    def _on_mouse_move(self, event: Event) -> None:
        """If enough time passed, request a SAM mask from our widget, crop it and draw to the overlay"""
        current_t = time()
        enough_time_passed = (current_t - self.prev_t) > MIN_TIME_S
        if self.widget is None:
            return
        img_set = self.widget.img_set
        if not enough_time_passed or not img_set:
            return
        y, x = int(event.value[0]), int(event.value[1]) # napari events are (y, x)
        whole_mask: np.ndarray = self.widget._live_sam_prompt(x, y)
        cropped_mask, offset = self._get_cropped_mask(whole_mask)
        self._update_img_from_mask(cropped_mask, offset)
        self.prev_t = current_t



def add_custom_overlay(
    layer: Layer, viewer: napari.Viewer
) -> Tuple[ImageOverlay, VispyImageOverlay]:
    """Init live SAM overlay and add it to the given layer. This involves creating
    custom overlay, an associated custom vispy overlay and updating the relevant
    overlay -> visual mappings in the layer and the viewer. We then need to copy
    all the transforms that bleong to the vispy overlay node's parent and apply
    them to the vispy node to prevent an error on startup and ensure the node 
    is aligned as the canvas is transformed. 

    :param layer: napari layer we want the live overlay on, usually the image layer
    :type layer: Layer
    :param viewer: parent viewer
    :type viewer: napari.Viewer
    :return: the overlay model (napari layer) and associated vispy overlay
    :rtype: Tuple[ImageOverlay, VispyImageOverlay]
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        vispy_layer = viewer.window._qt_viewer.canvas.layer_to_visual[layer]
        custom_overlay_model = ImageOverlay()
        custom_overlay_visual = VispyImageOverlay(
            layer=layer, overlay=custom_overlay_model, parent=viewer
        )
        vispy_layer.overlays[custom_overlay_model] = custom_overlay_visual
        viewer.window._qt_viewer.canvas._overlay_to_visual[
            custom_overlay_model
        ] = custom_overlay_visual
        layer._overlays["live_SAM"] = custom_overlay_model
        custom_overlay_visual.node.parent = vispy_layer.node
        custom_overlay_model.enabled = True
        custom_overlay_model.visible = True
        custom_overlay_visual.reset() # this is necessary
        vispy_layer._on_matrix_change()

        # per attribute transform copying - just using deepcopy or copy on the TransformSystem
        # itself doesn't work (canvas will follow the node transforms)
        p_sys: TransformSystem = custom_overlay_visual.node.parent.transforms
        c_sys = TransformSystem(p_sys.canvas)
        c_sys.canvas_transform = copy(p_sys.canvas_transform)
        c_sys.scene_transform = copy(p_sys.scene_transform)
        c_sys.visual_transform = copy(p_sys.visual_transform)
        c_sys.dpi = p_sys.dpi
        c_sys.framebuffer_transform = copy(p_sys.framebuffer_transform)
        c_sys.document_transform = copy(p_sys.document_transform)
        custom_overlay_visual.node.transforms = c_sys
    return custom_overlay_model, custom_overlay_visual

# when we add the custom overlay, this will trigger an event in the layer that tries to add a vispy overlay from
# the overlay_to_visual dict, which will fail for our custom overlay unless we modify it at runtime
overlay_to_visual[ImageOverlay] = VispyImageOverlay
