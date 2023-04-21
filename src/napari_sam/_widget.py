from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget, QLabel, QComboBox, QRadioButton, QGroupBox, QProgressBar, QApplication, QScrollArea, QLineEdit, QSpacerItem, QSizePolicy
from qtpy.QtGui import QIntValidator, QDoubleValidator
# from napari_sam.QCollapsibleBox import QCollapsibleBox
from qtpy import QtCore
from qtpy.QtCore import Qt
import napari
import numpy as np
from enum import Enum
from collections import deque, defaultdict
import inspect
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from napari_sam.utils import get_weights_path, get_cached_weight_types, normalize
import torch
from vispy.util.keys import CONTROL
import copy
import warnings
from tqdm import tqdm
from superqt.utils import qdebounced


class AnnotatorMode(Enum):
    NONE = 0
    CLICK = 1
    BBOX = 2
    AUTO = 3


class SegmentationMode(Enum):
    SEMANTIC = 0
    INSTANCE = 1


class SamWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.annotator_mode = AnnotatorMode.NONE
        self.segmentation_mode = SegmentationMode.SEMANTIC

        if not torch.cuda.is_available():
            if not torch.backends.mps.is_available():
                self.device = "cpu"
            else:
                self.device = "mps"
        else:
            self.device = "cuda"

        main_layout = QVBoxLayout()

        # self.scroll_area = QScrollArea()
        # self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        # self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        # self.scroll_area.setWidgetResizable(True)

        self.layer_types = {"image": napari.layers.image.image.Image, "labels": napari.layers.labels.labels.Labels}

        l_model_type = QLabel("Select model type:")
        main_layout.addWidget(l_model_type)

        self.cb_model_type = QComboBox()
        main_layout.addWidget(self.cb_model_type)

        self.btn_load_model = QPushButton("Load model")
        self.btn_load_model.clicked.connect(self._load_model)
        main_layout.addWidget(self.btn_load_model)
        self.is_model_loaded = False
        self.init_model_type_combobox()

        l_image_layer = QLabel("Select input image layer:")
        main_layout.addWidget(l_image_layer)

        self.cb_image_layers = QComboBox()
        self.cb_image_layers.addItems(self.get_layer_names("image"))
        self.cb_image_layers.currentTextChanged.connect(self.on_image_change)
        main_layout.addWidget(self.cb_image_layers)

        l_label_layer = QLabel("Select output labels layer:")
        main_layout.addWidget(l_label_layer)

        self.cb_label_layers = QComboBox()
        self.cb_label_layers.addItems(self.get_layer_names("labels"))
        main_layout.addWidget(self.cb_label_layers)

        self.comboboxes = [{"combobox": self.cb_image_layers, "layer_type": "image"}, {"combobox": self.cb_label_layers, "layer_type": "labels"}]

        self.g_annotation = QGroupBox("Annotation mode")
        self.l_annotation = QVBoxLayout()

        self.rb_click = QRadioButton("Click")
        self.rb_click.setChecked(True)
        self.rb_click.setToolTip("Positive Click: Middle Mouse Button\n \n"
                                 "Negative Click: Control + Middle Mouse Button \n \n"
                                 "Undo: Control + Z \n \n"
                                 "Select Point: Left Click \n \n"
                                 "Delete Selected Point: Delete")
        self.l_annotation.addWidget(self.rb_click)
        self.rb_click.clicked.connect(self.on_everything_mode_checked)

        self.rb_bbox = QRadioButton("Bounding Box (WIP)")
        self.rb_bbox.setEnabled(False)
        self.rb_bbox.setToolTip("This mode is still Work In Progress (WIP)")
        self.rb_bbox.setStyleSheet("color: gray")
        self.l_annotation.addWidget(self.rb_bbox)

        self.rb_auto = QRadioButton("Everything")
        # self.rb_auto.setEnabled(False)
        # self.rb_auto.setStyleSheet("color: gray")
        self.rb_auto.setToolTip("Creates automatically an instance segmentation \n"
                                            "of the entire image.\n"
                                            "No user interaction possible.")
        self.l_annotation.addWidget(self.rb_auto)
        self.rb_auto.clicked.connect(self.on_everything_mode_checked)

        self.g_annotation.setLayout(self.l_annotation)
        main_layout.addWidget(self.g_annotation)

        self.g_segmentation = QGroupBox("Segmentation mode")
        self.l_segmentation = QVBoxLayout()

        self.rb_semantic = QRadioButton("Semantic")
        self.rb_semantic.setChecked(True)
        self.rb_semantic.setToolTip("Enables the user to create a \n"
                                 "multi-label (semantic) segmentation of different classes.\n \n"
                                 "All objects from the same class \n"
                                 "should be given the same label by the user.\n \n"
                                 "The current label can be changed by the user \n"
                                 "on the labels layer pane after selecting the labels layer.")
        # self.rb_semantic.setEnabled(False)
        # self.rb_semantic.setStyleSheet("color: gray")
        self.l_segmentation.addWidget(self.rb_semantic)

        self.rb_instance = QRadioButton("Instance")
        self.rb_instance.setToolTip("Enables the user to create an \n"
                                 "instance segmentation of different objects.\n \n"
                                 "Objects can be from the same or different classes,\n"
                                 "but each object should be given a unique label by the user. \n \n"
                                 "The current label can be changed by the user \n"
                                 "on the labels layer pane after selecting the labels layer.")
        # self.rb_instance.setEnabled(False)
        # self.rb_instance.setStyleSheet("color: gray")
        self.l_segmentation.addWidget(self.rb_instance)

        self.g_segmentation.setLayout(self.l_segmentation)
        main_layout.addWidget(self.g_segmentation)

        self.btn_activate = QPushButton("Activate")
        self.btn_activate.clicked.connect(self._activate)
        self.btn_activate.setEnabled(False)
        self.is_active = False
        main_layout.addWidget(self.btn_activate)

        container_widget_info = QWidget()
        container_layout_info = QVBoxLayout(container_widget_info)

        self.g_info_tooltip = QGroupBox("Tooltip Information")
        self.l_info_tooltip = QVBoxLayout()
        self.label_info_tooltip = QLabel("Every mode shows further information when hovered over.")
        self.label_info_tooltip.setWordWrap(True)
        self.l_info_tooltip.addWidget(self.label_info_tooltip)
        self.g_info_tooltip.setLayout(self.l_info_tooltip)
        container_layout_info.addWidget(self.g_info_tooltip)

        self.g_info_contrast = QGroupBox("Contrast Limits")
        self.l_info_contrast = QVBoxLayout()
        self.label_info_contrast = QLabel("SAM computes its image embedding based on the current image contrast.\n"
                                          "Image contrast can be adjusted with the contrast slider of the image layer.")
        self.label_info_contrast.setWordWrap(True)
        self.l_info_contrast.addWidget(self.label_info_contrast)
        self.g_info_contrast.setLayout(self.l_info_contrast)
        container_layout_info.addWidget(self.g_info_contrast)

        self.g_info_click = QGroupBox("Click Mode")
        self.l_info_click = QVBoxLayout()
        self.label_info_click = QLabel("Positive Click: Middle Mouse Button\n \n"
                                 "Negative Click: Control + Middle Mouse Button\n \n"
                                 "Undo: Control + Z\n \n"
                                 "Select Point: Left Click\n \n"
                                 "Delete Selected Point: Delete\n \n")
        self.label_info_click.setWordWrap(True)
        self.l_info_click.addWidget(self.label_info_click)
        self.g_info_click.setLayout(self.l_info_click)
        container_layout_info.addWidget(self.g_info_click)

        scroll_area_info = QScrollArea()
        scroll_area_info.setWidget(container_widget_info)
        scroll_area_info.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        main_layout.addWidget(scroll_area_info)

        self.scroll_area_auto = self.init_auto_mode_settings()
        main_layout.addWidget(self.scroll_area_auto)

        self.setLayout(main_layout)

        self.image_name = None
        self.image_layer = None
        self.label_layer = None
        self.label_layer_changes = None
        self.label_color_mapping = None
        self.points_layer = None
        self.points_layer_name = "Ignore this layer"  # "Ignore this layer <hidden>"
        self.old_points = np.zeros(0)
        self.point_size = 10

        self.init_comboboxes()

        self.sam_model = None
        self.sam_predictor = None
        self.sam_logits = None
        self.sam_features = None

        self.points = defaultdict(list)
        self.point_label = None

        # self.viewer.window.qt_viewer.layers.model().filterAcceptsRow = self._myfilter

    def init_auto_mode_settings(self):
        container_widget_auto = QWidget()
        container_layout_auto = QVBoxLayout(container_widget_auto)

        # self.g_auto_mode_settings = QCollapsibleBox("Everything Mode Settings")
        self.g_auto_mode_settings = QGroupBox("Everything Mode Settings")
        self.l_auto_mode_settings = QVBoxLayout()

        l_points_per_side = QLabel("Points per side:")
        self.l_auto_mode_settings.addWidget(l_points_per_side)
        validator = QIntValidator()
        validator.setRange(0, 9999)
        self.le_points_per_side = QLineEdit()
        self.le_points_per_side.setText("32")
        self.le_points_per_side.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_points_per_side)

        l_points_per_batch = QLabel("Points per batch:")
        self.l_auto_mode_settings.addWidget(l_points_per_batch)
        validator = QIntValidator()
        validator.setRange(0, 9999)
        self.le_points_per_batch = QLineEdit()
        self.le_points_per_batch.setText("64")
        self.le_points_per_batch.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_points_per_batch)

        l_pred_iou_thresh = QLabel("Prediction IoU threshold:")
        self.l_auto_mode_settings.addWidget(l_pred_iou_thresh)
        validator = QDoubleValidator()
        validator.setRange(0.0, 1.0)
        validator.setDecimals(5)
        self.le_pred_iou_thresh = QLineEdit()
        self.le_pred_iou_thresh.setText("0.88")
        self.le_pred_iou_thresh.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_pred_iou_thresh)

        l_stability_score_thresh = QLabel("Stability score threshold:")
        self.l_auto_mode_settings.addWidget(l_stability_score_thresh)
        validator = QDoubleValidator()
        validator.setRange(0.0, 1.0)
        validator.setDecimals(5)
        self.le_stability_score_thresh = QLineEdit()
        self.le_stability_score_thresh.setText("0.95")
        self.le_stability_score_thresh.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_stability_score_thresh)

        l_stability_score_offset = QLabel("Stability score offset:")
        self.l_auto_mode_settings.addWidget(l_stability_score_offset)
        validator = QDoubleValidator()
        validator.setRange(0.0, 1.0)
        validator.setDecimals(5)
        self.le_stability_score_offset = QLineEdit()
        self.le_stability_score_offset.setText("1.0")
        self.le_stability_score_offset.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_stability_score_offset)

        l_box_nms_thresh = QLabel("Box NMS threshold:")
        self.l_auto_mode_settings.addWidget(l_box_nms_thresh)
        validator = QDoubleValidator()
        validator.setRange(0.0, 1.0)
        validator.setDecimals(5)
        self.le_box_nms_thresh = QLineEdit()
        self.le_box_nms_thresh.setText("0.7")
        self.le_box_nms_thresh.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_box_nms_thresh)

        l_crop_n_layers = QLabel("Crop N layers")
        self.l_auto_mode_settings.addWidget(l_crop_n_layers)
        validator = QIntValidator()
        validator.setRange(0, 9999)
        self.le_crop_n_layers = QLineEdit()
        self.le_crop_n_layers.setText("0")
        self.le_crop_n_layers.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_crop_n_layers)

        l_crop_nms_thresh = QLabel("Crop NMS threshold:")
        self.l_auto_mode_settings.addWidget(l_crop_nms_thresh)
        validator = QDoubleValidator()
        validator.setRange(0.0, 1.0)
        validator.setDecimals(5)
        self.le_crop_nms_thresh = QLineEdit()
        self.le_crop_nms_thresh.setText("0.7")
        self.le_crop_nms_thresh.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_crop_nms_thresh)

        l_crop_overlap_ratio = QLabel("Crop overlap ratio:")
        self.l_auto_mode_settings.addWidget(l_crop_overlap_ratio)
        validator = QDoubleValidator()
        validator.setRange(0.0, 1.0)
        validator.setDecimals(5)
        self.le_crop_overlap_ratio = QLineEdit()
        self.le_crop_overlap_ratio.setText("0.3413")
        self.le_crop_overlap_ratio.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_crop_overlap_ratio)

        l_crop_n_points_downscale_factor = QLabel("Crop N points downscale factor")
        self.l_auto_mode_settings.addWidget(l_crop_n_points_downscale_factor)
        validator = QIntValidator()
        validator.setRange(0, 9999)
        self.le_crop_n_points_downscale_factor = QLineEdit()
        self.le_crop_n_points_downscale_factor.setText("1")
        self.le_crop_n_points_downscale_factor.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_crop_n_points_downscale_factor)

        l_min_mask_region_area = QLabel("Min mask region area")
        self.l_auto_mode_settings.addWidget(l_min_mask_region_area)
        validator = QIntValidator()
        validator.setRange(0, 9999)
        self.le_min_mask_region_area = QLineEdit()
        self.le_min_mask_region_area.setText("0")
        self.le_min_mask_region_area.setValidator(validator)
        self.l_auto_mode_settings.addWidget(self.le_min_mask_region_area)

        # self.g_auto_mode_settings.setContentLayout(self.l_auto_mode_settings)
        self.g_auto_mode_settings.setLayout(self.l_auto_mode_settings)
        # main_layout.addWidget(self.g_auto_mode_settings)
        container_layout_auto.addWidget(self.g_auto_mode_settings)

        scroll_area_auto = QScrollArea()
        # scroll_area_info.setWidgetResizable(True)
        scroll_area_auto.setWidget(container_widget_auto)
        # Set the scrollbar policies for the scroll area
        scroll_area_auto.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # scroll_area_info.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll_area_auto.hide()
        return scroll_area_auto

    def on_everything_mode_checked(self):
        if self.rb_auto.isChecked():
            self.rb_semantic.setEnabled(False)
            self.rb_semantic.setChecked(False)
            self.rb_semantic.setStyleSheet("color: gray")
            self.rb_instance.setChecked(True)
            self.scroll_area_auto.show()
            self.btn_activate.setText("Run")
        else:
            self.rb_semantic.setEnabled(True)
            self.rb_semantic.setStyleSheet("")
            self.scroll_area_auto.hide()
            self.btn_activate.setText("Activate")

    def on_image_change(self):
        # image_name = self.cb_image_layers.currentText()
        # if image_name != "" and self.viewer.layers[image_name].ndim > 2:
        #     self.rb_auto.setEnabled(False)
        #     self.rb_auto.setChecked(False)
        #     self.rb_click.setChecked(True)
        #     self.rb_auto.setStyleSheet("color: gray")
        # else:
        #     self.rb_auto.setEnabled(True)
        #     self.rb_auto.setStyleSheet("")
        pass

    def init_model_type_combobox(self):
        model_types = list(sam_model_registry.keys())
        cached_weight_types = get_cached_weight_types(model_types)
        entries = []
        for name, is_cached in cached_weight_types.items():
            if is_cached:
                entries.append("{} (Cached)".format(name))
            else:
                entries.append("{} (Auto-Download)".format(name))
        self.cb_model_type.addItems(entries)

        if cached_weight_types[list(cached_weight_types.keys())[self.cb_model_type.currentIndex()]]:
            self.btn_load_model.setText("Load model")
        else:
            self.btn_load_model.setText("Download and load model")

        self.cb_model_type.currentTextChanged.connect(self.on_model_type_combobox_change)

    def on_model_type_combobox_change(self):
        model_types = list(sam_model_registry.keys())
        cached_weight_types = get_cached_weight_types(model_types)

        if cached_weight_types[list(cached_weight_types.keys())[self.cb_model_type.currentIndex()]]:
            self.btn_load_model.setText("Load model")
        else:
            self.btn_load_model.setText("Download and load model")

    def init_comboboxes(self):
        for combobox_dict in self.comboboxes:
            # If current active layer is of the same type of layer that the combobox accepts then set it as selected layer in the combobox.
            active_layer = self.viewer.layers.selection.active
            if combobox_dict["layer_type"] == "all" or isinstance(active_layer, self.layer_types[combobox_dict["layer_type"]]):
                index = combobox_dict["combobox"].findText(active_layer.name, QtCore.Qt.MatchFixedString)
                if index >= 0:
                    combobox_dict["combobox"].setCurrentIndex(index)

        # Inform all comboboxes on layer changes with the viewer.layer_change event
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.viewer.events.layers_change.connect(self._on_layers_changed)

        # viewer.layer_change event does not inform about layer name changes, so we have to register a separate event to each layer and each layer that will be created

        # Register an event to all existing layers
        for layer_name in self.get_layer_names():
            layer = self.viewer.layers[layer_name]

            @layer.events.name.connect
            def _on_rename(name_event):
                self._on_layers_changed()

        # Register an event to all layers that will be created
        @self.viewer.layers.events.inserted.connect
        def _on_insert(event):
            layer = event.value

            @layer.events.name.connect
            def _on_rename(name_event):
                self._on_layers_changed()

        self._init_comboboxes_callback()

    def _on_layers_changed(self):
        for combobox_dict in self.comboboxes:
            layer = combobox_dict["combobox"].currentText()
            layers = self.get_layer_names(combobox_dict["layer_type"])
            combobox_dict["combobox"].clear()
            combobox_dict["combobox"].addItems(layers)
            index = combobox_dict["combobox"].findText(layer, QtCore.Qt.MatchFixedString)
            if index >= 0:
                combobox_dict["combobox"].setCurrentIndex(index)
        self._on_layers_changed_callback()

    def get_layer_names(self, type="all", exclude_hidden=True):
        layers = self.viewer.layers
        filtered_layers = []
        for layer in layers:
            if (type == "all" or isinstance(layer, self.layer_types[type])) and ((not exclude_hidden) or (exclude_hidden and "<hidden>" not in layer.name)):
                filtered_layers.append(layer.name)
        return filtered_layers

    def _init_comboboxes_callback(self):
        self._check_activate_btn()
        self.on_image_change()

    def _on_layers_changed_callback(self):
        self._check_activate_btn()
        if (self.image_layer is not None and self.image_layer not in self.viewer.layers) or (self.label_layer is not None and self.label_layer not in self.viewer.layers):
            self._deactivate()

    def _check_activate_btn(self):
        if self.cb_image_layers.currentText() != "" and self.cb_label_layers.currentText() != "" and self.is_model_loaded:
            self.btn_activate.setEnabled(True)
        else:
            self.btn_activate.setEnabled(False)

    def _load_model(self):
        model_types = list(sam_model_registry.keys())
        model_type = model_types[self.cb_model_type.currentIndex()]
        self.sam_model = sam_model_registry[model_type](
            get_weights_path(model_type)
        )
        self.sam_model.to(self.device)
        self.sam_predictor = SamPredictor(self.sam_model)
        self.is_model_loaded = True
        self._check_activate_btn()

    def _activate(self):
        self.btn_activate.setEnabled(False)
        if not self.is_active:
            self.image_name = self.cb_image_layers.currentText()
            self.image_layer = self.viewer.layers[self.cb_image_layers.currentText()]
            self.label_layer = self.viewer.layers[self.cb_label_layers.currentText()]
            self.label_layer_changes = None

            if self.image_layer.ndim != 2 and self.image_layer.ndim != 3:
                raise RuntimeError("Only 2D and 3D images are supported.")

            if self.image_layer.ndim == 2:
                self.sam_logits = None
            else:
                self.sam_logits = [None] * self.image_layer.data.shape[0]

            if self.rb_click.isChecked():
                self.annotator_mode = AnnotatorMode.CLICK
                self.rb_bbox.setEnabled(False)
                self.rb_auto.setEnabled(False)
                self.rb_bbox.setStyleSheet("color: gray")
                self.rb_auto.setStyleSheet("color: gray")
            elif self.rb_bbox.isChecked():
                self.annotator_mode = AnnotatorMode.BBOX
                self.rb_click.setEnabled(False)
                self.rb_auto.setEnabled(False)
                self.rb_click.setStyleSheet("color: gray")
                self.rb_auto.setStyleSheet("color: gray")
            elif self.rb_auto.isChecked():
                self.annotator_mode = AnnotatorMode.AUTO
                self.rb_click.setEnabled(False)
                self.rb_bbox.setEnabled(False)
                self.rb_click.setStyleSheet("color: gray")
                self.rb_bbox.setStyleSheet("color: gray")
            else:
                raise RuntimeError("Annotator mode not implemented.")

            if self.annotator_mode != AnnotatorMode.AUTO:
                self.is_active = True
                self.btn_activate.setText("Deactivate")
                self.btn_load_model.setEnabled(False)
                self.cb_model_type.setEnabled(False)
                self.cb_image_layers.setEnabled(False)
                self.cb_label_layers.setEnabled(False)

            if self.rb_semantic.isChecked():
                self.segmentation_mode = SegmentationMode.SEMANTIC
                self.rb_instance.setEnabled(False)
                self.rb_instance.setStyleSheet("color: gray")
            elif self.rb_instance.isChecked():
                self.segmentation_mode = SegmentationMode.INSTANCE
                self.rb_semantic.setEnabled(False)
                self.rb_semantic.setStyleSheet("color: gray")
            else:
                raise RuntimeError("Segmentation mode not implemented.")

            if self.annotator_mode == AnnotatorMode.CLICK:
                self.create_label_color_mapping()

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    self._history_limit = self.label_layer._history_limit
                self._reset_history()

                self.image_layer.events.contrast_limits.connect(qdebounced(self.on_contrast_limits_change, timeout=1000))

                self.set_image()
                self.update_points_layer(None)

                self.viewer.mouse_drag_callbacks.append(self.callback_click)
                self.viewer.keymap['Delete'] = self.on_delete
                self.label_layer.keymap['Control-Z'] = self.on_undo
                self.label_layer.keymap['Control-Shift-Z'] = self.on_redo

            elif self.annotator_mode == AnnotatorMode.AUTO:
                self.sam_anything_predictor = SamAutomaticMaskGenerator(self.sam_model,
                                                                        points_per_side=int(self.le_points_per_side.text()),
                                                                        points_per_batch=int(self.le_points_per_batch.text()),
                                                                        pred_iou_thresh=float(self.le_pred_iou_thresh.text()),
                                                                        stability_score_thresh=float(self.le_stability_score_thresh.text()),
                                                                        stability_score_offset=float(self.le_stability_score_offset.text()),
                                                                        box_nms_thresh=float(self.le_box_nms_thresh.text()),
                                                                        crop_n_layers=int(self.le_crop_n_layers.text()),
                                                                        crop_nms_thresh=float(self.le_crop_nms_thresh.text()),
                                                                        crop_overlap_ratio=float(self.le_crop_overlap_ratio.text()),
                                                                        crop_n_points_downscale_factor=int(self.le_crop_n_points_downscale_factor.text()),
                                                                        min_mask_region_area=int(self.le_min_mask_region_area.text()),
                                                                        )
                prediction = self.predict_everything()
                self.label_layer.data = prediction
        else:
            self._deactivate()
        self.btn_activate.setEnabled(True)

    def _deactivate(self):
        self.is_active = False
        self.btn_activate.setText("Activate")
        self.btn_load_model.setEnabled(True)
        self.cb_model_type.setEnabled(True)
        self.cb_image_layers.setEnabled(True)
        self.cb_label_layers.setEnabled(True)
        self.remove_all_widget_callbacks(self.viewer)
        if self.label_layer is not None:
            self.remove_all_widget_callbacks(self.label_layer)
        if self.points_layer is not None and self.points_layer in self.viewer.layers:
            self.viewer.layers.remove(self.points_layer)
        self.image_name = None
        self.image_layer = None
        self.label_layer = None
        self.label_layer_changes = None
        self.points_layer = None
        self.annotator_mode = AnnotatorMode.NONE
        self.points = defaultdict(list)
        self.point_label = None
        self.sam_logits = None
        self.rb_click.setEnabled(True)
        self.rb_auto.setEnabled(True)
        self.rb_click.setStyleSheet("")
        self.rb_auto.setStyleSheet("")
        self.rb_semantic.setEnabled(True)
        self.rb_instance.setEnabled(True)
        self.rb_semantic.setStyleSheet("")
        self.rb_instance.setStyleSheet("")
        self._reset_history()

    def create_label_color_mapping(self, num_labels=1000):
        if self.label_layer is not None:
            self.label_color_mapping = {"label_mapping": {}, "color_mapping": {}}
            for label in range(num_labels):
                color = self.label_layer.get_color(label)
                self.label_color_mapping["label_mapping"][label] = color
                self.label_color_mapping["color_mapping"][str(color)] = label

    def callback_click(self, layer, event):
        if self.annotator_mode == AnnotatorMode.CLICK:
            data_coordinates = self.image_layer.world_to_data(event.position)
            coords = np.round(data_coordinates).astype(int)
            if (not CONTROL in event.modifiers) and event.button == 3:  # Positive middle click
                self.do_click(coords, 1)
                yield
            elif CONTROL in event.modifiers and event.button == 3:  # Negative middle click
                self.do_click(coords, 0)
                yield
            elif event.button == 1 and self.points_layer is not None and len(self.points_layer.data) > 0:
                # Find the closest point to the mouse click
                distances = np.linalg.norm(self.points_layer.data - coords, axis=1)
                closest_point_idx = np.argmin(distances)
                closest_point_distance = distances[closest_point_idx]

                # Select the closest point if it's within self.point_size pixels of the click
                if closest_point_distance <= self.point_size:
                    self.points_layer.selected_data = {closest_point_idx}
                else:
                    self.points_layer.selected_data = set()
                yield

    def on_delete(self, layer):
        selected_points = list(self.points_layer.selected_data)
        if len(selected_points) > 0:
            self.points_layer.data = np.delete(self.points_layer.data, selected_points[0], axis=0)
            self._save_history({"points": copy.deepcopy(self.points), "logits": self.sam_logits, "point_label": self.point_label})
            deleted_point, _ = self.find_changed_point(self.old_points, self.points_layer.data)
            label = self.find_point_label(deleted_point)
            index_to_remove = np.where((self.points[label] == deleted_point).all(axis=1))[0]
            self.points[label] = np.delete(self.points[label], index_to_remove, axis=0).tolist()
            if len(self.points[label]) == 0:
                del self.points[label]
            self.point_label = label
            if self.image_layer.ndim == 2:
                self.sam_logits = None
            elif self.image_layer.ndim == 3:
                self.sam_logits[deleted_point[0]] = None
            else:
                raise RuntimeError("Point deletion not implemented for this dimensionality.")
            self.run(self.points, self.point_label, deleted_point, label)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.label_layer._save_history((self.label_layer_changes["indices"], self.label_layer_changes["old_values"], self.label_layer_changes["new_values"]))

    def on_undo(self, layer):
        """Undo the last paint or fill action since the view slice has changed."""
        self.undo()
        self.label_layer.undo()
        self.label_layer.data = self.label_layer.data

    def on_redo(self, layer):
        """Redo any previously undone actions."""
        self.redo()
        self.label_layer.redo()
        self.label_layer.data = self.label_layer.data

    def on_contrast_limits_change(self):
        self.set_image()

    def set_image(self):
        if self.image_layer.ndim == 2:
            image = np.asarray(self.image_layer.data)
            if not self.image_layer.rgb:
                image = np.stack((image,)*3, axis=-1)  # Expand to 3-channel image
            image = image[..., :3]  # Remove a potential alpha channel
            self.sam_predictor.set_image(image)
            self.sam_features = self.sam_predictor.features
        elif self.image_layer.ndim == 3:
            l_creating_features= QLabel("Creating SAM image embedding:")
            self.layout().addWidget(l_creating_features)
            progress_bar = QProgressBar(self)
            progress_bar.setMaximum(self.image_layer.data.shape[0])
            progress_bar.setValue(0)
            self.layout().addWidget(progress_bar)
            self.sam_features = []
            for index in tqdm(range(self.image_layer.data.shape[0]), desc="Creating SAM image embedding"):
                image_slice = np.asarray(self.image_layer.data[index, ...])
                if not self.image_layer.rgb:
                    image_slice = np.stack((image_slice,) * 3, axis=-1)  # Expand to 3-channel image
                image_slice = image_slice[..., :3]  # Remove a potential alpha channel
                contrast_limits = self.image_layer.contrast_limits
                image_slice = normalize(image_slice, source_limits=contrast_limits, target_limits=(0, 255)).astype(np.uint8)
                self.sam_predictor.set_image(image_slice)
                self.sam_features.append(self.sam_predictor.features)
                progress_bar.setValue(index+1)
                QApplication.processEvents()
                progress_bar.deleteLater()
                l_creating_features.deleteLater()
        else:
            raise RuntimeError("Only 2D and 3D images are supported.")

    def do_click(self, coords, is_positive):
        # Check if there is already a point at these coordinates
        for label, points in self.points.items():
            if np.any((coords == points).all(1)):
                warnings.warn("There is already a point in this location. This click will be ignored.")
                return

        self._save_history({"points": copy.deepcopy(self.points), "logits": self.sam_logits, "point_label": self.point_label})

        self.point_label = self.label_layer.selected_label
        if not is_positive:
            self.point_label = 0

        self.points[self.point_label].append(coords)

        self.run(self.points, self.point_label, coords, self.point_label)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.label_layer._save_history((self.label_layer_changes["indices"], self.label_layer_changes["old_values"], self.label_layer_changes["new_values"]))

    def run(self, points, point_label, current_point, current_label):
        self.update_points_layer(points)

        if points:
            points_flattened = []
            labels_flattended = []
            for label, label_points in points.items():
                points_flattened.extend(label_points)
                label = int(label == point_label)
                labels = [label] * len(label_points)
                labels_flattended.extend(labels)

            prediction, predicted_slices = self.predict_click(points_flattened, labels_flattended, current_point, current_label)
        else:
            prediction = np.zeros_like(self.label_layer.data)
            predicted_slices = slice(None, None)

        if prediction is not None:
            label_layer = np.asarray(self.label_layer.data)
            changed_indices = np.where(prediction == 1)
            index_labels_old = label_layer[changed_indices]
            label_layer[predicted_slices][label_layer[predicted_slices] == point_label] = 0
            if self.segmentation_mode == SegmentationMode.SEMANTIC or point_label == 0:
                label_layer[prediction == 1] = point_label
            else:
                label_layer[(prediction == 1) & (label_layer == 0)] = point_label
            index_labels_new = label_layer[changed_indices]
            self.label_layer_changes = {"indices": changed_indices, "old_values": index_labels_old, "new_values": index_labels_new}
            self.label_layer.data = label_layer
            self.old_points = copy.deepcopy(self.points_layer.data)
            # self.label_layer.refresh()

    def predict_click(self, points, labels, current_point, current_label):
        points = np.asarray(points)
        if current_point is not None:
            if self.image_layer.ndim == 2:
                self.sam_predictor.features = self.sam_features
                prediction, _, self.sam_logits = self.sam_predictor.predict(
                    point_coords=np.flip(points, axis=-1),
                    point_labels=np.asarray(labels),
                    mask_input=self.sam_logits,
                    multimask_output=False,
                )
                prediction = prediction[0]
                predicted_slices = None
            elif self.image_layer.ndim == 3:
                x_coords = np.unique(points[:, 0])
                groups = {x_coord: list(points[points[:, 0] == x_coord]) for x_coord in x_coords}  # Group points if they are on the same image slice
                x_coord = current_point[0]
                prediction = np.zeros_like(self.label_layer.data)

                group_points = groups[x_coord]
                group_labels = [labels[np.argwhere(np.all(points == point, axis=1)).flatten()[0]] for point in group_points]
                group_points = [point[1:] for point in group_points]
                self.sam_predictor.features = self.sam_features[x_coord]
                prediction_yz, _, self.sam_logits[x_coord] = self.sam_predictor.predict(
                    point_coords=np.flip(group_points, axis=-1),
                    point_labels=np.asarray(group_labels),
                    mask_input=self.sam_logits[x_coord],
                    multimask_output=False,
                )
                prediction_yz = prediction_yz[0]
                prediction[x_coord, :, :] = prediction_yz
                predicted_slices = x_coord
            # elif self.image_layer.ndim == 3:
            #     z_coords = np.unique(points[:, 2])
            #     groups = {x_coord: list(points[points[:, 2] == x_coord]) for x_coord in z_coords}  # Group points if they are on the same image slice
            #     image_point_proposals, image_label_proposals = [], []
            #
            #     for x_coord, group_points in groups.items():
            #         group_labels = [labels[np.argwhere(np.all(points == point, axis=1)).flatten()[0]] for point in group_points]
            #         group_points = [point[:2] for point in group_points]
            #         self.sam_predictor.features = self.sam_features[x_coord - 1]
            #         prediction_yz, _, _ = self.sam_predictor.predict(
            #             point_coords=np.flip(group_points, axis=-1),
            #             point_labels=np.asarray(group_labels),
            #             mask_input=self.sam_logits,
            #             multimask_output=False,
            #         )
            #         prediction_yz = prediction_yz[0]
            #
            #         for i, point in enumerate(group_points):
            #             y_coord = point[1]
            #             prediction_x = prediction_yz[:, y_coord]
            #             point_proposals_x = np.asarray(list(zip(*np.where(prediction_x)))).flatten()
            #             point_proposals = [(point_proposal_x, y_coord, x_coord) for point_proposal_x in point_proposals_x]
            #             image_point_proposals.extend(point_proposals)
            #             image_label_proposals.extend([group_labels[i]] * len(point_proposals))
            #
            #     image_point_proposals = np.asarray(image_point_proposals)
            #     image_label_proposals = np.asarray(image_label_proposals)
            #     z_coords = np.unique(image_point_proposals[:, 0])
            #     groups = {x_coord: list(image_point_proposals[image_point_proposals[:, 0] == x_coord]) for x_coord in z_coords}  # Group points if they are on the same image slice
            #
            #     prediction = np.zeros_like(self.label_layer.data)
            #     for x_coord, group_points in groups.items():
            #         group_labels = [image_label_proposals[np.argwhere(np.all(image_point_proposals == point, axis=1)).flatten()[0]] for point in group_points]
            #         group_points = [point[1:] for point in group_points]
            #         self.sam_predictor.features = self.sam_features[x_coord - 1]
            #         prediction_yz, _, _ = self.sam_predictor.predict(
            #             point_coords=np.flip(group_points, axis=-1),
            #             point_labels=np.asarray(group_labels),
            #             mask_input=self.sam_logits,
            #             multimask_output=False,
            #         )
            #         prediction_yz = prediction_yz[0]
            #         prediction[x_coord, :, :] = prediction_yz  # prediction_yz is 2D
            #     print("")
            #     sam_logits = None  # TODO: Use sam_logits
            else:
                raise RuntimeError("Only 2D and 3D images are supported.")
        else:
            warnings.warn("Could not identify click position.")
            prediction = None
            predicted_slices = None
        return prediction, predicted_slices

    def predict_everything(self):
        if self.image_layer.ndim == 2:
            image = np.asarray(self.image_layer.data)
            if not self.image_layer.rgb:
                image = np.stack((image,) * 3, axis=-1)  # Expand to 3-channel image
            image = image[..., :3]  # Remove a potential alpha channel
            records = self.sam_anything_predictor.generate(image)
            masks = np.asarray([record["segmentation"] for record in records])
            prediction = np.argmax(masks, axis=0)
        elif self.image_layer.ndim == 3:
            l_creating_features= QLabel("Predicting everything:")
            self.layout().addWidget(l_creating_features)
            progress_bar = QProgressBar(self)
            progress_bar.setMaximum(self.image_layer.data.shape[0])
            progress_bar.setValue(0)
            self.layout().addWidget(progress_bar)
            prediction = []
            for index in tqdm(range(self.image_layer.data.shape[0]), desc="Predicting everything"):
                image_slice = np.asarray(self.image_layer.data[index, ...])
                if not self.image_layer.rgb:
                    image_slice = np.stack((image_slice,) * 3, axis=-1)  # Expand to 3-channel image
                image_slice = image_slice[..., :3]  # Remove a potential alpha channel
                contrast_limits = self.image_layer.contrast_limits
                image_slice = normalize(image_slice, source_limits=contrast_limits, target_limits=(0, 255)).astype(np.uint8)
                records_slice = self.sam_anything_predictor.generate(image_slice)
                masks_slice = np.asarray([record["segmentation"] for record in records_slice])
                prediction_slice = np.argmax(masks_slice, axis=0)
                prediction.append(prediction_slice)
                progress_bar.setValue(index+1)
                QApplication.processEvents()
                progress_bar.deleteLater()
                l_creating_features.deleteLater()
            prediction = np.asarray(prediction)
            prediction = self.merge_classes_over_slices(prediction)
        else:
            raise RuntimeError("Only 2D and 3D images are supported.")
        return prediction

    def merge_classes_over_slices(self, prediction, threshold=0.5):  # Currently only computes overlap from next_slice to current_slice but not vice versa
        for i in range(prediction.shape[0] - 1):
            current_slice = prediction[i]
            next_slice = prediction[i+1]
            next_labels, next_label_counts = np.unique(next_slice, return_counts=True)
            next_label_counts = next_label_counts[next_labels != 0]
            next_labels = next_labels[next_labels != 0]
            new_next_slice = np.zeros_like(next_slice)
            if len(next_labels) > 0:
                for next_label, next_label_count in zip(next_labels, next_label_counts):
                    current_roi_labels = current_slice[next_slice == next_label]
                    current_roi_labels, current_roi_label_counts = np.unique(current_roi_labels, return_counts=True)
                    current_roi_label_counts = current_roi_label_counts[current_roi_labels != 0]
                    current_roi_labels = current_roi_labels[current_roi_labels != 0]
                    if len(current_roi_labels) > 0:
                        current_max_count = np.max(current_roi_label_counts)
                        current_max_count_label = current_roi_labels[np.argmax(current_roi_label_counts)]
                        overlap = current_max_count / next_label_count
                        if overlap >= threshold:
                            new_next_slice[next_slice == next_label] = current_max_count_label
                        else:
                            new_next_slice[next_slice == next_label] = next_label
                    else:
                        new_next_slice[next_slice == next_label] = next_label
                prediction[i+1] = new_next_slice
        return prediction

    def update_points_layer(self, points):
        selected_layer = None
        if self.viewer.layers.selection.active != self.points_layer:
            selected_layer = self.viewer.layers.selection.active
        if self.points_layer is not None:
            self.viewer.layers.remove(self.points_layer)

        points_flattened = []
        colors_flattended = []
        if points is not None:
            for label, label_points in points.items():
                points_flattened.extend(label_points)
                color = self.label_color_mapping["label_mapping"][label]
                colors = [color] * len(label_points)
                colors_flattended.extend(colors)

        self.point_size = int(np.min(self.image_layer.data.shape[:2]) / 100)
        if self.point_size == 0:
            self.point_size = 1
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.points_layer = self.viewer.add_points(name=self.points_layer_name, data=np.asarray(points_flattened), face_color=colors_flattended, edge_color="white", size=self.point_size)
        self.points_layer.editable = False

        if selected_layer is not None:
            self.viewer.layers.selection.active = selected_layer
        self.points_layer.refresh()

    def find_changed_point(self, old_points, new_points):
        if len(new_points) == 0:
            old_point = old_points
        else:
            old_point = np.array([x for x in old_points if not np.any((x == new_points).all(1))])
        if len(old_points) == 0:
            new_point = new_points
        else:
            new_point = np.array([x for x in new_points if not np.any((x == old_points).all(1))])

        if len(old_point) == 0:
            deleted_point = None
        else:
            deleted_point = old_point[0]

        if len(new_point) == 0:
            new_point = None
        else:
            new_point = new_point[0]

        return deleted_point, new_point

    def find_point_label(self, point):
        for label, label_points in self.points.items():
            if np.in1d(point, label_points).all(axis=0):
                return label
        raise RuntimeError("Could not identify label.")

    def remove_all_widget_callbacks(self, layer):
        callback_types = ['mouse_double_click_callbacks', 'mouse_drag_callbacks', 'mouse_move_callbacks',
                          'mouse_wheel_callbacks', 'keymap']
        for callback_type in callback_types:
            callbacks = getattr(layer, callback_type)
            if isinstance(callbacks, list):
                for callback in callbacks:
                    if inspect.ismethod(callback) and callback.__self__ == self:
                        callbacks.remove(callback)
            elif isinstance(callbacks, dict):
                for key in list(callbacks.keys()):
                    if inspect.ismethod(callbacks[key]) and callbacks[key].__self__ == self:
                        del callbacks[key]
            else:
                raise RuntimeError("Could not determine callbacks type.")

    def _reset_history(self, event=None):
        self._undo_history = deque()
        self._redo_history = deque()

    def _save_history(self, history_item):
        """Save a history "atom" to the undo history.

        A history "atom" is a single change operation to the array. A history
        *item* is a collection of atoms that were applied together to make a
        single change. For example, when dragging and painting, at each mouse
        callback we create a history "atom", but we save all those atoms in
        a single history item, since we would want to undo one drag in one
        undo operation.

        Parameters
        ----------
        history_item : 2-tuple of region prop dicts
        """
        self._redo_history = deque()
        # if not self._block_saving:
        #     self._undo_history.append([value])
        # else:
        #     self._undo_history[-1].append(value)
        self._undo_history.append(history_item)

    def _load_history(self, before, after, undoing=True):
        """Load a history item and apply it to the array.

        Parameters
        ----------
        before : list of history items
            The list of elements from which we want to load.
        after : list of history items
            The list of element to which to append the loaded element. In the
            case of an undo operation, this is the redo queue, and vice versa.
        undoing : bool
            Whether we are undoing (default) or redoing. In the case of
            redoing, we apply the "after change" element of a history element
            (the third element of the history "atom").

        See Also
        --------
        Labels._save_history
        """
        if len(before) == 0:
            return

        history_item = before.pop()
        after.append(history_item)

        self.points = history_item["points"]
        self.point_label = history_item["point_label"]
        self.sam_logits = history_item["logits"]
        # self.run(history_item["points"], history_item["point_label"])
        self.update_points_layer(self.points)

    def undo(self):
        self._load_history(
            self._undo_history, self._redo_history, undoing=True
        )

    def redo(self):
        self._load_history(
            self._redo_history, self._undo_history, undoing=False
        )
        raise RuntimeError("Redo currently not supported.")

    # def _myfilter(self, row, parent):
    #     return "<hidden>" not in self.viewer.layers[row].name