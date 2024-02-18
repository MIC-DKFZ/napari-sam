import datetime
import glob

from qtpy.QtWidgets import QDialog, qApp, QVBoxLayout, QTabWidget, QHBoxLayout, QPushButton, QWidget, QLabel, QComboBox, QRadioButton, QGroupBox, QProgressBar, QApplication, QScrollArea, QLineEdit, QCheckBox
from qtpy.QtGui import QIntValidator, QDoubleValidator
from qtpy import QtCore
from qtpy.QtCore import Qt, QSettings
import napari
import numpy as np
from enum import Enum
from collections import deque, defaultdict
import inspect
from segment_anything import SamPredictor, build_sam_vit_h, build_sam_vit_l, build_sam_vit_b
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from .utils import normalize###############################
import torch
from vispy.util.keys import CONTROL
import copy
import warnings
from tqdm import tqdm
from superqt.utils import qdebounced
from .slicer import slicer
import urllib.request
from pathlib import Path
import os
from os.path import join
import pandas as pd
import warnings
import tifffile

class AnnotatorMode(Enum):
    NONE = 0
    CLICK = 1
    BBOX = 2
    AUTO = 3

class SegmentationMode(Enum):
    SEMANTIC = 0
    INSTANCE = 1

class BboxState(Enum):
    CLICK = 0
    DRAG = 1
    RELEASE = 2


SAM_MODELS = {
    "default": {"filename": "sam_vit_h_4b8939.pth", "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", "model": build_sam_vit_h},
    "vit_h": {"filename": "sam_vit_h_4b8939.pth", "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", "model": build_sam_vit_h},
    "vit_l": {"filename": "sam_vit_l_0b3195.pth", "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", "model": build_sam_vit_l},
    "vit_b": {"filename": "sam_vit_b_01ec64.pth", "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", "model": build_sam_vit_b},
    "MedSAM": {"filename": "sam_vit_b_01ec64_medsam.pth", "url": "https://syncandshare.desy.de/index.php/s/yLfdFbpfEGSHJWY/download/medsam_20230423_vit_b_0.0.1.pth", "model": build_sam_vit_b},
}


class SamManager():  ##TODO Makes this outside class
    def __init__(self):
        self.image_basename = None
        self.channels = None
        self.z_stack_id = None
        self.max_z_in_stack = 100
        self.features = None
        self.model = None
        self.predictor = None
        self.logits = None
        self.features = None

    def generate_embedding(self, samwidget, samwidget_image_basename):
        self.image_basename = samwidget_image_basename
        #self.channels = samwidget.channels

        # use presaved embedding if it exists
        self.z_stack_id = samwidget.viewer.dims.current_step[0] // self.max_z_in_stack
        z = self.z(samwidget)
        embedding_fname = f"{self.image_basename}_zmax{self.max_z_in_stack}-zstack{self.z_stack_id}.pt"
        embedding_fp = samwidget.le_embedding_fp.text().strip()
        presaved = os.path.join(embedding_fp, embedding_fname)

        if os.path.exists(presaved):
            print(f"  z={z}, using presaved embedding", presaved)
            self.features = torch.load(presaved,
                                       map_location=f'cuda:{torch.cuda.current_device()}')
            # TODO make above work no matter if it's running on cpu...?
            # [f.to(torch.cuda.device(torch.cuda.current_device())) for f in torch.load(presaved)]
            self.predictor.features = self.features[z]
            image_slice = samwidget.image_layer.data[z, ...]
            # TODO save pickle file and load here instead of assuming original and input size same
            self.predictor.original_size = image_slice.shape
            self.predictor.input_size = image_slice.shape
            self.predictor.is_image_set = True

            self.start_z = self.z_stack_id * self.max_z_in_stack
            self.end_z = (self.z_stack_id + 1) * self.max_z_in_stack - 1

        else:
            # create embedding
            l_creating_features = QLabel("Creating SAM image embedding:")
            samwidget.layout().addWidget(l_creating_features)
            progress_bar = QProgressBar(samwidget)
            progress_bar.setMaximum(samwidget.image_layer.data.shape[0])
            progress_bar.setValue(0)
            samwidget.layout().addWidget(progress_bar)
            self.features = []
            for index in tqdm(range(samwidget.image_layer.data.shape[0]),
                              desc="Creating SAM image embedding"):
                image_slice = np.asarray(samwidget.image_layer.data[index, ...])
                if not samwidget.image_layer.rgb:
                    image_slice = np.stack((image_slice,) * 3,
                                           axis=-1)  # Expand to 3-channel image
                image_slice = image_slice[...,
                              :3]  # Remove a potential alpha channel
                contrast_limits = samwidget.image_layer.contrast_limits
                image_slice = normalize(image_slice,
                                        source_limits=contrast_limits,
                                        target_limits=(0, 255)).astype(
                    np.uint8)
                self.predictor.set_image(image_slice)
                # TODO does it matter that setting image on last slice
                self.features.append(self.predictor.features)
                progress_bar.setValue(index + 1)
                QApplication.processEvents()
                progress_bar.deleteLater()
                l_creating_features.deleteLater()

    def check_set(self, samwidget):
        # TODO handle when image has no source path as generated from inside console...
        samwidget_image_basename = os.path.splitext(os.path.basename(samwidget.image_layer.source.path))[0]
        if samwidget_image_basename != self.image_basename:
            self.generate_embedding(samwidget, samwidget_image_basename)
        if not (self.start_z <= samwidget.viewer.dims.current_step[0] <= self.end_z):
            self.generate_embedding(samwidget, samwidget_image_basename)

    def z(self, samwidget):
        return samwidget.viewer.dims.current_step[0] - self.max_z_in_stack*self.z_stack_id

class SamWidget(QDialog):
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

        # create top level layout
        main_layout = QVBoxLayout()

        # self.scroll_area = QScrollArea()
        # self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        # self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        # self.scroll_area.setWidgetResizable(True)

        self.image_name = None
        self.image_name_cache = None
        self.image_layer = None
        self.metadata = {}

        # create tab widget
        tabs = QTabWidget()
        tabs.addTab(self.SAMTabUI(), "SAM")
        self.settings_tab = self.SettingsTabUI()
        tabs.addTab(self.settings_tab, "Settings")
        # tabs.addTab(self.IOTabUI(), "I/O")
        main_layout.addWidget(tabs)
        self.setLayout(main_layout)

        self.label_layer = None
        self.label_layer_changes = None
        self.label_color_mapping = None
        self.points_layer = None
        self.points_layer_name = "Ignore this layer1"  # "Ignore this layer <hidden>"
        self.old_points = np.zeros(0)
        self.point_size = 10

        self.le_point_size.setText(str(self.point_size))
        self.bbox_layer = None
        self.bbox_layer_name = "Ignore this layer2"
        self.bbox_edge_width = 10
        self.le_bbox_edge_width.setText(str(self.bbox_edge_width))

        self.init_comboboxes()
        self.sam = SamManager()
        self.viewer.layers.selection.events.active.connect(
            self._select_labels_layer)
        self.adding_multiple_labels = False

        self.points = defaultdict(list)
        self.point_label = None

        self.bboxes = defaultdict(list)

        # self.viewer.window.qt_viewer.layers.model().filterAcceptsRow = self._myfilter


    def SAMTabUI(self):
        self.layer_types = {"image": napari.layers.image.image.Image, "labels": napari.layers.labels.labels.Labels}
        tab = QWidget()
        layout = QVBoxLayout()

        l_model_type = QLabel("Select model type:")
        layout.addWidget(l_model_type)

        self.cb_model_type = QComboBox()
        layout.addWidget(self.cb_model_type)

        self.btn_load_model = QPushButton("Load model")
        self.btn_load_model.clicked.connect(self._load_model)
        layout.addWidget(self.btn_load_model)
        self.loaded_model = None
        self.init_model_type_combobox()


        l_image_layer = QLabel("Select input image layer:")
        layout.addWidget(l_image_layer)

        self.cb_image_layers = QComboBox()
        self.cb_image_layers.addItems(self.get_layer_names("image"))
        self.cb_image_layers.currentTextChanged.connect(self.on_image_change)
        layout.addWidget(self.cb_image_layers)

        l_label_layer = QLabel("Select output labels layer:")
        layout.addWidget(l_label_layer)

        self.cb_label_layers = QComboBox()
        self.cb_label_layers.addItems(self.get_layer_names("labels"))
        layout.addWidget(self.cb_label_layers)

        self.comboboxes = [{"combobox": self.cb_image_layers, "layer_type": "image"}, {"combobox": self.cb_label_layers, "layer_type": "labels"}]

        self.g_annotation = QGroupBox("Annotation mode")
        self.l_annotation = QVBoxLayout()

        self.rb_click = QRadioButton("Click && Bounding Box")
        self.rb_click.setChecked(True)
        self.rb_click.setToolTip("Positive Click: Middle Mouse Button\n \n"
                                    "Negative Click: Control + Middle Mouse Button \n \n"
                                    "Undo: Control + Z \n \n"
                                    "Select Point: Left Click \n \n"
                                    "Delete Selected Point: Delete")
        self.l_annotation.addWidget(self.rb_click)
        self.rb_click.clicked.connect(self.on_everything_mode_checked)

        self.rb_auto = QRadioButton("Everything")
        # self.rb_auto.setEnabled(False)
        # self.rb_auto.setStyleSheet("color: gray")
        self.rb_auto.setToolTip("Creates automatically an instance segmentation \n"
                                               "of the entire image.\n"
                                               "No user interaction possible.")
        self.l_annotation.addWidget(self.rb_auto)
        self.rb_auto.clicked.connect(self.on_everything_mode_checked)

        self.g_annotation.setLayout(self.l_annotation)
        layout.addWidget(self.g_annotation)

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

        self.rb_semantic.clicked.connect(self.on_segmentation_mode_changed)
        self.rb_instance.clicked.connect(self.on_segmentation_mode_changed)

        self.g_segmentation.setLayout(self.l_segmentation)
        layout.addWidget(self.g_segmentation)

        self.btn_add_annot_layers = QPushButton("Add annotation layers")
        self.btn_add_annot_layers.clicked.connect(self._add_annot_layers_activate)
        self.btn_add_annot_layers.setEnabled(False)
        layout.addWidget(self.btn_add_annot_layers)

        self.btn_activate = QPushButton("Activate")
        self.btn_activate.clicked.connect(self._activate)
        self.btn_activate.setEnabled(False)
        self.is_active = False
        layout.addWidget(self.btn_activate)

        self.btn_mode_switch = QPushButton("Switch to BBox Mode")
        self.btn_mode_switch.clicked.connect(self._switch_mode)
        self.btn_mode_switch.setEnabled(False)
        layout.addWidget(self.btn_mode_switch)

        self.btn_finish_image = QPushButton("Finished annotating image")
        self.btn_finish_image.clicked.connect(self._on_finish_image)
        self.btn_finish_image.setEnabled(False)
        layout.addWidget(self.btn_finish_image)

        self.check_prev_mask = QCheckBox('Use previous SAM prediction (recommended)')
        self.check_prev_mask.setEnabled(False)
        self.check_prev_mask.setChecked(True)
        layout.addWidget(self.check_prev_mask)

        self.check_auto_inc_bbox= QCheckBox('Auto increment bounding box label')
        self.check_auto_inc_bbox.setEnabled(False)
        self.check_auto_inc_bbox.setChecked(True)
        layout.addWidget(self.check_auto_inc_bbox)

        container_widget_info = QWidget()
        container_layout_info = QVBoxLayout(container_widget_info)

        self.g_size = QGroupBox("Point && Bounding Box Settings")
        self.l_size = QVBoxLayout()

        l_point_size = QLabel("Point Size:")
        self.l_size.addWidget(l_point_size)
        validator = QIntValidator()
        validator.setRange(0, 9999)
        self.le_point_size = QLineEdit()
        self.le_point_size.setText("1")
        self.le_point_size.setValidator(validator)
        self.l_size.addWidget(self.le_point_size)

        l_bbox_edge_width = QLabel("Bounding Box Edge Width:")
        self.l_size.addWidget(l_bbox_edge_width)
        validator = QIntValidator()
        validator.setRange(0, 9999)
        self.le_bbox_edge_width = QLineEdit()
        self.le_bbox_edge_width.setText("1")
        self.le_bbox_edge_width.setValidator(validator)
        self.l_size.addWidget(self.le_bbox_edge_width)
        self.g_size.setLayout(self.l_size)
        container_layout_info.addWidget(self.g_size)

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
                                       "Delete Selected Point: Delete\n \n"
                                       "Pick Label: Control + Left Click\n \n"
                                       "Increment Label: M\n \n")
        self.label_info_click.setWordWrap(True)
        self.l_info_click.addWidget(self.label_info_click)
        self.g_info_click.setLayout(self.l_info_click)
        container_layout_info.addWidget(self.g_info_click)

        scroll_area_info = QScrollArea()
        scroll_area_info.setWidget(container_widget_info)
        scroll_area_info.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(scroll_area_info)

        self.scroll_area_auto = self.init_auto_mode_settings()
        layout.addWidget(self.scroll_area_auto)

        tab.setLayout(layout)

        return tab

    def SettingsTabUI(self):
        tab = QWidget()
        layout = QVBoxLayout()
        self.settings_tab_cache = {}

        # ANNOTATION TYPES
        self.g_annotation_settings = QGroupBox("Data")
        self.l_annotation_settings = QVBoxLayout()

        # user writes down segmentation classes
        l_annot_classes = QLabel("List of labels to segment (comma separated NO SPACE)")
        l_annot_classes.setWordWrap(True)
        l_annot_classes.setToolTip("Separate labels with , only (e.g. Label1,Label2). "
                                   "Do not use extra space after comma. "
                                   "Do not use \"-\" "
                                   "unless delineating paint numbers within "
                                   "one label layer e.g. graft-host-background")
        self.l_annotation_settings.addWidget(l_annot_classes)
        self.le_annot_classes = QLineEdit()
        self.le_annot_classes.setText("Label1,Label2")
        self.settings_tab_cache['le_annot_classes'] = self.le_annot_classes
        # validator = QValidator() #TODO validate the separator
        #self.le_annot_classes.setValidator(validator)
        self.l_annotation_settings.addWidget(self.le_annot_classes)

        l_embedding_fp = QLabel("Folder containing presaved image embeddings")
        l_embedding_fp.setWordWrap(True)
        self.l_annotation_settings.addWidget(l_embedding_fp)
        self.le_embedding_fp = QLineEdit()
        self.le_embedding_fp.setText("")
        self.settings_tab_cache['le_embedding_fp'] = self.le_embedding_fp
        self.l_annotation_settings.addWidget(self.le_embedding_fp)

        self.g_annotation_settings.setLayout(self.l_annotation_settings)
        layout.addWidget(self.g_annotation_settings)

        # METRICS & GRAPHS
        self.g_output_settings = QGroupBox("Metrics && Graphs")
        self.l_output_settings = QVBoxLayout()

        l_metadata_fp = QLabel("Filepath for image information spreadsheet")
        l_metadata_fp.setWordWrap(True)
        # l_annot_classes.setToolTip("Separate labels with , only (e.g. Label1,Label2) - do not use extra space after comma")
        self.l_output_settings.addWidget(l_metadata_fp)
        self.le_metadata_fp = QLineEdit()
        # validator_spreadsheet = QValidator() #TODO validate the separator as filepath of type spreadsheet EXT
        # self.le_metadata_fp.setValidator(validator_spreadsheet)
        self.settings_tab_cache['le_metadata_fp'] = self.le_metadata_fp
        self.l_output_settings.addWidget(self.le_metadata_fp)

        l_collated_metrics_fp = QLabel("Filepath for collated output metrics files (optional)")
        l_collated_metrics_fp.setWordWrap(True)
        self.l_output_settings.addWidget(l_collated_metrics_fp)
        self.le_collated_metrics_fp = QLineEdit()
        #validator_csv = QValidator()
        #self.le_collated_metrics_fp.setValidator(validator_csv)
        self.le_collated_metrics_fp.textChanged.connect(self.check_input_image_matching_metadata_record)
        self.settings_tab_cache['le_collated_metrics_fp'] = self.le_collated_metrics_fp
        self.l_output_settings.addWidget(self.le_collated_metrics_fp)

        self.l_percentage_of_annot = QLabel("Label(s) to record other label areas as a percentage of (optional)")
        self.l_percentage_of_annot.setWordWrap(True)
        self.l_output_settings.addWidget(self.l_percentage_of_annot)
        self.le_percentage_of_annot = QLineEdit()
        #self.cb_label_layers_percentage_of = QComboBox()
        #self.cb_label_layers_percentage_of.addItems(self.get_layer_names("labels"))
        self.settings_tab_cache['le_percentage_of_annot'] = self.le_percentage_of_annot
        self.l_output_settings.addWidget(self.le_percentage_of_annot)

        self.l_percentage_of_annot_label = QLabel(
            "Paint number of label to record other labels as a pecentage of (optional integer). If ALL or blank will consider all paint numbers.")
        self.l_percentage_of_annot_label.setWordWrap(True)
        self.l_output_settings.addWidget(self.l_percentage_of_annot_label)
        self.le_percentage_of_annot_label = QLineEdit()
        # self.cb_label_layers_percentage_of = QComboBox()
        # self.cb_label_layers_percentage_of.addItems(self.get_layer_names("labels"))
        self.settings_tab_cache[
            'le_percentage_of_annot_label'] = self.le_percentage_of_annot_label
        self.l_output_settings.addWidget(self.le_percentage_of_annot_label)

        self.l_mindist_label = QLabel(
            "Label to calculate min distance from other labels, with optional =[INTEGER] to specify paint number of label to use")
        self.l_mindist_label.setWordWrap(True)
        self.l_output_settings.addWidget(self.l_mindist_label)
        self.le_mindist_label = QLineEdit()
        self.settings_tab_cache['le_mindist_label'] = self.le_mindist_label
        self.l_output_settings.addWidget(self.le_mindist_label)

        self.l_measure_empty_labels_slice = QLabel("In each annotated slice, record non-empty label layers only")
        self.l_measure_empty_labels_slice.setWordWrap(True)
        self.l_output_settings.addWidget(self.l_measure_empty_labels_slice)
        self.b_measure_empty_labels_slice = QCheckBox()
        self.b_measure_empty_labels_slice.setChecked(False)
        self.settings_tab_cache['b_measure_empty_labels_slice'] = self.b_measure_empty_labels_slice
        self.l_output_settings.addWidget(self.b_measure_empty_labels_slice)

        self.g_output_settings.setLayout(self.l_output_settings)
        layout.addWidget(self.g_output_settings)

        # setup saving settings
        self.appInstance = QApplication.instance()
        self.appInstance.lastWindowClosed.connect(self.on_close_callback)
        self.getSettingValues(tab)

        tab.setLayout(layout)
        return tab


    def getSettingValues(self, qwidget):
        self.setting_variables = QSettings("napari-sam", "variables")
        #qwidget = self.settings_tab
        for k,w in self.settings_tab_cache.items():
            val = self.setting_variables.value(f"{qwidget.objectName()}/{k}")
            if isinstance(w, QLineEdit):
                w.setText(val)
            elif isinstance(w, QComboBox):
                w.setCurrentText(val)
            elif isinstance(w, QCheckBox):
                w.setChecked(eval(val.capitalize())) #convert val to boolean
            else:
                raise Warning("Settings tab has widget type that not currently supported for caching")
    
        """
        for w in QtWidgets.qApp.allWidgets():
            mo = w.metaObject()
            #if qwidget.objectName() != "":
            for i in range(mo.propertyCount()):
                name = mo.property(i).name()
                val = self.setting_variables.value("{}/{}".format(qwidget.objectName(), name),
                                                qwidget.property(name))
                qwidget.setProperty(name, val)
        """

    def on_close_callback(self):
        #print("closing")
        # caching settings tab values to file
        qwidget = self.settings_tab
        for k,w in self.settings_tab_cache.items():
            if isinstance(w, QLineEdit):
                val = w.text()
            elif isinstance(w, QComboBox):
                val = w.currentText()
            elif isinstance(w, QCheckBox):
                val = w.isChecked()
            else:
                raise Warning("Settings tab has widget type that not currently supported for caching")
            self.setting_variables.setValue(
                f"{qwidget.objectName()}/{k}",
                val)

        """
        for i in range(qwidget.layout().count()):
            w = self.layout().itemAt(i).layout().widget()

            mo = w.metaObject()
            print("CLOSE:", w.objectName(), mo)
            #if qwidget.objectName() != "":
            for i in range(mo.propertyCount()):
                name = mo.property(i).name()
                print("  CLOSE CALLBACK:", mo.objectName(), name)
                #self.setting_variables.setValue(f"{w.objectName()}/{name}",
                #                                w.property(name))
            #self.setting_variables.setValue('le_annot_classes', self.le_annot_classes.text())
        """

    def _select_labels_layer(self):
        """
        Triggered whenever different layer is selected. If it's a labels layer
        and model is currently active, will change the widget labels layer
        for model input.
        :return:
        """
        current_layer = self.viewer.layers.selection.active
        if isinstance(current_layer, napari.layers.Labels) and \
                (current_layer.name != self.cb_label_layers.currentText()):
            if self.is_active:
                self._deactivate()
                # switch output to that labels layer
                self.cb_label_layers.setCurrentText(current_layer.name)
                self.label_layer = current_layer
                self._activate()

            # activate if not yet active and not adding initial set of layers
            elif (not self.is_active) and (not self.adding_multiple_labels):
                self.cb_label_layers.setCurrentText(current_layer.name)
                self.label_layer = current_layer
            #    self._activate()

    def select_layer_while_active(self, layer):
        """
        This function avoids triggering switching model to different label output
        layer when a labels layer is selected by the code instead of the person
        when the widget annotation is active
        :param layer: layer to be set to active
        """
        # WARNING: be careful when changing is_active type to not be a boolean
        # or hooking events up to it as it gets temporarily switched to False
        # when a layer needs to be selected during widget annotation activation.
        original_active_status = self.is_active
        if self.is_active:
            self.is_active = False
        self.viewer.layers.selection.active = layer
        self.is_active = original_active_status

    def _on_finish_image(self):
        print("FINISH IMAGE BUTTON PRESSED")
        if self.is_active:
            self._deactivate()
        self._save_labels()
        self._measure()
        # generate graphs
        self.viewer.layers.clear()

    def check_input_image_matching_metadata_record(self):

        EXT_LIST = ['csv', 'xls', 'xlsx', 'xlsm', 'xlsb']
        image_layer_name = self.cb_image_layers.currentText()
        # only proceed if image layer has been selected already
        if (image_layer_name == ""):
            return

        metadata_fp = self.le_metadata_fp.text().strip()
        image_layer = self.viewer.layers[image_layer_name]
        path = image_layer.source.path

        # no path found for image
        if path is None:
            return

        image_name = os.path.basename(path)

        # if metadata record already read in, do not read again
        # warning: assumes metadata record will not have changed since last read in
        if image_name in self.metadata.keys():
            return

        # read metadata file
        if metadata_fp.endswith("csv"):
            info_df = pd.read_csv(metadata_fp)
        elif any([metadata_fp.endswith(ext) for ext in EXT_LIST]):
            info_df = pd.read_excel(metadata_fp)
        elif (metadata_fp == '') or (metadata_fp is None):
            # metadata filepath hasn't been set yet so do nothing
            return
        else:
            warnings.warn(f"Image information spreadsheet "
                          f"{metadata_fp} is not of filetype "
                          f"{EXT_LIST} so cannot be processed")
            return


        image_info = info_df[info_df["Image"] == image_name]
        if image_info.empty:
            warnings.warn(f"{image_name} does not have a metadata record in "
                          f"{metadata_fp}. Check that column Image contains a "
                          f"record for {image_name}. Very minimal metrics file or graphs will be output")
        else:
            self.metadata[image_name] = image_info
            # print metadata to command line if different from previous metadata or metadata never been set by napari-sam
            #if (self.current_image_metadata is None) or (not self.metadata[image_name].equals(self.current_image_metadata)):
            print("_______________________________________________________")
            print(f"METADATA READ IN FROM {metadata_fp}")
            for col in image_info:
                print(f"  {col}: {image_info.iloc[0][col]}")
            # print("_______________________________________________________")
            #self.current_image_metadata = image_info

        return

    def init_auto_mode_settings(self):
        container_widget_auto = QWidget()
        container_layout_auto = QVBoxLayout(container_widget_auto)

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

    def on_segmentation_mode_changed(self):
        if self.rb_semantic.isChecked():
            self.segmentation_mode = SegmentationMode.SEMANTIC
        if self.rb_instance.isChecked():
            self.segmentation_mode = SegmentationMode.INSTANCE

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
        #pass

        # try and read in metadata entry
        image_name = self.cb_image_layers.currentText()
        if (image_name != ""):
            # check if image source is not most recently checked for metadata
            if (self.image_layer == None) or (self.viewer.layers[image_name].source.path != self.image_layer.source.path):
                self.check_input_image_matching_metadata_record()
            # cache the current image layer to allow checking if image source changed in future calls to this function
            self.image_layer = self.viewer.layers[self.cb_image_layers.currentText()]

    def init_model_type_combobox(self):
        model_types = list(SAM_MODELS.keys())
        cached_weight_types = self.get_cached_weight_types(model_types)
        # entries = []
        # for name, is_cached in cached_weight_types.items():
        #     if is_cached:
        #         entries.append("{} (Cached)".format(name))
        #     else:
        #         entries.append("{} (Auto-Download)".format(name))
        # self.cb_model_type.addItems(entries)
        self.update_model_type_combobox()

        if cached_weight_types[list(cached_weight_types.keys())[self.cb_model_type.currentIndex()]]:
            self.btn_load_model.setText("Load model")
        else:
            self.btn_load_model.setText("Download and load model")

        self.cb_model_type.currentTextChanged.connect(self.on_model_type_combobox_change)

    def update_model_type_combobox(self):
        model_types = list(SAM_MODELS.keys())
        cached_weight_types = self.get_cached_weight_types(model_types)
        entries = []
        for name, is_cached in cached_weight_types.items():
            if name == self.loaded_model:
                entries.append("{} (Loaded)".format(name))
            elif is_cached:
                entries.append("{} (Cached)".format(name))
            else:
                entries.append("{} (Auto-Download)".format(name))
        self.cb_model_type.clear()
        self.cb_model_type.addItems(entries)
        if self.loaded_model is not None:
            loaded_model_index = self.cb_model_type.findText("{} (Loaded)".format(self.loaded_model))
            self.cb_model_type.setCurrentIndex(loaded_model_index)

    def on_model_type_combobox_change(self):
        model_types = list(SAM_MODELS.keys())
        cached_weight_types = self.get_cached_weight_types(model_types)

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
        #self.on_image_change()

    def _on_layers_changed_callback(self):
        self._check_activate_btn()
        if (self.image_layer is not None and self.image_layer not in self.viewer.layers) or (self.label_layer is not None and self.label_layer not in self.viewer.layers):
            self._deactivate()

    def _check_activate_btn(self):
        if self.cb_image_layers.currentText() != "" and self.cb_label_layers.currentText() != "" and self.loaded_model is not None:
            self.btn_activate.setEnabled(True)
        else:
            self.btn_activate.setEnabled(False)

    def _load_model(self):
        self.cb_model_type.setEnabled(False)
        self.btn_load_model.setEnabled(False)
        model_types = list(SAM_MODELS.keys())
        model_type = model_types[self.cb_model_type.currentIndex()]
        self.sam.model = SAM_MODELS[model_type]["model"](
            self.get_weights_path(model_type)
        )
        self.sam.model.to(self.device)
        self.sam.predictor = SamPredictor(self.sam.model)
        self.loaded_model = model_type
        self.update_model_type_combobox()
        self.cb_model_type.setEnabled(True)
        self.btn_load_model.setEnabled(True)
        self.btn_add_annot_layers.setEnabled(True)
        self._check_activate_btn()

    def _add_annot_layers_activate(self):

        # autocontrast image layers
        image_layers = [x for x in self.viewer.layers if
                        isinstance(x, napari.layers.Image)]
        for l in image_layers:
            l._keep_auto_contrast = True
        print(
            f"All image layers are on continuous contrast: {[l.name for l in image_layers]}")

        # add annot layers
        self.adding_multiple_labels = True
        image_layer_name = self.cb_image_layers.currentText()
        if not image_layer_name:
            raise ValueError("No image open: load image into viewer before adding annotation layers.")
        save_path = self.viewer.layers[image_layer_name].source.path
        save_folder = os.path.dirname(save_path)
        img_name = os.path.splitext(os.path.basename(save_path))[0]

        # load/create annotation specified in settings
        if self.le_annot_classes.text() is not None:
            annot_classes = self.le_annot_classes.text().strip().split(",")
            for name in annot_classes:
                # load existing saved labels layer
                fp = os.path.join(save_folder, f"{img_name}_{name}.tif")
                if os.path.exists(fp):
                    im = tifffile.imread(fp)
                    self.viewer.add_labels(im, name=name)
                # create empty labels layer
                else:
                    current_image_layer = self.viewer.layers[
                        self.cb_image_layers.currentText()]
                    self.viewer.add_labels(
                        np.zeros(current_image_layer.data.shape,
                                 dtype='uint8'), name=name)

        # load any other saved annotations not already loaded into viewer
        saved_tifs = os.path.join(save_folder, f"{img_name}_*.tif")
        viewer_layers = [l.name for l in self.viewer.layers]
        for fp in glob.glob(saved_tifs):
            name = os.path.splitext(os.path.basename(fp))[0].split("_")[-1]
            if name not in viewer_layers:
                im = tifffile.imread(fp)
                self.viewer.add_labels(im, name=name)

        self.adding_multiple_labels = False
        self.btn_finish_image.setEnabled(True)

        self._activate()

    def _activate(self):
        self.btn_activate.setEnabled(False)
        if not self.is_active:
            # activate
            self.image_name = self.cb_image_layers.currentText()
            self.image_layer = self.viewer.layers[self.cb_image_layers.currentText()]
            self.label_layer = self.viewer.layers[self.cb_label_layers.currentText()]
            self.label_layer_changes = None
            # Fixes shape adjustment by napari
            if self.image_layer.ndim == 3:
                self.image_layer_affine_scale = self.image_layer.affine.scale
                self.image_layer_scale = self.image_layer.scale
                self.image_layer_scale_factor = self.image_layer.scale_factor
                self.label_layer_affine_scale = self.label_layer.affine.scale
                self.label_layer_scale = self.label_layer.scale
                self.label_layer_scale_factor = self.label_layer.scale_factor
                self.image_layer.affine.scale = np.array([1, 1, 1])
                self.image_layer.scale = np.array([1, 1, 1])
                self.image_layer.scale_factor = 1
                self.label_layer.affine.scale = np.array([1, 1, 1])
                self.label_layer.scale = np.array([1, 1, 1])
                self.label_layer.scale_factor = 1
                pos = self.viewer.dims.point
                self.viewer.dims.set_point(0, 0)
                self.viewer.dims.set_point(0, pos[0])
                self.viewer.reset_view()

            if self.image_layer.ndim != 2 and self.image_layer.ndim != 3:
                raise RuntimeError("Only 2D and 3D images are supported.")

            if self.image_layer.ndim == 2:
                self.sam.logits = None
            else:
                self.sam.logits = [None] * self.image_layer.data.shape[0]

            if self.rb_click.isChecked():
                self.annotator_mode = AnnotatorMode.CLICK
                # self.rb_bbox.setEnabled(False)
                self.rb_auto.setEnabled(False)
                # self.rb_bbox.setStyleSheet("color: gray")
                self.rb_auto.setStyleSheet("color: gray")
            # elif self.rb_bbox.isChecked():
            #     self.annotator_mode = AnnotatorMode.BBOX
            #     self.rb_click.setEnabled(False)
            #     self.rb_auto.setEnabled(False)
            #     self.rb_click.setStyleSheet("color: gray")
            #     self.rb_auto.setStyleSheet("color: gray")
            elif self.rb_auto.isChecked():
                self.annotator_mode = AnnotatorMode.AUTO
                self.rb_click.setEnabled(False)
                # self.rb_bbox.setEnabled(False)
                self.rb_click.setStyleSheet("color: gray")
                # self.rb_bbox.setStyleSheet("color: gray")
            else:
                raise RuntimeError("Annotator mode not implemented.")
            if self.annotator_mode != AnnotatorMode.AUTO:
                self.is_active = True
                self.btn_activate.setText("Deactivate")
                self.btn_load_model.setEnabled(False)
                self.cb_model_type.setEnabled(False)
                self.cb_image_layers.setEnabled(False)
                self.cb_label_layers.setEnabled(False)
                self.btn_mode_switch.setEnabled(True)
                self.check_prev_mask.setEnabled(True)
                self.check_auto_inc_bbox.setEnabled(True)
                self.check_auto_inc_bbox.setChecked(True)
                self.btn_mode_switch.setText("Switch to BBox Mode")
                self.annotator_mode = AnnotatorMode.CLICK

                # add bbox layer while retaining same selected layer
                selected_layer = None
                if self.viewer.layers.selection.active != self.points_layer:
                    selected_layer = self.viewer.layers.selection.active
                self.bbox_layer = self.viewer.add_shapes(
                    name=self.bbox_layer_name)
                if selected_layer is not None:
                    # must change layer when self.is_active = False otherwise will trigger _select_labels_layer that will deactivate
                    self.select_layer_while_active(selected_layer)

                if self.image_layer.ndim == 3:
                    self.check_auto_inc_bbox.setChecked(False)
                    # This "fixes" the problem that the first drawn bbox is not visible.
                    self.update_bbox_layer({}, bbox_tmp=[[self.viewer.dims.current_step[0], 0, 0], [self.viewer.dims.current_step[0], 0, 10], [self.viewer.dims.current_step[0], 10, 10], [self.viewer.dims.current_step[0], 10, 0]])
                    self.update_bbox_layer({}, bbox_tmp=None)
                    self.viewer.dims.set_point(0, 0)
                    self.viewer.dims.set_point(0, pos[0])
                self.bbox_layer.editable = False
                self.bbox_first_coords = None
                self.prev_segmentation_mode = SegmentationMode.SEMANTIC

                if self.image_layer.ndim == 2:
                    self.point_size = int(np.min(self.image_layer.data.shape[:2]) / 100)
                    if self.point_size == 0:
                        self.point_size = 1
                    self.bbox_edge_width = 10
                else:
                    self.point_size = 2
                    self.bbox_edge_width = 1
                self.le_point_size.setText(str(self.point_size))
                self.le_bbox_edge_width.setText(str(self.bbox_edge_width))

            if self.rb_semantic.isChecked():
                self.segmentation_mode = SegmentationMode.SEMANTIC
                # self.rb_instance.setEnabled(False)
                # self.rb_instance.setStyleSheet("color: gray")
            elif self.rb_instance.isChecked():
                self.segmentation_mode = SegmentationMode.INSTANCE
                # self.rb_semantic.setEnabled(False)
                # self.rb_semantic.setStyleSheet("color: gray")
            else:
                raise RuntimeError("Segmentation mode not implemented.")

            if self.annotator_mode == AnnotatorMode.CLICK or self.annotator_mode == AnnotatorMode.BBOX:
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
                self.sam_anything_predictor = SamAutomaticMaskGenerator(self.sam.model,
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
        self.btn_mode_switch.setEnabled(False)
        self.btn_mode_switch.setText("Switch to BBox Mode")
        self.check_prev_mask.setEnabled(False)
        self.check_auto_inc_bbox.setEnabled(False)
        self.prev_segmentation_mode = SegmentationMode.SEMANTIC
        self.annotator_mode = AnnotatorMode.CLICK

        # Undo: Fixes shape adjustment by napari  # TODO: Not working correctly atm
        # self.image_layer.affine.scale = self.image_layer_affine_scale
        # self.image_layer.scale = self.image_layer_scale
        # self.image_layer.scale_factor = self.image_layer_scale_factor
        # self.label_layer.affine.scale = self.label_layer_affine_scale
        # self.label_layer.scale = self.label_layer_scale
        # self.label_layer.scale_factor = self.label_layer_scale_factor
        # self.image_layer.refresh()
        # self.label_layer.refresh()

        self.remove_all_widget_callbacks(self.viewer)
        if self.label_layer is not None:
            self.remove_all_widget_callbacks(self.label_layer)
        if self.points_layer is not None and self.points_layer in self.viewer.layers:
            self.viewer.layers.remove(self.points_layer)
        if self.bbox_layer is not None and self.bbox_layer in self.viewer.layers:
            self.viewer.layers.remove(self.bbox_layer)
        self.image_name = None
        self.image_layer = None
        self.label_layer = None
        self.label_layer_changes = None
        self.points_layer = None
        self.bbox_layer = None
        self.bbox_first_coords = None
        self.annotator_mode = AnnotatorMode.NONE
        self.points = defaultdict(list)
        self.bboxes = defaultdict(list)
        self.point_label = None
        self.sam.logits = None
        self.rb_click.setEnabled(True)
        self.rb_auto.setEnabled(True)
        self.rb_click.setStyleSheet("")
        self.rb_auto.setStyleSheet("")
        self.rb_semantic.setEnabled(True)
        self.rb_instance.setEnabled(True)
        self.rb_semantic.setStyleSheet("")
        self.rb_instance.setStyleSheet("")
        self._reset_history()

    def _switch_mode(self):
        if self.annotator_mode == AnnotatorMode.CLICK:
            self.btn_mode_switch.setText("Switch to Click Mode")
            self.annotator_mode = AnnotatorMode.BBOX
            self.rb_semantic.setEnabled(False)
            self.rb_semantic.setChecked(False)
            self.rb_instance.setChecked(True)
            self.rb_semantic.setStyleSheet("color: gray")
            self.prev_segmentation_mode = self.segmentation_mode
            self.segmentation_mode = SegmentationMode.INSTANCE
        else:
            self.btn_mode_switch.setText("Switch to BBox Mode")
            self.annotator_mode = AnnotatorMode.CLICK
            self.rb_semantic.setEnabled(True)
            self.rb_semantic.setStyleSheet("")
            self.segmentation_mode = self.prev_segmentation_mode
            if self.segmentation_mode == SegmentationMode.SEMANTIC:
                self.rb_semantic.setChecked(True)
            else:
                self.rb_instance.setChecked(True)

    def create_label_color_mapping(self, num_labels=1000):
        if self.label_layer is not None:
            self.label_color_mapping = {"label_mapping": {}, "color_mapping": {}}
            for label in range(num_labels):
                color = self.label_layer.get_color(label)
                self.label_color_mapping["label_mapping"][label] = color
                self.label_color_mapping["color_mapping"][str(color)] = label

    def callback_click(self, layer, event):
        data_coordinates = self.image_layer.world_to_data(event.position)
        coords = np.round(data_coordinates).astype(int)
        if self.annotator_mode == AnnotatorMode.CLICK:
            if (not CONTROL in event.modifiers) and event.button == 3:  # Positive middle click
                self.do_point_click(coords, 1)
                yield
            elif CONTROL in event.modifiers and event.button == 3:  # Negative middle click
                self.do_point_click(coords, 0)
                yield
            elif (not CONTROL in event.modifiers) and event.button == 1 and self.points_layer is not None and len(self.points_layer.data) > 0:
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
            elif (CONTROL in event.modifiers) and event.button == 1:
                picked_label = self.label_layer.data[slicer(self.label_layer.data, coords)]
                self.label_layer.selected_label = picked_label
                yield
        elif self.annotator_mode == AnnotatorMode.BBOX:
            if (not CONTROL in event.modifiers) and event.button == 3:  # Positive middle click
                self.do_bbox_click(coords, BboxState.CLICK)
                yield
                while event.type == 'mouse_move':
                    data_coordinates = self.image_layer.world_to_data(event.position)
                    coords = np.round(data_coordinates).astype(int)
                    self.do_bbox_click(coords, BboxState.DRAG)
                    yield
                data_coordinates = self.image_layer.world_to_data(event.position)
                coords = np.round(data_coordinates).astype(int)
                self.do_bbox_click(coords, BboxState.RELEASE)

    def on_delete(self, layer):
        selected_points = list(self.points_layer.selected_data)
        if len(selected_points) > 0:
            self.points_layer.data = np.delete(self.points_layer.data, selected_points[0], axis=0)
            self._save_history({"mode": AnnotatorMode.CLICK, "points": copy.deepcopy(self.points), "bboxes": copy.deepcopy(self.bboxes), "logits": self.sam.logits, "point_label": self.point_label})
            deleted_point, _ = self.find_changed_point(self.old_points, self.points_layer.data)
            label = self.find_point_label(deleted_point)
            index_to_remove = np.where((self.points[label] == deleted_point).all(axis=1))[0]
            self.points[label] = np.delete(self.points[label], index_to_remove, axis=0).tolist()
            if len(self.points[label]) == 0:
                del self.points[label]
            self.point_label = label
            if self.image_layer.ndim == 2:
                self.sam.logits = None
            elif self.image_layer.ndim == 3:
                self.sam.logits[deleted_point[0]] = None
            else:
                raise RuntimeError("Point deletion not implemented for this dimensionality.")
            self.predict_click(self.points, self.point_label, deleted_point, label)
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
            if image.dtype != np.uint8:
                contrast_limits = self.image_layer.contrast_limits
                image = normalize(image, source_limits=contrast_limits, target_limits=(0, 255)).astype(np.uint8)
            self.sam.predictor.set_image(image)
            self.sam.features = self.sam.predictor.features
        elif self.image_layer.ndim == 3:
            self.sam.check_set(self)

        else:
            raise RuntimeError("Only 2D and 3D images are supported.")

    def do_point_click(self, coords, is_positive):
        # Check if there is already a point at these coordinates
        for label, points in self.points.items():
            if np.any((coords == points).all(1)):
                warnings.warn("There is already a point in this location. This click will be ignored.")
                return

        self._save_history({"mode": AnnotatorMode.CLICK, "points": copy.deepcopy(self.points), "bboxes": copy.deepcopy(self.bboxes), "logits": self.sam.logits, "point_label": self.point_label})

        self.point_label = self.label_layer.selected_label
        if not is_positive:
            self.point_label = 0

        self.points[self.point_label].append(coords)

        self.predict_click(self.points, self.point_label, coords, self.point_label)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.label_layer._save_history((self.label_layer_changes["indices"], self.label_layer_changes["old_values"], self.label_layer_changes["new_values"]))

    def predict_click(self, points, point_label, current_point, current_label):
        self.update_points_layer(points)

        if points:
            points_flattened = []
            labels_flattended = []
            for label, label_points in points.items():
                points_flattened.extend(label_points)
                label = int(label == point_label)
                labels = [label] * len(label_points)
                labels_flattended.extend(labels)

            x_coord = current_point[0]
            prediction = self.predict_sam(points=copy.deepcopy(points_flattened), labels=copy.deepcopy(labels_flattended), bbox=None, x_coord=copy.deepcopy(current_point[0]))
            if self.image_layer.ndim == 2:
                x_coord = slice(None, None)
        else:
            prediction = np.zeros_like(self.label_layer.data)
            x_coord = slice(None, None)

        if prediction is not None:
            label_layer = np.asarray(self.label_layer.data)
            changed_indices = np.where(prediction == 1)
            index_labels_old = label_layer[changed_indices]
            label_layer[x_coord][label_layer[x_coord] == point_label] = 0
            if self.segmentation_mode == SegmentationMode.SEMANTIC or point_label == 0:
                label_layer[prediction == 1] = point_label
            else:
                label_layer[(prediction == 1) & (label_layer == 0)] = point_label
            index_labels_new = label_layer[changed_indices]
            self.label_layer_changes = {"indices": changed_indices, "old_values": index_labels_old, "new_values": index_labels_new}
            self.label_layer.data = label_layer
            self.old_points = copy.deepcopy(self.points_layer.data)
            # self.label_layer.refresh()

    def do_bbox_click(self, coords, bbox_state):
        if bbox_state == BboxState.CLICK:
            if not (self.image_layer.ndim == 2 or self.image_layer.ndim == 3):
                raise RuntimeError("Only 2D and 3D images are supported.")
            self.bbox_first_coords = coords
        elif bbox_state == BboxState.DRAG:
            if self.image_layer.ndim == 2:
                bbox_tmp = np.asarray([self.bbox_first_coords, (self.bbox_first_coords[0], coords[1]), coords, (coords[0], self.bbox_first_coords[1])])
            elif self.image_layer.ndim == 3:
                bbox_tmp = np.asarray([self.bbox_first_coords, (self.bbox_first_coords[0], self.bbox_first_coords[1], coords[2]), coords, (self.bbox_first_coords[0], coords[1], self.bbox_first_coords[2])])
            else:
                raise RuntimeError("Only 2D and 3D images are supported.")
            bbox_tmp = np.rint(bbox_tmp).astype(np.int32)
            self.update_bbox_layer(self.bboxes, bbox_tmp=bbox_tmp)
        else:
            self._save_history({"mode": AnnotatorMode.BBOX, "points": copy.deepcopy(self.points), "bboxes": copy.deepcopy(self.bboxes), "logits": self.sam.logits, "point_label": self.point_label})
            if self.image_layer.ndim == 2:
                x_coord = slice(None, None)
                bbox_final = np.asarray([self.bbox_first_coords, (self.bbox_first_coords[0], coords[1]), coords, (coords[0], self.bbox_first_coords[1])])
            elif self.image_layer.ndim == 3:
                x_coord = self.bbox_first_coords[0]
                bbox_final = np.asarray([self.bbox_first_coords, (self.bbox_first_coords[0], self.bbox_first_coords[1], coords[2]), coords, (self.bbox_first_coords[0], coords[1], self.bbox_first_coords[2])])
            else:
                raise RuntimeError("Only 2D and 3D images are supported.")

            new_label = self.label_layer.selected_label
            if self.check_auto_inc_bbox.isChecked():
                new_label = np.max(self.label_layer.data) + 1
                self.label_layer.selected_label = new_label

            bbox_final = np.rint(bbox_final).astype(np.int32)
            self.bboxes[new_label].append(bbox_final)
            self.update_bbox_layer(self.bboxes)

            prediction = self.predict_sam(points=None, labels=None, bbox=copy.deepcopy(bbox_final), x_coord=x_coord)

            label_layer = np.asarray(self.label_layer.data)
            changed_indices = np.where(prediction == 1)
            index_labels_old = label_layer[changed_indices]
            # label_layer[x_coord][label_layer[x_coord] == point_label] = 0
            label_layer[(prediction == 1) & (label_layer == 0)] = new_label
            index_labels_new = label_layer[changed_indices]
            self.label_layer_changes = {"indices": changed_indices, "old_values": index_labels_old, "new_values": index_labels_new}
            self.label_layer.data = label_layer
            self.old_points = copy.deepcopy(self.points_layer.data)
            # self.label_layer.refresh()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.label_layer._save_history((self.label_layer_changes["indices"], self.label_layer_changes["old_values"], self.label_layer_changes["new_values"]))

            if self.check_auto_inc_bbox.isChecked():
                # Update the label here too. This way the label stays incremented when switching to click mode
                new_label = np.max(self.label_layer.data) + 1
                self.label_layer.selected_label = new_label

    def predict_sam(self, points, labels, bbox, x_coord=None):
        if self.image_layer.ndim == 2:
            if points is not None:
                points = np.flip(points, axis=-1)
                labels = np.asarray(labels)
            if bbox is not None:
                top_left_coord, bottom_right_coord = self.find_corners(bbox)
                bbox = [np.flip(top_left_coord), np.flip(bottom_right_coord)]
                bbox = np.asarray(bbox).flatten()
            logits = self.sam.logits
            if not self.check_prev_mask.isChecked():
                logits = None
            self.sam.predictor.features = self.sam.features
            prediction, _, self.sam.logits = self.sam.predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=bbox,
                mask_input=logits,
                multimask_output=False,
            )
            prediction = prediction[0]
        elif self.image_layer.ndim == 3:
            prediction = np.zeros_like(self.label_layer.data)
            if points is not None:
                points = np.asarray(points)
                x_coords = np.unique(points[:, 0])
                groups = {x_coord: list(points[points[:, 0] == x_coord]) for x_coord in x_coords}  # Group points if they are on the same image slice
                group_points = groups[x_coord]
                group_labels = [labels[np.argwhere(np.all(points == point, axis=1)).flatten()[0]] for point in group_points]
                group_points = [point[1:] for point in group_points]
                points = np.flip(group_points, axis=-1)
                labels = np.asarray(group_labels)
            if bbox is not None:
                bbox = bbox[:, 1:]
                top_left_coord, bottom_right_coord = self.find_corners(bbox)
                bbox = [np.flip(top_left_coord), np.flip(bottom_right_coord)]
                bbox = np.asarray(bbox).flatten()

            # check if current z is in range of loaded embedding, and try reloading if not
            self.sam.check_set(self)

            self.sam.predictor.features = self.sam.features[self.sam.z(self)]
            logits = self.sam.logits[self.sam.z(self)]
            if not self.check_prev_mask.isChecked():
                logits = None
            prediction_yz, _, self.sam.logits[self.sam.z(self)] = self.sam.predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=bbox,
                mask_input=logits,
                multimask_output=False,
            )
            prediction_yz = prediction_yz[0]
            prediction[x_coord, :, :] = prediction_yz
        else:
            raise RuntimeError("Only 2D and 3D images are supported.")
        return prediction

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
        self.point_size = int(self.le_point_size.text())
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

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.points_layer = self.viewer.add_points(name=self.points_layer_name, data=np.asarray(points_flattened), face_color=colors_flattended, edge_color="white", size=self.point_size)
        self.points_layer.editable = False

        if selected_layer is not None:
            self.select_layer_while_active(selected_layer)
        self.points_layer.refresh()

    def update_bbox_layer(self, bboxes, bbox_tmp=None):
        self.bbox_edge_width = int(self.le_bbox_edge_width.text())
        bboxes_flattened = []
        edge_colors = []
        for _, bbox in bboxes.items():
            bboxes_flattened.extend(bbox)
            edge_colors.extend(['skyblue'] * len(bbox))
        if bbox_tmp is not None:
            bboxes_flattened.append(bbox_tmp)
            edge_colors.append('steelblue')
        self.bbox_layer.data = bboxes_flattened
        self.bbox_layer.edge_width = [self.bbox_edge_width] * len(bboxes_flattened)
        self.bbox_layer.edge_color = edge_colors
        self.bbox_layer.face_color = [(0, 0, 0, 0)] * len(bboxes_flattened)

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

    def find_corners(self, coords):
        # convert the coordinates to numpy arrays
        coords = np.array(coords)

        # find the indices of the leftmost, rightmost, topmost, and bottommost coordinates
        left_idx = np.min(coords[:, 0])
        right_idx = np.max(coords[:, 0])
        top_idx = np.min(coords[:, 1])
        bottom_idx = np.max(coords[:, 1])

        # determine the top left and bottom right coordinates
        # top_left_coord = coords[top_idx, :] if left_idx != top_idx else coords[right_idx, :]
        # bottom_right_coord = coords[bottom_idx, :] if right_idx != bottom_idx else coords[left_idx, :]

        top_left_coord = [left_idx, top_idx]
        bottom_right_coord = [right_idx, bottom_idx]

        return top_left_coord, bottom_right_coord

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
        self.sam.logits = history_item["logits"]
        self.bboxes = history_item["bboxes"]
        self.update_points_layer(self.points)
        self.update_bbox_layer(self.bboxes)

    def undo(self):
        self._load_history(
            self._undo_history, self._redo_history, undoing=True
        )

    def redo(self):
        self._load_history(
            self._redo_history, self._undo_history, undoing=False
        )
        raise RuntimeError("Redo currently not supported.")

    def download_with_progress(self, url, output_file):
        # Open the URL and get the content length
        req = urllib.request.urlopen(url)
        content_length = int(req.headers.get('Content-Length'))

        l_creating_features = QLabel("Downloading model:")
        self.layout().addWidget(l_creating_features)
        progress_bar = QProgressBar(self)
        progress_bar.setMaximum(int(content_length / 1024))
        progress_bar.setValue(0)
        self.layout().addWidget(progress_bar)

        # Set up the progress bar
        progress_bar_tqdm = tqdm(total=content_length, unit='B', unit_scale=True, desc="Downloading model")

        # Download the file and update the progress bar
        with open(output_file, 'wb') as f:
            downloaded_bytes = 0
            while True:
                buffer = req.read(8192)
                if not buffer:
                    break
                downloaded_bytes += len(buffer)
                f.write(buffer)
                progress_bar_tqdm.update(len(buffer))
                progress_bar.setValue(int(downloaded_bytes / 1024))
                QApplication.processEvents()
                progress_bar.deleteLater()
                l_creating_features.deleteLater()

        # Close the progress bar and the URL
        progress_bar_tqdm.close()
        req.close()

    def get_weights_path(self, model_type):
        weight_url = SAM_MODELS[model_type]["url"]

        cache_dir = Path.home() / ".cache/napari-segment-anything"
        cache_dir.mkdir(parents=True, exist_ok=True)

        weight_path = cache_dir / SAM_MODELS[model_type]["filename"]

        if not weight_path.exists():
            print("Downloading {} to {} ...".format(weight_url, weight_path))
            self.download_with_progress(weight_url, weight_path)

        return weight_path

    def get_cached_weight_types(self, model_types):
        cached_weight_types = {}
        cache_dir = str(Path.home() / ".cache/napari-segment-anything")

        for model_type in model_types:
            filename = os.path.basename(SAM_MODELS[model_type]["filename"])
            if os.path.isfile(join(cache_dir, filename)):
                cached_weight_types[model_type] = True
            else:
                cached_weight_types[model_type] = False

        return cached_weight_types

    def _save_labels(self):
        image_layer_name = self.cb_image_layers.currentText()
        save_path = self.viewer.layers[image_layer_name].source.path
        save_folder = os.path.dirname(save_path)

        all_label_layers = [x for x in self.viewer.layers if
                            isinstance(x, napari.layers.Labels)]
        #need to save all layers not just one
        for layer in all_label_layers:

            # don't save if no annotations in layer
            if layer.data.sum() == 0:
                continue

            # save layer
            label_name = layer.name
            image_layer = self.viewer.layers[image_layer_name]
            img_name = os.path.splitext(os.path.basename(image_layer.source.path))[0]
            if img_name in label_name:
                save_file = os.path.join(save_folder, f"{label_name}.tif")
            else:
                save_file = os.path.join(save_folder, f"{img_name}_{label_name}.tif")
            print("saved:", save_file)
            layer.save(save_file)

    #function for measuring annotations (manual annotating)
    def _measure(self):
        print("Measuring...")
        all_label_layers = [x for x in self.viewer.layers
                            if isinstance(x, napari.layers.Labels)]
        object_output_dfs = []
        slice_output_dfs = []
        image_layer = self.viewer.layers[self.cb_image_layers.currentText()]
        image_path = image_layer.source.path
        dirname, image_name = os.path.split(image_path)

        # returns z values with any annotations - reduce looping through unannotated z slices
        if image_layer.ndim == 3:
            annotated_z = np.nonzero(
                np.any(np.any(np.any(np.stack([l.data for l in all_label_layers]),
                              axis=0), axis=1),axis=1))[0]
        elif image_layer.ndim == 2:
            annotated_z = 0
        else:
            warnings.warn(f"Currently do not support generating metrics for image dimensions {image_layer.ndim}, so none were generated")
            return

        # mindist measurements if specified
        if self.le_mindist_label.text() != "":
            if "=" in self.le_mindist_label.text():
                mindist_layer_name, mindist_layer_i = self.le_mindist_label.text().strip().split(
                    "=")
                mindist_layer_i = int(mindist_layer_i)
                mindist_layer = self.viewer.layers[mindist_layer_name].data
                # make background nonzero for euclidean transform
                mindist_layer = np.where(mindist_layer == 0,
                                         np.max(mindist_layer) + 1,
                                         mindist_layer)
                # make specified mindist layer label zero e.g. host
                mindist_layer = np.where(mindist_layer == mindist_layer_i,
                                         0,
                                         mindist_layer)
                mindist_labels = mindist_layer_name.split("-")
                mindist_label_name = mindist_labels[mindist_layer_i - 1]
            else:
                mindist_layer_name = self.le_mindist_label.text().strip()
                mindist_layer = self.viewer.layers[mindist_layer_name].data
                # make mindist layer label zero and background nonzero
                mindist_layer = np.logical_not(mindist_layer)
                mindist_labels = mindist_layer_name.split("-")
                mindist_label_name = mindist_labels[0]

        #print(annotated_z)
        for z in annotated_z:#range(0, self.image_layer_name.shape[0]):
            label_slice_sums = {}
            print(f"Slice {z}")

            # find area of the annotation label that want to take other annots as percentage of
            # WARNING: assumes all annotations are inside PERCENTAGE_OF_LABEL annotation
            # what happens when e.g. annotate mouse gloms not just graft gloms
            percentage_of_annot = self.le_percentage_of_annot.text()
            # self.cb_label_layers_percentage_of.currentText()
            if percentage_of_annot != "":
                percentage_of_annot_slice_area = {}
                for i,percentage_annot in enumerate(percentage_of_annot.split(",")):
                    percentage_of_annot_slice = self.viewer.layers[percentage_annot].data[z, ...]
                    percentage_of_annot_label = self.le_percentage_of_annot_label.text().upper().split(",")
                    # check annot_label is specified for this percentage_of_annot layer
                    # (assuming specified in order when there are multiple layers specified)
                    if (len(percentage_of_annot_label)>=i+1) and (percentage_of_annot_label[i] != "") and (percentage_of_annot_label[i] != "ALL"):
                        percentage_annot_label = int(percentage_of_annot_label[i])
                        percentage_annot = percentage_annot.split("-")[percentage_annot_label-1]
                        percentage_of_annot_slice_area[percentage_annot] = np.count_nonzero(percentage_of_annot_slice==percentage_annot_label)
                    else:
                        percentage_of_annot_slice_area[percentage_annot] = np.count_nonzero(percentage_of_annot_slice)

            if self.le_mindist_label.text() != "":
                from scipy import ndimage
                euclidean_dist_to_label = ndimage.distance_transform_edt(
                    mindist_layer[z, ...])

            for label_layer in all_label_layers:

                # deal with 2D or 3D
                if image_layer.ndim == 3:
                    label_layer_data = label_layer.data[z, ...]
                elif image_layer.ndim == 2:
                    label_layer_data = label_layer.data

                if self.b_measure_empty_labels_slice.isChecked():
                    if label_layer_data.sum() <= 0: # skips layers with no label annotation
                        continue
                    #annotated_z.append(z)

                # label slice sums considering possible dash in label name to
                # indicate different classes in one layer
                csv_label_name = lambda x: x.name.split("-") if "-" in x.name else x.name
                if "-" not in label_layer.name:
                    label_slice_sums[label_layer.name] = np.count_nonzero(label_layer_data)
                else:
                    label_names = csv_label_name(label_layer)
                    for i in range(len(label_names)):
                        label_slice_sums[label_names[i]] = np.count_nonzero(label_layer_data==i+1)

                    # if label layer name has "-" check that it's semantic seg
                    if len(label_names) != np.max(label_layer_data):
                        raise Warning(f"Name of layer {label_layer.name} "
                                      f"contains \"-\" so ensure it is semantic"
                                      f" segmentation with label numbers "
                                      f"strictly corresponding to names between dash "
                                      f"e.g. graft-host has label 1=graft, "
                                      f"2=host, 0=unlabelled/default "
                                      f"and no other label numbers")

                print(f"  Label: {label_layer.name}")

                object_ids = []
                object_areas = []
                min_dist_to_label = []

                # per object
                for i in range(1, np.max(label_layer_data) + 1):
                    area = (label_layer_data == i).sum()
                    if area <= 0: # skips label values with no annotation
                        continue
                    object_ids.append(i)
                    object_areas.append(area)
                    #print(f"    objectID {i}: {area}")

                    # mindist calculation
                    if self.le_mindist_label.text() != "":
                        if not any([x in label_layer.name for x in mindist_labels]):
                        # exclude measuring distance between specificed labels objects as will be 0
                            mindist = np.min(
                                euclidean_dist_to_label[(label_layer_data == i)])
                            # note that euclidean distance is from pixel centre
                            # so neighbouring pixels have distance 1, diagnoal pixels distance sqrt(2)
                            min_dist_to_label.append(mindist)
                        else:
                            min_dist_to_label.append("NA")

                # check if metadata record exists and if so read in
                if image_name in self.metadata.keys():
                    image_info = self.metadata[image_name]
                else:
                    image_info_no_metadata = {"Image": [image_name], "Folder": [dirname]}
                    image_info = pd.DataFrame(image_info_no_metadata)
                object_output_df = image_info.loc[np.repeat(image_info.index, len(object_ids))]

                # generate this rows in object spreadsheet for this slice and label
                shape = "x".join([str(x) for x in self.viewer.layers[0].data.shape])
                object_output_df["image_res"] = shape
                object_output_df["z"] = z
                object_output_df["label"] = csv_label_name(label_layer)
                object_output_df["object ID"] = object_ids
                object_output_df["pixel area"] = object_areas
                object_output_df[
                    f"min euclidean distance from {mindist_label_name}"] = min_dist_to_label
                if percentage_of_annot != "":
                    for percentage_annot,slice_area in percentage_of_annot_slice_area.items():
                        object_output_df[f"% {percentage_annot} pixel area"] = [100*o/slice_area for o in object_areas]
                object_output_dfs.append(object_output_df)

            # generate rows in slice spreadsheet for each label in this slice
            slice_output_df = image_info.loc[np.repeat(image_info.index, len(label_slice_sums))]
            slice_output_df["z"] = z
            slice_output_df["label"] = label_slice_sums.keys()
            slice_output_df["pixel area"] = label_slice_sums.values()
            if percentage_of_annot != "":
                for percentage_annot, slice_area in percentage_of_annot_slice_area.items():
                    slice_output_df[f"% {percentage_annot} pixel area"] = 100 * slice_output_df[
                                                                             "pixel area"] / slice_area
            slice_output_dfs.append(slice_output_df)
            print("area annotations per slice:", label_slice_sums)

            # print("mean pixels annotated per slice {}".format(
            #    np.mean(vals[vals > 0])))

        all_object_output_df = pd.concat(object_output_dfs)
        all_slice_output_df = pd.concat(slice_output_dfs)
        print(f"Total {len(annotated_z)} slices annotated: {annotated_z}")

        # write to spreadsheet for that image
        this_image_output = os.path.splitext(image_layer.source.path)[0]+"_annotation_measurements.xlsx"
        if os.path.exists(this_image_output):
            warnings.warn(f"{this_image_output} already exists, will be writing over that file")
        with pd.ExcelWriter(this_image_output) as writer:
            all_object_output_df.to_excel(writer, sheet_name="objects", index=False)
            all_slice_output_df.to_excel(writer, sheet_name="slices", index=False)
            print(f"saved this image's metrics to  {this_image_output}")

        # write to spreadsheet with all image data, if metadata was read in and output csv specified
        if (image_name in self.metadata.keys()) and (self.le_collated_metrics_fp.text().strip() != ""):
            csv_collated_save_folder = self.le_collated_metrics_fp.text().strip()
            csv_collated_save_objects_fp = os.path.join(csv_collated_save_folder, "annotation_measurements_objects.csv")
            csv_collated_save_slices_fp = os.path.join(csv_collated_save_folder, "annotation_measurements_slices.csv")
            all_object_output_df.to_csv(csv_collated_save_objects_fp,
                                        mode='a',
                                        index=False,
                                        # avoid writing header to existing file#(not os.path.exists(csv_collated_save_objects_fp))
                                        header=True
                                        )
            all_slice_output_df.to_csv(csv_collated_save_slices_fp,
                                       mode='a',
                                       index=False,
                                       header =True# avoid writing header to existing file#(not os.path.exists(csv_collated_save_slices_fp))
                                   )
            print(f"saved this image's metrics to collated sheets {csv_collated_save_objects_fp} and {csv_collated_save_slices_fp}")
        """# can't append to xlsx sheet that already exists
        with pd.ExcelWriter(CSV_COLLATED_SAVE_FP, mode='a') as writer:
            all_object_output_df.to_excel(writer, sheet_name="objects", index=False)
            all_slice_output_df.to_excel(writer, sheet_name="slices", index=False)
        """
        #active_label = self.viewer.layers.selected
        #export_data = da.array(self.viewer.layers.selected.data)
        #label_name = self.viewer.layers.selected.title
        #print(export_data.shape)
        #print(label_name)
        #export_path = string(self.viewer.layers.Image[0].metadata) + "_" + label_name
        #print(export_path)

    # def _myfilter(self, row, parent):
    #     return "<hidden>" not in self.viewer.layers[row].name