# -*- coding: utf-8 -*-

"""
# File name:    fe_util_sai.py
# Time :        2025/5/13
# Author:       shay.moshe@gmail.com
# Description:  This is the frontend app.
"""

import sys

import auto1111_if
from vers_image import VersImage

sys.path.append('.')
#sys.path.append('./external_code/my_cython')

from global_value_utils import TEMP_FOLDER
import os

from ui.backend import Backend
from util.imutil import read_rgb, write_rgb

from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QApplication, QLabel, QGridLayout, \
    QSlider, QFileDialog, QGraphicsPixmapItem, QTabWidget, QCheckBox
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QMouseEvent, QPalette, QColor
from PyQt5.QtCore import Qt
from ui.ui_async_operations import *

import colorsys
import numpy as np
from sai_if import change_style
from PIL import Image, ImageOps

# Define hair color palettes
PALETTES = {
    'Natural': {
        'Black': '#251510', # '#000000',
        'Dark Brown': '#4B3621',
        'Medium Brown': '#855E42',
        'Light Brown': '#A0522D',
        'Blonde': '#F0E2B6',
        'Platinum Blonde': '#E5E4E2',
        'Red / Ginger': '#B45144',
        'Gray / White': '#D1C7C7'
    },
    'Dyed / Enhanced': {
        'Ash Brown': '#B2A38E',
        'Chestnut': '#954535',
        'Honey Blonde': '#DDB67D',
        'Caramel': '#A16B4E',
        'Burgundy': '#800020',
        'Mahogany': '#C04000',
        'Blue‑Black': '#080808'
    }
}

class DragLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # start with tracking off so move events only fire when pressed
        self.setMouseTracking(False)
        self.dir = 1 # [-1,0,1] # 0/1 - x/y, -1 - both

    def set_parent(self, parent):
        self.parent = parent

    def mousePressEvent(self, e):
        # enable move events now that a button is down
        self.setMouseTracking(True)
        self.pos = (e.pos().x(),e.pos().y())
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        # this fires only while a button is down
        if not self.parent.is_overlay_segment_on_output:
            return
        pos = e.pos()
        if self.dir == 0:
            delta = pos.x() - self.pos[0]
        elif self.dir == 1:
            delta = pos.y() - self.pos[1]
        else:
            raise Exception('not implemented')

        print("Dragging at", pos.x(), pos.y(), delta)
        # update slider & value
        vm = self.parent.val2sld[5].value()  + delta * 2
        self.parent.val2sld[5].setValue(vm)
        self.parent.backend.change_shape(vm/100.0, 1)
        # update mask
        input_parsing_show = self.parent.backend.get_cur_mask()
        mask_vimage = VersImage.from_numpy(input_parsing_show)
        self.parent.data['output']['mask'] = mask_vimage
        # update output with merged mask
        output_vimage = self.parent.data['output']['raw_image'].merge_image(mask_vimage)
        output_vimage.set_pixmap(self.parent.lbl_out_img)

        super().mouseMoveEvent(e)


    def mouseReleaseEvent(self, e):
        # turn it back off so hover stops firing
        self.setMouseTracking(False)
        super().mouseReleaseEvent(e)
        self.parent.is_overlay_segment_on_output = False
        if self.parent.config_flexible_edit:
            inpaint_mask, _ = self.generate_masks(self.parent.data['original']['mask'],self.parent.data['output']['mask'])
            self.parent.evt_output() # recalc output using CtrlHair
            if self.parent.config_fix_shape_on_output:
                # new_output_image = auto1111_if.shape_modification(self.parent.data['original']['image'],
                #                    self.parent.data['output']['raw_image'],
                #                    inpaint_mask)
                # self.parent.data['output']['raw_image'] = new_output_image
                # new_output_image.set_pixmap(self.parent.lbl_out_img)
                self.worker = OperationWorker(auto1111_if.shape_modification,
                              (self.parent.data['original']['image'],
                                                  self.parent.data['output']['raw_image'],
                                                  inpaint_mask),
                                              auto1111_if.get_progress, auto1111_if.cancell_process,
                                              self.parent.evt_image_generation_result,
                                              self.parent)
                self.worker.execute()
        else:
            # Get updated mask
            current_vmask = self.parent.data['output']['mask']
            output_vimage = change_style(self.parent.data['original']['image'], current_vmask)
            # output_vimage = VersImage(self.parent.temp_path + r'\edited_input_img_2825922642.jpeg') # Tmp
            output_vimage = output_vimage.resize((self.parent.present_resolution, self.parent.present_resolution))
            if self.parent.config_enfore_identity_based_on_mask:
                binary_mask_inp = self.parent.data['original']['mask'].to_numpy()[:,:,2] > 127
                binary_mask_cur = current_vmask.to_numpy()[:,:,2] > 127
                binary_mask_comb = np.logical_or(binary_mask_inp , binary_mask_cur)
                m_exp = np.expand_dims(binary_mask_comb, 2).astype(int)
                new_img_np = self.parent.data['original']['image'].resize((self.parent.present_resolution, self.parent.present_resolution)).to_numpy() * (1.0 - m_exp) + output_vimage.to_numpy() * m_exp
                output_vimage = VersImage.from_numpy(np.uint8( new_img_np ))

            output_vimage.set_pixmap(self.parent.lbl_out_img)

        # # Save
        self.parent.data['original']['image'].resize((512,512)).image.save('D:/projects/output_images_data/original_image.jpg')
        self.parent.data['output']['raw_image'].resize((512,512)).image.save('D:/projects/output_images_data/output_image.jpg')

    def generate_masks(self, mask_org, mask_new):
        # Inpaint mask should be B&W. Structure segmentation map for control net is colored: Skin-Red, Hair-Green, Clothes-Blue, Background-000
        # Inpaint is joint mask from org | new masks
        # Seg map is from mask_new
        output_res = (512,512)
        binary_mask_org = mask_org.to_numpy()[:, :, 2] > 127
        binary_mask_new = mask_new.to_numpy()[:, :, 2] > 127
        binary_mask_comb = np.logical_or(binary_mask_org, binary_mask_new)
        # inpaint_mask_np = np.expand_dims(binary_mask_comb, 2).astype(int) * 255
        inpaint_mask_np = binary_mask_comb.astype(int) * 255 
        inpaint_vmask = VersImage.from_numpy(inpaint_mask_np.astype(np.uint8)).resize(output_res)

        mask_new_np = mask_new.to_numpy()
        mask_new_recolor_np = mask_new_np[:, :, [0, 2, 1]]
        mask_new_recolor_np[:,:,2] = 0 # Zero background
        seg_vmask = VersImage.from_numpy(mask_new_recolor_np.astype(np.uint8)).resize(output_res)

        # # Save
        inpaint_vmask.image.save('D:/projects/output_images_data/inpaint_vmask1.jpg')
        seg_vmask.image.save('D:/projects/output_images_data/segmap_vmask.jpg')
        return inpaint_vmask, seg_vmask

def mask_image_to_binary_mask(mask_image):
    """
    Convert a mask image to a binary mask.
    :param mask_image: A VersImage object containing the mask image.
    :return: A binary mask as a numpy array.
    """
    # Convert the mask image to grayscale
    gray_mask = mask_image.to_numpy()[:, :, 2]  # Assuming the 3rd channel is the hair
    # Create a binary mask where non-zero values are set to 255
    binary_mask = (gray_mask > 127).astype(np.uint8) * 255
    binary_vmask = VersImage.from_numpy(binary_mask.astype(np.uint8))

    return binary_vmask

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.data = {'original' : {'image':None, 'mask':None, 'slider':None},
                     'postprocessed' : {}, 'output' : {}}

        self.temp_path = os.path.join(TEMP_FOLDER, 'demo_output')
        self.maximum_value = 2.0
        self.blending = True
        self.config_flexible_edit = True
        self.config_enfore_identity_based_on_mask = True
        self.config_fix_shape_on_output = True
        self.backend = Backend(self.maximum_value, blending=self.blending)
        self.target_size = 256
        self.present_resolution = 256
        self.initUI()
        self.need_crop = False
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)
        self.font = QFont()
        self.font.setPointSize(15)
        self.setFont(self.font)

        self.input_name = None
        self.output_pixmap = None

    def initUI(self):
        self.lbl_input_img = QLabel(self)
        self.lbl_input_seg = QLabel(self)
        self.lbl_out_img = DragLabel(self) # QLabel(self)

        self.labels = [self.lbl_input_img,
                       self.lbl_input_seg, self.lbl_out_img]

        self.grid1 = QGridLayout()
        for idx in range(len(self.labels)):
            self.grid1.addWidget(self.labels[idx], 1, idx, alignment=Qt.AlignTop)
            self.labels[idx].setFixedSize(self.present_resolution, self.present_resolution)

        self.btn_open_input = QPushButton('Input Image', self)
        self.btn_open_input.clicked[bool].connect(self.evt_open_input)
        self.grid1.addWidget(self.btn_open_input, 0, 0)

        self.grid1.addWidget(QLabel('Hair Shape'), 0, 1, alignment=Qt.AlignCenter)

        self.btn_output = QPushButton('Output', self)
        self.btn_output.clicked[bool].connect(self.evt_output)
        self.grid1.addWidget(self.btn_output, 0, 2)
        self.btn_output.setEnabled(False)

        self.grid2 = QGridLayout()

        self.sld2val = {}
        self.val2sld = {}

        self.but2val = {}
        self.val2but = {}

        self.label_color = ['Color: Hue', 'Color: Saturation', 'Color: Brightness',
                            'Color: Variance']
        self.label_shape = ['Shape: Volume', 'Shape: Bangs', 'Shape: Length', 'Shape: Direction']
        self.label_curliness = ['Texture: Curliness']
        self.label_app = ['Texture: Smoothness', 'Texture: Thickness']
        self.label_total = self.label_color + self.label_shape + self.label_curliness + self.label_app

        col_num = 4
        row_num = 3
        for row in range(row_num):
            for col in range(col_num):
                if col == 3 and row == 2:
                    continue
                num = col_num * row + col
                sld = QSlider(Qt.Horizontal, self)
                sld.setMinimum(int(-self.maximum_value * 100))
                sld.setMaximum(int(self.maximum_value * 100))
                sld.sliderMoved[int].connect(self.evt_change_value)
                self.sld2val[sld] = num
                self.val2sld[num] = sld
                new_button = QPushButton(self.label_total[num], self)
                self.but2val[new_button] = num
                self.val2but[num] = new_button
                self.grid2.addWidget(new_button, row * 2 + 2, col)
                new_button.clicked[bool].connect(self.evt_push_controls)
                self.grid2.addWidget(sld, row * 2 + 2 + 1, col)
                sld.setEnabled(False)

        self.lbl_out_img.set_parent(self)

        self.grid2.addWidget(QLabel(), 10, 3)

        self.gridColors = QGridLayout()
        self.color_btns = []
        colors = PALETTES['Natural']
        # create buttons in a grid
        cols = 4
        for idx, (name, hex_) in enumerate(colors.items()):
            btn = QPushButton(
                'abc',
            )
            # btn.setFixedSize(32, 16)
            btn.setStyleSheet(f"background-color: {hex_};color: white;")#border: 3px solid #007bff;border-radius: 5px; padding: 10px;")
            # palette = btn.palette()
            # palette.setColor(QPalette.Button, QColor(hex_))
            # btn.setAutoFillBackground(True)
            # btn.setPalette(palette)
            btn.setProperty("tag", name)
            btn.setProperty("hex_color", hex_)
            btn.clicked[bool].connect(self.evt_btn_color_click)
            self.color_btns.append(btn)
            self.gridColors.addWidget(btn, idx // cols, idx % cols)

        # Adding a checkbox for resizing, default is True
        self.checkbox_resize = QCheckBox('Resize to GR', checked=True)

        # Create the tab widget
        tabs = QTabWidget()

        # Create tab 1
        tab1 = QWidget()
        tab1_layout = QVBoxLayout()
        tab1_layout.addLayout(self.grid2) #QLabel("This is Tab 1")
        tab1.setLayout(tab1_layout)

        # Create tab 2
        tab2 = QWidget()
        tab2_layout = QVBoxLayout()
        tab2_layout.addLayout(self.gridColors)
        tab2.setLayout(tab2_layout)

        # Create tab 3
        tab3 = QWidget()
        tab3_layout = QVBoxLayout()
        tab3_layout.addWidget(self.checkbox_resize)
        tab3.setLayout(tab3_layout)

        # Add tabs
        tabs.addTab(tab1, "Hair Shape")
        tabs.addTab(tab2, "Colors")
        tabs.addTab(tab3, "Advanced")

        # # Add the tab widget to the main layout
        # layout.addWidget(tabs)
        # self.setLayout(layout)

        whole_vbox = QVBoxLayout(self)
        whole_vbox.addLayout(self.grid1)
        # whole_vbox.addLayout(self.gridColors)
        # whole_vbox.addLayout(self.grid2)
        whole_vbox.addWidget(tabs)

        container = QWidget()
        container.setLayout(whole_vbox)
        container .setLayout(whole_vbox)
        self.setGeometry(100, 100, 900, 600)
        self.setWindowTitle('Change Hair Style')
        self.setCentralWidget(container)

        self.is_overlay_segment_on_output = False

    def evt_push_controls(self):
        self.is_overlay_segment_on_output = not self.is_overlay_segment_on_output
        if self.output_pixmap is None:
            self.evt_output()
        ctrl_num = self.but2val[self.sender()]
        # TODO: Add ctrl per sender

    def evt_btn_color_click(self):
        btn = self.sender()
        hex_color = btn.property("hex_color")
        rgb_val = np.array([int(hex_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)])
        hsv_vals = colorsys.rgb_to_hsv(rgb_val[0]/255.0,rgb_val[1]/255.0, rgb_val[2]/255.0)
        hsv_colors = hsv_vals[0]*360.0, hsv_vals[1]*255.0,hsv_vals[2]*255.0
        for idx, val in enumerate(hsv_colors):
            self.backend.cur_latent.color['hsv'][0][idx] = val
        # self.evt_output()
        self._update_rgb_btn()
        color_name = btn.property("tag")
        print('Selected color:', color_name)
        mask = self.data['output']['mask']
        if mask is None:
            print('No mask found. Using original mask')
            mask = self.data['original']['mask']
        self.worker = OperationWorker(auto1111_if.color_modification, (self.data['original']['image'], mask_image_to_binary_mask(mask), color_name),
                                      auto1111_if.get_progress, auto1111_if.cancell_process, self.evt_image_generation_result,
                                      self)
        self.worker.execute()

    def evt_image_generation_result(self, result_image):
        self.data['output']['raw_image'] = result_image
        result_image.set_pixmap(self.lbl_out_img)

    def evt_open_input(self):
        fname = QFileDialog.getOpenFileName(self, 'Open image file')
        if fname[0]:
            input_name = fname[0]
            self.data['original']['image'] = VersImage(input_name)
            self.input_name = input_name
            self.load_input_image()
            self.btn_output.setEnabled(True)

            for kk in self.sld2val:
                kk.setEnabled(True)

    def evt_output(self):
        output_img = self.backend.output()
        output_raw_vimage = VersImage.from_numpy(output_img)
        # img_path = os.path.join(self.temp_path, 'out_img.png')
        # write_rgb(img_path, output_img)
        # self.output_pixmap = QPixmap(img_path)
        if self.is_overlay_segment_on_output:
            output_vimage = output_raw_vimage.merge_image(self.data['output']['mask'])
        else:
            output_vimage = output_raw_vimage
        output_vimage.set_pixmap(self.lbl_out_img)
        self.data['output']['raw_image'] = output_raw_vimage
        self.data['output']['image'] = output_vimage

    def load_input_image(self):
        if self.checkbox_resize.isChecked():
            old_ratio = 1.2
            new_ratio = (1.6+old_ratio) / 2.0
            nw = int(self.data['original']['image'].image.width * new_ratio / old_ratio)
            # Create a new blank RGB image with black padding
            pad_width_offset = (nw - self.data['original']['image'].image.width) // 2
            padded_img = ImageOps.expand(self.data['original']['image'].image, border=(pad_width_offset, 0, pad_width_offset, 0), fill='black')
            self.data['original']['image'] = VersImage.from_image(padded_img).resize((self.present_resolution, self.present_resolution))
        img = self.data['original']['image'].to_numpy()
        if self.need_crop:
            img = self.backend.crop_face(img)

        input_img, input_parsing_show = self.backend.set_input_img(img_rgb=img)
        self.data['original']['mask'] = VersImage.from_numpy(input_parsing_show)
        self.data['output']['mask'] = self.data['original']['mask'] # Starting with output mask as in
        self.data['original']['image'].set_pixmap(self.lbl_input_img)

        self.data['original']['mask'].set_pixmap(self.lbl_input_seg)
        self.refresh_slider_from_be(self.data['original'])

        self.lbl_out_img.setPixmap(QPixmap(None))
        self._update_rgb_btn()

    def _update_rgb_btn(self):
        def _hex_to_rgb(hex_color):
            # Remove '#' and convert hex to RGB integers
            hex_color = hex_color.lstrip('#')
            return np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)])
        def _create_palette_as_np_array(palette):
            palette_np = np.zeros((len(palette),3))
            for idx, hex_color in enumerate(palette):
                palette_np[idx,:] = _hex_to_rgb(hex_color)
            return palette_np
        list_palette = list(PALETTES['Natural'].values())
        palette_np = _create_palette_as_np_array(list_palette)
        hsv_color = self.backend.cur_latent.color['hsv'].cpu().numpy()
        rgb_color = colorsys.hsv_to_rgb(hsv_color[0,0]/360.0, hsv_color[0,1]/255.0, hsv_color[0,2]/255.0)
        rgb_color_scaled_np = np.array(tuple(int(c * 255) for c in rgb_color))
        color_idx = np.argmin(np.square(rgb_color_scaled_np-palette_np).sum(axis=1))
        self._select_btn(list_palette, color_idx)
        print(rgb_color_scaled_np, color_idx)

    def _select_btn(self,list_palette, color_idx):
        # clear all borders
        for ind, hex_color in enumerate(list_palette):
            self.color_btns[ind].setStyleSheet(
                f"background-color: {hex_color};color: white; border: 3px solid #000000")
        self.color_btns[color_idx].setStyleSheet(
            f"background-color: {list_palette[color_idx]};color: white; border: 3px solid #007bff")


    def refresh_slider_from_be(self, dict_to_update):
        idx = 0
        # color
        color_val = self.backend.get_color_be2fe()
        dict_to_update['color_val'] = color_val
        for ii in range(4):
            self.val2sld[idx + ii].setValue(int(color_val[ii] * 100))

        # shape
        idx += len(self.label_color)
        shape_val = self.backend.get_shape_be2fe()
        dict_to_update['shape_val'] = shape_val
        for ii in range(4):
            self.val2sld[idx + ii].setValue(int(shape_val[ii] * 100))

        # curliness
        idx += len(self.label_shape)
        curliness_val = self.backend.get_curliness_be2fe()
        self.val2sld[idx].setValue(int(curliness_val * 100))
        dict_to_update['curliness'] = curliness_val

        #  texture
        idx += len(self.label_curliness)
        texture_val = self.backend.get_texture_be2fe()
        dict_to_update['texture'] = texture_val
        for ii in range(2):
            self.val2sld[idx + ii].setValue(int(texture_val[ii] * 100))

    def evt_change_value(self, sld_v):
        """
        change all sliders value
        :param v: 0-100
        :return:
        """
        v = sld_v / 100.0
        sld_idx = self.sld2val[self.sender()]
        if sld_idx < len(self.label_color):
            self.backend.change_color(v, sld_idx)
            return
        sld_idx -= len(self.label_color)
        if sld_idx < len(self.label_shape):
            self.backend.change_shape(v, sld_idx)
            input_parsing_show = self.backend.get_cur_mask()
            input_parsing_path = os.path.join(self.temp_path, 'input_parsing.png')
            write_rgb(input_parsing_path, input_parsing_show)
            self.lbl_input_seg.setPixmap((QPixmap(input_parsing_path)))
            return
        sld_idx -= len(self.label_shape)
        if sld_idx < len(self.label_curliness):
            self.backend.change_curliness(v)
            return
        sld_idx -= len(self.label_curliness)
        if sld_idx < len(self.label_app):
            self.backend.change_texture(v, sld_idx)
            return

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    app = QApplication(sys.argv)
    main_window  = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
