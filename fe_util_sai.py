# -*- coding: utf-8 -*-

"""
# File name:    frontend.py
# Time :        2022/2/20 15:58
# Author:       xyguoo@163.com
# Description:  This is the demo frontend
"""

import sys
sys.path.append('.')
sys.path.append('..\\')

from global_value_utils import TEMP_FOLDER
import argparse
import os

from util.common_options import ctrl_hair_parser_options

parser = argparse.ArgumentParser()
ctrl_hair_parser_options(parser)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
from ui.backend import Backend
from util.imutil import read_rgb, write_rgb

from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QApplication, QLabel, QGridLayout, \
    QSlider, QFileDialog, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QMouseEvent, QPalette, QColor
from PyQt5.QtCore import Qt
import colorsys
import numpy as np
from sai_if import change_style
from PIL import Image

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
        vm = self.parent.val2sld[5].value()  + delta * 2
        self.parent.val2sld[5].setValue(vm)
        self.parent.backend.change_shape(vm/100.0, 1)
        input_parsing_show = self.parent.backend.get_cur_mask()
        input_parsing_path = os.path.join(self.parent.temp_path, 'input_parsing.png')
        write_rgb(input_parsing_path, input_parsing_show)
        self.parent.lbl_input_seg.setPixmap((QPixmap(input_parsing_path)))

        output_img_merged = merge_pixmaps(self.parent.output_pixmap, self.parent.lbl_input_seg.pixmap())
        self.parent.lbl_out_img.setPixmap(output_img_merged)


        super().mouseMoveEvent(e)


    def mouseReleaseEvent(self, e):
        # turn it back off so hover stops firing
        self.setMouseTracking(False)
        super().mouseReleaseEvent(e)
        self.parent.is_overlay_segment_on_output = False
        # self.parent.evt_output()
        input_mask = self.parent.backend.get_cur_mask()
        input_mask_fpn = os.path.join(self.parent.temp_path, 'input_parsing.png')
        write_rgb(input_mask_fpn, input_mask)
        input_img_fpn = os.path.join(self.parent.temp_path, 'input_img.png')
        output_fpn = change_style(input_img_fpn, input_mask_fpn, self.parent.temp_path)
        # output_fpn = self.parent.temp_path + r'\edited_input_img_2825922642.jpeg'
        img = Image.open(output_fpn).resize((self.parent.present_resolution, self.parent.present_resolution), Image.Resampling.LANCZOS)

        # mask = Image.open(input_mask_fpn)
        # img_org = Image.open(input_img_fpn)
        # img_gt_np = np.array(img_org)
        # img_np = np.array(img)
        # mask_np = np.array(mask) / 255.0
        #
        # m_exp = np.expand_dims(mask_np[:, :, 2], 2)
        # new_img = img_gt_np * (1.0 - m_exp) + img_np * m_exp
        new_img = img

        data = Image.fromarray(np.uint8(new_img)).tobytes("raw", "RGBA")
        qimage = QImage(data, img.width, img.height, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimage)
        self.parent.output_pixmap = pixmap
        self.parent.lbl_out_img.setPixmap(self.parent.output_pixmap)


def merge_pixmaps(base_pixmap: QPixmap,
                  overlay_pixmap: QPixmap,
                  x: int = 0, y: int = 0,
                  opacity: float = 0.3) -> QPixmap:
    """
    Draws overlay_pixmap on top of base_pixmap at (x,y) with the given opacity.
    Returns a new QPixmap.
    """
    # Make a copy of the base so we don’t modify the originals
    result = QPixmap(base_pixmap)
    painter = QPainter(result)
    painter.setOpacity(opacity)                # how transparent the overlay is
    painter.drawPixmap(x, y, overlay_pixmap)   # paint the overlay
    painter.end()
    return result

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.temp_path = os.path.join(TEMP_FOLDER, 'demo_output')
        self.maximum_value = 2.0
        self.blending = not args.no_blending
        self.backend = Backend(self.maximum_value, blending=self.blending)
        self.target_size = 256
        self.present_resolution = 256
        self.initUI()
        self.need_crop = False # args.need_crop
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)
        self.font = QFont()
        self.font.setPointSize(15)
        self.setFont(self.font)

        self.input_name = None
        self.target_name = None
        self.output_pixmap = None

    def initUI(self):
        # self.lbl_target_img = QLabel(self)
        self.lbl_input_img = QLabel(self)
        self.lbl_input_seg = QLabel(self)
        self.lbl_out_img = DragLabel(self) # QLabel(self)

        self.labels = [self.lbl_input_img,
                       self.lbl_input_seg, self.lbl_out_img]

        self.grid1 = QGridLayout()
        # tags = ['target image', 'input image', 'hair shape', 'color_texture']
        # for idx in range(len(self.labels)):
        #     self.grid1.addWidget(QLabel(tags[idx]), 0, idx)
        for idx in range(len(self.labels)):
            self.grid1.addWidget(self.labels[idx], 1, idx, alignment=Qt.AlignTop)
            self.labels[idx].setFixedSize(self.present_resolution, self.present_resolution)

        # self.btn_open_target = QPushButton('Target Image', self)
        # self.btn_open_target.clicked[bool].connect(self.evt_open_target)
        # self.grid1.addWidget(self.btn_open_target, 0, 0)

        self.btn_open_input = QPushButton('Input Image', self)
        self.btn_open_input.clicked[bool].connect(self.evt_open_input)
        self.grid1.addWidget(self.btn_open_input, 0, 0)

        self.grid1.addWidget(QLabel('Hair Shape'), 0, 1, alignment=Qt.AlignCenter)

        self.btn_output = QPushButton('Output', self)
        self.btn_output.clicked[bool].connect(self.evt_output)
        self.grid1.addWidget(self.btn_output, 0, 2)
        self.btn_output.setEnabled(False)

        self.grid2 = QGridLayout()

        self.btn_trans_color = QPushButton('Transfer Color', self)
        self.btn_trans_color.clicked[bool].connect(self.evt_trans_color)
        self.grid2.addWidget(self.btn_trans_color, 10, 0)
        self.btn_trans_color.setEnabled(False)

        self.btn_trans_texture = QPushButton('Transfer Texture', self)
        self.btn_trans_texture.clicked[bool].connect(self.evt_trans_texture)
        self.grid2.addWidget(self.btn_trans_texture, 10, 1)
        self.btn_trans_texture.setEnabled(False)

        self.btn_trans_shape = QPushButton('Transfer Shape', self)
        self.btn_trans_shape.clicked[bool].connect(self.evt_trans_shape)
        self.grid2.addWidget(self.btn_trans_shape, 10, 2)
        self.btn_trans_shape.setEnabled(False)

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
                # self.grid2.addWidget(QLabel(self.label_total[num]), row * 2 + 2, col)
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

        whole_vbox = QVBoxLayout(self)
        whole_vbox.addLayout(self.grid1)
        whole_vbox.addLayout(self.gridColors)
        whole_vbox.addLayout(self.grid2)

        self.setLayout(whole_vbox)
        self.setGeometry(100, 100, 900, 600)
        self.setWindowTitle('Change Hair Style')

        self.is_overlay_segment_on_output = False

        self.show()

    def evt_push_controls(self):
        if self.output_pixmap is None:
            self.evt_output()
        ctrl_num = self.but2val[self.sender()]
        self.is_overlay_segment_on_output = not self.is_overlay_segment_on_output
        if self.is_overlay_segment_on_output:
            output_img_merged = merge_pixmaps(self.output_pixmap, self.lbl_input_seg.pixmap())
        else:
            output_img_merged = self.output_pixmap
        self.lbl_out_img.setPixmap(output_img_merged)

    def evt_btn_color_click(self):
        btn = self.sender()
        hex_color = btn.property("hex_color")
        rgb_val = np.array([int(hex_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)])
        hsv_vals = colorsys.rgb_to_hsv(rgb_val[0]/255.0,rgb_val[1]/255.0, rgb_val[2]/255.0)
        hsv_colors = hsv_vals[0]*360.0, hsv_vals[1]*255.0,hsv_vals[2]*255.0
        for idx, val in enumerate(hsv_colors):
            self.backend.cur_latent.color['hsv'][0][idx] = val
        self.evt_output()
        self._update_rgb_btn()


    def evt_open_target(self):
        fname = QFileDialog.getOpenFileName(self, 'Open image file')
        if fname[0]:
            self.target_name = fname[0]
            self.load_target_image(fname[0])
            if self.input_name is not None:
                self.btn_trans_color.setEnabled(True)
                self.btn_trans_shape.setEnabled(True)
                self.btn_trans_texture.setEnabled(True)

    def evt_open_input(self):
        fname = QFileDialog.getOpenFileName(self, 'Open image file')
        if fname[0]:
            input_name = fname[0]
            self.input_name = input_name
            self.load_input_image(input_name)
            self.btn_output.setEnabled(True)
            if self.target_name is not None:
                self.btn_trans_color.setEnabled(True)
                self.btn_trans_shape.setEnabled(True)
                self.btn_trans_texture.setEnabled(True)

            for kk in self.sld2val:
                kk.setEnabled(True)

    def evt_output(self):
        output_img = self.backend.output()
        img_path = os.path.join(self.temp_path, 'out_img.png')
        write_rgb(img_path, output_img)
        self.output_pixmap = QPixmap(img_path)
        if self.is_overlay_segment_on_output:
            output_img_merged = merge_pixmaps(self.output_pixmap, self.lbl_input_seg.pixmap())
        else:
            output_img_merged = self.output_pixmap
        self.lbl_out_img.setPixmap(output_img_merged)

    def evt_trans_color(self):
        self.backend.transfer_latent_representation('color')
        self.refresh_slider()

    def evt_trans_texture(self):
        self.backend.transfer_latent_representation('texture')
        self.refresh_slider()

    def evt_trans_shape(self):
        self.backend.transfer_latent_representation('shape', refresh=True)
        self.refresh_slider()
        input_parsing_show = self.backend.get_cur_mask()
        input_parsing_path = os.path.join(self.temp_path, 'input_parsing.png')
        write_rgb(input_parsing_path, input_parsing_show)
        self.lbl_input_seg.setPixmap((QPixmap(input_parsing_path)))

    def load_input_image(self, img_path):
        img = read_rgb(img_path)
        if self.need_crop:
            img = self.backend.crop_face(img)
        input_img, input_parsing_show = self.backend.set_input_img(img_rgb=img)
        input_path = os.path.join(self.temp_path, 'input_img.png')
        write_rgb(input_path, input_img)
        self.lbl_input_img.setPixmap((QPixmap(input_path)))

        input_parsing_path = os.path.join(self.temp_path, 'input_parsing.png')
        write_rgb(input_parsing_path, input_parsing_show)
        self.lbl_input_seg.setPixmap((QPixmap(input_parsing_path)))
        self.refresh_slider()

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

    def load_target_image(self, img_path):
        img = read_rgb(img_path)
        if self.need_crop:
            img = self.backend.crop_face(img)
        input_img, input_parsing_show = self.backend.set_target_img(img_rgb=img)
        input_path = os.path.join(self.temp_path, 'target_img.png')
        write_rgb(input_path, input_img)
        self.lbl_target_img.setPixmap((QPixmap(input_path)))

    def refresh_slider(self):
        idx = 0
        # color
        color_val = self.backend.get_color_be2fe()
        for ii in range(4):
            self.val2sld[idx + ii].setValue(int(color_val[ii] * 100))

        # shape
        idx += len(self.label_color)
        shape_val = self.backend.get_shape_be2fe()
        for ii in range(4):
            self.val2sld[idx + ii].setValue(int(shape_val[ii] * 100))

        # curliness
        idx += len(self.label_shape)
        self.val2sld[idx].setValue(int(self.backend.get_curliness_be2fe() * 100))
        #  texture
        idx += len(self.label_curliness)
        app_val = self.backend.get_texture_be2fe()
        for ii in range(2):
            self.val2sld[idx + ii].setValue(int(app_val[ii] * 100))

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


def main():
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
