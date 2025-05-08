# -*- coding: utf-8 -*-

"""
# File name:    frontend.py
# Time :        2022/2/20 15:58
# Author:       xyguoo@163.com
# Description:  This is the demo frontend
"""

import sys
from global_value_utils import TEMP_FOLDER
import argparse
import os
import time

from util.common_options import ctrl_hair_parser_options

parser = argparse.ArgumentParser()
ctrl_hair_parser_options(parser)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
from ui.backend import Backend
from util.imutil import read_rgb, write_rgb

from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QApplication, QLabel, QGridLayout, \
    QSlider, QFileDialog, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QFont, QPainter, QMouseEvent
from PyQt5.QtCore import Qt
from PIL import Image


# Go over original images folder.
# per image, generate the segment map, change the hairline until segment map changes for ~1cm in both directions (generate additional 2 x segmentation and output images)
# convert the image to 512 resolution
# save all images in directories

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.input_folder = r'D:\projects\CtrlHair\data\arranged'
        self.output_folder = r'D:\projects\CtrlHair\data\output'
        self.temp_path = os.path.join(TEMP_FOLDER, 'demo_output')
        self.maximum_value = 2.0*2
        self.resolution = 256
        self.output_resolution = 512
        self.blending = not args.no_blending
        self.backend = Backend(self.maximum_value, blending=self.blending)

        self.lbl_input_img = QLabel(self)
        self.lbl_input_seg = {0: QLabel(self),
                              -1: QLabel(self),
                              1: QLabel(self)}
        self.lbl_out_img = {0: QLabel(self),
                              -1: QLabel(self),
                              1: QLabel(self)}

        self.grid = QGridLayout()
        def _add_label_to_gui(label, loc_idxs):
            self.grid.addWidget(label, loc_idxs[0], loc_idxs[1])
            label.setFixedSize(self.resolution, self.resolution)

        _add_label_to_gui(self.lbl_input_img, (1,1))
        for idx, lbl in enumerate(self.lbl_input_seg.values()):
            _add_label_to_gui(lbl, (2, idx))
        for idx, lbl in enumerate(self.lbl_out_img.values()):
            _add_label_to_gui(lbl, (3, idx))


        whole_vbox = QVBoxLayout(self)
        whole_vbox.addLayout(self.grid)

        self.setLayout(whole_vbox)
        self.setGeometry(100, 100, 900, 600)
        self.setWindowTitle('Change Hair Style')

        self.is_overlay_segment_on_output = False

        self.show()

        self.run_all()

    def run_all(self):
        for index, filename in enumerate(os.listdir(self.input_folder)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(self.input_folder, filename)
                img_input_res_path = self._resample_from_fn(img_path, self.resolution)
                self.lbl_input_img.setPixmap(QPixmap(img_input_res_path))

                delta = 0
                # convert to segment map
                img = read_rgb(img_input_res_path)
                input_img, input_parsing_show = self.backend.set_input_img(img_rgb=img)
                self._update_label_from_image(input_parsing_show, self.lbl_input_seg[delta])

                output_img = self.backend.output()
                self._update_label_from_image(output_img, self.lbl_out_img[delta])
                self._save_images(filename, input_parsing_show, output_img, delta)

                # Change bangs
                shape_val = self.backend.get_shape_be2fe()
                org_val = shape_val[1]
                for delta in (-1,1):
                    new_val = org_val + delta
                    self.backend.change_shape(new_val, 1)
                    input_parsing_show = self.backend.get_cur_mask()
                    self._update_label_from_image(input_parsing_show, self.lbl_input_seg[delta])

                    # Generate output
                    output_img = self.backend.output()
                    self._update_label_from_image(output_img, self.lbl_out_img[delta])

                    self._save_images(filename, input_parsing_show, output_img, delta)


    def _save_images(self, base_image_fn, input_parsing_show, output_img, delta):
        target_folder_seg = os.path.join(self.output_folder, 'segments'
                                     'bangs'+str(delta).replace('-','_'))
        target_folder_output = os.path.join(self.output_folder, 'output'
                                     'bangs'+str(delta).replace('-','_'))

        os.makedirs(target_folder_output, exist_ok=True)

        # Resample and save segmentation
        os.makedirs(target_folder_seg, exist_ok=True)
        input_parsing_show_res_fn = self._resample_from_rgb(input_parsing_show,
                                                         self.output_resolution)
        input_parsing_show_res = read_rgb(input_parsing_show_res_fn)
        write_rgb(os.path.join(target_folder_seg, base_image_fn),
                  input_parsing_show_res)

        # Resample and save output
        os.makedirs(target_folder_output, exist_ok=True)
        output_img_res_fn = self._resample_from_rgb(output_img,
                                                         self.output_resolution)
        output_img_res = read_rgb(output_img_res_fn)
        write_rgb(os.path.join(target_folder_output, base_image_fn),
                  output_img_res)


    def _resample_from_fn(self, inp_image_fn, resolution):
        new_size = (resolution,resolution)
        img = Image.open(inp_image_fn)
        resampled_img = img.resize(new_size, Image.Resampling.LANCZOS)
        res_img_path = os.path.join(self.temp_path, 'input_img_res.png')
        resampled_img.save(res_img_path)
        return res_img_path

    def _resample_from_rgb(self, inp_image_rgb, resolution):
        tmp_path = os.path.join(self.temp_path, 'tmp.png')
        write_rgb(tmp_path, inp_image_rgb)

        return self._resample_from_fn(tmp_path, resolution)


    def _update_label_from_image(self, img_rgb, label):
        tmp_path = os.path.join(self.temp_path, 'tmp.png')
        write_rgb(tmp_path, img_rgb)
        label.setPixmap((QPixmap(tmp_path)))


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
        vm = self.parent.val2sld[5].value()  + delta * 5
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
        self.parent.evt_output()

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
        self.initUI()
        self.target_size = 256
        self.need_crop = args.need_crop
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)
        self.font = QFont()
        self.font.setPointSize(15)
        self.setFont(self.font)

        self.input_name = None
        self.target_name = None

    def initUI(self):
        self.lbl_target_img = QLabel(self)
        self.lbl_input_img = QLabel(self)
        self.lbl_input_seg = QLabel(self)
        self.lbl_out_img = DragLabel(self) # QLabel(self)

        self.labels = [self.lbl_target_img, self.lbl_input_img,
                       self.lbl_input_seg, self.lbl_out_img]

        self.grid1 = QGridLayout()
        # tags = ['target image', 'input image', 'hair shape', 'color_texture']
        # for idx in range(len(self.labels)):
        #     self.grid1.addWidget(QLabel(tags[idx]), 0, idx)
        for idx in range(len(self.labels)):
            self.grid1.addWidget(self.labels[idx], 1, idx, alignment=Qt.AlignTop)
            self.labels[idx].setFixedSize(256, 256)

        self.btn_open_target = QPushButton('Target Image', self)
        self.btn_open_target.clicked[bool].connect(self.evt_open_target)
        self.grid1.addWidget(self.btn_open_target, 0, 0)

        self.btn_open_input = QPushButton('Input Image', self)
        self.btn_open_input.clicked[bool].connect(self.evt_open_input)
        self.grid1.addWidget(self.btn_open_input, 0, 1)

        self.grid1.addWidget(QLabel('Hair Shape'), 0, 2, alignment=Qt.AlignCenter)

        self.btn_output = QPushButton('Output', self)
        self.btn_output.clicked[bool].connect(self.evt_output)
        self.grid1.addWidget(self.btn_output, 0, 3)
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

        whole_vbox = QVBoxLayout(self)
        whole_vbox.addLayout(self.grid1)
        whole_vbox.addLayout(self.grid2)

        self.setLayout(whole_vbox)
        self.setGeometry(100, 100, 900, 600)
        self.setWindowTitle('Change Hair Style')

        self.is_overlay_segment_on_output = False

        self.show()

    def evt_push_controls(self):
        ctrl_num = self.but2val[self.sender()]
        self.is_overlay_segment_on_output = not self.is_overlay_segment_on_output
        if self.is_overlay_segment_on_output:
            output_img_merged = merge_pixmaps(self.output_pixmap, self.lbl_input_seg.pixmap())
        else:
            output_img_merged = self.output_pixmap
        self.lbl_out_img.setPixmap(output_img_merged)

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
    ex = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
