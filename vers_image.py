from PIL import Image
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
import io

class VersImage():
    def __init__(self, filename = None):
        super().__init__()
        self.filename = filename
        self.image = None
        if filename:
            self.image = Image.open(filename)

    @classmethod
    def open(cls, filename):
        obj = cls()
        obj.filename = filename
        obj.image = Image.open(filename)
        return obj

    @classmethod
    def from_numpy(cls, np_array):
        obj = cls()
        obj.image = Image.fromarray(np_array)
        return obj

    @classmethod
    def from_image(cls, image : Image):
        obj = cls()
        obj.image = image
        return obj

    @classmethod
    def from_binary(cls, binary_data):
        obj = cls()
        # Wrap it in a BytesIO buffer
        img_stream = io.BytesIO(binary_data)
        # Open as a PIL Image
        obj.image = Image.open(img_stream)
        return obj

    def to_qimage(self, resolution = None):
        if resolution:
            img_present_resolution = self.image.resize(resolution)
            img_present_resolution = Image.fromarray(np.uint8(img_present_resolution))
        else:
            img_present_resolution = self.image
        data = img_present_resolution.tobytes("raw", "RGBA")
        qimage = QImage(data, img_present_resolution.width, img_present_resolution.height, QImage.Format_RGBA8888)
        return qimage

    def to_numpy(self):
        return np.array(self.image)

    def set_pixmap(self, label):
        qimage = self.to_qimage((label.size().width(), label.size().height()))
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap)

    def merge_image(self,
                      overlay_image,
                      opacity: float = 0.3):
        """
        Draws overlay_pixmap on top of base_pixmap at (x,y) with the given opacity.
        Returns a new QPixmap.
        """
        # Make a copy of the base so we don’t modify the originals
        # Load both images
        background = self.image.convert("RGBA")
        overlay = overlay_image.image.convert("RGBA")

        # make sure both are the same size
        overlay = overlay.resize(background.size)

        # Adjust transparency
        overlay.putalpha(int(opacity*255))

        # merge the images
        result = Image.alpha_composite(background, overlay)
        return VersImage.from_image(result)

    def to_streamio(self, format='PNG'):
        img_byte_arr = io.BytesIO()
        self.image.save(img_byte_arr, format=format)

        # Move to beginning of the stream
        img_byte_arr.seek(0)
        return img_byte_arr

    def resize(self, resolution):
        return VersImage.from_image(self.image.resize(resolution))

    def to_pil(self):
        return self.image