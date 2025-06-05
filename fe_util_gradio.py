import gradio as gr
import numpy as np
import colorsys
import os
import sys
from PIL import Image, ImageOps
import time
from threading import Thread
import queue

# Import your existing modules
import auto1111_if
from vers_image import VersImage
from ui.backend import Backend
from sai_if import change_style
from global_value_utils import TEMP_FOLDER

# Define hair color palettes
PALETTES = {
    'Natural': {
        'Black': '#251510',
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


class HairStyleEditor:
    def __init__(self):
        self.data = {'original': {'image': None, 'mask': None},
                     'output': {'image': None, 'mask': None, 'raw_image': None}}

        self.temp_path = os.path.join(TEMP_FOLDER, 'demo_output')
        self.maximum_value = 2.0
        self.blending = True
        self.config_flexible_edit = True
        self.config_enfore_identity_based_on_mask = True
        self.config_fix_shape_on_output = True
        self.backend = Backend(self.maximum_value, blending=self.blending)
        self.target_size = 256
        self.present_resolution = 256
        self.need_crop = False

        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

        # Initialize slider values
        self.color_values = [0, 0, 0, 0]  # Hue, Saturation, Brightness, Variance
        self.shape_values = [0, 0, 0, 0]  # Volume, Bangs, Length, Direction
        self.texture_values = [0, 0, 0]  # Curliness, Smoothness, Thickness

    def mask_image_to_binary_mask(self, mask_image):
        """Convert a mask image to a binary mask."""
        gray_mask = mask_image.to_numpy()[:, :, 2]
        binary_mask = (gray_mask > 127).astype(np.uint8) * 255
        return VersImage.from_numpy(binary_mask.astype(np.uint8))

    def process_input_image(self, image, resize_to_gr):
        """Process the input image and generate mask."""
        if image is None:
            return None, None, gr.update(interactive=False)

        # Convert PIL image to VersImage
        self.data['original']['image'] = VersImage.from_image(image)

        # Apply resizing if needed
        if resize_to_gr:
            old_ratio = 1.2
            new_ratio = (1.6 + old_ratio) / 2.0
            nw = int(image.width * new_ratio / old_ratio)
            pad_width_offset = (nw - image.width) // 2
            padded_img = ImageOps.expand(image, border=(pad_width_offset, 0, pad_width_offset, 0), fill='black')
            self.data['original']['image'] = VersImage.from_image(padded_img).resize(
                (self.present_resolution, self.present_resolution))
        else:
            self.data['original']['image'] = self.data['original']['image'].resize(
                (self.present_resolution, self.present_resolution))

        # Process with backend
        img = self.data['original']['image'].to_numpy()
        if self.need_crop:
            img = self.backend.crop_face(img)

        input_img, input_parsing_show = self.backend.set_input_img(img_rgb=img)
        self.data['original']['mask'] = VersImage.from_numpy(input_parsing_show)
        self.data['output']['mask'] = self.data['original']['mask']

        # Update slider values from backend
        self._update_sliders_from_backend()

        return (self.data['original']['image'].to_pil(),
                self.data['original']['mask'].to_pil(),
                gr.update(interactive=True))

    def _update_sliders_from_backend(self):
        """Update slider values from backend."""
        # Color values
        color_val = self.backend.get_color_be2fe()
        self.color_values = [int(v * 100) for v in color_val]

        # Shape values
        shape_val = self.backend.get_shape_be2fe()
        self.shape_values = [int(v * 100) for v in shape_val]

        # Texture values
        curliness_val = self.backend.get_curliness_be2fe()
        texture_val = self.backend.get_texture_be2fe()
        self.texture_values = [int(curliness_val * 100),
                               int(texture_val[0] * 100),
                               int(texture_val[1] * 100)]

    def update_color(self, hue, sat, bright, var):
        """Update color values."""
        for idx, value in enumerate([hue, sat, bright, var]):
            self.backend.change_color(value / 100.0, idx)
        return self.generate_output()

    def update_shape(self, volume, bangs, length, direction):
        """Update shape values and regenerate mask."""
        for idx, value in enumerate([volume, bangs, length, direction]):
            self.backend.change_shape(value / 100.0, idx)

        # Get updated mask
        input_parsing_show = self.backend.get_cur_mask()
        mask_vimage = VersImage.from_numpy(input_parsing_show)
        self.data['output']['mask'] = mask_vimage

        # Generate output with new mask
        output_img = self.generate_output()
        return output_img, mask_vimage.to_pil()

    def update_texture(self, curliness, smoothness, thickness):
        """Update texture values."""
        self.backend.change_curliness(curliness / 100.0)
        self.backend.change_texture(smoothness / 100.0, 0)
        self.backend.change_texture(thickness / 100.0, 1)
        return self.generate_output()

    def generate_output(self):
        """Generate output image."""
        if self.data['original']['image'] is None:
            return None

        output_img = self.backend.output()
        output_vimage = VersImage.from_numpy(output_img)
        self.data['output']['raw_image'] = output_vimage
        return output_vimage.to_pil()

    def apply_color_preset(self, color_name, progress=gr.Progress()):
        """Apply a preset color."""
        if self.data['original']['image'] is None:
            return None

        # Find the color in palettes
        hex_color = None
        for palette in PALETTES.values():
            if color_name in palette:
                hex_color = palette[color_name]
                break

        if hex_color is None:
            return None

        # Convert hex to RGB then HSV
        rgb_val = np.array([int(hex_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)])
        hsv_vals = colorsys.rgb_to_hsv(rgb_val[0] / 255.0, rgb_val[1] / 255.0, rgb_val[2] / 255.0)
        hsv_colors = hsv_vals[0] * 360.0, hsv_vals[1] * 255.0, hsv_vals[2] * 255.0

        for idx, val in enumerate(hsv_colors):
            self.backend.cur_latent.color['hsv'][0][idx] = val

        # Apply color modification with progress
        progress(0, desc="Starting color modification...")

        mask = self.data['output']['mask']
        if mask is None:
            mask = self.data['original']['mask']

        # Simulate progress (in real implementation, you'd get actual progress from auto1111_if)
        for i in range(10):
            progress((i + 1) * 10, desc=f"Processing... {(i + 1) * 10}%")
            time.sleep(0.5)  # Simulate processing time

        # In the actual implementation, this would call auto1111_if.color_modification
        # result_image = auto1111_if.color_modification(
        #     self.data['original']['image'],
        #     self.mask_image_to_binary_mask(mask),
        #     color_name
        # )

        # For now, return the generated output
        return self.generate_output()

    def generate_masks(self, mask_org, mask_new):
        """Generate inpaint and segmentation masks."""
        output_res = (512, 512)
        binary_mask_org = mask_org.to_numpy()[:, :, 2] > 127
        binary_mask_new = mask_new.to_numpy()[:, :, 2] > 127
        binary_mask_comb = np.logical_or(binary_mask_org, binary_mask_new)

        # Inpaint mask
        inpaint_mask_np = binary_mask_comb.astype(int) * 255
        inpaint_vmask = VersImage.from_numpy(inpaint_mask_np.astype(np.uint8)).resize(output_res)

        # Segmentation mask
        mask_new_np = mask_new.to_numpy()
        mask_new_recolor_np = mask_new_np[:, :, [0, 2, 1]]
        mask_new_recolor_np[:, :, 2] = 0
        seg_vmask = VersImage.from_numpy(mask_new_recolor_np.astype(np.uint8)).resize(output_res)

        return inpaint_vmask, seg_vmask


# Create the Gradio interface
def create_interface():
    editor = HairStyleEditor()

    with gr.Blocks(title="Hair Style Editor") as demo:
        gr.Markdown("# Hair Style Editor")
        gr.Markdown("Upload an image and adjust hair properties using the controls below.")

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Input Image", type="pil")
                resize_checkbox = gr.Checkbox(label="Resize to GR", value=True)
                process_btn = gr.Button("Process Image", variant="primary")

            with gr.Column(scale=1):
                mask_display = gr.Image(label="Hair Mask", interactive=False)

            with gr.Column(scale=1):
                output_image = gr.Image(label="Output", interactive=False)
                generate_btn = gr.Button("Generate Output", interactive=False)

        with gr.Tabs():
            with gr.Tab("Hair Shape"):
                with gr.Row():
                    color_hue = gr.Slider(-200, 200, value=0, label="Color: Hue", interactive=True)
                    color_sat = gr.Slider(-200, 200, value=0, label="Color: Saturation", interactive=True)
                    color_bright = gr.Slider(-200, 200, value=0, label="Color: Brightness", interactive=True)
                    color_var = gr.Slider(-200, 200, value=0, label="Color: Variance", interactive=True)

                with gr.Row():
                    shape_volume = gr.Slider(-200, 200, value=0, label="Shape: Volume", interactive=True)
                    shape_bangs = gr.Slider(-200, 200, value=0, label="Shape: Bangs", interactive=True)
                    shape_length = gr.Slider(-200, 200, value=0, label="Shape: Length", interactive=True)
                    shape_direction = gr.Slider(-200, 200, value=0, label="Shape: Direction", interactive=True)

                with gr.Row():
                    texture_curliness = gr.Slider(-200, 200, value=0, label="Texture: Curliness", interactive=True)
                    texture_smoothness = gr.Slider(-200, 200, value=0, label="Texture: Smoothness", interactive=True)
                    texture_thickness = gr.Slider(-200, 200, value=0, label="Texture: Thickness", interactive=True)

            with gr.Tab("Colors"):
                gr.Markdown("### Natural Colors")
                natural_colors = gr.Radio(
                    choices=list(PALETTES['Natural'].keys()),
                    label="Select a natural hair color",
                    value=None
                )

                gr.Markdown("### Dyed / Enhanced Colors")
                dyed_colors = gr.Radio(
                    choices=list(PALETTES['Dyed / Enhanced'].keys()),
                    label="Select a dyed hair color",
                    value=None
                )

                apply_color_btn = gr.Button("Apply Selected Color", variant="primary")

            with gr.Tab("Advanced"):
                gr.Markdown("### Advanced Settings")
                gr.Markdown("Additional configuration options can be added here.")

        # Event handlers
        process_btn.click(
            fn=editor.process_input_image,
            inputs=[input_image, resize_checkbox],
            outputs=[input_image, mask_display, generate_btn]
        )

        # Color sliders
        color_sliders = [color_hue, color_sat, color_bright, color_var]
        for slider in color_sliders:
            slider.change(
                fn=editor.update_color,
                inputs=color_sliders,
                outputs=output_image
            )

        # Shape sliders
        shape_sliders = [shape_volume, shape_bangs, shape_length, shape_direction]
        for slider in shape_sliders:
            slider.change(
                fn=editor.update_shape,
                inputs=shape_sliders,
                outputs=[output_image, mask_display]
            )

        # Texture sliders
        texture_sliders = [texture_curliness, texture_smoothness, texture_thickness]
        for slider in texture_sliders:
            slider.change(
                fn=editor.update_texture,
                inputs=texture_sliders,
                outputs=output_image
            )

        # Generate button
        generate_btn.click(
            fn=editor.generate_output,
            inputs=[],
            outputs=output_image
        )

        # Color preset selection
        def get_selected_color(natural, dyed):
            return natural if natural else dyed

        apply_color_btn.click(
            fn=lambda n, d: editor.apply_color_preset(get_selected_color(n, d)),
            inputs=[natural_colors, dyed_colors],
            outputs=output_image
        )

    return demo


# Main entry point
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    demo = create_interface()
    demo.launch(share=True)
    # demo.launch(share=True, server_name="0.0.0.0", server_port=7862)
