# -*- coding: utf-8 -*-

"""
# File name:    gradio_frontend.py
# Time :        2023/XX/XX HH:MM  # <-- Update time/date
# Author:       xyguoo@163.com, Gradio Conversion by AI
# Description:  Gradio frontend for CtrlHair demo
"""

import sys
import gradio as gr
import numpy as np
import torch

# --- Add project root to PYTHONPATH ---
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# --- End Path Addition ---

from global_value_utils import TEMP_FOLDER
import argparse
from util.common_options import ctrl_hair_parser_options
from ui.backend import Backend
from util.imutil import read_rgb, write_rgb

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
ctrl_hair_parser_options(parser)
default_args = ['--gpu', '0']
args, _ = parser.parse_known_args(default_args)

# --- Environment Setup ---
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if not torch.cuda.is_available():
    print("Warning: CUDA not available, running on CPU may be slow.")
    args.gpu = "-1"

TEMP_PATH = os.path.join(TEMP_FOLDER, 'gradio_demo_output')
if not os.path.exists(TEMP_PATH):
    os.makedirs(TEMP_PATH)
print(f"Using temp path: {TEMP_PATH}")

# --- Backend Initialization ---
MAXIMUM_VALUE = 2.0  # Used for Shape/Texture sliders
BLENDING = not args.no_blending
NEED_CROP = args.need_crop

try:
    backend = Backend(MAXIMUM_VALUE, blending=BLENDING)
    print("Backend initialized successfully.")
except Exception as e:
    print(f"Error initializing Backend: {e}")
    print("Please ensure models and configurations are correctly specified and accessible.")
    sys.exit(1)

# --- Color Palettes ---
PALETTES = {
    'Natural': {
        'Black': '#000000', 'Dark Brown': '#4B3621', 'Medium Brown': '#855E42',
        'Light Brown': '#A0522D', 'Blonde': '#F0E2B6', 'Platinum Blonde': '#E5E4E2',
        'Red / Ginger': '#B45144', 'Gray / White': '#D1C7C7'
    },
    'Dyed / Enhanced': {
        'Ash Brown': '#B2A38E', 'Chestnut': '#954535', 'Honey Blonde': '#DDB67D',
        'Caramel': '#A16B4E', 'Burgundy': '#800020', 'Mahogany': '#C04000',
        'Blue-Black': '#080808'
    }
}


# --- Gradio Helper Functions ---
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def save_temp_image(img_arr, filename_base):
    if img_arr.dtype != np.uint8:
        img_arr = (img_arr.clip(0, 1) * 255).astype(np.uint8)
    if img_arr.ndim == 3 and img_arr.shape[2] == 4:
        img_arr = img_arr[:, :, :3]
    elif img_arr.ndim == 2:
        img_arr = np.stack([img_arr] * 3, axis=-1)
    import uuid
    unique_id = uuid.uuid4()
    filepath = os.path.join(TEMP_PATH, f"{filename_base}_{unique_id}.png")
    try:
        write_rgb(filepath, img_arr)
        return filepath
    except Exception as e:
        print(f"Error saving image {filepath}: {e}")
        return None


def update_sliders_from_backend():
    sliders_values = {}
    shape_val = backend.get_shape_be2fe()
    sliders_values['shape_vol_slider'] = shape_val[0]
    sliders_values['shape_bang_slider'] = shape_val[1]
    sliders_values['shape_len_slider'] = shape_val[2]
    sliders_values['shape_dir_slider'] = shape_val[3]
    sliders_values['tex_curl_slider'] = backend.get_curliness_be2fe()
    app_val = backend.get_texture_be2fe()
    sliders_values['tex_smooth_slider'] = app_val[0]
    sliders_values['tex_thick_slider'] = app_val[1]
    return list(sliders_values.values())  # Ordered list of 7 values


# --- Gradio Event Handlers ---
current_input_image_path = None
current_input_processed_path = None
current_mask_path = None


def load_input_image(img_path):
    global current_input_image_path, current_input_processed_path, current_mask_path
    print(f"Loading input image: {img_path}")
    num_shape_texture_sliders = 7

    base_updates_on_clear = [
        gr.update(value=None),  # input_img display
        gr.update(value=None),  # hair_shape_mask display
        gr.update(value=None),  # output_img display (clear)
        gr.update(interactive=False),  # generate_output_btn
        gr.update(interactive=False, choices=list(PALETTES.keys()), value=None),  # palette_category_dd
        gr.update(interactive=False, choices=[], value=None),  # palette_color_dd
    ]
    slider_updates_on_clear = [gr.update(value=0.0, interactive=False) for _ in range(num_shape_texture_sliders)]
    total_updates_on_clear = tuple(base_updates_on_clear + slider_updates_on_clear)  # 6 + 7 = 13 updates

    if img_path is None:
        current_input_image_path = None
        return total_updates_on_clear

    current_input_image_path = img_path
    try:
        img = read_rgb(img_path)
        if NEED_CROP:
            print("Cropping input face...")
            img = backend.crop_face(img)
            if img is None:
                gr.Warning("Face detection failed for input image. Please try another image or disable cropping.")
                current_input_image_path = None
                return total_updates_on_clear

        print("Processing input image with backend...")
        input_img_processed, input_parsing_show = backend.set_input_img(img_rgb=img)

        input_processed_path = save_temp_image(input_img_processed, "input_processed")
        mask_path = save_temp_image(input_parsing_show, "input_mask")
        current_input_processed_path = input_processed_path
        current_mask_path = mask_path
        print("Input image processed.")

        slider_values = update_sliders_from_backend()  # List of 7 values

        output_updates_list = [
            gr.update(value=input_processed_path),  # 1. input_img
            gr.update(value=mask_path),  # 2. hair_shape_mask
            gr.update(value=None),  # 3. output_img
            gr.update(interactive=True),  # 4. generate_output_btn
            gr.update(interactive=True, choices=list(PALETTES.keys()), value=None),  # 5. palette_category_dd
            gr.update(interactive=True, choices=[], value=None),  # 6. palette_color_dd
        ]
        for val in slider_values:
            output_updates_list.append(gr.update(value=val, interactive=True))

        return tuple(output_updates_list)  # Expected 6 + 7 = 13 updates

    except Exception as e:
        print(f"Error processing input image: {e}")
        gr.Error(f"Failed to process input image: {e}")
        current_input_image_path = None
        return total_updates_on_clear


def update_color_choices(category):
    if category:
        return gr.Dropdown.update(choices=list(PALETTES[category].keys()), value=None, interactive=True)
    return gr.Dropdown.update(choices=[], value=None, interactive=False)


def on_palette_color_selected(category, color_name):
    if not current_input_image_path:
        gr.Warning("Please load an Input image first before selecting a color.")
        return  # No UI update needed here
    if not category or not color_name:
        return  # No UI update needed here

    print(f"Palette color selected: {category} - {color_name}")
    hex_color = PALETTES[category][color_name]
    rgb_tuple_255 = hex_to_rgb(hex_color)
    dummy_rgb_image = np.array([[rgb_tuple_255]], dtype=np.uint8)

    try:
        if hasattr(backend, 'apply_direct_color_from_rgb_array'):
            success = backend.apply_direct_color_from_rgb_array(dummy_rgb_image)
            if success:
                gr.Info(f"Color '{color_name}' applied. Press 'Generate Output' to see changes.")
            else:
                gr.Error(f"Failed to apply color '{color_name}'. Backend issue.")
        else:
            gr.Error("Backend method 'apply_direct_color_from_rgb_array' not found. Cannot apply palette color.")
            print("ERROR: Backend method 'apply_direct_color_from_rgb_array' is missing.")
    except Exception as e:
        print(f"Error applying palette color: {e}")
        gr.Error(f"Error applying color '{color_name}': {e}")
    # No return needed as this function now only has side effects (Info/Error popups)


def on_slider_change(slider_val, slider_id):
    global current_mask_path
    if current_input_image_path is None:
        return gr.update()

    try:
        if slider_id == 'shape_vol':
            backend.change_shape(slider_val, 0)
        elif slider_id == 'shape_bang':
            backend.change_shape(slider_val, 1)
        elif slider_id == 'shape_len':
            backend.change_shape(slider_val, 2)
        elif slider_id == 'shape_dir':
            backend.change_shape(slider_val, 3)
        elif slider_id == 'tex_curl':
            backend.change_curliness(slider_val)
        elif slider_id == 'tex_smooth':
            backend.change_texture(slider_val, 0)
        elif slider_id == 'tex_thick':
            backend.change_texture(slider_val, 1)
        else:
            print(f"Warning: Unknown slider ID '{slider_id}'")
            return gr.update()

        if 'shape' in slider_id:
            input_parsing_show = backend.get_cur_mask()
            mask_path = save_temp_image(input_parsing_show, "input_mask_slider_updated")
            current_mask_path = mask_path
            return gr.update(value=mask_path)
        else:
            return gr.update()
    except Exception as e:
        print(f"Error handling slider change for {slider_id}: {e}")
        return gr.update()


def generate_output():
    if current_input_image_path is None:
        gr.Warning("Please load an Input image first.")
        return None
    print("Generating output...")
    try:
        output_img_arr = backend.output()
        out_path = save_temp_image(output_img_arr, "output")
        print(f"Output generated: {out_path}")
        return gr.update(value=out_path)
    except Exception as e:
        print(f"Error generating output: {e}")
        gr.Error(f"Output generation failed: {e}")
        return None


# --- Build Gradio UI ---
def build_ui():
    global shape_vol_slider, shape_bang_slider, shape_len_slider, shape_dir_slider
    global tex_curl_slider, tex_smooth_slider, tex_thick_slider

    with gr.Blocks(title="CtrlHair Gradio", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# CtrlHair Demo")
        gr.Markdown(
            "Upload an Input image. Use palettes to set hair color and sliders to adjust other hair properties.")

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(label="Input Image", type="filepath", height=256, width=256)
            with gr.Column(scale=1):
                hair_shape_mask = gr.Image(label="Hair Shape (from Input)", type="filepath", interactive=False,
                                           height=256, width=256)
            with gr.Column(scale=1):
                output_img = gr.Image(label="Output Image", type="filepath", interactive=False, height=256, width=256)

        with gr.Row():
            generate_output_btn = gr.Button("Generate Output", variant="primary", interactive=False)

        with gr.Accordion("Hair Manipulation", open=True):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Color (Palette)")
                    palette_category_dd = gr.Dropdown(label="Palette Category", choices=list(PALETTES.keys()),
                                                      interactive=False)
                    palette_color_dd = gr.Dropdown(label="Color", choices=[], interactive=False)
                    # selected_color_swatch removed

                with gr.Column(scale=1):
                    gr.Markdown("### Shape")
                    shape_vol_slider = gr.Slider(minimum=-MAXIMUM_VALUE, maximum=MAXIMUM_VALUE, step=0.01,
                                                 label="Volume", value=0, interactive=False)
                    shape_bang_slider = gr.Slider(minimum=-MAXIMUM_VALUE, maximum=MAXIMUM_VALUE, step=0.01,
                                                  label="Bangs", value=0, interactive=False)
                    shape_len_slider = gr.Slider(minimum=-MAXIMUM_VALUE, maximum=MAXIMUM_VALUE, step=0.01,
                                                 label="Length", value=0, interactive=False)
                    shape_dir_slider = gr.Slider(minimum=-MAXIMUM_VALUE, maximum=MAXIMUM_VALUE, step=0.01,
                                                 label="Direction", value=0, interactive=False)
                with gr.Column(scale=1):
                    gr.Markdown("### Texture")
                    tex_curl_slider = gr.Slider(minimum=-MAXIMUM_VALUE, maximum=MAXIMUM_VALUE, step=0.01,
                                                label="Curliness", value=0, interactive=False)
                    tex_smooth_slider = gr.Slider(minimum=-MAXIMUM_VALUE, maximum=MAXIMUM_VALUE, step=0.01,
                                                  label="Smoothness", value=0, interactive=False)
                    tex_thick_slider = gr.Slider(minimum=-MAXIMUM_VALUE, maximum=MAXIMUM_VALUE, step=0.01,
                                                 label="Thickness", value=0, interactive=False)

        shape_texture_sliders = [
            shape_vol_slider, shape_bang_slider, shape_len_slider, shape_dir_slider,
            tex_curl_slider, tex_smooth_slider, tex_thick_slider
        ]
        shape_texture_slider_ids = [
            'shape_vol', 'shape_bang', 'shape_len', 'shape_dir',
            'tex_curl', 'tex_smooth', 'tex_thick'
        ]

        # --- Connect Events ---
        # This list must be flat and contain 6 base components + 7 slider components = 13 components
        # Matching the 13 gr.update objects returned by load_input_image
        outputs_for_load_input = [
                                     input_img, hair_shape_mask, output_img,
                                     generate_output_btn,
                                     palette_category_dd, palette_color_dd,
                                     # selected_color_swatch removed
                                 ] + shape_texture_sliders

        input_img.upload(
            fn=load_input_image,
            inputs=[input_img],
            outputs=outputs_for_load_input
        )

        palette_category_dd.change(
            fn=update_color_choices,
            inputs=[palette_category_dd],
            outputs=[palette_color_dd]
        )
        palette_color_dd.change(
            fn=on_palette_color_selected,
            inputs=[palette_category_dd, palette_color_dd],
            outputs=None  # No UI component is directly updated by this function anymore
        )

        generate_output_btn.click(
            fn=generate_output,
            inputs=None,
            outputs=[output_img]
        )

        for slider, slider_id_val in zip(shape_texture_sliders, shape_texture_slider_ids):
            slider.release(
                fn=on_slider_change,
                inputs=[slider, gr.Textbox(value=slider_id_val, visible=False)],
                outputs=[hair_shape_mask]
            )
    return demo


# --- Main Execution ---
if __name__ == "__main__":
    print("Building Gradio UI...")
    gradio_ui = build_ui()
    print("Launching Gradio App...")
    gradio_ui.queue().launch()