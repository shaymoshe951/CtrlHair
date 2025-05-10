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
# Adjust these paths if your directory structure is different
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# print(f"Project Root: {project_root}")
# print(f"Sys Path: {sys.path}")
# --- End Path Addition ---

from global_value_utils import TEMP_FOLDER
import argparse
from util.common_options import ctrl_hair_parser_options
from ui.backend import Backend # Assuming backend.py is in ui directory relative to project root
from util.imutil import read_rgb, write_rgb

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
ctrl_hair_parser_options(parser)
# Provide default arguments for Gradio if not running from command line
# You might need to adjust these defaults
default_args = [
    '--gpu', '0',
    # Add other necessary args with defaults if needed, e.g.
    # '--model_path', 'path/to/your/model'
    # '--config_path', 'path/to/your/config'
    # '--blending', # Add if you want blending by default
    # '--need_crop' # Add if you want cropping by default
]
args, _ = parser.parse_known_args(default_args) # Use parse_known_args if running in environments like Jupyter

# --- Environment Setup ---
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if not torch.cuda.is_available():
    print("Warning: CUDA not available, running on CPU may be slow.")
    # Optionally modify args or backend initialization for CPU
    args.gpu = "-1" # Or however your backend handles CPU


TEMP_PATH = os.path.join(TEMP_FOLDER, 'gradio_demo_output')
if not os.path.exists(TEMP_PATH):
    os.makedirs(TEMP_PATH)
print(f"Using temp path: {TEMP_PATH}")

# --- Backend Initialization ---
MAXIMUM_VALUE = 2.0
BLENDING = not args.no_blending
NEED_CROP = args.need_crop

try:
    backend = Backend(MAXIMUM_VALUE, blending=BLENDING)
    print("Backend initialized successfully.")
except Exception as e:
    print(f"Error initializing Backend: {e}")
    print("Please ensure models and configurations are correctly specified and accessible.")
    # Optionally exit or provide a dummy backend for UI testing
    sys.exit(1)


# --- Gradio Helper Functions ---

def save_temp_image(img_arr, filename_base):
    """Saves a numpy array image to the temp folder and returns the path."""
    # Ensure it's uint8
    if img_arr.dtype != np.uint8:
        img_arr = (img_arr.clip(0, 1) * 255).astype(np.uint8)

    # Ensure 3 channels (handle potential alpha channel from backend)
    if img_arr.ndim == 3 and img_arr.shape[2] == 4:
        img_arr = img_arr[:, :, :3] # Drop alpha
    elif img_arr.ndim == 2: # Handle grayscale mask
        img_arr = np.stack([img_arr]*3, axis=-1) # Convert to RGB


    # Add a unique identifier to prevent caching issues in browser/Gradio
    import uuid
    unique_id = uuid.uuid4()
    filepath = os.path.join(TEMP_PATH, f"{filename_base}_{unique_id}.png")
    try:
        write_rgb(filepath, img_arr)
        # print(f"Saved image: {filepath}, Shape: {img_arr.shape}, Dtype: {img_arr.dtype}")
        return filepath
    except Exception as e:
        print(f"Error saving image {filepath}: {e}")
        return None

def update_sliders_from_backend():
    """Gets current values from backend and formats for Gradio sliders."""
    sliders_values = {}
    # Color
    color_val = backend.get_color_be2fe()
    sliders_values['color_hue_slider'] = color_val[0]
    sliders_values['color_sat_slider'] = color_val[1]
    sliders_values['color_brt_slider'] = color_val[2]
    sliders_values['color_var_slider'] = color_val[3]
    # Shape
    shape_val = backend.get_shape_be2fe()
    sliders_values['shape_vol_slider'] = shape_val[0]
    sliders_values['shape_bang_slider'] = shape_val[1]
    sliders_values['shape_len_slider'] = shape_val[2]
    sliders_values['shape_dir_slider'] = shape_val[3]
    # Curliness
    sliders_values['tex_curl_slider'] = backend.get_curliness_be2fe()
    # Texture App
    app_val = backend.get_texture_be2fe()
    sliders_values['tex_smooth_slider'] = app_val[0]
    sliders_values['tex_thick_slider'] = app_val[1]
    return sliders_values

def create_slider_update_dict(updated_values):
    """Creates a dictionary mapping slider components to their new values."""
    # This function assumes the Gradio components (sliders) are accessible
    # via their variable names defined in the build_ui scope.
    # It's often easier to just return the values in the correct order.
    # This approach is kept for conceptual clarity but returning a tuple/list
    # matched to the `outputs` list in the event listener is usually preferred.
    return {
        color_hue_slider: gr.update(value=updated_values['color_hue_slider']),
        color_sat_slider: gr.update(value=updated_values['color_sat_slider']),
        color_brt_slider: gr.update(value=updated_values['color_brt_slider']),
        color_var_slider: gr.update(value=updated_values['color_var_slider']),
        shape_vol_slider: gr.update(value=updated_values['shape_vol_slider']),
        shape_bang_slider: gr.update(value=updated_values['shape_bang_slider']),
        shape_len_slider: gr.update(value=updated_values['shape_len_slider']),
        shape_dir_slider: gr.update(value=updated_values['shape_dir_slider']),
        tex_curl_slider: gr.update(value=updated_values['tex_curl_slider']),
        tex_smooth_slider: gr.update(value=updated_values['tex_smooth_slider']),
        tex_thick_slider: gr.update(value=updated_values['tex_thick_slider']),
    }

# --- Gradio Event Handlers ---

# Global variables to store image paths
current_input_image_path = None
current_input_processed_path = None
current_mask_path = None


def load_input_image(img_path):
    """Handles input image upload."""
    global current_input_image_path, current_input_processed_path, current_mask_path
    print(f"Loading input image: {img_path}")
    if img_path is None:
        # Return: input_img_val, mask_val, output_img_val, gen_btn_interactive, sliders_interactive, slider_vals (11)
        return (
            gr.update(value=None),  # input_img
            gr.update(value=None),  # hair_shape_mask
            gr.update(value=None),  # output_img
            gr.update(interactive=False),  # generate_output_btn
            gr.update(interactive=False),  # for all 11 sliders' interactive state
        ) + tuple(gr.update(value=0.0) for _ in range(11)) # for all 11 sliders' values


    current_input_image_path = img_path
    try:
        img = read_rgb(img_path)
        if NEED_CROP:
            print("Cropping input face...")
            img = backend.crop_face(img)
            if img is None: # Check if cropping failed
                 gr.Warning("Face detection failed for input image. Please try another image or disable cropping.")
                 return (
                    gr.update(value=None),
                    gr.update(value=None),
                    gr.update(value=None),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                 ) + tuple(gr.update(value=0.0) for _ in range(11))


        print("Processing input image with backend...")
        input_img_processed, input_parsing_show = backend.set_input_img(img_rgb=img)

        # Save processed images to temp files
        input_processed_path = save_temp_image(input_img_processed, "input_processed")
        mask_path = save_temp_image(input_parsing_show, "input_mask")
        current_input_processed_path = input_processed_path
        current_mask_path = mask_path

        print("Input image processed.")

        # Update sliders
        slider_updates = update_sliders_from_backend()

        output_updates = [
            gr.update(value=input_processed_path), # Update input image display
            gr.update(value=mask_path),            # Update mask display
            gr.update(value=None),                 # Clear output image display
            gr.update(interactive=True),           # Enable output button
            gr.update(interactive=True)            # Enable sliders (broadcast to all slider interactive controls)
        ]
        # Append slider value updates
        output_updates.extend(slider_updates.values())

        return tuple(output_updates)

    except Exception as e:
        print(f"Error processing input image: {e}")
        gr.Error(f"Failed to process input image: {e}")
        return (
            gr.update(value=None),
            gr.update(value=None),
            gr.update(value=None),
            gr.update(interactive=False),
            gr.update(interactive=False),
        ) + tuple(gr.update(value=0.0) for _ in range(11))


def on_slider_change(slider_val, slider_id):
    """Handles changes in any slider value."""
    global current_mask_path
    if current_input_image_path is None:
        # Don't do anything if input image isn't loaded
        # This prevents errors if sliders are moved before loading
        return gr.update() # No change to mask

    # print(f"Slider change: ID={slider_id}, Value={slider_val}")
    try:
        if slider_id == 'color_hue': backend.change_color(slider_val, 0)
        elif slider_id == 'color_sat': backend.change_color(slider_val, 1)
        elif slider_id == 'color_brt': backend.change_color(slider_val, 2)
        elif slider_id == 'color_var': backend.change_color(slider_val, 3)
        elif slider_id == 'shape_vol': backend.change_shape(slider_val, 0)
        elif slider_id == 'shape_bang': backend.change_shape(slider_val, 1)
        elif slider_id == 'shape_len': backend.change_shape(slider_val, 2)
        elif slider_id == 'shape_dir': backend.change_shape(slider_val, 3)
        elif slider_id == 'tex_curl': backend.change_curliness(slider_val)
        elif slider_id == 'tex_smooth': backend.change_texture(slider_val, 0)
        elif slider_id == 'tex_thick': backend.change_texture(slider_val, 1)
        else:
            print(f"Warning: Unknown slider ID '{slider_id}'")
            return gr.update() # No change

        # Update mask only if a shape slider was changed
        if 'shape' in slider_id:
            input_parsing_show = backend.get_cur_mask()
            mask_path = save_temp_image(input_parsing_show, "input_mask_slider_updated")
            current_mask_path = mask_path
            # print(f"Shape slider changed, updating mask to: {mask_path}")
            return gr.update(value=mask_path)
        else:
            # print("Non-shape slider changed, no mask update needed.")
            return gr.update() # No change needed for mask display

    except Exception as e:
        print(f"Error handling slider change for {slider_id}: {e}")
        # Optionally show a Gradio warning/error
        return gr.update() # No change on error


def generate_output():
    """Generates the final output image."""
    if current_input_image_path is None:
        gr.Warning("Please load an Input image first.")
        return None

    print("Generating output...")
    try:
        output_img = backend.output()
        out_path = save_temp_image(output_img, "output")
        print(f"Output generated: {out_path}")
        return gr.update(value=out_path)
    except Exception as e:
        print(f"Error generating output: {e}")
        gr.Error(f"Output generation failed: {e}")
        return None

# --- Build Gradio UI ---

def build_ui():
    global color_hue_slider, color_sat_slider, color_brt_slider, color_var_slider # Make them accessible for create_slider_update_dict if used
    global shape_vol_slider, shape_bang_slider, shape_len_slider, shape_dir_slider
    global tex_curl_slider, tex_smooth_slider, tex_thick_slider

    with gr.Blocks(title="CtrlHair Gradio", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# CtrlHair Demo")
        gr.Markdown("Upload an Input image, use sliders to adjust hair properties.")

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(label="Input Image", type="filepath", height=256, width=256)
            with gr.Column(scale=1):
                hair_shape_mask = gr.Image(label="Hair Shape (from Input)", type="filepath", interactive=False, height=256, width=256)
            with gr.Column(scale=1):
                output_img = gr.Image(label="Output Image", type="filepath", interactive=False, height=256, width=256)

        with gr.Row():
            generate_output_btn = gr.Button("Generate Output", variant="primary", interactive=False)

        with gr.Accordion("Hair Manipulation Sliders", open=True):
             with gr.Row():
                  with gr.Column():
                       gr.Markdown("### Color")
                       color_hue_slider = gr.Slider(minimum=-MAXIMUM_VALUE, maximum=MAXIMUM_VALUE, step=0.01, label="Hue", value=0, interactive=False)
                       color_sat_slider = gr.Slider(minimum=-MAXIMUM_VALUE, maximum=MAXIMUM_VALUE, step=0.01, label="Saturation", value=0, interactive=False)
                       color_brt_slider = gr.Slider(minimum=-MAXIMUM_VALUE, maximum=MAXIMUM_VALUE, step=0.01, label="Brightness", value=0, interactive=False)
                       color_var_slider = gr.Slider(minimum=-MAXIMUM_VALUE, maximum=MAXIMUM_VALUE, step=0.01, label="Variance", value=0, interactive=False)
                  with gr.Column():
                       gr.Markdown("### Shape")
                       shape_vol_slider = gr.Slider(minimum=-MAXIMUM_VALUE, maximum=MAXIMUM_VALUE, step=0.01, label="Volume", value=0, interactive=False)
                       shape_bang_slider = gr.Slider(minimum=-MAXIMUM_VALUE, maximum=MAXIMUM_VALUE, step=0.01, label="Bangs", value=0, interactive=False)
                       shape_len_slider = gr.Slider(minimum=-MAXIMUM_VALUE, maximum=MAXIMUM_VALUE, step=0.01, label="Length", value=0, interactive=False)
                       shape_dir_slider = gr.Slider(minimum=-MAXIMUM_VALUE, maximum=MAXIMUM_VALUE, step=0.01, label="Direction", value=0, interactive=False)
                  with gr.Column():
                       gr.Markdown("### Texture")
                       tex_curl_slider = gr.Slider(minimum=-MAXIMUM_VALUE, maximum=MAXIMUM_VALUE, step=0.01, label="Curliness", value=0, interactive=False)
                       tex_smooth_slider = gr.Slider(minimum=-MAXIMUM_VALUE, maximum=MAXIMUM_VALUE, step=0.01, label="Smoothness", value=0, interactive=False)
                       tex_thick_slider = gr.Slider(minimum=-MAXIMUM_VALUE, maximum=MAXIMUM_VALUE, step=0.01, label="Thickness", value=0, interactive=False)

        # Define all sliders in a list for easier connections
        all_sliders = [
            color_hue_slider, color_sat_slider, color_brt_slider, color_var_slider,
            shape_vol_slider, shape_bang_slider, shape_len_slider, shape_dir_slider,
            tex_curl_slider, tex_smooth_slider, tex_thick_slider
        ]
        slider_ids = [ # Corresponding IDs for the handler function
            'color_hue', 'color_sat', 'color_brt', 'color_var',
            'shape_vol', 'shape_bang', 'shape_len', 'shape_dir',
            'tex_curl', 'tex_smooth', 'tex_thick'
        ]

        # --- Connect Events ---
        input_img_outputs_list = [
            input_img,              # Display processed input (value update)
            hair_shape_mask,        # Mask display (value update)
            output_img,             # Clear output (value update)
            generate_output_btn,    # Enable/disable (interactive update)
            # Sliders for interactive state update (11 components)
            # The single gr.update(interactive=True/False) from load_input_image will be broadcast to these
            color_hue_slider, color_sat_slider, color_brt_slider, color_var_slider,
            shape_vol_slider, shape_bang_slider, shape_len_slider, shape_dir_slider,
            tex_curl_slider, tex_smooth_slider, tex_thick_slider
        ] + all_sliders # These are for slider value updates (11 components)

        input_img.upload(
            fn=load_input_image,
            inputs=[input_img],
            outputs=input_img_outputs_list
        )

        generate_output_btn.click(
            fn=generate_output,
            inputs=None,
            outputs=[output_img]
        )

        # Connect sliders using release event (triggers after user stops dragging)
        for slider, slider_id_val in zip(all_sliders, slider_ids):
            slider.release( # Use release instead of change for better performance
                fn=on_slider_change,
                inputs=[slider, gr.Textbox(value=slider_id_val, visible=False)], # Pass value and ID
                outputs=[hair_shape_mask] # Only potentially updates the mask
            )

    return demo

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure required libraries are imported within functions or globally
    # import torch # Import torch here if not already done globally (already imported)

    print("Building Gradio UI...")
    gradio_ui = build_ui()
    print("Launching Gradio App...")
    # queue() enables handling multiple requests potentially
    gradio_ui.queue().launch() # Add share=True if you need a public link