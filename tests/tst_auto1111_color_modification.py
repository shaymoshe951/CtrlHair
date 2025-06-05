import json

import requests
import base64

from vers_image import VersImage

def save_payload(payload, filename):
    # Save payload (with base64 images stripped for readability)
    payload_to_save = json.loads(json.dumps(payload))  # Deep copy
    payload_to_save["init_images"] = ["<base64 image omitted>"]
    payload_to_save["mask"] = "<base64 mask omitted>"
    for ctrl in payload_to_save["alwayson_scripts"]["controlnet"]["args"]:
        ctrl["input_image"] = "<base64 control image omitted>"

    with open(filename, "w") as f:
        json.dump(payload_to_save, f, indent=2)

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

resolution = (512, 512)

# Input image for img2img
# init_image = encode_image("C:/Users/Lab/Downloads/my_pics/Shay0_ChatGPT Image Apr 25, 2025, 07_45_53 PM_part1.jpg")
source_image_b64 = encode_image("D:/projects/output_images_data/original_image.jpg")

# Adding Mask Image for inpaint
inpaint_mask_b64 = encode_image("D:/projects/output_images_data/inpaint_vmask.jpg")          # White = editable area
# # Adding Mask Image for controlnet segmentation structure
# hed_input_image = encode_image("D:/projects/output_images_data/segmap_vmask.jpg")          # White = editable area
# Softedge image for softedge controlnet
reference_shape_b64 = encode_image("D:/projects/output_images_data/output_image.jpg")          # White = editable area

# Reference image for IP-Adapter (same or different image)
ip_adapter_image = source_image_b64 # encode_image("C:/Users/Lab/Downloads/my_pics/Shay0_ChatGPT Image Apr 25, 2025, 07_45_53 PM_part1.jpg")

color_text = "VERY LIGHT BLONDE - DEEP IRIS"

payload = {
    "init_images": [source_image_b64],
    "mask": inpaint_mask_b64,
    # "prompt": f"Modify the color of the hair in the source image to be {color_text}.",
    # "negative_prompt": "No unwanted artifacts, maintain original style.",
    "prompt": f"hair color {color_text}. cinematic photo. 35mm photograph, film, bokeh, professional, highly detailed",
    "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed",
    "steps": 30,
    "sampler_name": "DPM++ 2M Karras",
    "cfg_scale": 7,
    "width": resolution[0],
    "height": resolution[1],
    "seed": -1,
    "denoising_strength": 0.65,
    "mask_blur": 4,
    "inpainting_fill": 1,
    "inpaint_full_res": True,
    "inpaint_full_res_padding": 32,
    "inpainting_mask_invert": 0,
    "alwayson_scripts": {
        "controlnet": {
            "args": [
                {
                    "enabled": True,
                    "module": "canny",
                    "model": "control_sd15_canny [fef5e48e]",
                    "weight": 1.0,
                    "resize_mode": "Crop and Resize",
                    "processor_res": 512,
                    "threshold_a": 100,
                    "threshold_b": 200,
                    "guidance_start": 0.0,
                    "guidance_end": 1.0,
                    "control_mode": "Balanced",
                    "pixel_perfect": True
                }
            ]
        }
    }
}

output_folder = "D:/projects/tmp/"

response = requests.post("http://127.0.0.1:7860/sdapi/v1/img2img", json=payload)
result = response.json()
vimg1 = VersImage.from_binary(base64.b64decode(result['images'][0]))


vimg1.image.show("img1: softedge_pidinet")

