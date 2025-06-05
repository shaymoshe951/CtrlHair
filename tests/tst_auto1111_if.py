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

# Input image for img2img
# init_image = encode_image("C:/Users/Lab/Downloads/my_pics/Shay0_ChatGPT Image Apr 25, 2025, 07_45_53 PM_part1.jpg")
init_image = encode_image("D:/projects/output_images_data/original_image.jpg")

# Adding Mask Image for inpaint
mask_image = encode_image("D:/projects/output_images_data/inpaint_vmask.jpg")          # White = editable area
# # Adding Mask Image for controlnet segmentation structure
# hed_input_image = encode_image("D:/projects/output_images_data/segmap_vmask.jpg")          # White = editable area
# Softedge image for softedge controlnet
softedge_image = encode_image("D:/projects/output_images_data/output_image.jpg")          # White = editable area

# Reference image for IP-Adapter (same or different image)
ip_adapter_image = init_image # encode_image("C:/Users/Lab/Downloads/my_pics/Shay0_ChatGPT Image Apr 25, 2025, 07_45_53 PM_part1.jpg")



# payload = {
#     "init_images": [init_image],
#     "mask": mask_image,
#     "prompt": "cinematic photo. 35mm photograph, film, bokeh, professional, 4k, highly detailed",
#     "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed",
#     "denoising_strength": 0.55,
#     "steps": 20,
#     "sampler_name": "DPM++ 2M Karras",
#     "cfg_scale": 5,
#     "seed": -1,
#     "width": 512,
#     "height": 512,
#     "mask_blur": 4,
#     "inpainting_fill": 1,  # 0=fill, 1=original, 2=latent noise
#     "inpaint_full_res": True,
#     "inpaint_full_res_padding": 32,
#     "inpainting_mask_invert": 0,  # 0 = white = inpainted, 1 = black = inpainted
#     "alwayson_scripts": {
#         "controlnet": {
#             "args": [
#                 {
#                     "enabled": True,
#                     "module": "ip-adapter-auto",
#                     "model": "ip-adapter-plus_sd15 [c817b455]",
#                     "input_image": ip_adapter_image,
#                     "weight": 1.0,
#                     "resize_mode": "Crop and Resize",
#                     "processor_res": 512,
#                     "threshold_a": 0.5,
#                     "threshold_b": 0.5,
#                     "guidance_start": 0.0,
#                     "guidance_end": 1.0,
#                     "control_mode": "Balanced",  # Balanced
#                     "pixel_perfect": False
#                 },
#                 {
#                     "enabled": True,
#                     "module": "softedge_hed",
#                     "model": "control_sd15_softedge",  # Use correct model name if available
#                     "input_image": hed_input_image,
#                     "weight": 1.0,
#                     "resize_mode": "Crop and Resize",
#                     "processor_res": 512,
#                     "threshold_a": 0.5,
#                     "threshold_b": 0.5,
#                     "guidance_start": 0.0,
#                     "guidance_end": 1.0,
#                     "control_mode": "Balanced",
#                     "pixel_perfect": True
#                 }
#             ]
#         }
#     }
# }

payload = {
    "init_images": [init_image],
    "mask": mask_image,
    # "prompt": "cinematic photo. 35mm photograph, film, bokeh, professional, 4k, highly detailed",
    # "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed",
    "prompt": "Modify the shape of the source image based on the reference shape.",
    "negative_prompt": "No unwanted artifacts, maintain original style.",
    "steps": 30,
    "sampler_name": "DPM++ 2M Karras",
    "cfg_scale": 5.5,
    "width": 512,
    "height": 512,
    "seed": 2245589321,
    "denoising_strength": 0.7,
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
                    "module": "softedge_pidinet",
                    "model": "control_v11p_sd15_softedge [a8575a2a]",
                    "input_image": softedge_image,
                    "weight": 1.0,
                    "resize_mode": "Crop and Resize",
                    "processor_res": 512,
                    "threshold_a": 0.5,
                    "threshold_b": 0.5,
                    "guidance_start": 0.0,
                    "guidance_end": 1.0,
                    "control_mode": "Balanced",
                    "pixel_perfect": True
                },
                {
                    "enabled": True,
                    "module": "ip-adapter-auto",
                    "model": "ip-adapter-plus_sd15 [c817b455]",
                    "input_image": ip_adapter_image,
                    "weight": 1.0,
                    "resize_mode": "Crop and Resize",
                    "processor_res": 512,
                    "threshold_a": 0.5,
                    "threshold_b": 0.5,
                    "guidance_start": 0.0,
                    "guidance_end": 1.0,
                    "control_mode": "Balanced", #"ControlNet is more important",
                    "pixel_perfect": True,
                    "weight_type": "style and composition",
                    "weight_apply_to": "composition",
                    "weight_values": [1, 1, 1, 1, 0.25, 1, 0.0, 0.0, 0.0, 1, 1, 1, 1, 1, 1]
                }
            ]
        }
    }
}

output_folder = "D:/projects/tmp/"
save_payload(payload, output_folder+"payload_img2img.json")

response = requests.post("http://127.0.0.1:7860/sdapi/v1/img2img", json=payload)
result = response.json()


# Save the output image
image_data = result['images'][0]
image_bytes = base64.b64decode(image_data)

vimage = VersImage.from_binary(image_bytes)
vimage.image.show()


with open(output_folder+"output_img2img.png", "wb") as f:
    f.write(image_bytes)

print("Image saved as output_img2img.png")
