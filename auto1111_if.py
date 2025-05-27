import requests
import base64

from vers_image import VersImage

# encode image from vers_image class
def encode_image(vimage : VersImage):
    return base64.b64encode(vimage.to_streamio().getvalue()).decode()

def shape_modification(source_image, reference_shape_image, inpaint_mask_bw_image, resolution = (512,512)):
    """
    Modify the shape of the source image based on the reference shape image and inpaint mask.

    :param source_image: Path to the source image to be modified.
    :param reference_shape_image: Path to the reference shape image.
    :param inpaint_mask_image: Path to the inpaint mask image.
    :param output_image_path: Path where the modified image will be saved.
    """
    # Encode images to base64
    source_image_b64 = encode_image(source_image.resize(resolution))
    reference_shape_b64 = encode_image(reference_shape_image.resize(resolution))
    inpaint_mask_b64 = encode_image(inpaint_mask_bw_image.resize(resolution))
    ip_adapter_image = source_image_b64

    # Create payload
    payload = {
        "init_images": [source_image_b64],
        "mask": inpaint_mask_b64,
        "prompt": "Modify the shape of the source image based on the reference shape.",
        "negative_prompt": "No unwanted artifacts, maintain original style.",
        # "prompt": "cinematic photo. 35mm photograph, film, bokeh, professional, highly detailed",
        # "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed",
        "steps": 30,
        "sampler_name": "DPM++ 2M Karras",
        "cfg_scale": 5.5,
        "width": resolution[0],
        "height": resolution[1],
        "seed": -1,
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
                        "module": "ip-adapter-auto",
                        "model": "ip-adapter-plus_sd15 [c817b455]",
                        "input_image": reference_shape_b64,
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
                        "control_mode": "Balanced",  # "ControlNet is more important",
                        "pixel_perfect": True,
                        "weight_type": "style and composition",
                        "weight_apply_to": "composition",
                        "weight_values": [1, 1, 1, 1, 0.25, 1, 0.0, 0.0, 0.0, 1, 1, 1, 1, 1, 1]
                    }
                ]
            },
        }
    }

    # Send request to the API
    response = requests.post("http://127.0.0.1:7860/sdapi/v1/img2img", json=payload)
    result = response.json()

    # Save the output image
    image_data = result['images'][0]
    image_bytes = base64.b64decode(image_data)

    vimage = VersImage.from_binary(image_bytes)
    # vimage.image.show()
    return vimage

def color_modification(source_image, inpaint_mask_bw_image, color_text, resolution = (512,512)):
    """
    Modify the shape of the source image based on the reference shape image and inpaint mask.

    :param source_image: Path to the source image to be modified.
    :param reference_shape_image: Path to the reference shape image.
    :param inpaint_mask_image: Path to the inpaint mask image.
    :param output_image_path: Path where the modified image will be saved.
    """
    # Encode images to base64
    source_image_b64 = encode_image(source_image.resize(resolution))
    inpaint_mask_b64 = encode_image(inpaint_mask_bw_image.resize(resolution))

    # Create payload
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
                        "threshold_b": 200  ,
                        "guidance_start": 0.0,
                        "guidance_end": 1.0,
                        "control_mode": "Balanced",
                        "pixel_perfect": True
                    }
                ]
            }
        }
    }

    # Send request to the API
    response = requests.post("http://127.0.0.1:7860/sdapi/v1/img2img", json=payload)
    result = response.json()

    # Save the output image
    image_data = result['images'][0]
    image_bytes = base64.b64decode(image_data)

    vimage = VersImage.from_binary(image_bytes)
    # vimage.image.show()
    return vimage


def get_progress():
    API_URL = "http://127.0.0.1:7860/sdapi/v1/progress"
    r = requests.get(API_URL)
    data = r.json()
    progress = int(data['progress'] * 100)
    return progress

def cancell_process():
    API_URL = "http://127.0.0.1:7860/sdapi/v1/interrupt"
    r = requests.post(API_URL)
    if r.status_code == 200:
        print("Process cancelled successfully.")
    else:
        print(f"Failed to cancel process: {r.status_code} - {r.text}")