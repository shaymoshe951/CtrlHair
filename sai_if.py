from io import BytesIO
import IPython
import json
import os
from PIL import Image
import requests
import time

from vers_image import VersImage

STABILITY_KEY = 'sk-NgoF97Yb0gQNiCDdgM0tTe5f927fuZbJi141jkphYleDOoHk'



def send_generation_request(
    host,
    params,
    files = None
):
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }

    if files is None:
        files = {}

    # Encode parameters
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != '':
        files["image"] = open(image, 'rb')
    if mask is not None and mask != '':
        files["mask"] = open(mask, 'rb')
    if len(files)==0:
        files["none"] = ''

    # Send request
    print(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    return response

def send_async_generation_request(
    host,
    params,
    files = None
):
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }

    if files is None:
        files = {}

    # Encode parameters
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != '':
        files["image"] = open(image, 'rb')
    if mask is not None and mask != '':
        files["mask"] = open(mask, 'rb')
    if len(files)==0:
        files["none"] = ''

    # Send request
    print(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    # Process async response
    response_dict = json.loads(response.text)
    generation_id = response_dict.get("id", None)
    assert generation_id is not None, "Expected id in response"

    # Loop until result or timeout
    timeout = int(os.getenv("WORKER_TIMEOUT", 500))
    start = time.time()
    status_code = 202
    while status_code == 202:
        print(f"Polling results at https://api.stability.ai/v2beta/results/{generation_id}")
        response = requests.get(
            f"https://api.stability.ai/v2beta/results/{generation_id}",
            headers={
                **headers,
                "Accept": "*/*"
            },
        )

        if not response.ok:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        status_code = response.status_code
        time.sleep(1)
        if time.time() - start > timeout:
            raise Exception(f"Timeout after {timeout} seconds")

    return response

def change_style_using_image_filenames(input_img_fn, style_img_fn):
    # input_img = r"C:\Temp\pics\Different_hairline_db\outputbangs0\ChatGPT Image Apr 14, 2025, 07_59_20 AM.png" #@param {type:"string"}
    # style_img = r"C:\Temp\pics\Different_hairline_db\segmentsbangs1\ChatGPT Image Apr 14, 2025, 07_59_20 AM.png" #@param {type:"string"}
    input_img = VersImage(input_img_fn)
    style_img = VersImage(style_img_fn)
    return change_style(input_img, style_img)

#@title Style Transfer

#@markdown - Drag and drop image to file folder on left
#@markdown - Right click it and choose Copy path
#@markdown - Paste that path into image field below
#@markdown <br><br>
def change_style(input_img, style_img, resolution=(512,512)):
    prompt = "" #@param {type:"string"}
    negative_prompt = "" #@param {type:"string"}
    style_strength = 0.8   #@param {type:"slider", min:0, max:1, step:0.05}
    composition_fidelity = 0.9   #@param {type:"slider", min:0, max:1, step:0.05}
    change_strength = 0.1   #@param {type:"slider", min:0.1, max:1, step:0.05}
    seed = 0 #@param {type:"integer"}
    output_format = "jpeg" #@param ["webp", "jpeg", "png"]

    # host = f"https://api.stability.ai/v2beta/stable-image/control/style-transfer"
    host = f"https://172.64.153.32/v2beta/stable-image/control/style-transfer"

    params = {
        "change_strength" : change_strength,
        "composition_fidelity" : composition_fidelity,
        "output_format": output_format,
        "prompt" : prompt,
        "negative_prompt" : negative_prompt,
        "seed" : seed,
        "style_strength" : style_strength,
        # "output_resolution" : f"{resolution[0]}x{resolution[1]}"
    }

    files = {}
    # files["init_image"] = input_img.to_streamio(output_format) # open(input_img, 'rb')
    # files["style_image"] = style_img.to_streamio(output_format) # open(style_img, 'rb')
    files["init_image"] = open(r"D:\projects\CtrlHair\temp_folder\demo_output\input_img.png", 'rb')
    files["style_image"] = open(r"D:\projects\CtrlHair\temp_folder\demo_output\input_parsing.png", 'rb')

    response = send_generation_request(
        host,
        params,
        files
    )

    # Decode response
    output_image = VersImage.from_binary(response.content)
    finish_reason = response.headers.get("finish-reason")
    # seed = response.headers.get("seed")

    # Check for NSFW classification
    if finish_reason == 'CONTENT_FILTERED':
        raise Warning("Generation failed NSFW classifier")

    return output_image

