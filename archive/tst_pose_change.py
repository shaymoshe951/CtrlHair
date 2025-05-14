from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import OpenposeDetector

# Load models
# controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose")
controlnet = ControlNetModel.from_pretrained(r"C:\Users\Lab\Downloads\sd_controlnet_openpose_model")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet
)

# Detect pose from reference image
# openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
openpose = OpenposeDetector.from_pretrained(r"C:\Users\Lab\Downloads\sd_controlnet_openpose_model\control_sd15_openpose.pth")
pose = openpose(r"C:\Users\Lab\Downloads\output_split_imgs\Shay0_rp0_part1.jpg")

# Generate image
img = pipe(prompt="change pose to profile (side view). keep same identity", image=pose, num_inference_steps=50).images[0]
img.show()
