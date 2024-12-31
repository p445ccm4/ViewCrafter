import os
import upscale
from rife_model import load_rife_model, rife_inference_with_latents
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import export_to_video
import torch
import torchvision

input_video_path = "outputs/c2/20241231_1434_c2/diffusion.mp4"

# load
device = 'cuda'
upscale_model = upscale.load_sd_upscale("./checkpoints/models/model_real_esran/RealESRGAN_x4.pth", device)
frame_interpolation_model = load_rife_model("./checkpoints/models/model_rife")
diffusion_result, _, _ = torchvision.io.read_video(input_video_path)

# preprocess
diffusion_result = diffusion_result.to(device) / 255.0
diffusion_result = torch.permute(diffusion_result, (0, 3, 1, 2))
diffusion_result = torch.unsqueeze(diffusion_result, 0)

# inference
diffusion_result = rife_inference_with_latents(frame_interpolation_model, diffusion_result)
diffusion_result = upscale.upscale_batch_and_concatenate(upscale_model, diffusion_result.to(device), device)

# postprocess
diffusion_result = VaeImageProcessor.pt_to_numpy(diffusion_result[0])

# export
save_video_path = os.path.join(os.path.dirname(input_video_path), 'interpolated_upscaled_output2.mp4')
export_to_video(diffusion_result, save_video_path, fps=30)
