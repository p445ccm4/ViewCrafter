from viewcrafter import ViewCrafter
import os
from configs.infer_config import get_parser
from utils.pvd_utils import *
from datetime import datetime
import upscale
from rife_model import load_rife_model, rife_inference_with_latents
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import export_to_video

if __name__=="__main__":
    parser = get_parser() # infer config.py
    opts = parser.parse_args()

    # opts.image_dir = "inputs/a0"
    # opts.out_dir = "outputs/a0"
    # opts.mode = 'sparse_view_horiz'
    # opts.bg_trd = 0.2
    # opts.ckpt_path = "./checkpoints/model_sparse.ckpt"
    # opts.config = "configs/inference_pvd_1024.yaml"
    # opts.ddim_steps = 50
    # opts.video_length = 25
    # opts.device = 'cuda:0'
    # opts.height = 576 
    # opts.width = 1024
    # opts.model_path = "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

    if opts.exp_name == None:
        prefix = datetime.now().strftime("%Y%m%d_%H%M")
        opts.exp_name = f'{prefix}_{os.path.splitext(os.path.basename(opts.image_dir))[0]}'
    opts.save_dir = os.path.join(opts.out_dir,opts.exp_name)
    os.makedirs(opts.save_dir,exist_ok=True)
    pvd = ViewCrafter(opts)

    if opts.mode == 'single_view_target':
        pvd.nvs_single_view()

    elif opts.mode == 'single_view_txt':
        pvd.nvs_single_view()

    elif opts.mode == 'single_view_eval':
        pvd.nvs_single_view_eval()

    elif opts.mode == 'sparse_view_interp':
        diffusion_result = pvd.nvs_sparse_view_interp()

    elif opts.mode == 'sparse_view_horiz':
        diffusion_result = pvd.nvs_sparse_view_horiz()

    else:
        raise KeyError(f"Invalid Mode: {opts.mode}")
    
    # diffusion_result.shape = (25, 576, 1024, 3) [f, h, w, c]
    diffusion_result = torch.permute(diffusion_result, (0, 3, 1, 2))
    diffusion_result = torch.unsqueeze(diffusion_result, 0)

    device = 'cuda'
    upscale_model = upscale.load_sd_upscale("./checkpoints/models/model_real_esran/RealESRGAN_x4.pth", device)
    frame_interpolation_model = load_rife_model("./checkpoints/models/model_rife")

    diffusion_result = rife_inference_with_latents(frame_interpolation_model, diffusion_result)
    diffusion_result = upscale.upscale_batch_and_concatenate(upscale_model, diffusion_result, device)
    
    diffusion_result = VaeImageProcessor.pt_to_numpy(diffusion_result[0])
    save_video_path = os.path.join(opts.out_dir, 'upscaled_interpolated_output.mp4')
    export_to_video(diffusion_result, save_video_path, fps=30)
