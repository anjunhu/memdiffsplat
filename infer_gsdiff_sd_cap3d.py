"""
This script is obsolete
python3 infer_gsdiff_sd_cap3d.py --allow_tf32  --config_file configs/gsdiff_sd15.yaml 

For each processed prompt, we save:
{index}_{seed}_{prompt}_traj.json - trajectory data
{index}_{seed}_{prompt}_multiview.png - multi-view generated image
{index}_{seed}_{prompt}_comparison.png - target vs generated comparison
{index}_{seed}_{prompt}_noise_diff_plot.png - noise difference plot
"""

import warnings
warnings.filterwarnings("ignore")  # ignore all warnings

from typing import *

import os
import re
import json
import glob
import argparse
import logging
import time

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import imageio
import torch
import torch.nn.functional as tF
from einops import rearrange
import accelerate
from accelerate import load_checkpoint_and_dispatch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler
from diffusers.models.autoencoders import AutoencoderKL
from kiui.cam import orbit_camera
from skimage.metrics import structural_similarity as ssim
from PIL import Image

from src.options import opt_dict
from src.models import GSAutoencoderKL, GSRecon, ElevEst
import src.utils.util as util
import src.utils.op_util as op_util
import src.utils.geo_util as geo_util
import src.utils.vis_util as vis_util

from extensions.diffusers_diffsplat import UNetMV2DConditionModel, StableMVDiffusionPipeline, MVControlNetModel, StableMVDiffusionControlNetPipeline

# Memorization thresholds (similar to SPAD script)
MSE_MEM_THRESHOLD = 1000.0
SSIM_MEM_THRESHOLD = 0.85

def load_prompts_from_json_and_csv(json_path, csv_path):
    """Load prompts from JSON clusters and CSV captions."""
    # Load cluster UUIDs
    with open(json_path, 'r') as f:
        clusters = json.load(f)
    
    uuid_to_prompt = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                uuid, prompt = parts
                uuid_to_prompt[uuid.strip()] = prompt.strip()
    
    prompts = []
    for category, cluster in clusters.items():
        for cluster_id, uuid_list in cluster.items():
            for uuid in uuid_list:
                if uuid in uuid_to_prompt:
                    prompts.append({
                        "uuid": uuid,
                        "caption": uuid_to_prompt[uuid],
                        "category": category,
                        "cluster_id": cluster_id
                    })
    return prompts

def calculate_image_similarity(generated_img, target_img):
    """Calculate MSE and SSIM between generated and target images."""
    # Ensure both images are the same size
    if generated_img.size != target_img.size:
        generated_img = generated_img.resize(target_img.size, Image.BILINEAR)
    
    # Convert to numpy arrays
    gen_np = np.array(generated_img).astype(np.float32)
    target_np = np.array(target_img).astype(np.float32)
    
    # Calculate MSE
    mse = np.mean((gen_np - target_np) ** 2)
    
    # Calculate SSIM
    if len(gen_np.shape) == 3:  # RGB image
        ssim_val = ssim(target_np, gen_np, multichannel=True, channel_axis=-1, data_range=255)
    else:  # Grayscale
        ssim_val = ssim(target_np, gen_np, data_range=255)
    
    return mse, ssim_val

def process_memorized_prompts_gsdiff(pipeline, gsvae, gsrecon, opt, data, savedir, args, mem_curves, device, logger):
    """Process memorized prompts from dataset and evaluate memorization for GSDiff."""
    
    # Create output directory
    output_dir = os.path.join(savedir, "gsdiff_cap3d_multiview")
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    
    # Get camera parameters similar to original script
    fxfycxcy = torch.tensor([opt.fxfy, opt.fxfy, 0.5, 0.5], device=device).float()
    V_in = opt.num_input_views
    
    for i in tqdm(range(12, len(data)), desc="Processing prompts"):
        prompt_data = data[i]
        prompt = prompt_data['caption']
        safe_prompt = re.sub(r'\W+', '_', prompt)[:50]  # Sanitize and truncate
        
        # Check if target image exists
        target_glob = os.path.join(args.target_root, "original", "*", f"{i:04d}_*.png")
        target_files = glob.glob(target_glob)
        logger.info(f"For <{safe_prompt}>, Found {len(target_files)} training images: {target_files}")
        has_target = True
        if not target_files:
            # Try alternative naming
            target_glob = os.path.join(args.target_root, f"{i:04d}_*.png")
            target_files = glob.glob(target_glob)
            if not target_files:
                has_target = False
        
        logger.info(f"Processing memorized prompt: {prompt}")
        
        # Process multiple seeds for robustness
        for seed in range(2025, 2030):
            if seed >= 0:
                generator = torch.Generator(device=device).manual_seed(seed)
            else:
                generator = None
            
            # Setup camera poses for multi-view generation
            elevation = args.elevation if args.elevation is not None else 10.
            elevations = torch.tensor([-elevation] * V_in, device=device).deg2rad().float()
            azimuths = torch.tensor([0., 90., 180., 270.][:V_in], device=device).deg2rad().float()
            radius = torch.tensor([args.distance] * V_in, device=device).float()
            input_C2W = geo_util.orbit_camera(elevations, azimuths, radius, is_degree=False)  # (V_in, 4, 4)
            input_C2W[:, :3, 1:3] *= -1  # OpenGL -> OpenCV
            input_fxfycxcy = fxfycxcy.unsqueeze(0).repeat(input_C2W.shape[0], 1)  # (V_in, 4)
            
            # Get plucker embeddings if needed
            plucker = None
            if opt.input_concat_plucker:
                H = W = opt.input_res
                plucker, _ = geo_util.plucker_ray(H, W, input_C2W.unsqueeze(0), input_fxfycxcy.unsqueeze(0))
                plucker = plucker.squeeze(0)  # (V_in, 6, H, W)
                if opt.view_concat_condition:
                    plucker = torch.cat([plucker[0:1, ...], plucker], dim=0)  # (V_in+1, 6, H, W)
            
            # Generate with noise difference tracking
            with torch.no_grad():
                with torch.autocast("cuda", torch.bfloat16 if args.half_precision else torch.float32):
                    out = pipeline(
                        None,  # No input image for text-to-3D
                        prompt=prompt, 
                        negative_prompt=args.negative_prompt,
                        num_inference_steps=args.num_inference_steps, 
                        guidance_scale=args.guidance_scale,
                        triangle_cfg_scaling=args.triangle_cfg_scaling,
                        min_guidance_scale=args.min_guidance_scale, 
                        max_guidance_scale=args.guidance_scale,
                        output_type="latent", 
                        eta=args.eta, 
                        generator=generator,
                        plucker=plucker, 
                        num_views=V_in,
                        init_std=args.init_std, 
                        init_noise_strength=args.init_noise_strength, 
                        init_bg=args.init_bg,
                        guess_mode=args.guess_mode,
                        controlnet_conditioning_scale=float(args.controlnet_scale),
                    )
                    
                    # Extract noise differences (assuming pipeline returns this)
                    noise_diffs = [diff.norm(p=2).item() for diff in out.noise_diffs] if hasattr(out, 'noise_diffs') else []
                    
                    # Decode and render
                    latents = out.images
                    latents = latents / gsvae.scaling_factor + gsvae.shift_factor
                    render_outputs = gsvae.decode_and_render_gslatents(
                        gsrecon,
                        latents, input_C2W.unsqueeze(0), input_fxfycxcy.unsqueeze(0),
                        height=args.render_res, width=args.render_res,
                        opacity_threshold=args.opacity_threshold,
                    )
                    images = render_outputs["image"].squeeze(0)  # (V_in, 3, H, W)
            
            # Convert rendered images to PIL format for comparison
            generated_views = []
            for view_idx in range(V_in):
                view_tensor = images[view_idx]  # (3, H, W)
                view_array = vis_util.tensor_to_image(view_tensor)  # Convert to numpy
                generated_views.append(Image.fromarray(view_array))
            
            # Load target image for comparison or create dummy placeholder
            if has_target:
                target_img = Image.open(target_files[0]).convert("RGB")
                # Resize target to match generated image if needed
                if target_img.size != generated_views[0].size:
                    target_img = target_img.resize(generated_views[0].size, Image.BILINEAR)
            else:
                # Create dummy zero target image matching generated image size
                target_img = Image.new('RGB', generated_views[0].size, (0, 0, 0))
            
            # Calculate similarity metrics for all views and find the best match
            best_mse = float('inf')
            best_ssim = -1.0
            best_view_idx = 0
            
            view_similarities = []
            for view_idx, generated_view in enumerate(generated_views):
                mse, ssim_val = calculate_image_similarity(generated_view, target_img)
                view_similarities.append({
                    "view_idx": int(view_idx), 
                    "mse": float(mse), 
                    "ssim": float(ssim_val)
                })
                
                # Track best match (minimal MSE and maximal SSIM)
                if mse < best_mse:
                    best_mse = mse
                if ssim_val > best_ssim:
                    best_view_idx = view_idx
                    best_ssim = ssim_val
            
            # Determine if memorized based on best matching view
            label = 0
            if best_mse < MSE_MEM_THRESHOLD or best_ssim > SSIM_MEM_THRESHOLD:
                label = 1
            
            # Create filename components
            filename_base = f"{i:04d}_{seed}_{safe_prompt}"
            
            # Save trajectory data as JSON
            traj_path = os.path.join(output_dir, f"{filename_base}_traj.json")
            traj_data = {
                "prompt": prompt,
                "index": int(i),
                "seed": int(seed),
                "memorized": int(label),
                "best_mse": float(best_mse),
                "best_ssim": float(best_ssim),
                "best_view_idx": int(best_view_idx),
                "noise_diff_norms": [float(x) for x in noise_diffs] if noise_diffs else [],
                "all_view_similarities": view_similarities,
                "cfg_scale": float(args.guidance_scale),
                "num_inference_steps": int(args.num_inference_steps),
                "num_views": int(V_in),
                "elevation": float(elevation),
                "distance": float(args.distance),
                "target_image_path": target_files[0] if target_files else None,
            }
            
            with open(traj_path, 'w') as f:
                json.dump(traj_data, f, indent=2)
            
            # Store curves for plotting
            if f"noise_diff_{label}" not in mem_curves:
                mem_curves[f"noise_diff_{label}"] = []
            mem_curves[f"noise_diff_{label}"].append(noise_diffs)
            
            # Save multi-view generated image
            multiview_tensor = rearrange(images, "v c h w -> c h (v w)")  # Concatenate views horizontally
            multiview_array = vis_util.tensor_to_image(multiview_tensor)
            multiview_path = os.path.join(output_dir, f"{filename_base}_multiview.png")
            # imageio.imsave(multiview_path, multiview_array)
            
            # Create comparison image (target + all generated views)
            total_width = target_img.width + multiview_array.shape[1]  # target + all views
            comparison_height = max(target_img.height, multiview_array.shape[0])
            comparison_img = Image.new('RGB', (total_width, comparison_height), (255, 255, 255))
            
            # Paste target image
            comparison_img.paste(target_img, (0, 0))
            # Paste all generated views
            all_views_img = Image.fromarray(multiview_array)
            comparison_img.paste(all_views_img, (target_img.width, 0))
            
            # Save comparison image
            comparison_path = os.path.join(output_dir, f"{filename_base}_comparison.png")
            comparison_img.save(comparison_path)
            
            # Create and save noise difference plot
            if noise_diffs:
                plt.figure(figsize=(10, 6))
                plt.plot(noise_diffs, 'b-', linewidth=2, label='Conditional-Unconditional Noise Diff Norms')
                plt.xlabel("Denoising Step")
                plt.ylabel("Noise Difference Norm (L2)")
                plt.title(f"GSDiff Classifier-Free Guidance Noise Differences\nPrompt: {prompt[:50]}...")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Add memorization indicator
                color = 'red' if label == 1 else 'green'
                plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                plt.text(0.02, 0.98, f"Memorized: {'Yes' if label == 1 else 'No'}", 
                        transform=plt.gca().transAxes, fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3),
                        verticalalignment='top')
                
                plot_path = os.path.join(output_dir, f"{filename_base}_noise_diff_plot.png")
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            logger.info(f"Processed {filename_base}: Best MSE={best_mse:.1f}, Best SSIM={best_ssim:.4f}, Best View={best_view_idx}, Memorized={label}")
            processed_count += 1
    
    logger.info(f"Processed {processed_count} samples total")
    return mem_curves


def main():
    parser = argparse.ArgumentParser(
        description="Infer a diffusion model for 3D object generation with Cap3D memorization tracking"
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/gsdiff_sd15.yaml",
        help="Path to the config file"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="gsdiff_gobj83k_sd15__render",
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--hdfs_dir",
        type=str,
        default=None,
        help="Path to the HDFS directory to save checkpoints"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the PRNG"
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use"
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Use half precision for inference"
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 for faster training on Ampere GPUs"
    )

    # Cap3D dataset arguments
    parser.add_argument(
        "--mem_data_path", 
        type=str, 
        required=False, 
        help="Path to memorization dataset",
        default="../one-step-extraction"
    )
    parser.add_argument(
        "--target_root", 
        type=str, 
        help="Root directory for target images",
        default="."
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to the image for reconstruction"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to the directory of images for reconstruction"
    )
    parser.add_argument(
        "--infer_from_iter",
        type=int,
        default=-1,
        help="The iteration to load the checkpoint from"
    )
    parser.add_argument(
        "--rembg_and_center",
        action="store_true",
        help="Whether or not to remove background and center the image"
    )
    parser.add_argument(
        "--rembg_model_name",
        default="u2net",
        type=str,
        help="Rembg model, see https://github.com/danielgatis/rembg#models"
    )
    parser.add_argument(
        "--border_ratio",
        default=0.2,
        type=float,
        help="Rembg output border ratio"
    )

    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="sde-dpmsolver++",
        help="Type of diffusion scheduler"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Diffusion steps for inference"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale for inference"
    )
    parser.add_argument(
        "--triangle_cfg_scaling",
        action="store_true",
        help="Whether or not to use triangle classifier-free guidance scaling"
    )
    parser.add_argument(
        "--min_guidance_scale",
        type=float,
        default=1.,
        help="Minimum of triangle cfg scaling"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.,
        help="The weight of noise for added noise in diffusion step"
    )
    parser.add_argument(
        "--init_std",
        type=float,
        default=0.,
        help="Standard deviation of Gaussian grids (cf. Instant3D) for initialization"
    )
    parser.add_argument(
        "--init_noise_strength",
        type=float,
        default=0.98,
        help="Noise strength of Gaussian grids (cf. Instant3D) for initialization"
    )
    parser.add_argument(
        "--init_bg",
        type=float,
        default=0.,
        help="Gray background of Gaussian grids for initialization"
    )
    parser.add_argument(
        "--guess_mode",
        action="store_true",
        help="Enable guess mode for ControlNet"
    )
    parser.add_argument(
        "--controlnet_scale",
        type=float,
        default=1.0,
        help="Scaling factor for ControlNet"
    )

    parser.add_argument(
        "--elevation",
        type=float,
        default=None,
        help="The elevation of rendering"
    )
    parser.add_argument(
        "--use_elevest",
        action="store_true",
        help="Whether or not to use an elevation estimation model"
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=1.4,
        help="The distance of rendering"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Caption prompt for generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt for better classifier-free guidance"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Path to the file of text prompts for generation"
    )

    parser.add_argument(
        "--render_res",
        type=int,
        default=None,
        help="Resolution of GS rendering"
    )
    parser.add_argument(
        "--opacity_threshold",
        type=float,
        default=0.,
        help="The min opacity value for filtering floater Gaussians"
    )
    parser.add_argument(
        "--opacity_threshold_ply",
        type=float,
        default=0.,
        help="The min opacity value for filtering floater Gaussians in ply file"
    )
    parser.add_argument(
        "--save_ply",
        action="store_true",
        help="Whether or not to save the generated Gaussian ply file"
    )
    parser.add_argument(
        "--output_video_type",
        type=str,
        default=None,
        help="Type of the output video"
    )

    parser.add_argument(
        "--name_by_id",
        action="store_true",
        help="Whether or not to name the output by the prompt/image ID"
    )
    parser.add_argument(
        "--eval_text_cond",
        action="store_true",
        help="Whether or not to evaluate text-conditioned generation"
    )

    parser.add_argument(
        "--load_pretrained_gsrecon",
        type=str,
        default="gsrecon_gobj265k_cnp_even4",
        help="Tag of a pretrained GSRecon in this project"
    )
    parser.add_argument(
        "--load_pretrained_gsrecon_ckpt",
        type=int,
        default=-1,
        help="Iteration of the pretrained GSRecon checkpoint"
    )
    parser.add_argument(
        "--load_pretrained_gsvae",
        type=str,
        default="gsvae_gobj265k_sd",
        help="Tag of a pretrained GSVAE in this project"
    )
    parser.add_argument(
        "--load_pretrained_gsvae_ckpt",
        type=int,
        default=-1,
        help="Iteration of the pretrained GSVAE checkpoint"
    )
    parser.add_argument(
        "--load_pretrained_controlnet",
        type=str,
        default=None,
        help="Tag of a pretrained ControlNet in this project"
    )
    parser.add_argument(
        "--load_pretrained_controlnet_ckpt",
        type=int,
        default=-1,
        help="Iteration of the pretrained ControlNet checkpoint"
    )
    parser.add_argument(
        "--load_pretrained_elevest",
        type=str,
        default="elevest_gobj265k_b_C25",
        help="Tag of a pretrained GSRecon in this project"
    )
    parser.add_argument(
        "--load_pretrained_elevest_ckpt",
        type=int,
        default=-1,
        help="Iteration of the pretrained GSRecon checkpoint"
    )

    # Parse the arguments
    args, extras = parser.parse_known_args()

    # Parse the config file
    configs = util.get_configs(args.config_file, extras)  # change yaml configs by `extras`

    # Parse the option dict
    opt = opt_dict[configs["opt_type"]]
    if "opt" in configs:
        for k, v in configs["opt"].items():
            setattr(opt, k, v)

    # Create an experiment directory using the `tag`
    if args.tag is None:
        args.tag = time.strftime("%Y-%m-%d_%H:%M") + "_" + \
            os.path.split(args.config_file)[-1].split()[0]  # config file name

    # Create the experiment directory
    exp_dir = os.path.join(args.output_dir, args.tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    infer_dir = os.path.join(exp_dir, "inference")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(infer_dir, exist_ok=True)
    if args.hdfs_dir is not None:
        args.project_hdfs_dir = args.hdfs_dir
        args.hdfs_dir = os.path.join(args.hdfs_dir, args.tag)

    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(os.path.join(args.output_dir, args.tag, "log_infer.txt"))  # output to file
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)
    logger.propagate = True  # propagate to the root logger (console)

    # Set the random seed
    if args.seed >= 0:
        accelerate.utils.set_seed(args.seed)
        logger.info(f"You have chosen to seed([{args.seed}]) the experiment [{args.tag}]\n")

    # Enable TF32 for faster training on Ampere GPUs
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Set options for image-conditioned models
    if (args.image_path is not None or args.image_dir is not None) and args.load_pretrained_controlnet is None:
        opt.prediction_type = "v_prediction"
        opt.view_concat_condition = True
        opt.input_concat_binary_mask = True
        if args.guidance_scale > 3.:
            logger.info(
                f"WARNING: guidance scale ({args.guidance_scale}) is too large for image-conditioned models. " +
                "Please set it to a smaller value (e.g., 2.0) for better results.\n"
            )

    # Initialize the model, optimizer and lr scheduler
    in_channels = 4  # hard-coded for SD 1.5/2.1
    if opt.input_concat_plucker:
        in_channels += 6
    if opt.input_concat_binary_mask:
        in_channels += 1
    unet_from_pretrained_kwargs = {
        "sample_size": opt.input_res // 8,  # `8` hard-coded for SD 1.5/2.1
        "in_channels": in_channels,
        "zero_init_conv_in": opt.zero_init_conv_in,
        "view_concat_condition": opt.view_concat_condition,
        "input_concat_plucker": opt.input_concat_plucker,
        "input_concat_binary_mask": opt.input_concat_binary_mask,
    }
    tokenizer = CLIPTokenizer.from_pretrained(opt.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(opt.pretrained_model_name_or_path, subfolder="text_encoder", variant="fp16")
    vae = AutoencoderKL.from_pretrained(opt.pretrained_model_name_or_path, subfolder="vae")

    gsvae = GSAutoencoderKL(opt)
    gsrecon = GSRecon(opt)

    if args.scheduler_type == "ddim":
        noise_scheduler = DDIMScheduler.from_pretrained(opt.pretrained_model_name_or_path, subfolder="scheduler")
    elif "dpmsolver" in args.scheduler_type:
        noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(opt.pretrained_model_name_or_path, subfolder="scheduler")
        noise_scheduler.config.algorithm_type = args.scheduler_type
    elif args.scheduler_type == "edm":
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(opt.pretrained_model_name_or_path, subfolder="scheduler")
    else:
        raise NotImplementedError(f"Scheduler [{args.scheduler_type}] is not supported by now")
    if opt.common_tricks:
        noise_scheduler.config.timestep_spacing = "trailing"
        noise_scheduler.config.rescale_betas_zero_snr = True
    if opt.prediction_type is not None:
        noise_scheduler.config.prediction_type = opt.prediction_type
    if opt.beta_schedule is not None:
        noise_scheduler.config.beta_schedule = opt.beta_schedule

    # Load checkpoint
    logger.info(f"Load checkpoint from iteration [{args.infer_from_iter}]\n")
    logger.info(os.path.join(ckpt_dir, f"{args.infer_from_iter:06d}"))
    if not os.path.exists(os.path.join(ckpt_dir, f"{args.infer_from_iter:06d}")):
        args.infer_from_iter = util.load_ckpt(
            ckpt_dir,
            args.infer_from_iter,
            args.hdfs_dir,
            None,  # `None`: not load model ckpt here
        )
    path = os.path.join(ckpt_dir, f"{args.infer_from_iter:06d}")
    os.system(f"python3 extensions/merge_safetensors.py {path}/unet_ema")  # merge safetensors for loading
    unet, loading_info = UNetMV2DConditionModel.from_pretrained_new(path, subfolder="unet_ema",
        low_cpu_mem_usage=False, ignore_mismatched_sizes=True, output_loading_info=True, **unet_from_pretrained_kwargs)
    for key in loading_info.keys():
        assert len(loading_info[key]) == 0  # no missing_keys, unexpected_keys, mismatched_keys, error_msgs

    # Load ControlNet checkpoint
    if args.load_pretrained_controlnet is not None:
        logger.info(f"Load MVUNet ControlNet checkpoint from [{args.load_pretrained_controlnet}] iteration [{args.load_pretrained_controlnet_ckpt:06d}]\n")
        path = f"out/{args.load_pretrained_controlnet}/checkpoints/{args.load_pretrained_controlnet_ckpt:06d}"
        if not os.path.exists(path):
            args.load_pretrained_controlnet_ckpt = util.load_ckpt(
                os.path.join(args.output_dir, args.load_pretrained_controlnet, "checkpoints"),
                args.load_pretrained_controlnet_ckpt,
                None if args.hdfs_dir is None else os.path.join(args.project_hdfs_dir, args.load_pretrained_controlnet),
                None,  # `None`: not load model ckpt here
            )
        controlnet = MVControlNetModel.from_unet(unet, conditioning_channels=opt.controlnet_input_channels)
        path = f"out/{args.load_pretrained_controlnet}/checkpoints/{args.load_pretrained_controlnet_ckpt:06d}"
        ckpt_path = os.path.join(path, "controlnet", "diffusion_pytorch_model.safetensors")
        load_checkpoint_and_dispatch(controlnet, ckpt_path)
    else:
        controlnet = None

    # Freeze all models
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    gsvae.requires_grad_(False)
    gsrecon.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.eval()
    vae.eval()
    gsvae.eval()
    gsrecon.eval()
    unet.eval()
    if controlnet is not None:
        controlnet.requires_grad_(False)
        controlnet.eval()

    # Load pretrained reconstruction and gsvae models
    logger.info(f"Load GSVAE checkpoint from [{args.load_pretrained_gsvae}] iteration [{args.load_pretrained_gsvae_ckpt:06d}]\n")
    gsvae = util.load_ckpt(
        os.path.join(args.output_dir, args.load_pretrained_gsvae, "checkpoints"),
        args.load_pretrained_gsvae_ckpt,
        None if args.hdfs_dir is None else os.path.join(args.project_hdfs_dir, args.load_pretrained_gsvae),
        gsvae,
    )
    logger.info(f"Load GSRecon checkpoint from [{args.load_pretrained_gsrecon}] iteration [{args.load_pretrained_gsrecon_ckpt:06d}]\n")
    gsrecon = util.load_ckpt(
        os.path.join(args.output_dir, args.load_pretrained_gsrecon, "checkpoints"),
        args.load_pretrained_gsrecon_ckpt,
        None if args.hdfs_dir is None else os.path.join(args.project_hdfs_dir, args.load_pretrained_gsrecon),
        gsrecon,
    )

    device = f"cuda:{args.gpu_id}"
    text_encoder = text_encoder.to(device)
    vae = vae.to(device)
    gsvae = gsvae.to(device)
    gsrecon = gsrecon.to(device)
    unet = unet.to(device)
    if controlnet is not None:
        controlnet = controlnet.to(device)

    # Set diffusion pipeline
    V_in = opt.num_input_views
    pipeline = StableMVDiffusionPipeline(
        text_encoder=text_encoder, tokenizer=tokenizer,
        vae=vae, unet=unet,
        scheduler=noise_scheduler,
    )
    if controlnet is not None:
        pipeline = StableMVDiffusionControlNetPipeline(
            text_encoder=text_encoder, tokenizer=tokenizer,
            vae=vae, unet=unet, controlnet=controlnet,
            scheduler=noise_scheduler,
        )
    pipeline.set_progress_bar_config(disable=False)

    # Set rendering resolution
    if args.render_res is None:
        args.render_res = opt.input_res

    # Load elevation estimation model
    if args.use_elevest:
        elevest = ElevEst(opt)
        elevest.requires_grad_(False)
        elevest.eval()

        logger.info(f"Load ElevEst checkpoint from [{args.load_pretrained_elevest}] iteration [{args.load_pretrained_elevest_ckpt:06d}]\n")
        elevest = util.load_ckpt(
            os.path.join(args.output_dir, args.load_pretrained_elevest, "checkpoints"),
            args.load_pretrained_elevest_ckpt,
            None if args.hdfs_dir is None else os.path.join(args.project_hdfs_dir, args.load_pretrained_elevest),
            elevest,
        )
        elevest = elevest.to(device)

    # Save all experimental parameters of this run to a file (args and configs)
    _ = util.save_experiment_params(args, configs, opt, infer_dir)

    # Load Cap3D dataset
    logger.info("Loading Cap3D dataset...")
    json_path = os.path.join(args.mem_data_path, "aggregated_clusters.json")
    csv_path = os.path.join(args.mem_data_path, "Cap3D_automated_Objaverse_full.csv")
    data = load_prompts_from_json_and_csv(json_path, csv_path)
    logger.info(f"Loaded {len(data)} prompts from JSON and CSV.")

    # Initialize memorization curves storage
    mem_curves = {}

    # Process memorized prompts with Cap3D dataset
    logger.info("Processing memorized prompts...")
    mem_curves = process_memorized_prompts_gsdiff(
        pipeline, gsvae, gsrecon, opt, data, infer_dir, args, mem_curves, device, logger
    )

    # Save aggregated results
    aggregated_results_path = os.path.join(infer_dir, "aggregated_memorization_results.json")
    with open(aggregated_results_path, 'w') as f:
        json.dump({
            "mem_curves": mem_curves,
            "config": vars(args),
            "total_samples": len(data)
        }, f, indent=2)

    logger.info(f"Memorization analysis complete. Results saved to {infer_dir}")

    # Create summary plot of memorization curves
    if mem_curves:
        plt.figure(figsize=(15, 10))
        
        for i, (curve_type, curves) in enumerate(mem_curves.items()):
            plt.subplot(2, 2, i + 1)
            for curve in curves:
                if curve:  # Only plot non-empty curves
                    plt.plot(curve, alpha=0.6)
            
            # Calculate and plot average
            if curves and any(curve for curve in curves):
                non_empty_curves = [curve for curve in curves if curve]
                if non_empty_curves:
                    max_len = max(len(curve) for curve in non_empty_curves)
                    padded_curves = []
                    for curve in non_empty_curves:
                        padded = curve + [curve[-1]] * (max_len - len(curve))
                        padded_curves.append(padded)
                    avg_curve = np.mean(padded_curves, axis=0)
                    plt.plot(avg_curve, 'k-', linewidth=3, label='Average')
            
            plt.title(f'{curve_type.replace("_", " ").title()} (n={len(curves)})')
            plt.xlabel('Denoising Step')
            plt.ylabel('Noise Difference Norm')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        summary_plot_path = os.path.join(infer_dir, "memorization_summary_curves.png")
        plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
        plt.close()



if __name__ == "__main__":
    main()