# DiffSplat/run_baseline.py - Updated for DiffSplat attention controller

import os
import sys
import argparse
import re
import json
import time
import logging
import gc

import torch
import wandb
from tqdm import tqdm
from omegaconf import OmegaConf
from PIL import Image
import numpy as np

# --- Path Setup ---
sys.path.append(os.path.join(os.path.dirname(__file__)))

# --- DiffSplat & Diffusers Imports ---
from extensions.diffusers_diffsplat import UNetMV2DConditionModel, StableMVDiffusionPipeline, MVControlNetModel, StableMVDiffusionControlNetPipeline
from src.models import GSAutoencoderKL, GSRecon, ElevEst
from src.options import opt_dict
import src.utils.util as util
import src.utils.geo_util as geo_util

from diffusers.schedulers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler
from diffusers.models.autoencoders import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel

# --- Memorization Framework Imports ---
from memorization.data.dataloaders import DatasetManager
from memorization.evaluation.evaluator import DiffSplatEvaluator, save_run_outputs, multiview_tensor_to_images
from memorization.metrics import (
    NoiseDiffNormMetric, HessianMetric,
    DiversityMetric, BrightEndingMetric, XAttnEntropyMetric,
    InvMMMetric, PLaplaceMetric
)
# --- CORRECTED IMPORTS ---
from memorization.controller import AttentionStore

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def cleanup_memory():
    """Aggressive memory cleanup."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def setup_camera_parameters(opt, device, num_views=4, elevation=10.0, distance=1.4):
    """Setup camera parameters for multi-view generation."""
    
    # Get camera intrinsics
    fxfycxcy = torch.tensor([opt.fxfy, opt.fxfy, 0.5, 0.5], device=device).float()
    
    # Setup camera poses for multi-view generation
    elevations = torch.tensor([-elevation] * num_views, device=device).deg2rad().float()
    azimuths = torch.tensor([0., 90., 180., 270.][:num_views], device=device).deg2rad().float()
    radius = torch.tensor([distance] * num_views, device=device).float()
    
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
    
    return {
        'input_C2W': input_C2W,
        'input_fxfycxcy': input_fxfycxcy, 
        'plucker': plucker,
        'num_views': num_views,
        'elevation': elevation,
        'distance': distance,
        'fxfycxcy': fxfycxcy
    }


def load_diffsplat_models(cfg, device):
    """Load all DiffSplat models and pipeline components."""
    
    # Parse the config file and options
    configs = util.get_configs(cfg.config_file, [])
    opt = opt_dict[configs["opt_type"]]
    if "opt" in configs:
        for k, v in configs["opt"].items():
            setattr(opt, k, v)
    
    # Initialize UNet
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
    
    # Load base models from opt.pretrained_model_name_or_path (from config)
    tokenizer = CLIPTokenizer.from_pretrained(opt.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(opt.pretrained_model_name_or_path, subfolder="text_encoder", variant="fp16")
    vae = AutoencoderKL.from_pretrained(opt.pretrained_model_name_or_path, subfolder="vae")
    
    # Load custom models
    gsvae = GSAutoencoderKL(opt)
    gsrecon = GSRecon(opt)
    
    # Load scheduler
    if cfg.scheduler_type == "ddim":
        noise_scheduler = DDIMScheduler.from_pretrained(opt.pretrained_model_name_or_path, subfolder="scheduler")
    elif "dpmsolver" in cfg.scheduler_type:
        noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(opt.pretrained_model_name_or_path, subfolder="scheduler")
        noise_scheduler.config.algorithm_type = cfg.scheduler_type
    elif cfg.scheduler_type == "edm":
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(opt.pretrained_model_name_or_path, subfolder="scheduler")
    else:
        raise NotImplementedError(f"Scheduler [{cfg.scheduler_type}] is not supported")
        
    if opt.common_tricks:
        noise_scheduler.config.timestep_spacing = "trailing"
        noise_scheduler.config.rescale_betas_zero_snr = True
    if opt.prediction_type is not None:
        noise_scheduler.config.prediction_type = opt.prediction_type
    if opt.beta_schedule is not None:
        noise_scheduler.config.beta_schedule = opt.beta_schedule

    cfg.exp_dir = os.path.join(cfg.output_dir, cfg.tag)
    cfg.ckpt_dir = os.path.join(cfg.exp_dir, "checkpoints")

    # Load UNet checkpoint
    logger.info(f"Loading UNet from checkpoint: {cfg.ckpt_dir}")
    ckpt_dir = cfg.ckpt_dir
    print(ckpt_dir, cfg.infer_from_iter)
    path = os.path.join(ckpt_dir, f"{cfg.infer_from_iter:06d}")
        
    if not os.path.exists(path):
        # Try to load using util.load_ckpt logic
        infer_from_iter = util.load_ckpt(
            ckpt_dir,
            cfg.infer_from_iter,
            cfg.get('hdfs_dir'),
            None,  # `None`: not load model ckpt here
        )
        path = os.path.join(ckpt_dir, f"{infer_from_iter:06d}")
    
    os.system(f"python3 extensions/merge_safetensors.py {path}/unet_ema")  # merge safetensors for loading
    unet, loading_info = UNetMV2DConditionModel.from_pretrained_new(
        path, subfolder="unet_ema",
        low_cpu_mem_usage=False, 
        ignore_mismatched_sizes=True, 
        output_loading_info=True, 
        **unet_from_pretrained_kwargs
    )
    
    for key in loading_info.keys():
        assert len(loading_info[key]) == 0  # no missing_keys, unexpected_keys, mismatched_keys, error_msgs
    
    # Load ControlNet if specified
    controlnet = None
    if cfg.get('load_pretrained_controlnet'):
        logger.info(f"Loading ControlNet from {cfg.load_pretrained_controlnet}")
        controlnet_path = f"out/{cfg.load_pretrained_controlnet}/checkpoints/{cfg.load_pretrained_controlnet_ckpt:06d}"
        controlnet = MVControlNetModel.from_unet(unet, conditioning_channels=opt.controlnet_input_channels)
        ckpt_path = os.path.join(controlnet_path, "controlnet", "diffusion_pytorch_model.safetensors")
        from accelerate import load_checkpoint_and_dispatch
        load_checkpoint_and_dispatch(controlnet, ckpt_path)
    
    # Load pretrained GSVAE and GSRecon
    logger.info(f"Loading GSVAE from {cfg.load_pretrained_gsvae}")
    gsvae = util.load_ckpt(
        os.path.join("out", cfg.load_pretrained_gsvae, "checkpoints"),
        cfg.load_pretrained_gsvae_ckpt,
        None,
        gsvae,
    )
    
    logger.info(f"Loading GSRecon from {cfg.load_pretrained_gsrecon}")
    gsrecon = util.load_ckpt(
        os.path.join("out", cfg.load_pretrained_gsrecon, "checkpoints"),
        cfg.load_pretrained_gsrecon_ckpt,
        None,
        gsrecon,
    )
    
    # Move to device
    text_encoder = text_encoder.to(device)
    vae = vae.to(device)
    gsvae = gsvae.to(device)
    gsrecon = gsrecon.to(device)
    unet = unet.to(device)
    if controlnet is not None:
        controlnet = controlnet.to(device)
    
    # Set to eval mode
    text_encoder.eval()
    vae.eval()
    gsvae.eval()
    gsrecon.eval()
    unet.eval()
    if controlnet is not None:
        controlnet.eval()
    
    # Freeze all parameters
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    gsvae.requires_grad_(False)
    gsrecon.requires_grad_(False)
    unet.requires_grad_(False)
    if controlnet is not None:
        controlnet.requires_grad_(False)
    # unet = patch_diffsplat_attention_storage(unet)

    # Set diffusion pipeline
    if controlnet is not None:
        pipeline = StableMVDiffusionControlNetPipeline(
            text_encoder=text_encoder, tokenizer=tokenizer,
            vae=vae, unet=unet, controlnet=controlnet,
            scheduler=noise_scheduler,
        )
    else:
        pipeline = StableMVDiffusionPipeline(
            text_encoder=text_encoder, tokenizer=tokenizer,
            vae=vae, unet=unet,
            scheduler=noise_scheduler,
        )
    
    pipeline.set_progress_bar_config(disable=False)
    
    return pipeline, gsvae, gsrecon, opt


def patch_diffsplat_attention_storage(unet):
    """
    Patch DiffSplat attention modules to store their computed attention weights.
    This enables passive capture without recomputation.
    """
    def patch_attention_forward(module, original_forward):
        def patched_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            
            # Store attention weights if they were computed
            # Look for common patterns in DiffSplat attention modules
            if hasattr(module, 'last_attn_slice'):
                module.attn_weights = module.last_attn_slice
            elif hasattr(module, 'attention_probs'):
                module.attn_weights = module.attention_probs
            elif hasattr(module, '_attn_weights'):
                module.attn_weights = module._attn_weights
            
            return result
        return patched_forward
    
    patched_count = 0
    for name, module in unet.named_modules():
        if 'Attention' in module.__class__.__name__:
            try:
                original_forward = module.forward
                module.forward = patch_attention_forward(module, original_forward)
                patched_count += 1
            except Exception as e:
                print(f"Failed to patch attention module {name}: {e}")
    
    print(f"Patched {patched_count} attention modules for storage")
    return unet


def main(cfg):
    """
    Main function to run baseline memorization evaluation on the DiffSplat model.
    """
    script_start_time = time.time()

    # --- 1. Configuration & Setup ---
    torch.manual_seed(cfg.seed if 'seed' in cfg and cfg.seed is not None else 42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Create dataset manager and load data
    dataset_config = getattr(cfg, 'dataset_config', {
        'laion_memorized': {
            'path': 'data/nemo-prompts/memorized_laion_prompts.csv',  # Update this path
            'max_prompts_per_cluster': 4,
            'max_clusters': 100,
            'source_type': 'csv'
        },
        'cap3d': {
            'clusters_json_path': 'data/objaverse-dupes/aggregated_clusters.json',
            'captions_csv_path': 'data/objaverse-dupes/Cap3D_automated_Objaverse_full.csv',
            'concept_key': 'teddy_bear',
            'max_prompts_per_cluster': 4,
            'max_clusters': 20
        },
        'cap3d': {
            'clusters_json_path': 'data/objaverse-dupes/aggregated_clusters.json',
            'captions_csv_path': 'data/objaverse-dupes/Cap3D_automated_Objaverse_full.csv',
            'concept_key': 'fazbear_',
            'max_prompts_per_cluster': 4,
            'max_clusters': 20
        },
        'cap3d': {
            'clusters_json_path': 'data/objaverse-dupes/aggregated_clusters.json',
            'captions_csv_path': 'data/objaverse-dupes/Cap3D_automated_Objaverse_full.csv',
            'concept_key': "backpack",
            'max_prompts_per_cluster': 4,
            'max_clusters': 20,
        },
        'cap3d': {
            'clusters_json_path': 'data/objaverse-dupes/aggregated_clusters.json',
            'captions_csv_path': 'data/objaverse-dupes/Cap3D_automated_Objaverse_full.csv',
            'concept_key': "kirby_",
            'max_prompts_per_cluster': 4,
            'max_clusters': 20,
        },
        'cap3d': {
            'clusters_json_path': 'data/objaverse-dupes/aggregated_clusters.json',
            'captions_csv_path': 'data/objaverse-dupes/Cap3D_automated_Objaverse_full.csv',
            'concept_key': "pokemon_",
            'max_prompts_per_cluster': 4,
            'max_clusters': 20,
        },
        # 'cap3d': {
        #     'clusters_json_path': 'data/objaverse-dupes/aggregated_clusters.json',
        #     'captions_csv_path': 'data/objaverse-dupes/Cap3D_automated_Objaverse_full.csv',
        #     'concept_key': "_",
        #     'max_prompts_per_cluster': 4,
        #     'max_clusters': 20,
        # },
        'cap3d': {
            'clusters_json_path': 'data/objaverse-dupes/aggregated_clusters.json',
            'captions_csv_path': 'data/objaverse-dupes/Cap3D_automated_Objaverse_full.csv',
            'concept_key': "sonic_",
            'max_prompts_per_cluster': 4,
            'max_clusters': 20,
        },
        'cap3d': {
            'clusters_json_path': 'data/objaverse-dupes/aggregated_clusters.json',
            'captions_csv_path': 'data/objaverse-dupes/Cap3D_automated_Objaverse_full.csv',
            'concept_key': "avocado",
            'max_prompts_per_cluster': 4,
            'max_clusters': 20,
        },
        # 'cap3d': {
        #     'clusters_json_path': 'data/objaverse-dupes/aggregated_clusters.json',
        #     'captions_csv_path': 'data/objaverse-dupes/Cap3D_automated_Objaverse_full.csv',
        #     'concept_key': "automatic_washer",
        #     'max_prompts_per_cluster': 4,
        #     'max_clusters': 20,
        # },
        # 'cap3d': {
        #     'clusters_json_path': 'data/objaverse-dupes/aggregated_clusters.json',
        #     'captions_csv_path': 'data/objaverse-dupes/Cap3D_automated_Objaverse_full.csv',
        #     'concept_key': "armchair",
        #     'max_prompts_per_cluster': 4,
        #     'max_clusters': 20,
        # },
    })
    
    manager = DatasetManager(dataset_config)
    datasets = manager.load_datasets()
    dataloaders = manager.create_dataloaders(datasets, batch_size=1)
    
    # --- 2. W&B Initialization ---
    if cfg.use_wandb:
        wandb.init(
                project="diffsplat-memorization-evaluation",
                name=f"eval-baseline-{time.strftime('%Y%m%d-%H%M%S')}",
                config=OmegaConf.to_container(cfg, resolve=True)
            )

    # --- 3. Model Loading ---
    pipeline, gsvae, gsrecon, opt = load_diffsplat_models(cfg, device)
    logger.info("DiffSplat models loaded successfully.")
    
    # --- Initialize Metrics ---
    per_seed_metrics = [
        NoiseDiffNormMetric(), 
        BrightEndingMetric(), 
        XAttnEntropyMetric(),
        HessianMetric(),
        InvMMMetric(),
        PLaplaceMetric(),
    ]
    
    if cfg.get('include_slow_hessian', False):
        per_seed_metrics.append(HessianMetricSlow())
    
    diversity_metric = DiversityMetric()
    
    evaluator = DiffSplatEvaluator(pipeline, gsvae, gsrecon, per_seed_metrics, device=device)
    logger.info("Evaluator initialized successfully.")

    # --- Setup camera and rendering parameters ---
    camera_params_base = setup_camera_parameters(
        opt, device, 
        num_views=cfg.num_views,
        elevation=cfg.elevation,
        distance=cfg.distance
    )
    
    render_params = {
        'height': cfg.render_res or opt.input_res,
        'width': cfg.render_res or opt.input_res,
        'opacity_threshold': cfg.get('opacity_threshold', 0.0)
    }
    
    # Add generation parameters to camera_params
    camera_params_base.update({
        'negative_prompt': cfg.get('negative_prompt', ''),
        'triangle_cfg_scaling': cfg.get('triangle_cfg_scaling', False),
        'min_guidance_scale': cfg.get('min_guidance_scale', 1.0),
        'eta': cfg.get('eta', 1.0),
        'init_std': cfg.get('init_std', 0.0),
        'init_noise_strength': cfg.get('init_noise_strength', 0.98),
        'init_bg': cfg.get('init_bg', 0.0),
        'guess_mode': cfg.get('guess_mode', False),
        'controlnet_scale': cfg.get('controlnet_scale', 1.0)
    })

    # --- 4. Main Evaluation Loop ---
    for dataset_name, dataloader in dataloaders.items():
        logger.info(f"Starting evaluation for dataset: {dataset_name}")
        output_base_dir = os.path.join(cfg.output_folder, dataset_name)
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing {dataset_name}")):
            try:
                prompt = batch['prompt'][0]
                is_memorized = "memorized" in batch['label'][0]
                prompt_idx = batch.get('idx', torch.tensor([batch_idx])).item()
                
                # Extract metadata for ordinal-based naming (Cap3D style)
                metadata = batch.get('metadata')
                cluster_id = metadata.get('cluster_id', str(batch_idx))[0]
                uuid = metadata.get('uuid', f'unknown_{batch_idx}')[0]
                
                safe_prompt = re.sub(r'\W+', '_', prompt)[:50]
                
                generated_images = []  # For diversity metric (collect all views from all seeds)
                generated_intermediates = [] 

                # --- 5. Per-Seed Generation and Evaluation ---
                for seed_idx in range(cfg.num_seeds):
                    seed = (cfg.seed if 'seed' in cfg and cfg.seed is not None else 42) + seed_idx
                    
                    # Create ordinal-based filename like the original infer_gsdiff script
                    base_filename = f"{prompt_idx:04d}_{seed_idx:02d}_{safe_prompt}"
                    

                    try:
                        # --- SIMPLIFIED ATTENTION CONTROLLER SETUP ---
                        # Just create the controller - pipeline will handle everything else
                        controller = AttentionStore(
                            input_height=render_params['height'], 
                            input_width=render_params['width'], 
                            store_timesteps=[999, 49],
                        )
                        
                        artifacts = {
                            "controller": controller,
                            "attention_map_dir": os.path.join(output_base_dir, "attention_maps")
                        }
                        
                        # Reset controller state for this generation
                        controller.reset()
                        
                        logger.info(f"Starting generation for prompt {prompt_idx}, seed {seed_idx}")
                        
                        # --- PIPELINE HANDLES EVERYTHING AUTOMATICALLY ---
                        # The evaluator passes controller directly to pipeline
                        result = evaluator.process_single_prompt_single_seed(
                            prompt=prompt,
                            seed=seed,
                            num_inference_steps=cfg.num_inference_steps,
                            guidance_scale=cfg.guidance_scale,
                            camera_params=camera_params_base,
                            render_params=render_params,
                            unlearning_artifacts=artifacts
                        )
                        
                        # --- DEBUG ATTENTION CAPTURE ---
                        if controller:
                            attention_maps = controller.get_attention_maps()
                            logger.info(f"Timesteps captured: {list(attention_maps.keys())}")
                            
                            if attention_maps:
                                for t, layers in attention_maps.items():
                                    logger.info(f"  Timestep {t}: {len(layers)} layers - {list(layers.keys())}")
                            else:
                                logger.warning("  NO ATTENTION CAPTURED!")

                        # Check if generation was successful
                        if "error" in result:
                            logger.error(f"Generation failed for seed {seed_idx}: {result['error']}")
                            continue
                        
                        # Convert multi-view tensor to images for diversity metric
                        images_from_multiview = multiview_tensor_to_images(result["rendered_images"])
                        generated_images.extend(images_from_multiview)
                        generated_intermediates.append(result) 
                        
                        # Add metadata to metrics
                        result["metrics"].update({
                            "prompt": prompt, 
                            "memorized": is_memorized,
                            "cluster_id": cluster_id,
                            "uuid": uuid,
                            "prompt_idx": prompt_idx
                        })
                        
                        # Save attention maps if captured
                        if controller and hasattr(controller, 'get_attention_maps'):
                            try:
                                attention_maps = controller.get_attention_maps()
                                if attention_maps:
                                    attention_dir = os.path.join(output_base_dir, "attention_maps")
                                    os.makedirs(attention_dir, exist_ok=True)
                                    attention_path = os.path.join(attention_dir, f"{base_filename}_attention.pt")
                                    torch.save(attention_maps, attention_path)
                                    result["attention_maps_path"] = attention_path
                                    logger.info(f"Saved attention maps to {attention_path}")
                            except Exception as e:
                                logger.warning(f"Failed to save attention maps: {e}")
                        
                        # Save all outputs for this run
                        save_run_outputs(result, output_base_dir, base_filename)

                        # --- W&B Logging (Per-Seed) ---
                        if cfg.use_wandb:
                            try:
                                log_data = {}
                                
                                # Log scalar metrics
                                for metric_name, metric_value in result["metrics"].items():
                                    if isinstance(metric_value, dict):
                                        # Handle nested metric values
                                        for sub_key, sub_value in metric_value.items():
                                            if isinstance(sub_value, (int, float)):
                                                log_data[f"{dataset_name}/metrics/{metric_name}_{sub_key}"] = sub_value
                                    elif isinstance(metric_value, (int, float)):
                                        log_data[f"{dataset_name}/metrics/{metric_name}"] = metric_value
                                
                                # Log multi-view montage if available
                                if "montage_path" in result and os.path.exists(result["montage_path"]):
                                    log_data[f"{dataset_name}/multiview_images"] = wandb.Image(
                                        result["montage_path"], caption=f"Prompt: {prompt}\nSeed: {seed}"
                                    )
                                
                                # Log individual views as well
                                if "view_paths" in result and result["view_paths"]:
                                    view_images = []
                                    for view_idx, view_path in enumerate(result["view_paths"]):
                                        if os.path.exists(view_path):
                                            view_images.append(wandb.Image(view_path, caption=f"View {view_idx}"))
                                    if view_images:
                                        log_data[f"{dataset_name}/individual_views"] = view_images
                                
                                if log_data:
                                    wandb.log(log_data)
                                    
                            except Exception as e:
                                logger.warning(f"W&B logging failed for seed {seed_idx}: {e}")
                        
                        logger.info(f"Completed seed {seed_idx} for prompt {prompt_idx} (cluster {cluster_id})")
                        
                    except Exception as e:
                        logger.error(f"Error processing seed {seed_idx} for prompt {prompt_idx}: {e}", exc_info=True)
                        continue
                        
                    finally:
                        # --- SIMPLIFIED CLEANUP ---
                        # No manual hook management needed - pipeline handles it
                        
                        # Clean up memory
                        if 'result' in locals():
                            del result
                        if controller:
                            del controller
                        cleanup_memory()

                # --- 6. Cross-Seed Metrics (Diversity) ---
                if generated_images and len(generated_images) >= 2:
                    try:
                        diversity_score = diversity_metric.measure(images=generated_images, intermediates=generated_intermediates)
                        cross_seed_json_path = os.path.join(output_base_dir, f"{prompt_idx:04d}_{safe_prompt}_cross_seed.json")
                        
                        cross_seed_data = {
                            "prompt": prompt, 
                            "memorized": is_memorized, 
                            "cluster_id": cluster_id,
                            "uuid": uuid,
                            "prompt_idx": prompt_idx,
                            diversity_metric.name: diversity_score,
                            "total_images_analyzed": len(generated_images),
                            "num_seeds": cfg.num_seeds,
                            "num_views_per_seed": cfg.num_views
                        }
                        
                        with open(cross_seed_json_path, 'w') as f:
                            json.dump(cross_seed_data, f, indent=2)
                        
                        logger.info(f"Cross-seed diversity for prompt {prompt_idx}: {diversity_score}")
                        
                        if cfg.use_wandb:
                            try:
                                # Log diversity scores
                                diversity_log = {}
                                if isinstance(diversity_score, dict):
                                    for key, value in diversity_score.items():
                                        if isinstance(value, (int, float)):
                                            diversity_log[f"{dataset_name}/diversity_{key}"] = value
                                else:
                                    diversity_log[f"{dataset_name}/diversity_score"] = diversity_score
                                    
                                if diversity_log:
                                    wandb.log(diversity_log)
                            except Exception as e:
                                logger.warning(f"W&B diversity logging failed: {e}")
                                
                    except Exception as e:
                        logger.error(f"Error computing diversity for prompt {prompt_idx}: {e}", exc_info=True)
                else:
                    logger.warning(f"Not enough valid images generated for diversity calculation for prompt {prompt_idx}")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}", exc_info=True)
            finally:
                # Clean up batch-level resources
                if 'generated_images' in locals():
                    del generated_images
                    del generated_intermediates
                cleanup_memory()

    logger.info(f"Total evaluation time: {(time.time() - script_start_time) / 60:.2f} minutes.")
    
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gsdiff_sd15.yaml", help="Path to the evaluation config file.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--num_seeds", type=int, default=4, help="Number of seeds to run per prompt.")
    parser.add_argument("--num_views", type=int, default=4, help="Number of camera views to render.")
    
    args = parser.parse_args()
    
    try:
        cfg = OmegaConf.load(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        exit(1)
    
    # Allow command-line args to override config file settings
    cfg.use_wandb = args.use_wandb
    cfg.num_seeds = args.num_seeds
    cfg.num_views = args.num_views
    cfg.config_file = args.config
    cfg.scheduler_type = getattr(cfg, 'scheduler_type', 'sde-dpmsolver++')
    cfg.half_precision = getattr(cfg, 'half_precision', False)
    cfg.triangle_cfg_scaling = getattr(cfg, 'triangle_cfg_scaling', False)
    cfg.min_guidance_scale = getattr(cfg, 'min_guidance_scale', 1.0)
    cfg.eta = getattr(cfg, 'eta', 1.0)
    cfg.init_std = getattr(cfg, 'init_std', 0.0)
    cfg.init_noise_strength = getattr(cfg, 'init_noise_strength', 0.98)
    cfg.init_bg = getattr(cfg, 'init_bg', 0.0)
    cfg.guess_mode = getattr(cfg, 'guess_mode', False)
    cfg.controlnet_scale = getattr(cfg, 'controlnet_scale', 1.0)
    cfg.elevation = getattr(cfg, 'elevation', 10.0)
    cfg.distance = getattr(cfg, 'distance', 1.4)
    cfg.negative_prompt = getattr(cfg, 'negative_prompt', '')
    cfg.opacity_threshold = getattr(cfg, 'opacity_threshold', 0.0)
    cfg.load_pretrained_gsrecon = getattr(cfg, 'load_pretrained_gsrecon', 'gsrecon_gobj265k_cnp_even4')
    cfg.load_pretrained_gsrecon_ckpt = getattr(cfg, 'load_pretrained_gsrecon_ckpt', -1)
    cfg.load_pretrained_gsvae = getattr(cfg, 'load_pretrained_gsvae', 'gsvae_gobj265k_sd')
    cfg.load_pretrained_gsvae_ckpt = getattr(cfg, 'load_pretrained_gsvae_ckpt', -1)
    cfg.load_pretrained_controlnet = getattr(cfg, 'load_pretrained_controlnet', None)
    cfg.load_pretrained_controlnet_ckpt = getattr(cfg, 'load_pretrained_controlnet_ckpt', -1)
    cfg.output_dir = getattr(cfg, 'output_dir', 'out')
    cfg.tag = getattr(cfg, 'tag', 'gsdiff_gobj83k_sd15__render')
    cfg.infer_from_iter = getattr(cfg, 'infer_from_iter', 13020)
    cfg.hdfs_dir = getattr(cfg, 'hdfs_dir', None)
    cfg.render_res = 256
    cfg.output_folder = 'output'
    cfg.num_inference_steps = 20
    cfg.guidance_scale = 7.5

    main(cfg)