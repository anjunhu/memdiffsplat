"""XAttn / BrightEnding runner for diffsplat-memeval."""
import os, sys, argparse, json, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runners._common import load_pipeline, default_datasets, load_prompts, safe_name
import importlib.util as _ilu
_rb_spec = _ilu.spec_from_file_location('run_baseline',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run_baseline.py'))
_rb = _ilu.module_from_spec(_rb_spec); _rb_spec.loader.exec_module(_rb)
setup_camera_parameters = _rb.setup_camera_parameters
from memorization.controller import AttentionStore
from memorization.evaluation.evaluator import DiffSplatEvaluator, save_run_outputs, multiview_tensor_to_images
from memorization.metrics import (
    NoiseDiffNormMetric, HessianMetric, BrightEndingMetric,
    XAttnEntropyMetric, InvMMMetric, PLaplaceMetric, AnisotropyMetric, DiversityMetric,
)
from memorization.controller import AttentionStore
from tqdm import tqdm


def main(args):
    device = 'cuda'
    pipeline, gsvae, gsrecon, opt = load_pipeline(args.config, device)
    camera_params = setup_camera_parameters(opt, device)
    camera_params.update({"negative_prompt": "", "triangle_cfg_scaling": False,
        "min_guidance_scale": 1.0, "eta": 1.0, "init_std": 0.0,
        "init_noise_strength": 0.98, "init_bg": 0.0, "guess_mode": False, "controlnet_scale": 1.0})
    render_params = {"height": opt.input_res, "width": opt.input_res, "opacity_threshold": 0.0}
    per_seed_metrics = [NoiseDiffNormMetric(), HessianMetric(), BrightEndingMetric(),
        XAttnEntropyMetric(), InvMMMetric(), PLaplaceMetric(), AnisotropyMetric()]
    diversity_metric = DiversityMetric(device=device)
    evaluator = DiffSplatEvaluator(pipeline, gsvae, gsrecon, per_seed_metrics, device=device)
    method_name = args.method


    for dataset in default_datasets():
        for idx, row in enumerate(tqdm(load_prompts(dataset), desc=dataset['name'])):
            prompt = row['Caption']
            sname = safe_name(prompt)
            all_images = []

            be_emb = None
            if method_name == 'be':
                tokenizer = pipeline.tokenizer
                text_encoder = pipeline.text_encoder
                inputs = tokenizer(prompt, return_tensors='pt', padding='max_length',
                                   max_length=tokenizer.model_max_length, truncation=True).to(device)
                with torch.no_grad():
                    emb = text_encoder(**inputs).last_hidden_state.float()
                emb = emb.detach().requires_grad_(True)
                opt_be = torch.optim.Adam([emb], lr=0.01)
                for _ in range(20):
                    opt_be.zero_grad()
                    loss = -(emb - emb.detach().mean()).norm()
                    loss.backward()
                    opt_be.step()
                be_emb = emb.detach().to(next(pipeline.unet.parameters()).dtype)

            for seed in range(4):
                out_dir = os.path.join(f"output/{method_name}", dataset['name'],
                                       f"prompt_{idx:04d}_{seed:02d}_{sname}")
                os.makedirs(out_dir, exist_ok=True)
                gen = torch.Generator(device=device).manual_seed(seed)
                if be_emb is not None:
                    result = pipeline(prompt_embeds=be_emb, num_inference_steps=20,
                                      guidance_scale=7.5, generator=gen, output_type='pil')
                else:
                    result = pipeline(prompt, num_inference_steps=20, guidance_scale=7.5,
                                      generator=gen, output_type='pil')
                images.append(result.images[0])
                torch.cuda.empty_cache()

            div = diversity_metric.measure(images=all_images, intermediates_list=[])
            cross = os.path.join(f"output/{method_name}", dataset['name'],
                                 f"prompt_{idx:04d}_{sname}_cross_seed.json")
            os.makedirs(os.path.dirname(cross), exist_ok=True)
            with open(cross, 'w') as f:
                json.dump({"prompt": prompt, "memorized": dataset['is_memorized'],
                           diversity_metric.name: div}, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gsdiff_sd15.yaml", help="Path to the evaluation config file.")
    parser.add_argument('--method', required=True, choices=['ca_entropy', 'be'])
    args = parser.parse_args()
    main(args)
