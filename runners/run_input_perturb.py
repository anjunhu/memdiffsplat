"""Input Perturbation runner for diffsplat-memeval."""
import os, sys, argparse, json, torch
from functools import partial
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
from unlearning import (
    perturb_random_tokens, perturb_word_repetition, perturb_random_numbers,
    add_gaussian_noise, optimize_embedding_wen,
)
from tqdm import tqdm

METHODS = ['ip-rt', 'ip-rna', 'ip-cwr', 'ip-gni', 'ip-wen']


def main(args):
    device = 'cuda'
    pipeline, gsvae, gsrecon, opt = load_pipeline(args.config, device)
    tokenizer = pipeline.tokenizer
    camera_params = setup_camera_parameters(opt, device)
    camera_params.update({"negative_prompt": "", "triangle_cfg_scaling": False,
        "min_guidance_scale": 1.0, "eta": 1.0, "init_std": 0.0,
        "init_noise_strength": 0.98, "init_bg": 0.0, "guess_mode": False, "controlnet_scale": 1.0})
    render_params = {"height": opt.input_res, "width": opt.input_res, "opacity_threshold": 0.0}
    per_seed_metrics = [NoiseDiffNormMetric(), HessianMetric(), BrightEndingMetric(),
        XAttnEntropyMetric(), InvMMMetric(), PLaplaceMetric(), AnisotropyMetric()]
    diversity_metric = DiversityMetric(device=device)
    evaluator = DiffSplatEvaluator(pipeline, gsvae, gsrecon, per_seed_metrics, device=device)

    token_fn, embed_fn = None, None
    if args.method == 'ip-rt':
        token_fn = partial(perturb_random_tokens, tokenizer=tokenizer, num_tokens=4)
    elif args.method == 'ip-rna':
        token_fn = partial(perturb_random_numbers, num_numbers=4)
    elif args.method == 'ip-cwr':
        token_fn = partial(perturb_word_repetition, num_repeats=4)
    elif args.method == 'ip-gni':
        embed_fn = partial(add_gaussian_noise, std=5.0)
    elif args.method == 'ip-wen':
        embed_fn = partial(optimize_embedding_wen, model=pipeline, lr=0.1, steps=200)

    method_name = f"perturb_{args.method}"

    for dataset in default_datasets():
        for idx, row in enumerate(tqdm(load_prompts(dataset), desc=dataset['name'])):
            prompt = row['Caption']
            sname = safe_name(prompt)
            all_images = []
            run_prompt = token_fn(prompt) if token_fn else prompt

            for seed in range(4):
                out_dir = os.path.join(f"output/{method_name}", dataset['name'],
                                       f"prompt_{idx:04d}_{seed:02d}_{sname}")
                os.makedirs(out_dir, exist_ok=True)
                base_filename = f"prompt_{idx:04d}_{seed:02d}_{sname}"
                if os.path.exists(os.path.join(out_dir, f"{base_filename}_metrics.json")):
                    continue
                controller = AttentionStore()

                cam = dict(camera_params)
                eval_prompt = run_prompt
                if embed_fn is not None:
                    inputs = tokenizer(prompt, return_tensors='pt', padding='max_length',
                                       max_length=tokenizer.model_max_length, truncation=True).to(device)
                    with torch.no_grad():
                        emb = pipeline.text_encoder(**inputs).last_hidden_state
                    cam['prompt_embeds'] = embed_fn(emb, output_path=out_dir)
                    eval_prompt = None

                result = evaluator.process_single_prompt_single_seed(
                    prompt=eval_prompt, seed=seed, num_inference_steps=20, guidance_scale=7.5,
                    camera_params=cam, render_params=render_params,
                    unlearning_artifacts={"controller": controller},
                )
                if "error" not in result:
                    result["metrics"]["memorized"] = dataset["is_memorized"]
                    save_run_outputs(result, out_dir, f"prompt_{idx:04d}_{seed:02d}_{sname}")
                    all_images.extend(multiview_tensor_to_images(result["rendered_images"]))
                torch.cuda.empty_cache()

            div = diversity_metric.measure(images=all_images, intermediates_list=[])
            cross = os.path.join(f"output/{method_name}", dataset['name'],
                                 f"prompt_{idx:04d}_{sname}_cross_seed.json")
            os.makedirs(os.path.dirname(cross), exist_ok=True)
            with open(cross, 'w') as f:
                json.dump({"prompt": prompt, "perturbed_prompt": run_prompt,
                           "memorized": dataset["is_memorized"],
                           diversity_metric.name: div}, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/gsdiff_sd15.yaml')
    parser.add_argument('--method', required=True, choices=METHODS)
    args = parser.parse_args()
    main(args)
