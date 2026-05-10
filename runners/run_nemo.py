"""NeMo unlearning runner for diffsplat-memeval."""
import os, sys, argparse, json, torch
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util as _ilu
_rb_spec = _ilu.spec_from_file_location('run_baseline',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run_baseline.py'))
_rb = _ilu.module_from_spec(_rb_spec); _rb_spec.loader.exec_module(_rb)
setup_camera_parameters = _rb.setup_camera_parameters

from runners._common import load_pipeline, default_datasets, load_prompts, safe_name
from memorization.utils import resolve_data_path
from memorization.controller import AttentionStore
from memorization.editing.model_adapter import DiffSplatPipelineAdapter
from memorization.evaluation.evaluator import DiffSplatEvaluator, save_run_outputs, multiview_tensor_to_images
from memorization.metrics import (
    NoiseDiffNormMetric, HessianMetric, BrightEndingMetric,
    XAttnEntropyMetric, InvMMMetric, PLaplaceMetric, AnisotropyMetric, DiversityMetric,
)
from unlearning import NeMoEditor
from tqdm import tqdm


def main(args):
    device = 'cuda'
    pipeline, gsvae, gsrecon, opt = load_pipeline(args.config, device)
    adapter = DiffSplatPipelineAdapter(pipeline, image_size=opt.input_res)

    camera_params = setup_camera_parameters(opt, device)
    camera_params.update({
        'negative_prompt': '', 'triangle_cfg_scaling': False,
        'min_guidance_scale': 1.0, 'eta': 1.0,
        'init_std': 0.0, 'init_noise_strength': 0.98, 'init_bg': 0.0,
        'guess_mode': False, 'controlnet_scale': 1.0,
    })
    render_params = {'height': opt.input_res, 'width': opt.input_res, 'opacity_threshold': 0.0}

    per_seed_metrics = [
        NoiseDiffNormMetric(), HessianMetric(), BrightEndingMetric(),
        XAttnEntropyMetric(), InvMMMetric(), PLaplaceMetric(), AnisotropyMetric(),
    ]
    diversity_metric = DiversityMetric(device=device)
    evaluator = DiffSplatEvaluator(pipeline, gsvae, gsrecon, per_seed_metrics, device=device)

    non_mem_prompts = pd.read_csv(
        resolve_data_path("nemo-prompts/unmemorized_laion_prompts.csv"), sep=';'
    )['Caption'].tolist()[:100]
    editor = NeMoEditor(adapter, non_mem_prompts=non_mem_prompts)

    for dataset in default_datasets():
        for idx, row in enumerate(tqdm(load_prompts(dataset), desc=dataset['name'])):
            prompt = row['Caption']
            sname = safe_name(prompt)
            all_images = []

            neurons = editor.find_neurons(prompt)
            if not any(neurons.values()):
                continue

            editor.register_hooks(neurons)
            for seed in range(4):
                out_dir = os.path.join("output/nemo", dataset['name'],
                                       f"prompt_{idx:04d}_{seed:02d}_{sname}")
                os.makedirs(out_dir, exist_ok=True)

                base_filename = f"prompt_{idx:04d}_{seed:02d}_{sname}"
                if os.path.exists(os.path.join(out_dir, f"{base_filename}_metrics.json")):
                    continue
                controller = AttentionStore()
                result = evaluator.process_single_prompt_single_seed(
                    prompt=prompt, seed=seed,
                    num_inference_steps=20, guidance_scale=7.5,
                    camera_params=camera_params, render_params=render_params,
                    unlearning_artifacts={"controller": controller},
                )

                result["metrics"]["memorized"] = dataset["is_memorized"]
                if "error" in result:
                    print(f"[warn] seed {seed}: {result['error']}")
                    continue

                base_filename = f"prompt_{idx:04d}_{seed:02d}_{sname}"
                save_run_outputs(result, out_dir, base_filename)

                rendered_views = multiview_tensor_to_images(result["rendered_images"])
                all_images.extend(rendered_views)

            editor.remove_hooks()

            if all_images:
                div = diversity_metric.measure(images=all_images, intermediates_list=[])
                cross = os.path.join("output/nemo", dataset['name'],
                                     f"prompt_{idx:04d}_{sname}_cross_seed.json")
                with open(cross, 'w') as f:
                    json.dump({"prompt": prompt, "memorized": dataset['is_memorized'],
                               diversity_metric.name: div}, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gsdiff_sd15.yaml")
    args = parser.parse_args()
    main(args)
