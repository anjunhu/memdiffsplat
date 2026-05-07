"""Subspace Pruning runner for diffsplat-memeval."""
import os, sys, argparse, json, torch
from copy import deepcopy
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
from memorization.editing.model_adapter import DiffSplatPipelineAdapter
from unlearning import SubspacePruner
from tqdm import tqdm

BATCH_SIZE = 10


def main(args):
    device = 'cuda'
    baseline_pipeline, gsvae, gsrecon, opt = load_pipeline(args.config, device)


    for dataset in default_datasets():
        prompts_data = load_prompts(dataset)
        for batch_start in range(0, len(prompts_data), BATCH_SIZE):
            batch = prompts_data[batch_start:batch_start + BATCH_SIZE]
            batch_prompts = [r['Caption'] for r in batch]

            edited_pipe = deepcopy(baseline_pipeline)
            adapter = DiffSplatPipelineAdapter(edited_pipe, image_size=opt.input_res)
            pruner = SubspacePruner(adapter, sparsity=0.0008)
            masks = pruner.find_memorization_subspace(batch_prompts)
            pruner.prune_model_weights(masks)

            for local_idx, row in enumerate(tqdm(batch, desc=f"{dataset['name']} b{batch_start}")):
                idx = batch_start + local_idx
                prompt = row['Caption']
                sname = safe_name(prompt)
                all_images = []
                for seed in range(4):
                    out_dir = os.path.join("output/subspace_prune", dataset['name'],
                                           f"prompt_{idx:04d}_{seed:02d}_{sname}")
                    os.makedirs(out_dir, exist_ok=True)
                    gen = torch.Generator(device=device).manual_seed(seed)
                    result = edited_pipe(prompt, num_inference_steps=20, guidance_scale=7.5,
                                        generator=gen, output_type='pil')
                    images.append(result.images[0])
                    torch.cuda.empty_cache()

                div = diversity_metric.measure(images=all_images, intermediates_list=[])
                cross = os.path.join("output/subspace_prune", dataset['name'],
                                     f"prompt_{idx:04d}_{sname}_cross_seed.json")
                os.makedirs(os.path.dirname(cross), exist_ok=True)
                with open(cross, 'w') as f:
                    json.dump({"prompt": prompt, "memorized": dataset['is_memorized'],
                               diversity_metric.name: div}, f, indent=2)

            del edited_pipe, pruner, adapter
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gsdiff_sd15.yaml", help="Path to the evaluation config file.")
    args = parser.parse_args()
    main(args)
