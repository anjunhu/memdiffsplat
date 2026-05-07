"""UCE Multi-Concept runner for diffsplat-memeval."""
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
from unlearning import UCEEditor
from tqdm import tqdm


def main(args):
    device = 'cuda'
    baseline_pipeline, gsvae, gsrecon, opt = load_pipeline(args.config, device)


    for dataset in default_datasets():
        for idx, row in enumerate(tqdm(load_prompts(dataset), desc=dataset['name'])):
            prompt = row['Caption']
            sname = safe_name(prompt)
            all_images = []

            edited_pipe = deepcopy(baseline_pipeline)
            adapter = DiffSplatPipelineAdapter(edited_pipe, image_size=opt.input_res)
            editor = UCEEditor(adapter)
            editor.erase_concept(
                edit_concepts=[prompt],
                guide_concepts=["a high-quality photograph"],
                preserve_concepts=["car", "house", "tree"],
            )

            for seed in range(4):
                out_dir = os.path.join("output/uce", dataset['name'],
                                       f"prompt_{idx:04d}_{seed:02d}_{sname}")
                os.makedirs(out_dir, exist_ok=True)
                gen = torch.Generator(device=device).manual_seed(seed)
                result = edited_pipe(prompt, num_inference_steps=20, guidance_scale=7.5,
                                     generator=gen, output_type='pil')
                images.append(result.images[0])
                torch.cuda.empty_cache()

            del edited_pipe, editor, adapter
            torch.cuda.empty_cache()

            div = diversity_metric.measure(images=all_images, intermediates_list=[])
            cross = os.path.join("output/uce", dataset['name'],
                                 f"prompt_{idx:04d}_{sname}_cross_seed.json")
            os.makedirs(os.path.dirname(cross), exist_ok=True)
            with open(cross, 'w') as f:
                json.dump({"prompt": prompt, "memorized": dataset['is_memorized'],
                           diversity_metric.name: div}, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gsdiff_sd15.yaml", help="Path to the evaluation config file.")
    args = parser.parse_args()
    main(args)
