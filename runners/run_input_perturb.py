"""Input Perturbation runner for diffsplat-memeval."""
import os, sys, argparse, json, torch
from functools import partial
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runners._common import load_pipeline, default_datasets, load_prompts, safe_name, default_metrics
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
    metrics = default_metrics(device)
    diversity_metric = metrics[-1]

    for dataset in default_datasets():
        for idx, row in enumerate(tqdm(load_prompts(dataset), desc=dataset['name'])):
            prompt = row['Caption']
            sname = safe_name(prompt)
            images = []
            run_prompt = token_fn(prompt) if token_fn else prompt

            for seed in range(4):
                out_dir = os.path.join(f"output/{method_name}", dataset['name'],
                                       f"prompt_{idx:04d}_{seed:02d}_{sname}")
                os.makedirs(out_dir, exist_ok=True)
                gen = torch.Generator(device=device).manual_seed(seed)

                if embed_fn is not None:
                    inputs = tokenizer(prompt, return_tensors='pt', padding='max_length',
                                       max_length=tokenizer.model_max_length,
                                       truncation=True).to(device)
                    with torch.no_grad():
                        emb = pipeline.text_encoder(**inputs).last_hidden_state
                    perturbed_emb = embed_fn(emb, output_path=out_dir)
                    result = pipeline(prompt_embeds=perturbed_emb, num_inference_steps=20,
                                      guidance_scale=7.5, generator=gen, output_type='pil')
                else:
                    result = pipeline(run_prompt, num_inference_steps=20, guidance_scale=7.5,
                                      generator=gen, output_type='pil')
                images.append(result.images[0])
                torch.cuda.empty_cache()

            div = diversity_metric.measure(images=images, intermediates_list=[])
            cross = os.path.join(f"output/{method_name}", dataset['name'],
                                 f"prompt_{idx:04d}_{sname}_cross_seed.json")
            os.makedirs(os.path.dirname(cross), exist_ok=True)
            with open(cross, 'w') as f:
                json.dump({"prompt": prompt, "perturbed_prompt": run_prompt,
                           "memorized": dataset['is_memorized'],
                           diversity_metric.name: div}, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', required=True, choices=METHODS)
    args = parser.parse_args()
    main(args)
