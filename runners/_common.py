"""
Shared setup utilities for diffsplat-memeval unlearning runners.
"""
import os, sys, re, json, torch
import pandas as pd
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)

from memorization.utils import resolve_data_path
from memorization.controller import AttentionStore
from memorization.metrics import (
    NoiseDiffNormMetric, HessianMetric, DiversityMetric,
    BrightEndingMetric, XAttnEntropyMetric, InvMMMetric,
    PLaplaceMetric, AnisotropyMetric,
)


def default_datasets():
    import json as _json
    v2_path = resolve_data_path("objaverse-dupes/aggregated_clusters_v2.json")
    try:
        concepts = _json.load(open(v2_path))
        return [
            {"name": f"objaverse_{concept}", "type": "json",
             "path": v2_path, "concept_key": concept, "is_memorized": True}
            for concept in sorted(concepts.keys())
            if sum(len(uids) for uids in concepts[concept].values()) >= 4
        ]
    except Exception as e:
        print(f"Warning: could not load v2 clusters ({e}), falling back to defaults")
        return [
            {"name": "objaverse_backpack", "type": "json",
             "path": resolve_data_path("objaverse-dupes/aggregated_clusters.json"),
             "concept_key": "backpack", "is_memorized": True},
        ]


def default_metrics(device='cuda'):
    return [
        NoiseDiffNormMetric(), HessianMetric(), BrightEndingMetric(),
        XAttnEntropyMetric(), InvMMMetric(), PLaplaceMetric(),
        AnisotropyMetric(), DiversityMetric(device=device),
    ]


def load_prompts(dataset, max_prompts=50):
    path = dataset['path']
    if path.endswith('.json'):
        import json as _json
        from memorization.utils import load_uids_from_clusters, uids_to_prompts
        uids = load_uids_from_clusters(path, dataset['concept_key'])
        return uids_to_prompts(uids)[:max_prompts]
    df = pd.read_csv(path, sep=';')
    return df.to_dict('records')[:max_prompts]


def safe_name(prompt, maxlen=32):
    return re.sub(r'\W+', '_', prompt)[:maxlen]


def load_pipeline(config_path: str, device='cuda'):
    """Delegate to run_baseline.load_diffsplat_models for identical pipeline loading behaviour."""
    import importlib.util
    from omegaconf import OmegaConf
    spec = importlib.util.spec_from_file_location("run_baseline", os.path.join(_REPO, "run_baseline.py"))
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    cfg = OmegaConf.load(config_path)
    # Set defaults that run_baseline.__main__ normally injects
    cfg.config_file = config_path
    cfg.scheduler_type = getattr(cfg, 'scheduler_type', 'sde-dpmsolver++')
    cfg.half_precision = getattr(cfg, 'half_precision', False)
    cfg.output_dir = getattr(cfg, 'output_dir', 'out')
    cfg.tag = getattr(cfg, 'tag', 'gsdiff_gobj83k_sd15__render')
    cfg.infer_from_iter = getattr(cfg, 'infer_from_iter', 13020)
    cfg.hdfs_dir = getattr(cfg, 'hdfs_dir', None)
    cfg.load_pretrained_gsrecon = getattr(cfg, 'load_pretrained_gsrecon', 'gsrecon_gobj265k_cnp_even4')
    cfg.load_pretrained_gsrecon_ckpt = getattr(cfg, 'load_pretrained_gsrecon_ckpt', -1)
    cfg.load_pretrained_gsvae = getattr(cfg, 'load_pretrained_gsvae', 'gsvae_gobj265k_sd')
    cfg.load_pretrained_gsvae_ckpt = getattr(cfg, 'load_pretrained_gsvae_ckpt', -1)
    cfg.load_pretrained_controlnet = getattr(cfg, 'load_pretrained_controlnet', None)
    cfg.load_pretrained_controlnet_ckpt = getattr(cfg, 'load_pretrained_controlnet_ckpt', -1)
    return rb.load_diffsplat_models(cfg, device)
