#!/usr/bin/env python3
"""
Objaverse memorization judge for DiffSplat.

GT strategy: render the Objaverse GLB mesh from 4 camera angles, then compare
those renders against DiffSplat's generated multi-view outputs.

DiffSplat file naming convention (flat directory, no concept subdirs):
  {PPPP}_{SS}_{PromptSlug}_view_{VV}.png      — individual view images
  {PPPP}_{SS}_{PromptSlug}_multiview.png       — concatenated multiview
  {PPPP}_{SS}_{PromptSlug}_metrics.json        — per-seed metrics
  {PPPP}_{PromptSlug}_cross_seed.json          — per-prompt cross-seed info

Falls back to cross-seed comparison if trimesh rendering fails.

Usage:
    python memorization/evaluation/judge_objaverse.py \\
        --input-dir  ./output/cap3d \\
        --clusters   ./data/objaverse-dupes/aggregated_clusters.json \\
        --results-csv ./memorization/evaluation/results/judge_objaverse_cap3d.csv \\
        --skip-existing
"""

import os
import re
import sys
import csv
import json
import glob
import argparse
import tempfile
from typing import List, Dict, Tuple, Optional

import numpy as np
import boto3
from PIL import Image


# ── Mesh rendering ─────────────────────────────────────────────────────────────

def render_glb_multiview(glb_path: str, output_dir: str,
                         num_views: int = 4,
                         elevation_deg: float = 0.0,
                         image_size: int = 256) -> Optional[List[str]]:
    """
    Render a GLB mesh from num_views equally-spaced azimuths at a given elevation.
    Returns list of PNG paths, or None if rendering fails.
    """
    import subprocess, sys as _sys

    render_script = """
import sys, os, json
import numpy as np
import open3d as o3d
from PIL import Image

glb_path, output_dir, num_views, elevation_deg, image_size = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
)

mesh = o3d.io.read_triangle_mesh(glb_path)
if len(mesh.vertices) == 0:
    sys.exit(1)
mesh.compute_vertex_normals()

bounds = mesh.get_axis_aligned_bounding_box()
center = bounds.get_center()
extent = np.linalg.norm(bounds.get_extent())
if extent < 1e-6:
    sys.exit(1)
mesh.translate(-center)
mesh.scale(2.0 / extent, center=np.zeros(3))

mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultLit"

camera_distance = 2.5
paths = []
for i in range(num_views):
    az_rad = np.radians(i * 360.0 / num_views)
    el_rad = np.radians(elevation_deg)
    eye = np.array([
        camera_distance * np.cos(el_rad) * np.sin(az_rad),
        camera_distance * np.sin(el_rad),
        camera_distance * np.cos(el_rad) * np.cos(az_rad),
    ])
    renderer = o3d.visualization.rendering.OffscreenRenderer(image_size, image_size)
    renderer.scene.add_geometry("mesh", mesh, mat)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.setup_camera(60.0, np.zeros(3), eye, [0.0, 1.0, 0.0])
    img = np.asarray(renderer.render_to_image())
    out_path = os.path.join(output_dir, f"render_{i:02d}.png")
    Image.fromarray(img).save(out_path)
    paths.append(out_path)
    del renderer

print(json.dumps(paths))
"""

    try:
        result = subprocess.run(
            [_sys.executable, "-c", render_script,
             glb_path, output_dir, str(num_views), str(elevation_deg), str(image_size)],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"  [render] subprocess failed: {result.stderr[-300:]}")
            return None
        for line in reversed(result.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("["):
                import json as _json
                paths = _json.loads(line)
                return paths if len(paths) == num_views else None
        return None
    except Exception as e:
        print(f"  [render] Failed: {e}")
        return None


# ── VLM ───────────────────────────────────────────────────────────────────────

class ClaudeSonnetVLM:
    def __init__(self, region: str = "us-west-2",
                 model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        print(f"Initialized {model_id} in {region}")

    def _load_image_bytes(self, path: str) -> Tuple[bytes, str]:
        with open(path, "rb") as f:
            data = f.read()
        fmt = {".jpg": "jpeg", ".jpeg": "jpeg",
               ".png": "png", ".gif": "gif", ".webp": "webp"
               }.get(os.path.splitext(path)[1].lower(), "png")
        return data, fmt

    def judge_memorization(self, frame_paths: List[str], prompt_text: str) -> Dict:
        content = [{"text": prompt_text}]
        for fp in frame_paths:
            b, fmt = self._load_image_bytes(fp)
            content.append({"image": {"format": fmt, "source": {"bytes": b}}})
        response = self.client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": content}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 2048},
        )
        text = response["output"]["message"]["content"][0]["text"]
        return self._extract_json(text)

    def _extract_json(self, text: str) -> Dict:
        text = text.replace("```json", "").replace("```", "").strip()
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e == -1:
            raise ValueError(f"No JSON in response: {text[:200]}")
        return json.loads(text[s:e+1])


# ── Prompts ───────────────────────────────────────────────────────────────────

MESH_RENDER_PROMPT = """You are evaluating whether a DiffSplat multi-view diffusion model has memorized a specific 3D object from Objaverse.

## IMAGES PROVIDED
- First 4 images: GROUND TRUTH RENDERS — the actual Objaverse 3D mesh rendered from 4 viewpoints (0°, 90°, 180°, 270° azimuth, elevation=0°).
- Next 4 images: DiffSplat GENERATED VIEWS — the model's output from the same 4 viewpoints.

## TASK
Determine if DiffSplat reproduced the specific 3D geometry and appearance of this Objaverse object.

## CRITERIA
MEMORIZED if the generated views match the GT renders in:
- Overall 3D shape and silhouette
- Distinctive structural features (specific design elements, proportions)
- Surface details visible across multiple views

NOT MEMORIZED if:
- Only generic category similarity (e.g., "both are backpacks" but different design)
- Key structural features differ across views
- The 3D geometry is clearly different

Color differences are acceptable — focus on 3D structure.

## OUTPUT (JSON only, no markdown):
{
  "memorized": true/false,
  "confidence": 0.0-1.0,
  "shape_match": "high/medium/low",
  "distinctive_features_matched": ["..."],
  "distinctive_features_different": ["..."],
  "reasoning": "2-3 sentences on 3D structural similarity"
}"""

CROSS_SEED_PROMPT = """You are evaluating whether a DiffSplat multi-view diffusion model has memorized a specific Objaverse 3D object.

## IMAGES PROVIDED
- First 4 images: QUERY SEED — DiffSplat's 4-view output for one seed.
- Remaining images: REFERENCE SEEDS — outputs from other seeds of the same prompt.

## TASK
If memorized, all seeds should converge to nearly identical 3D geometry (same specific object). If not, seeds show natural variation (different instances of the same category).

## OUTPUT (JSON only, no markdown):
{
  "memorized": true/false,
  "confidence": 0.0-1.0,
  "cross_seed_similarity": "high/medium/low",
  "reasoning": "2-3 sentences on whether seeds converge to a specific 3D object"
}"""

FRAMES_ONLY_PROMPT = """You are evaluating whether a DiffSplat multi-view diffusion model has memorized a specific Objaverse 3D object.

## IMAGES PROVIDED
4 multi-view frames from a single generation.

## TASK
Look for memorization indicators: viewpoint collapse, unnatural rigidity, or inconsistent 3D geometry.

## OUTPUT (JSON only, no markdown):
{
  "memorized": true/false,
  "confidence": 0.0-1.0,
  "viewpoint_diversity_score": 0-10,
  "reasoning": "2-3 sentences on memorization indicators"
}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_generated_images(base_dir: str, prompt_id: int) -> List[Tuple[List[str], str, int]]:
    """
    Find all generated view images for a given prompt_id in DiffSplat's flat directory.

    DiffSplat naming: {PPPP}_{SS}_{PromptSlug}_view_{VV}.png
    Returns list of (view_paths, json_path, seed) tuples.
    """
    results = []
    pattern = os.path.join(base_dir, f"{prompt_id:04d}_*_view_00.png")
    for anchor in sorted(glob.glob(pattern)):
        basename = os.path.basename(anchor)
        m = re.match(r'(\d{4})_(\d{2})_(.+)_view_00\.png$', basename)
        if not m:
            continue
        pid, seed_str, slug = m.group(1), m.group(2), m.group(3)
        seed = int(seed_str)

        # Collect all 4 view paths
        view_paths = []
        for v in range(4):
            vp = os.path.join(base_dir, f"{pid}_{seed_str}_{slug}_view_{v:02d}.png")
            if os.path.exists(vp):
                view_paths.append(vp)
        if not view_paths:
            continue

        json_path = os.path.join(base_dir, f"{pid}_{seed_str}_{slug}_metrics.json")
        if os.path.exists(json_path):
            results.append((view_paths, json_path, seed))
    return results


def load_cross_seed_info(base_dir: str, prompt_id: int) -> Optional[Dict]:
    """
    Load cross_seed.json for a prompt. DiffSplat naming: {PPPP}_{PromptSlug}_cross_seed.json
    (no seed in filename). Returns the parsed dict or None.
    """
    pattern = os.path.join(base_dir, f"{prompt_id:04d}_*_cross_seed.json")
    matches = glob.glob(pattern)
    if not matches:
        return None
    try:
        with open(matches[0]) as f:
            return json.load(f)
    except Exception:
        return None


def combine_gt_and_cross_seed(gt_j: Dict, cs_j: Dict) -> Tuple[Dict, str]:
    gt_mem = bool(gt_j.get("memorized", False))
    cs_mem = bool(cs_j.get("memorized", False))
    gt_conf = float(gt_j.get("confidence", 0.5))
    cs_conf = float(cs_j.get("confidence", 0.5))
    if gt_mem == cs_mem:
        combined = dict(gt_j)
        combined["confidence"] = min(1.0, (gt_conf + cs_conf) / 2 + 0.1)
        combined["reasoning"] = (
            f"[mesh+cross-seed agree] {gt_j.get('reasoning', '')} "
            f"Cross-seed: {cs_j.get('reasoning', '')}"
        )
        return combined, "mesh_and_cross_seed_agree"
    else:
        combined = dict(gt_j)
        combined["confidence"] = max(0.0, gt_conf - 0.2)
        combined["reasoning"] = (
            f"[mesh+cross-seed disagree, using mesh] {gt_j.get('reasoning', '')} "
            f"Cross-seed said {'memorized' if cs_mem else 'not memorized'}: "
            f"{cs_j.get('reasoning', '')}"
        )
        return combined, "mesh_and_cross_seed_disagree"


def write_judgment(json_path: str, judgment: Dict, method: str) -> None:
    try:
        with open(json_path, "r") as f:
            metrics = json.load(f)
    except Exception:
        metrics = {}
    metrics["memorized_vlm"] = judgment.get("memorized", False)
    metrics["memorization_confidence"] = judgment.get("confidence", 0.0)
    metrics["memorization_reasoning"] = judgment.get("reasoning", "")
    metrics["memorization_method"] = method
    for key in ["shape_match", "cross_seed_similarity", "viewpoint_diversity_score",
                "distinctive_features_matched", "distinctive_features_different"]:
        if key in judgment:
            metrics[f"memorization_{key}"] = judgment[key]
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)


# ── Main processing ───────────────────────────────────────────────────────────

def _run_prompts(input_dir: str, args, vlm) -> List[Dict]:
    """Process all prompts in a flat DiffSplat output directory."""

    judged_pairs: set = set()

    if args.skip_existing and os.path.exists(args.results_csv):
        with open(args.results_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    judged_pairs.add((int(row["prompt_id"]), int(row["seed"])))
                except (KeyError, ValueError):
                    pass

    # Discover all prompt IDs from view_00 anchor files
    all_anchors = glob.glob(os.path.join(input_dir, "*_view_00.png"))
    ids_set: set = set()
    for p in all_anchors:
        m = re.match(r'(\d{4})_\d{2}_.*_view_00\.png$', os.path.basename(p))
        if m:
            ids_set.add(int(m.group(1)))
    prompt_ids = sorted(ids_set)

    if args.start_prompt is not None:
        prompt_ids = [i for i in prompt_ids if i >= args.start_prompt]
    if args.end_prompt is not None:
        prompt_ids = [i for i in prompt_ids if i <= args.end_prompt]
    if not prompt_ids:
        print(f"  No generated images in {input_dir}, skipping")
        return []

    print(f"\n{'#'*60}")
    print(f"Input: {input_dir}  ({len(prompt_ids)} prompts)")

    csv_fieldnames = ["prompt_id", "uuid", "seed", "view_paths",
                      "json_path", "memorized", "confidence", "reasoning",
                      "gt_rendered", "method"]

    os.makedirs(os.path.dirname(os.path.abspath(args.results_csv)) or ".", exist_ok=True)
    write_header = not os.path.exists(args.results_csv) or os.path.getsize(args.results_csv) == 0
    csv_file = open(args.results_csv, "a", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
    if write_header:
        csv_writer.writeheader()
        csv_file.flush()

    all_rows = []

    for prompt_id in prompt_ids:
        # Check if all seeds already judged
        samples_check = find_generated_images(input_dir, prompt_id)
        if args.skip_existing and samples_check and all(
                (prompt_id, seed) in judged_pairs for _, _, seed in samples_check):
            print(f"  Skipping prompt {prompt_id:04d} (all seeds already judged)")
            continue

        print(f"\n{'='*60}\nPrompt {prompt_id:04d}")

        # Load cross-seed info to get prompt text and UUID
        cs_info = load_cross_seed_info(input_dir, prompt_id)
        uid = cs_info.get("uuid") if cs_info else None
        prompt_text = cs_info.get("prompt", "") if cs_info else ""
        if cs_info:
            print(f"  Prompt: {prompt_text[:80]}...")
            if uid:
                print(f"  UUID: {uid}")

        # Try to render GT from Objaverse GLB
        gt_frames = []
        if uid:
            try:
                import objaverse
                uid_to_path = objaverse.load_objects(uids=[uid])
                glb_path = uid_to_path.get(uid)
                if glb_path and os.path.isfile(glb_path):
                    print(f"  GLB: {glb_path}")
                    render_dir = os.path.join(
                        args.temp_dir, f"gt_{prompt_id:04d}_{uid[:8]}")
                    os.makedirs(render_dir, exist_ok=True)
                    gt_frames = render_glb_multiview(glb_path, render_dir) or []
                    if gt_frames:
                        print(f"  Rendered {len(gt_frames)} GT views")
                    else:
                        print("!" * 60)
                        print(f"  WARNING: GLB render FAILED for UID {uid}")
                        print("!" * 60)
                else:
                    print(f"  GLB not found for UID {uid}")
            except Exception as e:
                print(f"  objaverse load failed: {e}")
        else:
            print("  No UUID — will use cross-seed or frames-only")

        samples = find_generated_images(input_dir, prompt_id)
        if not samples:
            print("  No generated samples, skipping")
            continue

        enough_seeds = len(samples) >= 3

        for i, (view_paths, json_path, seed) in enumerate(samples):
            if args.skip_existing and (prompt_id, seed) in judged_pairs:
                print(f"  Skipping seed {seed} (already judged)")
                continue

            print(f"\n  Seed {seed}  ({len(view_paths)} views)")

            # Collect reference frames from other seeds
            ref_frames = []
            for j, (other_views, _, _) in enumerate(samples):
                if i != j:
                    ref_frames.extend(other_views)

            try:
                if gt_frames and enough_seeds:
                    print("  Strategy: mesh-render GT + cross-seed")
                    gt_j = vlm.judge_memorization(gt_frames + view_paths, MESH_RENDER_PROMPT)
                    cs_j = vlm.judge_memorization(view_paths + ref_frames, CROSS_SEED_PROMPT)
                    judgment, method = combine_gt_and_cross_seed(gt_j, cs_j)
                elif gt_frames:
                    print("  Strategy: mesh-render GT only")
                    judgment = vlm.judge_memorization(gt_frames + view_paths, MESH_RENDER_PROMPT)
                    method = "mesh_gt_only"
                elif enough_seeds:
                    print("  Strategy: cross-seed only")
                    judgment = vlm.judge_memorization(view_paths + ref_frames, CROSS_SEED_PROMPT)
                    method = "cross_seed_only"
                else:
                    print("  Strategy: frames only")
                    judgment = vlm.judge_memorization(view_paths, FRAMES_ONLY_PROMPT)
                    method = "frames_only"
            except Exception as e:
                print(f"  ERROR: {e}")
                judgment = {"memorized": False, "confidence": 0.0, "reasoning": f"ERROR: {e}"}
                method = "error"

            print(f"  {'MEMORIZED' if judgment.get('memorized') else 'NOT MEMORIZED'} "
                  f"(conf={judgment.get('confidence', 0):.2f})")

            write_judgment(json_path, judgment, method)

            row = {
                "prompt_id": prompt_id,
                "uuid": uid or "",
                "seed": seed,
                "view_paths": ";".join(view_paths),
                "json_path": json_path,
                "memorized": bool(judgment.get("memorized", False)),
                "confidence": float(judgment.get("confidence", 0.0)),
                "reasoning": judgment.get("reasoning", ""),
                "gt_rendered": bool(gt_frames),
                "method": method,
            }
            csv_writer.writerow(row)
            csv_file.flush()
            all_rows.append(row)

    csv_file.close()
    print(f"  → {args.results_csv}")
    return all_rows


def main():
    parser = argparse.ArgumentParser(
        description="Objaverse memorization judge for DiffSplat (mesh-render GT)"
    )
    parser.add_argument("--input-dir", "-i", required=True,
                        help="Flat directory with generated images "
                             "(e.g. output/cap3d or output/laion_memorized)")
    parser.add_argument("--clusters",
                        default="data/objaverse-dupes/aggregated_clusters.json",
                        help="Path to aggregated clusters JSON (unused for UID lookup "
                             "but kept for compatibility)")
    parser.add_argument("--results-csv", default=None,
                        help="Output CSV path. Defaults to "
                             "memorization/evaluation/results/judge_objaverse_{dirname}.csv")
    parser.add_argument("--temp-dir", default="/tmp/diffsplat_objaverse_judge")
    parser.add_argument("--aws-region", default="us-west-2")
    parser.add_argument("--model-id",
                        default="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--start-prompt", type=int, default=None)
    parser.add_argument("--end-prompt", type=int, default=None)
    args = parser.parse_args()

    # Default results-csv based on input directory name
    if args.results_csv is None:
        dirname = os.path.basename(os.path.normpath(args.input_dir))
        args.results_csv = os.path.join(
            "memorization", "evaluation", "results",
            f"judge_objaverse_{dirname}.csv"
        )

    vlm = ClaudeSonnetVLM(region=args.aws_region, model_id=args.model_id)
    os.makedirs(args.temp_dir, exist_ok=True)

    rows = _run_prompts(args.input_dir, args, vlm)
    print(f"\nDone. {len(rows)} total rows written to {args.results_csv}")


if __name__ == "__main__":
    main()
