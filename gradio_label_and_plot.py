import os
import glob
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
import gradio as gr
import trimesh
import objaverse
from typing import List
from scipy.spatial import distance
import tempfile
import difflib
import argparse
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve

def find_objaverse_cache_dir():
    """Try to find where Objaverse caches assets."""
    # Common locations for Objaverse cache
    potential_paths = [
        os.path.expanduser("~/.objaverse/objects"),
        os.path.expanduser("~/.cache/objaverse/objects"),
        "/tmp/objaverse/objects"
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            return path
    
    return None

def load_objaverse_mesh(uid: str) -> trimesh.Trimesh:
    mesh = objaverse.load_objects(uids=[uid])
    mesh = trimesh.load(list(mesh.values())[0])
    mesh = list(mesh.geometry.values())[0]
    return mesh

def objects_to_prompts(uids: List[str]) -> List[str]:
    annotations = objaverse.load_annotations(uids)
    prompts = []
    for uid in annotations:
        attributes = [annotations[uid]['name']]
        for tag in annotations[uid]['tags']:
            attributes.append(tag['name'])
        attributes.append('3d asset')
        p = ', '.join(attributes)
        prompts.append(p)
    return prompts

def normalize_mesh(mesh: trimesh.Trimesh):
    vertices = mesh.vertices
    pairwise_distances = distance.pdist(vertices)
    max_dist = np.max(pairwise_distances)
    normalized_vertices = vertices / max_dist
    normalized_mesh = trimesh.Trimesh(vertices=normalized_vertices, faces=mesh.faces)
    return normalized_mesh

def preprocess(mesh: trimesh.Trimesh, nsamples: int=5000, verbose=False):
    normalized_mesh = normalize_mesh(mesh)
    return normalized_mesh

def extract_info_from_filename(filename):
    """Extract index and prompt information from filename."""
    # Example: 0000_0__3D_scan_Wooden_statue_of_a_bigger_cute_bear_bear_wooden_statue_3dscan_3d_asset_0.png
    parts = os.path.basename(filename).split('__', 1)
    
    if len(parts) < 2:
        return None, None, None
    
    prefix = parts[0]  # e.g., "0000_0"
    index_parts = prefix.split('_')
    
    if len(index_parts) < 2:
        return None, None, None
    
    index = int(index_parts[0])
    label = int(index_parts[1])
    
    # Extract the safe prompt part (everything between __ and the last _number)
    safe_prompt = parts[1]
    safe_prompt = re.sub(r'_traj_\d+\.json$|_\d+\.png$|_noise_mag_\d+\.png$', '', safe_prompt)
    
    return index, label, safe_prompt

def get_file_pairs(input_files):
    """Get all the file pairs in the directory."""
    # Find all json files with traj pattern
    json_files = input_files
    
    pairs = []
    for json_file in json_files:
        # print("Processing file:", json_file)
        # Extract base pattern to find corresponding png files
        base_pattern = json_file.replace("_traj_", "_").rsplit(".", 1)[0]
        base_pattern = base_pattern.rsplit("_", 1)[0]
        
        # Check if corresponding image files exist
        img_file = f"{base_pattern}_{json_file.split('_traj_')[1].split('.')[0]}.png"
        # noise_img_file = f"{base_pattern}_noise_mag_{json_file.split('_traj_')[1].split('.')[0]}.png"
        
        if os.path.exists(img_file):
            pairs.append((json_file, img_file))
    
    return pairs

def load_json_data(json_file):
    """Load JSON data from file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def save_json_data(json_file, data):
    """Save JSON data to file."""
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)

def update_memorization_status(json_file, status):
    """Update the memorization status in the JSON file."""
    data = load_json_data(json_file)
    data["memorized"] = status
    save_json_data(json_file, data)
    return "JSON updated successfully!"

# 2. Add function to create noise trajectory plot (place after save_mesh_as_obj function)
def create_noise_trajectory_plot(json_data):
    """Create a Plotly figure from noise_diff_norms data."""
    fig = go.Figure()
    
    # Check if noise_diff_norms exists in the data
    if 'noise_diff_norms' in json_data:
        traj_data = json_data['noise_diff_norms']
        fig.add_trace(go.Scatter(
            x=list(range(len(traj_data))),
            y=traj_data,
            mode='lines',
            name='Noise Diff Norms',
            line=dict(color='black', width=3),
            opacity=1.0,
            hovertemplate='<b>Noise Trajectory</b><br>' +
                         'Step: %{x}<br>' +
                         'Value: %{y:.4f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title="Noise Diff Norms Trajectory",
            xaxis_title="Denoising Step",
            yaxis_title="Noise Diff Norm",
            template="plotly_white",
            height=400
        )
    else:
        # Create empty plot with message
        fig.add_annotation(
            text="No noise_diff_norms data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            template="plotly_white",
            height=400
        )
    
    return fig




def calculate_rocauc(inputs):
    """Calculate ROCAUC for average noise norm vs memorization status."""
    json_files = inputs[:700]
    
    # Store the average norms and their memorization labels
    avg_norms = []
    labels = []
    dataset_info = []  # Store which dataset each sample belongs to
    
    for json_file in json_files:
        try:
            data = load_json_data(json_file)
            
            # Check which noise norm field to use
            norm_field = "text_noise_norms"
            if "noise_diff_norms" in data:
                norm_field = "noise_diff_norms"  # Use this if available
            
            # Skip if no noise norms or not labeled
            if norm_field not in data or "memorized" not in data:
                continue
                
            # Calculate the average norm across timesteps
            avg_norm = np.mean(data[norm_field])
            avg_norms.append(avg_norm)
            labels.append(1 if data["memorized"] else 0)
            
            # Determine dataset
            if "objaverse" in str(json_file):
                dataset = "Objaverse"
            else:
                dataset = "LAION"
            dataset_info.append(dataset)
                
        except (KeyError, FileNotFoundError) as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    if not avg_norms or not labels:
        return None, None, None, None, None
    
    # Calculate overall ROCAUC
    fpr, tpr, thresholds = roc_curve(labels, avg_norms)
    roc_auc = auc(fpr, tpr)
    
    # Find the optimal threshold using Youden's J statistic (maximizing sensitivity + specificity - 1)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]
    
    # Calculate dataset-specific ROCAUCs
    dataset_results = {}
    for dataset_name in set(dataset_info):
        # Skip if it's just non-memorized samples (they're all treated the same)
        if sum(1 for i, l in enumerate(labels) if l == 1 and dataset_info[i] == dataset_name) == 0:
            continue
            
        # For each dataset, we compare memorized samples from this dataset vs all non-memorized
        dataset_labels = []
        dataset_norms = []
        
        for i, (norm, label) in enumerate(zip(avg_norms, labels)):
            # Include all non-memorized samples (label=0) regardless of dataset
            # For memorized samples (label=1), only include those from this dataset
            if label == 0 or (label == 1 and dataset_info[i] == dataset_name):
                dataset_labels.append(label)
                dataset_norms.append(norm)
                
        # Calculate ROCAUC for this dataset
        try:
            ds_fpr, ds_tpr, ds_thresholds = roc_curve(dataset_labels, dataset_norms)
            ds_roc_auc = auc(ds_fpr, ds_tpr)
            
            # Find optimal threshold
            ds_j_scores = ds_tpr - ds_fpr
            ds_best_idx = np.argmax(ds_j_scores)
            ds_optimal_threshold = ds_thresholds[ds_best_idx]
            
            dataset_results[dataset_name] = {
                'auc': ds_roc_auc,
                'fpr': ds_fpr,
                'tpr': ds_tpr,
                'threshold': ds_optimal_threshold
            }
        except Exception as e:
            print(f"Error calculating ROCAUC for {dataset_name}: {e}")
    
    return roc_auc, fpr, tpr, optimal_threshold, dataset_results

def plot_roc_curve(fpr, tpr, roc_auc, output_file="mvdream_roc_curve.png", dataset_results=None):
    """Create and save ROC curve plot with per-dataset breakdowns."""
    plt.figure(figsize=(10, 8))
    
    # Plot overall ROC curve
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Overall ROC (AUC = {roc_auc:.3f})')
    
    # Plot per-dataset ROC curves if available
    if dataset_results:
        colors = {'LAION': 'red', 'Objaverse': 'black'}
        for dataset_name, result in dataset_results.items():
            plt.plot(
                result['fpr'], 
                result['tpr'], 
                color=colors.get(dataset_name, 'purple'), 
                lw=1.5, 
                linestyle='-',
                label=f'{dataset_name} ROC (AUC = {result["auc"]:.3f})'
            )
    
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('MVDream - ROC Curves: Average Noise Norm as Memorization Predictor')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    return output_file

def plot_score_distributions(inputs, optimal_threshold=None, output_file="mvdream_score_distribution.png"):
    """Plot distributions of average noise norms for memorized vs non-memorized samples."""
    json_files = inputs[:700]
    
    # Separate scores by dataset and memorization status
    objaverse_memorized_scores = []
    laion_memorized_scores = []
    not_memorized_scores = []  # All non-memorized (combined across all datasets)
    
    for json_file in json_files:
        try:
            data = load_json_data(json_file)
            is_objaverse = "objaverse" in str(json_file)
            
            # Check which norm field to use
            norm_field = "text_noise_norms"
            if "noise_diff_norms" in data:
                norm_field = "noise_diff_norms"  # Use this if available
            
            # Skip if no noise norms or not labeled
            if norm_field not in data or "memorized" not in data:
                continue
                
            # Calculate the average norm across timesteps
            avg_norm = np.mean(data[norm_field])
            
            # Group all non-memorized samples together, regardless of dataset
            if not data["memorized"]:
                not_memorized_scores.append(avg_norm)
            else:
                # For memorized samples, maintain dataset distinction
                if is_objaverse:
                    objaverse_memorized_scores.append(avg_norm)
                else:
                    laion_memorized_scores.append(avg_norm)
                
        except (KeyError, FileNotFoundError):
            continue
    
    if not (objaverse_memorized_scores or laion_memorized_scores) and not not_memorized_scores:
        return None
    
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    max_value = max(
        max(objaverse_memorized_scores, default=0),
        max(laion_memorized_scores, default=0),
        max(not_memorized_scores, default=0)
    ) + 10
    bins = np.linspace(0, max_value, 30)
    
    # Plot all non-memorized samples together
    plt.hist(not_memorized_scores, bins=bins, alpha=0.5, label=f'Not Memorized (green)', color='green')
    
    # Plot memorized samples by dataset
    if laion_memorized_scores:
        plt.hist(laion_memorized_scores, bins=bins, alpha=0.5, label=f'LAION Memorized (red)', color='red')
    if objaverse_memorized_scores:
        plt.hist(objaverse_memorized_scores, bins=bins, alpha=0.7, label=f'Objaverse Memorized (black)', color='black')
    
    # Add vertical line for optimal threshold if provided
    if optimal_threshold is not None:
        plt.axvline(x=optimal_threshold, color='blue', linestyle='--', 
                   label=f'Optimal Threshold: {optimal_threshold:.2f}')
    
    plt.xlabel('Average Noise Norm')
    plt.ylabel('Frequency')
    plt.title('MVDream - Distribution of Average Noise Norms by Memorization Status')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return output_file

def plot_precision_recall_curve(inputs, output_file="mvdream_precision_recall.png"):
    """Create and save precision-recall curve."""
    json_files = inputs[:700]
    
    # Store the average norms and their memorization labels
    avg_norms = []
    labels = []
    
    for json_file in json_files:
        try:
            data = load_json_data(json_file)
            
            # Check which norm field to use
            norm_field = "text_noise_norms"
            if "noise_diff_norms" in data:
                norm_field = "noise_diff_norms"
                
            # Skip if no noise norms or not labeled
            if norm_field not in data or "memorized" not in data:
                continue
                
            # Calculate the average norm across timesteps
            avg_norm = np.mean(data[norm_field])
            avg_norms.append(avg_norm)
            labels.append(1 if data["memorized"] else 0)
                
        except (KeyError, FileNotFoundError):
            continue
    
    if not avg_norms or not labels or len(set(labels)) < 2:
        return None
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(labels, avg_norms)
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2)
    
    # Add labels and title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('MVDream - Precision-Recall Curve')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file)
    plt.close()
    
    return output_file



def create_aggregated_plot(inputs, output_file="mvdream_aggregated_plot.png"):
    """Create an aggregated plot of text_noise_norms for both classes."""
    json_files = inputs[:700]
    
    # Create four separate groups based on memorization status and path
    objaverse_memorized_norms = []
    objaverse_not_memorized_norms = []
    standard_memorized_norms = []
    standard_not_memorized_norms = []
    
    labeled_count = 0
    unlabeled_count = 0
    
    for json_file in json_files:
        try:
            data = load_json_data(json_file)
            is_objaverse = "objaverse" in str(json_file)
            
            if "memorized" in data:
                labeled_count += 1
                if data["memorized"]:
                    if is_objaverse:
                        objaverse_memorized_norms.append(data["text_noise_norms"])
                    else:
                        standard_memorized_norms.append(data["text_noise_norms"])
                else:
                    if is_objaverse:
                        objaverse_not_memorized_norms.append(data["text_noise_norms"])
                    else:
                        standard_not_memorized_norms.append(data["text_noise_norms"])
            else:
                unlabeled_count += 1
        except (KeyError, FileNotFoundError):
            continue
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Only create plot if we have data
    if any([objaverse_memorized_norms, objaverse_not_memorized_norms, 
            standard_memorized_norms, standard_not_memorized_norms]):
        # Get timesteps from the first available trajectory
        timesteps = None
        for norm_list in [objaverse_memorized_norms, objaverse_not_memorized_norms, 
                         standard_memorized_norms, standard_not_memorized_norms]:
            if norm_list:
                timesteps = list(range(len(norm_list[0])))
                break
        
        if timesteps is None:
            timesteps = []
        
        # Plot each group with its specific color
        legend_handles = []
        
        # LAION memorized (red)
        if standard_memorized_norms:
            line = plt.plot(timesteps, standard_memorized_norms[0], color="red", alpha=0.1, linewidth=1)[0]
            for traj in standard_memorized_norms[1:]:
                plt.plot(timesteps, traj, color="red", alpha=0.1, linewidth=1)
            legend_handles.append((line, f"LAION Memorized (red)"))
            
        # LAION not memorized (green)
        if standard_not_memorized_norms:
            line = plt.plot(timesteps, standard_not_memorized_norms[0], color="green", alpha=0.3, linewidth=1)[0]
            for traj in standard_not_memorized_norms[1:]:
                plt.plot(timesteps, traj, color="green", alpha=0.3, linewidth=1)
            legend_handles.append((line, f"LAION Not Memorized (green)"))

        # Objaverse not memorized (green)
        if objaverse_not_memorized_norms:
            line = plt.plot(timesteps, objaverse_not_memorized_norms[0], color="green", alpha=0.1, linewidth=1)[0]
            for traj in objaverse_not_memorized_norms[1:]:
                plt.plot(timesteps, traj, color="green", alpha=0.1, linewidth=1)
            legend_handles.append((line, f"Objaverse Not Memorized (green)"))

        # Objaverse memorized (black)
        if objaverse_memorized_norms:
            line = plt.plot(timesteps, objaverse_memorized_norms[0], color="black", alpha=0.8, linewidth=1)[0]
            for traj in objaverse_memorized_norms[1:]:
                plt.plot(timesteps, traj, color="black", alpha=0.8, linewidth=1)
            legend_handles.append((line, f"Objaverse Memorized (black)"))
            
        plt.title(f"MVDream | Aggregated Text Noise Norms by Memorization Status and Dataset\n"
                # f"(Labeled: {labeled_count}, Unlabeled: {unlabeled_count})"
                )
        plt.xlabel("Denoising Step")
        plt.ylabel("Text Noise Norm")
        plt.ylim(0, 40)
        plt.grid(True)

        # Add the legend with counts
        plt.legend([h for h, l in legend_handles], 
                  [l for h, l in legend_handles], 
                  loc="upper right")

    else:
        plt.text(0.5, 0.5, "No labeled data available", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14)
    
    plt.tight_layout()
    
    # Save to disk
    plt.savefig(output_file)
    plt.close()
    
    # Print statistics
    print(f"Aggregated plot saved to {output_file}")
    print(f"Statistics:")
    print(f"  - Objaverse Memorized: {len(objaverse_memorized_norms)}")
    print(f"  - Objaverse Not Memorized: {len(objaverse_not_memorized_norms)}")
    print(f"  - LAION Memorized: {len(standard_memorized_norms)}")
    print(f"  - LAION Not Memorized: {len(standard_not_memorized_norms)}")
    print(f"Total: {labeled_count} labeled, {unlabeled_count} unlabeled")
    
    # Add ROC AUC analysis after aggregated plot
    roc_auc, fpr, tpr, optimal_threshold, dataset_results = calculate_rocauc(inputs)
    
    # Create additional ROC analysis plots and export data
    results = {}
    
    if roc_auc is not None:
        # Create ROC curve plot
        roc_plot_path = plot_roc_curve(fpr, tpr, roc_auc, 
                                       output_file="mvdream_roc_curve.png", 
                                       dataset_results=dataset_results)
        results["roc_plot"] = roc_plot_path
        
        # Create distribution plot
        dist_plot_path = plot_score_distributions(inputs, optimal_threshold, 
                                                 output_file="mvdream_score_distribution.png")
        results["distribution_plot"] = dist_plot_path
        
        # Generate precision-recall curve
        if len(set([1 if data["memorized"] else 0 for data in 
                   [load_json_data(file) for file in json_files if "memorized" in load_json_data(file)]])) > 1:
            precision_recall_path = plot_precision_recall_curve(inputs, 
                                                              output_file="mvdream_precision_recall.png")
            results["precision_recall_plot"] = precision_recall_path
    
    return output_file, roc_auc, optimal_threshold, results


def check_prompt_consistency(filename, json_data):
    """Check if the prompt in the JSON matches the filename (accounting for safe_prompt transformations)."""
    _, _, safe_prompt_from_filename = extract_info_from_filename(filename)
    
    if "prompt" not in json_data:
        return "No prompt field in JSON", ""
    
    original_prompt = json_data["prompt"]
    
    # Convert original prompt to what would be a safe_prompt
    expected_safe_prompt = re.sub(r'\W+', '_', original_prompt)
    
    if safe_prompt_from_filename and safe_prompt_from_filename in expected_safe_prompt:
        return f"Prompt matches filename: {original_prompt}", original_prompt
    else:
        return f"WARNING: Prompt mismatch! {safe_prompt_from_filename} != {expected_safe_prompt}", original_prompt

def save_mesh_as_obj(mesh, output_file=None):
    """Save mesh as OBJ file for Gradio's Model3D component."""
    if output_file is None:
        # Create a temporary file if no output file is specified
        temp_file = tempfile.NamedTemporaryFile(suffix='.obj', delete=False)
        output_file = temp_file.name
        temp_file.close()
    
    # Export the mesh as OBJ
    mesh.export(output_file, file_type='obj')
    
    return output_file

def find_original_asset_path(uid):
    """Find the original asset path in the Objaverse cache."""
    cache_dir = find_objaverse_cache_dir()
    if not cache_dir:
        return None
    
    # The asset might be stored with its UID as the filename
    potential_paths = glob.glob(os.path.join(cache_dir, f"{uid}.*"))
    potential_paths += glob.glob(os.path.join(cache_dir, uid, "*.*"))
    
    for path in potential_paths:
        if os.path.exists(path) and path.lower().endswith(('.obj', '.glb', '.gltf')):
            return path
    
    return None

def find_best_matching_asset(safe_prompt, category='teddy_bear', top_n=5):
    """Find the best matching 3D asset based on prompt similarity."""
    try:
        # Load LVIS annotations
        lvis_annotations = objaverse.load_lvis_annotations()
        
        if category not in lvis_annotations:
            return None, None, f"Category '{category}' not found in LVIS annotations"
        
        # Get UIDs for the category
        uids = lvis_annotations[category]
        
        # Get prompts for the UIDs
        all_prompts = objects_to_prompts(uids)
        
        # Convert prompts to safe versions for comparison
        prompt_mapping = {}
        for uid, prompt in zip(uids, all_prompts):
            safe_version = re.sub(r'\W+', '_', prompt.lower())
            prompt_mapping[safe_version] = (uid, prompt)
        
        # Find the closest matches using difflib
        safe_prompts = list(prompt_mapping.keys())
        safe_prompt_lower = safe_prompt.lower()
        
        # Get top matches
        matches = difflib.get_close_matches(safe_prompt_lower, safe_prompts, n=top_n, cutoff=0.1)
        
        if not matches:
            # If no close matches, try substring matching
            submatches = []
            for sp in safe_prompts:
                if safe_prompt_lower in sp or sp in safe_prompt_lower:
                    submatches.append((sp, len(sp)))
            
            if submatches:
                # Sort by length (longer matches preferred)
                submatches.sort(key=lambda x: x[1], reverse=True)
                matches = [m[0] for m in submatches[:top_n]]
        
        if matches:
            # Return the best match
            best_match = matches[0]
            uid, original_prompt = prompt_mapping[best_match]
            
            # Calculate similarity score
            similarity = difflib.SequenceMatcher(None, safe_prompt_lower, best_match).ratio()
            
            return uid, original_prompt, f"Found match with {similarity:.1%} similarity"
        else:
            # Fallback to index if available
            index_match = re.search(r'(\d+)', safe_prompt)
            if index_match:
                index = int(index_match.group(1))
                if index < len(uids):
                    uid = uids[index]
                    original_prompt = all_prompts[uids.index(uid)]
                    return uid, original_prompt, f"Using index {index} from prompt"
                
            return None, None, "No matching assets found"
        
    except Exception as e:
        return None, None, f"Error finding matching asset: {str(e)}"

def load_3d_asset_by_prompt(safe_prompt, fallback_index=None, category='teddy_bear'):
    """Load a 3D asset based on prompt similarity with fallback to index."""
    # First try to find by prompt similarity
    uid, original_prompt, match_info = find_best_matching_asset(safe_prompt, category)
    
    if not uid and fallback_index is not None:
        # Try fallback to index
        try:
            lvis_annotations = objaverse.load_lvis_annotations()
            if category in lvis_annotations and fallback_index < len(lvis_annotations[category]):
                uid = lvis_annotations[category][fallback_index]
                match_info = f"Using fallback index {fallback_index}"
        except Exception as e:
            return None, f"Error using fallback index: {str(e)}"
    
    if not uid:
        return None, match_info
    
    try:
        # First try to find the original asset in the cache
        original_path = find_original_asset_path(uid)
        if original_path:
            return original_path, f"UID: {uid}\n{match_info}\nLoaded directly from cache"
        
        # Otherwise, load and process the mesh
        if not original_prompt:
            original_prompt = objects_to_prompts([uid])[0]
            
        mesh = load_objaverse_mesh(uid)
        normalized_mesh = preprocess(mesh)
        
        # Save as OBJ for the Model3D component
        obj_file = save_mesh_as_obj(normalized_mesh)
        
        return obj_file, f"UID: {uid}\nPrompt: {original_prompt}\n{match_info}"
        
    except Exception as e:
        return None, f"Error loading 3D asset: {str(e)}"

def app(input_files, enable_3d=False):
    pairs = get_file_pairs(input_files)
    if not pairs:
        print("No valid file pairs found.")
        return None
    
    current_index = 0
    current_3d_model = None
    
    def load_current_pair():
        nonlocal current_3d_model
        
        if current_index >= len(pairs):
            plot_path = create_aggregated_plot(input_files)
            return (
                None, None, "All files processed! Aggregated plot created.", 
                gr.update(value=plot_path, visible=True), 
                gr.update(visible=False), 
                gr.update(visible=False), 
                gr.update(visible=True),
                None, None
            )
        
        json_file, img_file = pairs[current_index]
        data = load_json_data(json_file)
        noise_plot_fig = create_noise_trajectory_plot(data)
        
        index, label, safe_prompt = extract_info_from_filename(json_file)
        prompt_message, original_prompt = check_prompt_consistency(json_file, data)

        model_path = None
        asset_info = "3D asset loading disabled" if not enable_3d else "No 3D asset information available"
        if safe_prompt and enable_3d:
            model_path, asset_info = load_3d_asset_by_prompt(safe_prompt, fallback_index=index)
            current_3d_model = model_path

        has_memorized_field = "memorized" in data
        default_value = None
        if has_memorized_field:
            default_value = "1 (Memorized)" if data["memorized"] else "0 (Not Memorized)"

        # Gather other attributes
        other_info = "\n".join(
            [f"- **{k}**: {v}" for k, v in data.items() if k not in ["prompt", "memorized", "text_noise_norms"]]
        )
        print(f"Other info: {other_info}")

        file_info_text = f"""
        **File {current_index + 1}/{len(pairs)}**: `{os.path.basename(json_file)}`

        **Prompt Check**: {prompt_message}

        **3D Asset Information**:   
        {asset_info}
        
        """

        return (
            img_file, 
            noise_plot_fig,
            file_info_text, 
            gr.update(value=None, visible=False),
            gr.update(value=default_value, visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            model_path if enable_3d else None,
            gr.update(visible=bool(model_path) if enable_3d else False)
        )
    
    def submit_label(choice):
        nonlocal current_index
        json_file = pairs[current_index][0]
        if choice is not None:
            update_memorization_status(json_file, choice)
        current_index += 1
        return load_current_pair()
    
    def skip():
        nonlocal current_index
        current_index += 1
        return load_current_pair()
    
    def go_back():
        nonlocal current_index
        current_index = max(0, current_index - 1)
        return load_current_pair()

    with gr.Blocks() as demo:
        gr.Markdown("# Memorization Labeling App with 3D Asset Visualization")
        gr.Markdown("Label images as memorized (1) or not memorized (0) and compare with 3D assets")
        
        with gr.Row():
            file_info = gr.Markdown("Loading...")
        
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(label="Image", show_label=True)
                noise_plot = gr.Plot(label="Noise Diff Norms", show_label=True)
            
            with gr.Column(scale=1):
                asset_3d = gr.Model3D(label="3D Asset", visible=False, 
                                     camera_position=[1.5, 1.5, 1.5],
                                     camera_target=[0, 0, 0]) if enable_3d else gr.Markdown("Disabled.")
        
        with gr.Row():
            choice = gr.Radio(["0 (Not Memorized)", "1 (Memorized)", "None (Unsure)"], label="Memorization Status")
        
        with gr.Row():
            submit_btn = gr.Button("Submit")
            skip_btn = gr.Button("Skip")
            back_btn = gr.Button("Back")
        
        with gr.Row():
            with gr.Column(scale=1):
                plot_display = gr.Image(label="Aggregated Plot", visible=False)
            
            with gr.Column(scale=1):
                asset_final = gr.Markdown(visible=False)
        
        # Initialize the UI with the first pair
        demo.load(
            load_current_pair, 
            [], 
            [image, noise_plot, file_info, plot_display, choice, submit_btn, plot_display, asset_3d, asset_3d]
        )
        
        # Set up event handlers
        submit_btn.click(
            fn=lambda c: submit_label(None if c == "None (Unsure)" else (1 if c == "1 (Memorized)" else 0)),
            inputs=[choice], 
            outputs=[image, noise_plot, file_info, plot_display, choice, submit_btn, plot_display, asset_3d, asset_3d]
        )
        skip_btn.click(
            fn=skip, 
            inputs=[], 
            outputs=[image, noise_plot, file_info, plot_display, choice, submit_btn, plot_display, asset_3d, asset_3d]
        )
        back_btn.click(
            fn=go_back,
            inputs=[],
            outputs=[image, noise_plot, file_info, plot_display, choice, submit_btn, plot_display, asset_3d, asset_3d]
        )
    return demo

def main():
    parser = argparse.ArgumentParser(description="Memorization Labeling App with 3D Asset Visualization")
    parser.add_argument("--inputs", "-i", nargs='+',
                        type=str, help="Regex of saved json files from attacks", 
                        default="./out/gsdiff_gobj83k_sd15__render/inference/gsdiff_cap3d_multiview_noisediffonly/")
    parser.add_argument("--port", "-d", type=int, default=7860, help="Port to run the app on")
    parser.add_argument("--output", "-o", type=str, default="mvdream_aggregated_plot.png", help="Output file for the aggregated plot")
    parser.add_argument("--plot-only", action="store_true", help="Only generate the aggregated plot without launching the UI")
    parser.add_argument("--enable-3d", action="store_true", default=True, help="Disable 3D model display")
    
    args = parser.parse_args()
    
    # Check if only plotting is requested
    if args.plot_only:
        create_aggregated_plot(args.inputs, args.output)
        return
    
    # Otherwise, launch the labeling app
    demo = app(args.inputs)
    demo.launch(server_port=args.port)

if __name__ == "__main__":
    main()