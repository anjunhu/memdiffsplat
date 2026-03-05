# DiffSplat/analysis.py
#
# Description:
# This script loads experimental results, processes them, and generates a variety of
# analysis plots. It uses a schema from constants.py to ensure only expected
# metrics are loaded. It automatically processes multiple Hessian metric types,
# generates plottable difference vectors, and derives scalar "sharpness gap" metrics
# from them for ROC analysis and distribution plots. The plotting logic is also
# enhanced to support a 'both' mode, which uses a hierarchical color and fill scheme to
# visualize multiple dimensions of the data simultaneously.
#
# Enhanced Features:
# - Multi-metric support: Can process multiple metrics at once or "all" available metrics
# - Fuzzy matching: Partial metric names match related metrics automatically
# - Wildcard support: Use wildcards in group specifications (e.g., baseline/*)
# - Summary Table: Automatically prints a ROC AUC summary table when using --metric all.
#
# Usage Examples:
#
# 1. Plot multiple metrics with wildcard groups:
#    python ./analysis.py --metric d_score lpips_score --group1 baseline/* --group2 subspace_prune/*
#
# 2. Plot all available metrics and get a summary table:
#    python ./analysis.py --metric all --group1 baseline/* ca_entropy/* --label_by memorized_field
#
# 3. Fuzzy matching - plot all BrightEnding metrics:
#    python ./analysis.py --metric BrightEnding --group1 baseline/* --group2 nemo/*
#
# 4. Generate Hessian plots with wildcards:
#    python ./analysis.py --group1 baseline/* ca_entropy/* --plot_hessian_diff --hessian_metric_type finidiff

import os
import glob
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from scipy.stats import gaussian_kde
from collections import defaultdict
import fnmatch
import re

from memorization.constants import (
    OUTPUT_ROOT,
    METHOD_COLORS,
    EXPECTED_PER_SEED_METRICS,
    EXPECTED_CROSS_SEED_METRICS
)
ANALYSIS_OUTPUT_ROOT = 'analysis/plots'

# --- Configuration for different Hessian metric types ---
HESSIAN_CONFIG = {
    'finidiff': {
        'cond_template': 'Hessian_SAIL_Metric_visualizations_{t}_cond_magnitudes',
        'uncond_template': 'Hessian_SAIL_Metric_visualizations_{t}_uncond_magnitudes',
        'diff_template': 'Hessian_SAIL_diff_{t}',
        'scalar_diff_template': 'Hessian_SAIL_diff_sum_{t}',
        'check_prefix': 'Hessian_SAIL_Metric_visualizations_',
        'check_suffix': '_cond_magnitudes',
        'plot_title': 'Hessian (FiniDiff) Eigenvalue Distribution Comparison'
    },
    'autograd': {
        'cond_template': 'HessianMetric_{t}_cond_eigvals',
        'uncond_template': 'HessianMetric_{t}_uncond_eigvals',
        'diff_template': 'HessianMetric_diff_{t}',
        'scalar_diff_template': 'HessianMetric_diff_sum_{t}',
        'check_prefix': 'HessianMetric_',
        'check_suffix': '_cond_eigvals',
        'plot_title': 'Hessian (AutoGrad) Eigenvalue Distribution Comparison'
    }
}


def detect_hessian_timesteps(df_columns, config):
    """
    Dynamically detect available Hessian timesteps from DataFrame columns.
    Returns list of timestep key strings sorted descending by numeric value.
    """
    prefix = config.get('check_prefix', '')
    suffix = config.get('check_suffix', '')
    timesteps = set()
    pattern = re.compile(rf'^{re.escape(prefix)}([a-zA-Z]?\d+){re.escape(suffix)}$')
    for col in df_columns:
        match = pattern.match(col)
        if match:
            timesteps.add(match.group(1))
    def sort_key(val):
        return int(re.sub(r'[^0-9]', '', val))
    return sorted(timesteps, key=sort_key, reverse=True)


# --- Utility Functions for Enhanced Features ---

def expand_wildcards(directory_patterns):
    """
    Expands wildcard patterns in directory specifications.
    Example: 'baseline/*' -> ['baseline/laion_memorized', 'baseline/objaverse_teddy_bear']
    """
    expanded_dirs = []
    
    for pattern in directory_patterns:
        if '*' in pattern:
            # Handle wildcard patterns
            full_pattern = os.path.join(OUTPUT_ROOT, pattern)
            matching_paths = glob.glob(full_pattern)
            
            for path in matching_paths:
                if os.path.isdir(path):
                    # Convert back to relative path format (method/dataset)
                    rel_path = os.path.relpath(path, OUTPUT_ROOT)
                    expanded_dirs.append(rel_path)
        else:
            # Handle exact directory specifications
            expanded_dirs.append(pattern)
    
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for item in expanded_dirs:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result
    

def fuzzy_match_metrics(user_metrics, available_metrics):
    """
    Performs fuzzy matching of user-specified metrics against available metrics.
    Returns a list of matched metric names.
    """
    if not user_metrics:
        return []
    
    if user_metrics == ['all']:
        return available_metrics
    
    matched_metrics = []
    
    for user_metric in user_metrics:
        if user_metric in available_metrics:
            # Exact match
            matched_metrics.append(user_metric)
        else:
            # Fuzzy matching - find metrics that contain the user input
            fuzzy_matches = [m for m in available_metrics if user_metric.lower() in m.lower()]
            
            if fuzzy_matches:
                matched_metrics.extend(fuzzy_matches)
                print(f"Fuzzy matched '{user_metric}' to: {fuzzy_matches}")
            else:
                print(f"Warning: No matches found for metric '{user_metric}'")
    
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for item in matched_metrics:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return result


def create_safe_filename(base_name, max_length=200):
    """
    Creates a filesystem-safe filename by truncating and adding a hash if needed.
    """
    import hashlib
    
    # Replace problematic characters
    safe_name = base_name.replace('/', '-').replace('\\', '-').replace(':', '-')
    
    if len(safe_name) <= max_length:
        return safe_name
    
    # If too long, truncate and add hash of full name
    hash_suffix = hashlib.md5(base_name.encode()).hexdigest()[:8]
    truncated = safe_name[:max_length-10]  # Leave room for hash and separator
    return f"{truncated}_{hash_suffix}"


def create_group_label(directories, max_dirs=3):
    """
    Creates a concise label for a group of directories.
    """
    if not directories:
        return "empty"
    
    if len(directories) <= max_dirs:
        return "+".join(d.replace('/', '-') for d in directories)
    else:
        # Show first few directories and indicate there are more
        shown = "+".join(d.replace('/', '-') for d in directories[:max_dirs])
        return f"{shown}+{len(directories)-max_dirs}more"


# --- Data Loading and Processing ---

def extract_defined_metrics(data, schema):
    """
    Traverses the data and extracts metrics ONLY if they are defined in the schema.
    This prevents loading unexpected or malformed data.
    """
    if 'metrics' in data and isinstance(data['metrics'], dict):
        data = data['metrics']

    out = {}
    def traverse(data_node, schema_node, prefix=""):
        if not isinstance(data_node, dict) or not isinstance(schema_node, dict):
            return

        for schema_key, schema_value in schema_node.items():
            if schema_key in data_node:
                data_value = data_node[schema_key]
                new_prefix = f"{prefix}{schema_key}_"
                
                if isinstance(schema_value, list):
                    if isinstance(data_value, dict):
                        for sub_key in schema_value:
                            if sub_key in data_value:
                                out[f"{new_prefix}{sub_key}".replace('-', '_')] = data_value[sub_key]
                    elif isinstance(data_value, list):
                         out[f"{prefix}{schema_key}".replace('-', '_')] = data_value
                elif isinstance(schema_value, dict):
                    traverse(data_value, schema_value, new_prefix)
                elif schema_value is None:
                     out[f"{prefix}{schema_key}".replace('-', '_')] = data_value

    traverse(data, schema)
    return out

def get_prompt_identifier(filepath):
    """Extracts a unique identifier for a prompt from its file path."""
    # Example path: output/method/dataset/prompt_0000_...
    # We want to group by 'method/dataset/prompt_0000'
    parts = filepath.split(os.sep)
    filename = parts[-1]
    prompt_base = "_".join(filename.split('_')[:2]) #e.g. prompt_0000
    return os.path.join(parts[-3], parts[-2], prompt_base)


def load_all_results_to_dataframe(target_dirs: list) -> pd.DataFrame:
    """
    Scans specified directories, loads all defined features into a DataFrame,
    and correctly assigns 'memorized' status to cross-seed metrics.
    """
    all_json_files = []
    print(f"Scanning {len(target_dirs)} specified directories for JSON files...")
    for dir_path in set(target_dirs):
        full_scan_path = os.path.join(OUTPUT_ROOT, dir_path)
        if not os.path.isdir(full_scan_path):
            print(f"Warning: Directory not found, awoiding: {full_scan_path}")
            continue
        files_in_dir = glob.glob(os.path.join(full_scan_path, '**', '*.json'), recursive=True)
        all_json_files.extend(files_in_dir)

    # --- First Pass: Determine memorization status for each prompt ---
    prompt_memorization_status = defaultdict(list)
    print("Pass 1: Determining memorization status for each prompt...")
    for f_path in all_json_files:
        if '_cross_seed.json' in os.path.basename(f_path) or 'cross_seed_metrics' in os.path.basename(f_path):
            continue
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
            prompt_id = get_prompt_identifier(f_path)
            if 'memorized' in data:
                prompt_memorization_status[prompt_id].append(data.get('memorized', False))
        except (json.JSONDecodeError, IndexError, IOError):
            continue
            
    # Aggregate status: if any seed is memorized, the whole prompt is.
    final_prompt_status = {
        prompt_id: any(statuses) for prompt_id, statuses in prompt_memorization_status.items()
    }

    # --- Second Pass: Load all data and assign correct memorized flag ---
    records = []
    print(f"Pass 2: Loading {len(all_json_files)} JSON files and assigning labels...")
    for f_path in all_json_files:
        try:
            parts = f_path.split(os.sep)
            if len(parts) < 4 or parts[-4] != os.path.basename(OUTPUT_ROOT):
                continue
            method, dataset = parts[-3], parts[-2]

            with open(f_path, 'r') as f:
                data = json.load(f)
            
            is_cross_seed = '_cross_seed.json' in os.path.basename(f_path) or 'cross_seed_metrics' in os.path.basename(f_path)
            schema = EXPECTED_CROSS_SEED_METRICS if is_cross_seed else EXPECTED_PER_SEED_METRICS
            defined_metrics = extract_defined_metrics(data, schema)

            # Determine the memorized flag
            if is_cross_seed:
                prompt_id = get_prompt_identifier(f_path)
                memorized_flag = final_prompt_status.get(prompt_id, None)
            else:
                memorized_flag = data.get('memorized')

            record = {
                'method': method,
                'dataset': dataset,
                'filepath': f_path,
                'memorized': memorized_flag,
                **defined_metrics
            }
            records.append(record)
        except (json.JSONDecodeError, IndexError, IOError) as e:
            print(f"Warning: Could not process file {f_path}. Reason: {e}")

    df = pd.DataFrame(records)
    print(f"Successfully loaded {len(df)} records into DataFrame using defined schemas.")
    return df


def filter_df_by_dirs(df, dir_list):
    """Filters a DataFrame to include only records from a list of 'method/dataset' strings."""
    if not dir_list: return pd.DataFrame()
    conditions = []
    for d in dir_list:
        try:
            method, dataset = d.split('/')
            conditions.append((df['method'] == method) & (df['dataset'] == dataset))
        except ValueError:
            print(f"Warning: Skipping invalid directory format: {d}")
            continue
    if not conditions:
        return pd.DataFrame()
    combined_condition = pd.concat(conditions, axis=1).any(axis=1)
    return df[combined_condition]

def apply_temporal_transform(df, metric, transform_type):
    """Applies a transformation to a temporal metric column."""
    if metric not in df.columns:
        print(f"Error: Metric '{metric}' not found for transformation.")
        return df, None

    new_metric_name = f"{metric}_{transform_type}"
    valid_series = df[metric].dropna().apply(isinstance, args=(list,))
    if not valid_series.any():
        print(f"Warning: No list data found for metric '{metric}' to transform.")
        return df, None

    transforms = {
        'mean': np.mean, 'median': np.median, 'max': np.max,
        'min': np.min, 'std': np.std, 'last': lambda x: x[-1] if x else None
    }
    
    df[new_metric_name] = df[metric].apply(
        lambda x: transforms[transform_type](x) if isinstance(x, list) and len(x) > 0 else None
    )
    print(f"Applied '{transform_type}' transform: new metric is '{new_metric_name}'")
    return df, new_metric_name

# --- Plotting and Analysis ---

def find_optimal_threshold(y_true, y_score):
    """
    Find the optimal threshold for a ROC curve using Youden's J statistic.
    It automatically handles cases where the positive class has lower scores.
    """
    # Check if the positive class (1) has lower scores than the negative class (0)
    mean_pos = np.mean(y_score[y_true == 1])
    mean_neg = np.mean(y_score[y_true == 0])

    # If positive class scores are lower, we invert the scores for calculation
    # so that "higher" always means more likely to be positive.
    if mean_pos < mean_neg:
        scores_for_calc = -y_score
    else:
        scores_for_calc = y_score
        
    fpr, tpr, thresholds = roc_curve(y_true, scores_for_calc)
    
    j_scores = tpr - fpr
    if len(j_scores) == 0: return None, None
    
    best_idx = np.argmax(j_scores)
    
    # The threshold is on the transformed scale, so we must untransform it
    optimal_threshold_transformed = thresholds[best_idx]
    
    if mean_pos < mean_neg:
        # If we inverted the scores, we must invert the threshold back
        optimal_threshold = -optimal_threshold_transformed
    else:
        optimal_threshold = optimal_threshold_transformed
        
    return optimal_threshold, j_scores[best_idx]

def plot_scalar_distributions_teaser(
    df1, df2, label1, label2, metric, output_base_path, title,
    color1="red", color2="green", log_x=True
):
    """
    Teaser-style distribution plot:
    - KDE (gaussian_kde) curves with translucent fill
    - optional log-x axis
    - minimal grid/labels
    Saves both PNG and PDF using output_base_path.{png,pdf}
    """
    # Extract scores
    s1 = df1[metric].dropna().astype(float)
    s2 = df2[metric].dropna().astype(float)
    if s1.empty and s2.empty:
        print(f"Warning: Not enough data to plot teaser distribution for '{metric}'.")
        return

    # Determine x-range across both sets
    all_scores = []
    if not s1.empty: all_scores.extend(s1.tolist())
    if not s2.empty: all_scores.extend(s2.tolist())
    if len(all_scores) == 0:
        print(f"Warning: No valid scores for teaser distribution '{metric}'.")
        return

    lo, hi = float(np.nanmin(all_scores)), float(np.nanmax(all_scores))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        print(f"Warning: Degenerate range for teaser distribution '{metric}'.")
        return

    # Build x-grid (log-friendly if requested)
    if log_x and lo > 0:
        x = np.logspace(np.log10(max(lo, 1e-6)), np.log10(hi), 1000)
    else:
        x = np.linspace(lo, hi, 1000)

    plt.figure(figsize=(8, 6))

    # KDE and plot (mirrors teaservisuals.py)
    if not s1.empty:
        kde1 = gaussian_kde(s1)
        y1 = kde1(x)
        plt.fill_between(x, y1, alpha=0.30, color=color1, label=label1)
        plt.plot(x, y1, color=color1, lw=1.5)
    if not s2.empty:
        kde2 = gaussian_kde(s2)
        y2 = kde2(x)
        plt.fill_between(x, y2, alpha=0.30, color=color2, label=label2)
        plt.plot(x, y2, color=color2, lw=1.5)

    if log_x and lo > 0:
        plt.xscale('log')

    plt.title(title, fontsize=14)
    plt.xlabel('Metric Score'); plt.ylabel('Density')
    # plt.legend(loc='upper left', fontsize=10)
    # plt.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()

    # Save both PNG + PDF
    png_path = output_base_path + ".png"
    pdf_path = output_base_path + ".pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print(f"Saved teaser distribution plots:\n  {png_path}\n  {pdf_path}")

def plot_scalar_distributions(df1, df2, label1, label2, metric, output_path, title):
    """Plots distribution histograms for a scalar metric from two dataframes."""
    plt.figure(figsize=(12, 8))
    scores1 = df1[metric].dropna().astype(float) # Class 0
    scores2 = df2[metric].dropna().astype(float) # Class 1

    if scores1.empty and scores2.empty:
        print(f"Warning: Not enough data to plot distribution for '{metric}'.")
        plt.close(); return

    # Prepare data for optimal threshold calculation
    y_true = np.concatenate([np.zeros(len(scores1)), np.ones(len(scores2))])
    y_score = pd.concat([scores1, scores2]).to_numpy()
    
    opt_thresh, j_stat = find_optimal_threshold(y_true, y_score)

    all_scores = y_score[~np.isnan(y_score)]
    if len(all_scores) == 0:
        print(f"Warning: No valid scores to determine bin range for '{metric}'.")
        plt.close(); return
        
    bins = np.linspace(all_scores.min(), all_scores.max(), 50)
    if not scores1.empty:
        plt.hist(scores1, bins=bins, alpha=0.7, label=f'{label1}', color='red', density=True)
    if not scores2.empty:
        plt.hist(scores2, bins=bins, alpha=0.7, label=f'{label2}', color='blue', density=True)
    
    if opt_thresh is not None:
        plt.axvline(opt_thresh, color='green', linestyle='--', lw=2.5,
                    label=f'Optimal Threshold = {opt_thresh:.4f}\n(Youden\'s J = {j_stat:.3f})')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Metric Score'); plt.ylabel('Density'); plt.legend()
    plt.grid(True, alpha=0.5); plt.tight_layout(); plt.savefig(output_path); plt.close()
    print(f"Saved distribution plot: {output_path}")

def plot_scalar_distributions_detailed(plot_groups, metric, output_path, title):
    """
    Plots distribution histograms for multiple groups, using distinct colors for each group.
    Each method/dataset/memorization combination gets its own unique color for accessibility.
    """
    plt.figure(figsize=(14, 10))
    
    all_scores_list = [g['df'][metric].dropna().astype(float) for g in plot_groups]
    if not any(not s.empty for s in all_scores_list):
        print(f"Warning: No data to plot detailed distribution for '{metric}'.")
        plt.close(); return
    all_scores = pd.concat(all_scores_list)

    # Calculate optimal threshold between all 'memorized' and 'non-memorized' samples
    y_true_list, y_score_list = [], []
    for g in plot_groups:
        is_memorized = '(Mem)' in g['label']  # Updated logic since we no longer use hatch
        scores = g['df'][metric].dropna().astype(float)
        if not scores.empty:
            y_true_list.append(np.full(len(scores), int(is_memorized)))
            y_score_list.append(scores)
    
    if not y_true_list:
        opt_thresh, j_stat = None, None
    else:
        y_true = np.concatenate(y_true_list)
        y_score = pd.concat(y_score_list).to_numpy()
        opt_thresh, j_stat = find_optimal_threshold(y_true, y_score)
    
    bins = np.linspace(all_scores.min(), all_scores.max(), 50)
    
    for group in plot_groups:
        scores = group['df'][metric].dropna().astype(float)
        if not scores.empty:
            plt.hist(scores, bins=bins, alpha=0.7, label=group['label'], 
                     color=group['color'], density=True,
                     edgecolor='black', linewidth=0.5)

    if opt_thresh is not None:
        plt.axvline(opt_thresh, color='black', linestyle='--', lw=2.5,
                    label=f'Overall Optimal Threshold = {opt_thresh:.4f}\n(Youden\'s J = {j_stat:.3f})')

    plt.title(title, fontsize=16)
    plt.xlabel('Metric Score'); plt.ylabel('Density')
    plt.grid(True, alpha=0.5)

    # Simple legend with all groups
    plt.legend(loc='upper left', fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    plt.savefig(output_path)
    plt.close()
    print(f"Saved detailed distribution plot: {output_path}")


def plot_temporal_comparison(df1, df2, label1, label2, metric, output_path):
    """Plots the mean and std deviation for a temporal metric."""
    plt.figure(figsize=(12, 8))
    
    def plot_single_dist(df, label, color):
        series = df[metric].dropna()
        if series.empty: return
        max_len = series.apply(len).max()
        data_padded = series.apply(lambda x: x + [np.nan] * (max_len - len(x))).tolist()
        data = np.array(data_padded)
        mean, std = np.nanmean(data, axis=0), np.nanstd(data, axis=0)
        x_axis = np.arange(len(mean))
        plt.plot(x_axis, mean, color=color, label=f'{label}', lw=2.5)
        plt.fill_between(x_axis, mean - std, mean + std, color=color, alpha=0.2)

    plot_single_dist(df1, label1, 'red'); plot_single_dist(df2, label2, 'blue')
    
    plt.title(f'Temporal Metric Comparison: {metric}', fontsize=16)
    plt.xlabel('Step / Index'); plt.ylabel('Metric Value'); plt.legend()
    plt.grid(True, alpha=0.5); plt.tight_layout(); plt.savefig(output_path); plt.close()
    print(f"Saved temporal plot: {output_path}")

def calculate_roc_auc(df, metric, label_col):
    """
    Calculates and returns the ROC AUC score, fpr, and tpr.
    Automatically handles cases where the positive class has lower scores.
    Returns (None, None, None) if calculation is not possible.
    """
    data = df.dropna(subset=[metric, label_col]).copy()
    data[label_col] = pd.to_numeric(data[label_col], errors='coerce')
    data.dropna(subset=[label_col], inplace=True)
    data[label_col] = data[label_col].astype(int)
    
    if data[label_col].nunique() < 2:
        return None, None, None

    y_true = data[label_col].to_numpy()
    y_score = data[metric].astype(float).to_numpy()
    
    # Handle cases with NaN/inf in scores
    if not np.all(np.isfinite(y_score)):
        print(f"Warning: Non-finite values found in '{metric}' for ROC calculation. Skipping.")
        return None, None, None

    mean_pos = np.mean(y_score[y_true == 1])
    mean_neg = np.mean(y_score[y_true == 0])
    
    scores_for_roc = y_score
    if not np.isnan(mean_pos) and not np.isnan(mean_neg) and mean_pos < mean_neg:
        scores_for_roc = -y_score

    try:
        fpr, tpr, _ = roc_curve(y_true, scores_for_roc)
        roc_auc = auc(fpr, tpr)
    except ValueError as e:
        print(f"Warning: Could not calculate ROC for metric '{metric}'. Error: {e}")
        return None, None, None
        
    return roc_auc, fpr, tpr

def plot_roc_curve(df, metric, label_col, output_path, title, ax=None):
    """
    Plots a ROC curve using the centralized calculation function.
    Can plot on a provided axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        standalone_plot = True
    else:
        standalone_plot = False

    roc_auc, fpr, tpr = calculate_roc_auc(df, metric, label_col)

    if roc_auc is None:
        warning_msg = f"Warning: For '{title}', ROC curve requires at least 2 classes in '{label_col}'. Skipping plot."
        print(warning_msg)
        if standalone_plot:
            ax.text(0.5, 0.5, "Not enough data for ROC curve", ha='center', va='center')
            plt.title(title)
            plt.savefig(output_path)
            plt.close()
        else:
             ax.text(0.5, 0.5, "N/A", ha='center', va='center')
             ax.set_title(title)
        return

    ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.5)

    if standalone_plot:
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Saved ROC plot: {output_path}")


def plot_hessian_eigenvalue_comparison(df1, df2, label1, label2, output_path, hessian_config, title_supplement=None):
    """
    Creates a multi-row plot for a comprehensive Hessian analysis.
    - Row 1: Eigenvalue distributions (mean +/- std dev).
    - Row 2: Distribution of the scalar sharpness gap metric.
    - Row 3: ROC curve for the scalar sharpness gap metric.
    """
    all_cols = set(df1.columns) | set(df2.columns)
    timesteps = detect_hessian_timesteps(all_cols, hessian_config)
    if not timesteps:
        print("Warning: No Hessian timesteps found for plotting.")
        return
    
    fig, axes = plt.subplots(3, len(timesteps), figsize=(6 * len(timesteps), 18))
    
    title = f"{hessian_config['plot_title']}\n'{label1}' vs '{label2}'"
    if title_supplement:
        import textwrap
        wrapped_supplement = '\n'.join(textwrap.wrap(f"(Data Sources: {title_supplement})", width=100))
        title += f"\n{wrapped_supplement}"
    fig.suptitle(title, fontsize=16)
    
    colors = {'df1': 'red', 'df2': 'blue'}

    for j, t in enumerate(timesteps):
        # --- Row 1: Eigenvalue Distributions ---
        ax_eig = axes[0, j]
        ax_eig.set_title(f'Eigenvalue Dist. (t={t})')
        ax_eig.grid(True, alpha=0.5)

        for df, label, color in [(df1, label1, colors['df1']), (df2, label2, colors['df2'])]:
            cond_col = hessian_config['cond_template'].format(t=t)
            uncond_col = hessian_config['uncond_template'].format(t=t)

            def plot_series(series, line_style, alpha):
                if series.empty: return
                sorted_series = series.apply(lambda x: sorted(x) if isinstance(x, list) else np.nan).dropna()
                if sorted_series.empty: return
                max_len = sorted_series.apply(len).max()
                data_padded = sorted_series.apply(lambda x: x + [np.nan] * (max_len - len(x))).tolist()
                data = np.array(data_padded)
                mean_vals, std_vals = np.nanmean(data, axis=0), np.nanstd(data, axis=0)
                x_axis = np.arange(len(mean_vals))
                ax_eig.plot(x_axis, mean_vals, color=color, linestyle=line_style, label=f'{label} ({alpha})')
                ax_eig.fill_between(x_axis, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.15)

            if cond_col in df.columns: plot_series(df[cond_col].dropna(), '-', 'Cond')
            if uncond_col in df.columns: plot_series(df[uncond_col].dropna(), '--', 'Uncond')
        
        ax_eig.legend()

        # --- Row 2 & 3: Scalar Metric Analysis ---
        scalar_metric = hessian_config['scalar_diff_template'].format(t=t)
        if scalar_metric not in df1.columns or scalar_metric not in df2.columns:
            axes[1, j].text(0.5, 0.5, 'Scalar data not found', ha='center', va='center')
            axes[2, j].text(0.5, 0.5, 'Scalar data not found', ha='center', va='center')
            continue

        # --- Row 2: Distribution of Scalar Metric ---
        ax_dist = axes[1, j]
        scores1 = df1[scalar_metric].dropna()
        scores2 = df2[scalar_metric].dropna()
        all_scores = pd.concat([scores1, scores2])
        if not all_scores.empty:
            bins = np.linspace(all_scores.min(), all_scores.max(), 30)
            ax_dist.hist(scores1, bins=bins, color=colors['df1'], alpha=0.7, label=label1, density=True)
            ax_dist.hist(scores2, bins=bins, color=colors['df2'], alpha=0.7, label=label2, density=True)
        ax_dist.set_title(f'Scalar Gap Dist. (t={t})')
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.5)

        # --- Row 3: ROC Curve for Scalar Metric ---
        ax_roc = axes[2, j]
        df_combined = pd.concat([df1.assign(roc_label=0), df2.assign(roc_label=1)])
        plot_roc_curve(df_combined, scalar_metric, 'roc_label', None, f'ROC (t={t})', ax=ax_roc)

    plt.tight_layout(rect=[0, 0, 1, 0.92]) # Adjust layout for suptitle
    plt.savefig(output_path)
    plt.close()
    print(f"Saved comprehensive Hessian analysis plot: {output_path}")

# --- Analysis Runners ---

def run_summary_table(args, df_full, all_dirs):
    """
    Calculates and prints a summary table of ROC AUC scores for key metrics.
    This is triggered when --metric all is used.
    """
    print("\n--- Generating ROC AUC Summary Table ---")
    
    df_to_analyze = filter_df_by_dirs(df_full, all_dirs)
    
    # Prepare the 'memorized' column
    if 'memorized' not in df_to_analyze.columns:
        print("Error: 'memorized' column not found. Cannot generate summary table.")
        return
    
    df_to_analyze = df_to_analyze.dropna(subset=['memorized']).copy()
    df_to_analyze['memorized'] = pd.to_numeric(df_to_analyze['memorized'], errors='coerce')
    df_to_analyze.dropna(subset=['memorized'], inplace=True)
    df_to_analyze['memorized'] = df_to_analyze['memorized'].astype(bool)

    # Defines the metrics and their display names for the summary table, in order.
    summary_metrics_list = [
        ("Diversity Tiled l2", "Image_Diversity_min_tiled_l2_distance"),
        ("Diversity Median SSCD", "Image_Diversity_median_sscd_similarity"),
        ("Local X-Attn Entropy", "CrossAttention_Entropy_entropy"),
        ("BE (Localized ||sθΔ(xt)||)", "BrightEnding_LD_Score_ld_score"),
        ("||sθΔ(xt)||", "Noise_Difference_Norm_noise_diff_norm_mean"),
    ]
    
    # Dynamically add Hessian metrics based on what's in the data
    for m_type, config in HESSIAN_CONFIG.items():
        detected_ts = detect_hessian_timesteps(df_to_analyze.columns, config)
        for t in detected_ts:
            col_name = config['scalar_diff_template'].format(t=t)
            display_name = f"Hessian {m_type.title()} ({t})"
            if col_name in df_to_analyze.columns:
                summary_metrics_list.append((display_name, col_name))
    
    # Dynamically add pLaplace metrics if present
    for col in sorted(df_to_analyze.columns):
        if col.startswith('pLaplace') and df_to_analyze[col].notna().any():
            display = col.replace('_', ' ')
            summary_metrics_list.append((display, col))
    
    results = {}
    table_title = ""

    if args.label_by == 'directory':
        if not args.group1 or not args.group2:
            print("Summary Table Warning: --label_by directory requires at least 2 groups for summary table.")
            return

        g1_label = create_group_label(args.group1)
        g2_label = create_group_label(args.group2)
        table_title = f"ROC AUC Summary ('{g1_label}' vs '{g2_label}')"
        
        df1 = filter_df_by_dirs(df_full, args.group1)
        df2 = filter_df_by_dirs(df_full, args.group2)
        
        if df1.empty or df2.empty:
            print(f"Summary Table Warning: Not enough data for one or both groups to generate directory comparison summary.")
            return
            
        df_to_analyze = pd.concat([df1.assign(summary_label=0), df2.assign(summary_label=1)])
        
        for display_name, metric_name in summary_metrics_list:
            if metric_name in df_to_analyze.columns:
                roc_auc, _, _ = calculate_roc_auc(df_to_analyze, metric_name, 'summary_label')
                results[display_name] = roc_auc if roc_auc is not None else "N/A"
            else:
                results[display_name] = "N/A"

    else: # 'memorized_field' or 'both'
        table_title = "ROC AUC Summary (Memorized vs. Non-Memorized)"
        all_dirs = [d for group in args.original_groups for d in group]
        df_to_analyze = filter_df_by_dirs(df_full, all_dirs)
        
        if 'memorized' not in df_to_analyze.columns:
            print("Summary Table Warning: 'memorized' column not found. Cannot generate summary table.")
            return
        
        df_to_analyze = df_to_analyze.dropna(subset=['memorized']).copy()
        df_to_analyze['memorized'] = pd.to_numeric(df_to_analyze['memorized'], errors='coerce').astype(bool)

        if df_to_analyze['memorized'].nunique() < 2:
            print("Summary Table Warning: Data for both memorized and non-memorized samples is required for summary table.")
            return
            
        for display_name, metric_name in summary_metrics_list:
            if metric_name in df_to_analyze.columns:
                roc_auc, _, _ = calculate_roc_auc(df_to_analyze, metric_name, 'memorized')
                results[display_name] = roc_auc if roc_auc is not None else "N/A"
            else:
                results[display_name] = "N/A"

    # Print the formatted table
    print("\n" + "="*60)
    print(f" {table_title}")
    print("="*60)
    print(f"{'Method':<35} | {'ROC AUC':<15}")
    print("-" * 60)
    for name, score in results.items():
        if isinstance(score, float):
            print(f"{name:<35} | {score:<15.4f}")
        else:
            print(f"{name:<35} | {str(score):<15}")
    print("="*60 + "\n")


def run_directory_comparison(args, df_full, metrics_to_plot):
    """
    Enhanced function to orchestrate plotting for comparing multiple groups of directories.
    Now handles both traditional 2-group comparison and multi-group analysis.
    """
    print("\n--- Running Comparison by Directory ---")
    
    if len(args.original_groups) > 2:
        print(f"Multi-group mode: Comparing {len(args.original_groups)} groups")
        return run_multi_group_directory_comparison(args, df_full, metrics_to_plot)
    else:
        print("Two-group mode: Traditional comparison")
        return run_two_group_directory_comparison(args, df_full, metrics_to_plot)


def run_two_group_directory_comparison(args, df_full, metrics_to_plot):
    """Original two-group comparison logic."""
    df1 = filter_df_by_dirs(df_full, args.group1)
    df2 = filter_df_by_dirs(df_full, args.group2)

    if df1.empty or df2.empty:
        print("Error: Could not find data for one or both specified groups. Please check directory names.")
        return

    g1_label = create_group_label(args.group1)
    g2_label = create_group_label(args.group2)
    
    # Plot Hessian diff if requested or if 'all' metrics are requested
    if args.plot_hessian_diff or (args.metric and 'all' in args.metric):
        output_dir = os.path.join(ANALYSIS_OUTPUT_ROOT, 'Hessian_Analysis')
        os.makedirs(output_dir, exist_ok=True)
        for m_type in args.hessian_metric_type:
            config = HESSIAN_CONFIG.get(m_type)
            if not detect_hessian_timesteps(df_full.columns, config):
                print(f"Warning: Data for Hessian metric type '{m_type}' not found. Skipping its plot.")
                continue
            
            print(f"--- Generating Hessian Plot for metric type: '{m_type}' ---")
            filename_base = f"Hessian_{m_type.upper()}_Comparison_{g1_label[:32]}_vs_{g2_label[:32]}.png"
            output_path = os.path.join(output_dir, filename_base)
            plot_hessian_eigenvalue_comparison(df1, df2, g1_label, g2_label, output_path, config)

    # Plot specific metrics if requested
    if metrics_to_plot:
        for metric_to_plot in metrics_to_plot:
            # Apply transformation if needed
            df_transformed = df_full.copy()
            current_metric = metric_to_plot
            if args.transform:
                df_transformed, new_metric = apply_temporal_transform(df_transformed, metric_to_plot, args.transform)
                if new_metric:
                    current_metric = new_metric
                else:
                    print(f"Could not apply transform to '{metric_to_plot}'. Skipping this metric.")
                    continue

            # Re-filter after transformation
            df1_transformed = filter_df_by_dirs(df_transformed, args.group1)
            df2_transformed = filter_df_by_dirs(df_transformed, args.group2)

            output_dir = os.path.join(ANALYSIS_OUTPUT_ROOT, current_metric)
            os.makedirs(output_dir, exist_ok=True)
            filename_base = create_safe_filename(f"{g1_label}_vs_{g2_label}")
            
            is_temporal = df_transformed[current_metric].dropna().apply(isinstance, args=(list,)).any()

            if is_temporal:
                output_path = os.path.join(output_dir, f"Temporal_{filename_base}.png")
                plot_temporal_comparison(df1_transformed, df2_transformed, g1_label, g2_label, current_metric, output_path)
            else: # Scalar metric
                dist_title = f"Distribution of {current_metric}\n'{g1_label}' vs '{g2_label}'"
                dist_title = dist_title[:32]
                output_path_dist = os.path.join(output_dir, f"Dist_{filename_base}.png")
                plot_scalar_distributions(df1_transformed, df2_transformed, g1_label, g2_label, current_metric, output_path_dist, dist_title)

                df_combined = pd.concat([df1_transformed.assign(roc_label=0), df2_transformed.assign(roc_label=1)])
                roc_title = f"ROC: '{g1_label}' (0) vs '{g2_label}' (1)\nMetric: {current_metric}"
                output_path_roc = os.path.join(output_dir, f"ROC_by_dir_{filename_base}.png")
                plot_roc_curve(df_combined, current_metric, 'roc_label', output_path_roc, roc_title)


def run_multi_group_directory_comparison(args, df_full, metrics_to_plot):
    """
    Enhanced multi-group comparison logic that creates comprehensive plots
    showing all groups simultaneously with distinct visual styling.
    """
    print(f"Processing {len(args.original_groups)} groups for multi-group analysis...")
    
    # Prepare group data and labels
    group_data = []
    all_group_dirs = []
    
    for i, group_dirs in enumerate(args.original_groups):
        df_group = filter_df_by_dirs(df_full, group_dirs)
        if df_group.empty:
            print(f"Warning: No data found for group {i+1}: {group_dirs}")
            continue
            
        group_label = create_group_label(group_dirs)
        all_group_dirs.extend(group_dirs)
        
        group_data.append({
            'df': df_group,
            'dirs': group_dirs,
            'label': group_label,
            'index': i
        })
    
    if len(group_data) < 2:
        print("Error: Need at least 2 groups with data for multi-group comparison.")
        return
    
    combined_label = create_safe_filename("_vs_".join([g['label'] for g in group_data]), max_length=80)
    
    # Plot Hessian diff if requested or if 'all' metrics are requested
    if args.plot_hessian_diff or (args.metric and 'all' in args.metric):
        output_dir = os.path.join(ANALYSIS_OUTPUT_ROOT, 'Hessian_Analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        for m_type in args.hessian_metric_type:
            config = HESSIAN_CONFIG.get(m_type)
            if not detect_hessian_timesteps(df_full.columns, config):
                print(f"Warning: Data for Hessian metric type '{m_type}' not found. Skipping its plot.")
                continue
            
            print(f"--- Generating Multi-Group Hessian Plot for metric type: '{m_type}' ---")
            filename_base = f"Hessian_{m_type.upper()}_MultiGroup_{combined_label}.png"
            output_path = os.path.join(output_dir, filename_base)
            plot_multi_group_hessian_comparison(group_data, output_path, config)

    # Plot specific metrics if requested
    if metrics_to_plot:
        for metric_to_plot in metrics_to_plot:
            # Apply transformation if needed
            df_transformed = df_full.copy()
            current_metric = metric_to_plot
            if args.transform:
                df_transformed, new_metric = apply_temporal_transform(df_transformed, metric_to_plot, args.transform)
                if new_metric:
                    current_metric = new_metric
                else:
                    print(f"Could not apply transform to '{metric_to_plot}'. Skipping this metric.")
                    continue

            # Re-filter all groups after transformation
            transformed_group_data = []
            for group in group_data:
                df_group_transformed = filter_df_by_dirs(df_transformed, group['dirs'])
                if not df_group_transformed.empty:
                    transformed_group_data.append({
                        **group,
                        'df': df_group_transformed
                    })

            if len(transformed_group_data) < 2:
                print(f"Warning: Not enough groups with data after transformation for metric '{current_metric}'. Skipping.")
                continue

            output_dir = os.path.join(ANALYSIS_OUTPUT_ROOT, current_metric)
            os.makedirs(output_dir, exist_ok=True)
            
            is_temporal = df_transformed[current_metric].dropna().apply(isinstance, args=(list,)).any()

            if is_temporal:
                # Multi-group temporal comparison
                output_path = os.path.join(output_dir, f"Temporal_MultiGroup_{combined_label}.png")
                plot_multi_group_temporal_comparison(transformed_group_data, current_metric, output_path)
            else:
                # Multi-group scalar comparison
                # 1. Distribution plot with all groups
                dist_title = f"Multi-Group Distribution of {current_metric}"
                output_path_dist = os.path.join(output_dir, f"Dist_MultiGroup_{combined_label}.png")
                plot_multi_group_scalar_distributions(transformed_group_data, current_metric, output_path_dist, dist_title)
                
                # 2. Pairwise ROC curves (optional - can be many plots)
                if len(transformed_group_data) <= 5:  # Only do pairwise ROCs for reasonable number of groups
                    plot_pairwise_roc_curves(transformed_group_data, current_metric, output_dir, combined_label)
                else:
                    print(f"Skipping pairwise ROC curves for {len(transformed_group_data)} groups (too many combinations)")



def plot_multi_group_hessian_comparison(group_data, output_path, hessian_config, title_supplement=None):
    """
    Creates a comprehensive multi-group Hessian analysis plot.
    Similar structure to the original but supports multiple groups.
    """
    all_cols = set()
    for group in group_data:
        all_cols |= set(group['df'].columns)
    timesteps = detect_hessian_timesteps(all_cols, hessian_config)
    if not timesteps:
        print("Warning: No Hessian timesteps found for multi-group plotting.")
        return
    n_groups = len(group_data)
    
    fig, axes = plt.subplots(3, len(timesteps), figsize=(6 * len(timesteps), 18))
    
    title = f"{hessian_config['plot_title']} - Multi-Group Analysis"
    if title_supplement:
        title += f"\n(Groups: {title_supplement})"
    fig.suptitle(title, fontsize=16)
    
    # Use a colormap that provides distinct colors
    if n_groups <= 10:
        cmap = plt.cm.get_cmap('tab10')
    else:
        cmap = plt.cm.get_cmap('hsv')

    for j, t in enumerate(timesteps):
        # --- Row 1: Eigenvalue Distributions ---
        ax_eig = axes[0, j]
        ax_eig.set_title(f'Eigenvalue Dist. (t={t})')
        ax_eig.grid(True, alpha=0.5)

        for i, group in enumerate(group_data):
            df = group['df']
            label = group['label']
            color = cmap(i / n_groups) if n_groups > 10 else cmap(i)
            
            cond_col = hessian_config['cond_template'].format(t=t)
            uncond_col = hessian_config['uncond_template'].format(t=t)

            def plot_series(series, line_style, suffix):
                if cond_col not in df.columns or series.empty: 
                    return
                sorted_series = series.apply(lambda x: sorted(x) if isinstance(x, list) else np.nan).dropna()
                if sorted_series.empty: 
                    return
                max_len = sorted_series.apply(len).max()
                data_padded = sorted_series.apply(lambda x: x + [np.nan] * (max_len - len(x))).tolist()
                data = np.array(data_padded)
                mean_vals, std_vals = np.nanmean(data, axis=0), np.nanstd(data, axis=0)
                x_axis = np.arange(len(mean_vals))
                ax_eig.plot(x_axis, mean_vals, color=color, linestyle=line_style, 
                           label=f'{label} ({suffix})', alpha=0.8)
                ax_eig.fill_between(x_axis, mean_vals - std_vals, mean_vals + std_vals, 
                                   color=color, alpha=0.1)

            if cond_col in df.columns: 
                plot_series(df[cond_col].dropna(), '-', 'Cond')
            if uncond_col in df.columns: 
                plot_series(df[uncond_col].dropna(), '--', 'Uncond')
        
        ax_eig.legend(fontsize='small')

        # --- Row 2: Scalar Metric Distribution ---
        ax_dist = axes[1, j]
        ax_dist.set_title(f'Scalar Gap Dist. (t={t})')
        ax_dist.grid(True, alpha=0.5)
        
        scalar_metric = hessian_config['scalar_diff_template'].format(t=t)
        
        # Collect all scores to determine bin range
        all_scores = []
        valid_groups = []
        for group in group_data:
            if scalar_metric in group['df'].columns:
                scores = group['df'][scalar_metric].dropna()
                if not scores.empty:
                    all_scores.extend(scores.tolist())
                    valid_groups.append(group)
        
        if not all_scores:
            ax_dist.text(0.5, 0.5, 'Scalar data not found', ha='center', va='center')
        else:
            bins = np.linspace(min(all_scores), max(all_scores), 30)
            
            for i, group in enumerate(valid_groups):
                scores = group['df'][scalar_metric].dropna()
                color = cmap(group['index'] / n_groups) if n_groups > 10 else cmap(group['index'])
                ax_dist.hist(scores, bins=bins, color=color, alpha=0.6, 
                           label=group['label'], density=True, edgecolor='black', linewidth=0.5)
            
            ax_dist.legend(fontsize='small')

        # --- Row 3: ROC Analysis ---
        ax_roc = axes[2, j]
        ax_roc.set_title(f'Pairwise ROC (t={t})')
        ax_roc.grid(True, alpha=0.5)
        
        if not valid_groups:
            ax_roc.text(0.5, 0.5, 'Scalar data not found', ha='center', va='center')
        else:
            # For multi-group ROC, we can either:
            # 1. One-vs-rest for each group
            # 2. All pairwise comparisons (can be overwhelming)
            # Let's do one-vs-rest for clarity
            
            plot_one_vs_rest_roc(valid_groups, scalar_metric, ax_roc, cmap)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path)
    plt.close()
    print(f"Saved multi-group Hessian analysis plot: {output_path}")


def plot_one_vs_rest_roc(group_data, metric, ax, cmap):
    """
    Plots one-vs-rest ROC curves for multiple groups on a single axis.
    Each group is compared against all other groups combined.
    """
    n_groups = len(group_data)
    if n_groups < 2:
        ax.text(0.5, 0.5, 'Need ≥2 groups for ROC', ha='center', va='center')
        return
    
    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    
    for i, target_group in enumerate(group_data):
        # Prepare one-vs-rest data
        target_scores = target_group['df'][metric].dropna()
        other_scores = pd.concat([g['df'][metric].dropna() for j, g in enumerate(group_data) if j != i])
        
        if target_scores.empty or other_scores.empty:
            continue
            
        y_true = np.concatenate([np.ones(len(target_scores)), np.zeros(len(other_scores))])
        y_score = pd.concat([target_scores, other_scores]).to_numpy()
        
        # Handle case where target group has lower scores
        mean_target = np.mean(target_scores)
        mean_others = np.mean(other_scores)
        
        if mean_target < mean_others:
            scores_for_roc = -y_score
        else:
            scores_for_roc = y_score
            
        try:
            fpr, tpr, _ = roc_curve(y_true, scores_for_roc)
            roc_auc = auc(fpr, tpr)
            
            color = cmap(i / n_groups) if n_groups > 10 else cmap(i)
            ax.plot(fpr, tpr, color=color, lw=2, alpha=0.8,
                   label=f'{target_group["label"]} (AUC={roc_auc:.3f})')
        except Exception as e:
            print(f"Warning: Could not plot ROC for group '{target_group['label']}': {e}")
            continue
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', fontsize='small')


def plot_multi_group_scalar_distributions(group_data, metric, output_path, title):
    """
    Enhanced version of plot_scalar_distributions_detailed specifically for directory groups.
    Uses consistent color scheme and clear labeling.
    """
    plt.figure(figsize=(14, 10))
    
    # Collect all scores to determine bin range
    all_scores = []
    valid_groups = []
    
    for group in group_data:
        scores = group['df'][metric].dropna().astype(float)
        if not scores.empty:
            all_scores.extend(scores.tolist())
            valid_groups.append(group)
    
    if not all_scores:
        print(f"Warning: No data to plot multi-group distribution for '{metric}'.")
        plt.close()
        return
    
    bins = np.linspace(min(all_scores), max(all_scores), 50)
    
    # Use consistent color scheme
    n_groups = len(valid_groups)
    if n_groups <= 10:
        cmap = plt.cm.get_cmap('tab10')
    else:
        cmap = plt.cm.get_cmap('hsv')
    
    for group in valid_groups:
        scores = group['df'][metric].dropna().astype(float)
        color = cmap(group['index'] / len(group_data)) if n_groups > 10 else cmap(group['index'])
        
        plt.hist(scores, bins=bins, alpha=0.7, label=group['label'], 
                color=color, density=True, edgecolor='black', linewidth=0.5)
    
    # Calculate and plot optimal threshold between first group vs all others
    if len(valid_groups) >= 2:
        first_group_scores = valid_groups[0]['df'][metric].dropna().astype(float)
        other_scores = pd.concat([g['df'][metric].dropna().astype(float) for g in valid_groups[1:]])
        
        if not first_group_scores.empty and not other_scores.empty:
            y_true = np.concatenate([np.ones(len(first_group_scores)), np.zeros(len(other_scores))])
            y_score = pd.concat([first_group_scores, other_scores]).to_numpy()
            opt_thresh, j_stat = find_optimal_threshold(y_true, y_score)
            
            if opt_thresh is not None:
                plt.axvline(opt_thresh, color='black', linestyle='--', lw=2.5,
                           label=f'Optimal Threshold = {opt_thresh:.4f}\n({valid_groups[0]["label"]} vs Others)')

    plt.title(title, fontsize=16)
    plt.xlabel('Metric Score')
    plt.ylabel('Density')
    plt.legend(loc='upper left', fontsize='small')
    plt.grid(True, alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path)
    plt.close()
    print(f"Saved multi-group distribution plot: {output_path}")


def plot_multi_group_temporal_comparison(group_data, metric, output_path):
    """
    Plots temporal comparison for multiple groups with distinct colors and styling.
    """
    plt.figure(figsize=(14, 10))
    
    n_groups = len(group_data)
    if n_groups <= 10:
        cmap = plt.cm.get_cmap('tab10')
    else:
        cmap = plt.cm.get_cmap('hsv')
    
    for group in group_data:
        df = group['df']
        label = group['label']
        color = cmap(group['index'] / n_groups) if n_groups > 10 else cmap(group['index'])
        
        series = df[metric].dropna()
        if series.empty:
            continue
            
        # Process temporal data
        max_len = series.apply(len).max()
        data_padded = series.apply(lambda x: x + [np.nan] * (max_len - len(x))).tolist()
        data = np.array(data_padded)
        mean_vals = np.nanmean(data, axis=0)
        std_vals = np.nanstd(data, axis=0)
        x_axis = np.arange(len(mean_vals))
        
        plt.plot(x_axis, mean_vals, color=color, label=label, lw=2.5, alpha=0.8)
        plt.fill_between(x_axis, mean_vals - std_vals, mean_vals + std_vals, 
                        color=color, alpha=0.2)
    
    plt.title(f'Multi-Group Temporal Comparison: {metric}', fontsize=16)
    plt.xlabel('Step / Index')
    plt.ylabel('Metric Value')
    plt.legend(loc='upper left', fontsize='small')
    plt.grid(True, alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path)
    plt.close()
    print(f"Saved multi-group temporal plot: {output_path}")


def plot_pairwise_roc_curves(group_data, metric, output_dir, combined_label):
    """
    Creates pairwise ROC curve comparisons between all groups.
    Only used when the number of groups is reasonable (≤5).
    """
    print(f"Generating pairwise ROC curves for {len(group_data)} groups...")
    
    for i in range(len(group_data)):
        for j in range(i + 1, len(group_data)):
            group1, group2 = group_data[i], group_data[j]
            
            df_combined = pd.concat([
                group1['df'].assign(roc_label=0),
                group2['df'].assign(roc_label=1)
            ])
            
            if df_combined.empty:
                continue
            
            roc_title = f"ROC: '{group1['label']}' vs '{group2['label']}'\nMetric: {metric}"
            safe_filename = create_safe_filename(f"ROC_Pairwise_{group1['label']}_vs_{group2['label']}")
            output_path_roc = os.path.join(output_dir, f"{safe_filename}.png")
            
            plot_roc_curve(df_combined, metric, 'roc_label', output_path_roc, roc_title)

            
def run_field_comparison(args, df_full, mode, metrics_to_plot):
    """
    Orchestrates plotting for comparing by 'memorized' field.
    Handles both 'memorized_field' and 'both' modes.
    """
    print(f"\n--- Running Comparison by 'memorized' Field (Mode: {mode}) ---")
    all_dirs = list(set(args.group1 + (args.group2 or [])))
    df_to_analyze = filter_df_by_dirs(df_full, all_dirs)

    # Clean and prepare the 'memorized' column
    if 'memorized' not in df_to_analyze.columns:
        print("Error: 'memorized' column not found in the data. Cannot run this analysis.")
        return
        
    df_to_analyze = df_to_analyze.dropna(subset=['memorized']).copy()
    df_to_analyze['memorized'] = pd.to_numeric(df_to_analyze['memorized'], errors='coerce')
    df_to_analyze.dropna(subset=['memorized'], inplace=True)
    df_to_analyze['memorized'] = df_to_analyze['memorized'].astype(bool)

    if df_to_analyze['memorized'].nunique() < 2:
        print("Error: For this mode, data for both memorized (True) and non-memorized (False) samples is required. Found only one class.")
        return
        
    df_memorized = df_to_analyze[df_to_analyze['memorized'] == True]
    df_non_memorized = df_to_analyze[df_to_analyze['memorized'] == False]
    
    g1_label = "+".join(args.group1).replace('/', '-')
    g2_label = "+".join(args.group2).replace('/', '-') if args.group2 else None
    combined_label = create_safe_filename(g1_label + (f"-VS-{g2_label}" if g2_label else ""), max_length=80)

    # Plot Hessian diff if requested or if 'all' metrics are requested
    if args.plot_hessian_diff or (args.metric and 'all' in args.metric):
        output_dir = os.path.join(ANALYSIS_OUTPUT_ROOT, 'Hessian_Analysis')
        os.makedirs(output_dir, exist_ok=True)
        for m_type in args.hessian_metric_type:
            config = HESSIAN_CONFIG.get(m_type)
            # Use the more robust check here as well
            if not detect_hessian_timesteps(df_full.columns, config):
                print(f"Warning: Data for Hessian metric type '{m_type}' not found. Skipping its plot.")
                continue
            
            print(f"--- Generating Hessian Plot for metric type: '{m_type}' by memorized_field ---")
            filename_base = f"Hessian_{m_type.upper()}_Comparison_by_field_in_{combined_label[:32]}.png"
            output_path = os.path.join(output_dir, filename_base)
            plot_hessian_eigenvalue_comparison(
                df_memorized, df_non_memorized, 
                'Memorized', 'Non-Memorized', 
                output_path, config, title_supplement=combined_label
            )

    # Plot specific metrics if requested
    if metrics_to_plot:
        for metric_to_plot in metrics_to_plot:
            # Apply transformation if needed
            df_transformed = df_to_analyze.copy()
            current_metric = metric_to_plot
            if args.transform:
                df_transformed, new_metric = apply_temporal_transform(df_transformed, metric_to_plot, args.transform)
                if new_metric:
                    current_metric = new_metric
                else:
                    print(f"Could not apply transform to '{metric_to_plot}'. Skipping this metric.")
                    continue

            # Re-filter memorized/non-memorized after transformation
            df_mem_transformed = df_transformed[df_transformed['memorized'] == True]
            df_non_mem_transformed = df_transformed[df_transformed['memorized'] == False]

            output_dir = os.path.join(ANALYSIS_OUTPUT_ROOT, current_metric)
            os.makedirs(output_dir, exist_ok=True)
            
            is_temporal = df_transformed[current_metric].dropna().apply(isinstance, args=(list,)).any()

            if is_temporal:
                # --- Temporal Metric Plotting ---
                if mode == 'memorized_field':
                    temporal_title = f"Temporal Comparison: {current_metric}\n({combined_label})"
                    temporal_filename = f"Temporal_by_field_in_{combined_label}_{current_metric}.png"
                    plot_temporal_comparison(
                        df_non_mem_transformed, df_mem_transformed, 
                        'Non-Memorized', 'Memorized', 
                        current_metric, 
                        os.path.join(output_dir, temporal_filename)
                    )
                else:  # mode == 'both'
                    print(f"Warning: Temporal plot for metric '{current_metric}' in 'both' mode is complex and not yet implemented. Skipping.")
                    continue
            else:
                # --- Scalar Metric Plotting ---
                roc_title = f"ROC by 'memorized' field in {combined_label}\nMetric: {current_metric}"
                roc_filename = f"ROC_by_field_in_{combined_label[:20]}_{current_metric[:20]}.png"
                plot_roc_curve(df_transformed, current_metric, 'memorized', os.path.join(output_dir, roc_filename), roc_title)

                if mode == 'memorized_field':
                    dist_title = f"{current_metric}\n({combined_label})"
                    dist_title = dist_title[:32]
                    dist_filename = f"Dist_by_field_in_{combined_label[:20]}_{current_metric[:20]}.png"
                    plot_scalar_distributions(
                        df_non_mem_transformed, df_mem_transformed, 
                        'Non-Memorized', 'Memorized', 
                        current_metric, 
                        os.path.join(output_dir, dist_filename),
                        dist_title
                    )
                    if args.teaser_style:
                        teaser_visual_name = f"Dist_Teaser_by_field_in_{combined_label[:20]}_{current_metric[:20]}"
                        output_base = os.path.join(output_dir, "teaser", teaser_visual_name)
                        os.makedirs(output_base, exist_ok=True)
                        plot_scalar_distributions_teaser(
                            df_non_mem_transformed, df_mem_transformed,
                            'Non-Memorized', 'Memorized',
                            current_metric, output_base, dist_title,
                            color1="green", color2="red",  # match teaser: non-mem green, mem red
                            log_x=True
                        )
                elif mode == 'both':
                    # Create unique combinations of method/dataset/memorization
                    unique_combinations = []
                    for _, row in df_transformed.iterrows():
                        combo = (row['method'], row['dataset'], row['memorized'])
                        if combo not in unique_combinations:
                            unique_combinations.append(combo)
                    
                    # Assign distinct colors to each combination for accessibility
                    n_combinations = len(unique_combinations)
                    if n_combinations <= 10:
                        cmap = plt.cm.get_cmap('tab10')
                    elif n_combinations <= 20:
                        cmap = plt.cm.get_cmap('tab20')
                    else:
                        cmap = plt.cm.get_cmap('hsv')
                    
                    plot_groups = []
                    for i, (method_name, dataset_name, is_memorized) in enumerate(unique_combinations):
                        df_group = df_transformed[
                            (df_transformed['method'] == method_name) & 
                            (df_transformed['dataset'] == dataset_name) & 
                            (df_transformed['memorized'] == is_memorized)
                        ]
                        
                        if df_group.empty:
                            continue
                        
                        if n_combinations <= 20:
                            color = cmap(i)
                        else:
                            color = cmap(i / n_combinations)
                        
                        mem_status = 'Mem' if is_memorized else 'Non-Mem'
                        label = f'{method_name}/{dataset_name} ({mem_status})'
                        
                        plot_groups.append({
                            'df': df_group,
                            'label': label,
                            'color': color
                        })
                    
                    dist_title = f"Detailed Distribution of {current_metric}\n(Each method/dataset/memorization combination has a unique color)"
                    dist_filename = f"Dist_detailed_in_{combined_label}_{current_metric}.png"
                    plot_scalar_distributions_detailed(
                        plot_groups, 
                        current_metric, 
                        os.path.join(output_dir, dist_filename), 
                        dist_title
                    )


def validate_and_transform_metrics(df_full, user_metrics):
    """Validates and expands user-provided metrics using fuzzy matching."""
    available_metrics = sorted([col for col in df_full.columns if col not in ['method', 'dataset', 'filepath', 'memorized', 'group_label']])
    
    if not user_metrics:
        return []
    
    # Use fuzzy matching to expand the metrics
    matched_metrics = fuzzy_match_metrics(user_metrics, available_metrics)
    
    if not matched_metrics:
        print(f"\n--- ERROR: No valid metrics found! ---\nAvailable metrics include:\n" + "\n".join(f"  - {m}" for m in available_metrics[:20]))
        if len(available_metrics) > 20:
            print(f"  ... and {len(available_metrics) - 20} more metrics")
        return []
    
    # Filter out Hessian-related metrics from regular processing
    hessian_metric_patterns = [
        'Hessian_SAIL_Metric_visualizations',
        'HessianMetric_',
        'Hessian_SAIL_diff',
        'HessianMetric_diff'
    ]
    
    filtered_metrics = []
    hessian_metrics_found = []
    
    for metric in matched_metrics:
        is_hessian = any(pattern in metric for pattern in hessian_metric_patterns)
        if is_hessian:
            hessian_metrics_found.append(metric)
        else:
            filtered_metrics.append(metric)
    
    if hessian_metrics_found:
        print(f"Found {len(hessian_metrics_found)} Hessian-related metrics that will be handled by specialized Hessian plotting:")
        for h_metric in hessian_metrics_found[:5]:
            print(f"  - {h_metric}")
        if len(hessian_metrics_found) > 5:
            print(f"  ... and {len(hessian_metrics_found) - 5} more Hessian metrics")
        print("Note: Use --plot_hessian_diff --hessian_metric_type finidiff autograd to plot these metrics.")
    
    print(f"Will process {len(filtered_metrics)} non-Hessian metric(s): {filtered_metrics}")
    return filtered_metrics

    
# --- Main Execution ---

def main():
    """Defines arguments, loads data, and orchestrates plotting."""
    parser = argparse.ArgumentParser(
        description="Generate analysis plots for memorization metrics.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--metric", type=str, nargs='*', help="The metric(s) to plot. Can be exact names, partial names for fuzzy matching, or 'all' for all available metrics (e.g., 'd_score', 'BrightEnding', 'all').")
    parser.add_argument("--group1", type=str, nargs='+', help="First group of 'method/dataset' directories to compare (supports wildcards like baseline/*).")
    parser.add_argument("--group2", type=str, nargs='*', help="Second group of 'method/dataset' directories to compare (supports wildcards, only used for --label_by directory).")
    parser.add_argument("--groups", type=str, nargs='*', action='append', help="Multiple groups for comparison. Use multiple times: --groups baseline/* --groups ca_entropy/* --groups method3/*. Takes precedence over --group1/--group2.")
    parser.add_argument("--transform", type=str, choices=['mean', 'median', 'max', 'min', 'std', 'last'], help="Apply a transformation to temporal metrics to make them scalar.")
    parser.add_argument("--label_by", type=str, choices=['directory', 'memorized_field', 'both'], default='directory',
                        help="How to determine labels for ROC and plots.\n"
                             "'directory': compares --group1 vs --group2.\n"
                             "'memorized_field': compares memorized vs. non-memorized samples across all specified groups.\n"
                             "'both': uses a hierarchical color/fill scheme for detailed comparison.")
    parser.add_argument("--metric_type", type=str, choices=['all', 'per_seed', 'cross_seed'], default='all',
                        help="Specify which type of metrics to load and analyze. 'per_seed' or 'cross_seed'.")
    parser.add_argument("--plot_hessian_diff", action='store_true', help="Generate detailed plots comparing conditional and unconditional Hessian eigenvalues.")
    parser.add_argument("--hessian_metric_type", type=str, nargs='+', choices=['finidiff', 'autograd'], default=['finidiff'], 
                        help="Which type(s) of Hessian metric to plot. Can specify multiple (e.g., 'finidiff autograd').")
    parser.add_argument(
        "--teaser_style",
        action="store_true",
        help="Use simplified teaser-style distribution plots (KDE, minimal styling) and save both PNG+PDF."
    )
    args = parser.parse_args()

    # --- Reworked the argument parsing logic for groups ---
    args.original_groups = []
    dirs_to_scan = []

    if args.groups:
        print("Processing using --groups argument...")
        for group_list in args.groups:
            expanded_group = expand_wildcards(group_list)
            if expanded_group:
                args.original_groups.append(expanded_group)
                dirs_to_scan.extend(expanded_group)

        if len(args.original_groups) < 2 and args.label_by == 'directory':
             parser.error("--label_by directory requires at least 2 --groups arguments.")
        
        if len(args.original_groups) == 2:
            args.group1 = args.original_groups[0]
            args.group2 = args.original_groups[1]
        elif len(args.original_groups) == 1:
            args.group1 = args.original_groups[0]
            args.group2 = []
        else:
            args.group1, args.group2 = [], []

    else:
        print("Processing using --group1/--group2 arguments...")
        if not args.group1:
            parser.error("Either --group1 or --groups must be specified.")
        
        if args.label_by == 'directory' and not args.group2:
            parser.error("--group2 is required when --label_by is 'directory' and not using --groups.")

        args.group1 = expand_wildcards(args.group1)
        dirs_to_scan.extend(args.group1)
        args.original_groups.append(args.group1)

        if args.group2:
            args.group2 = expand_wildcards(args.group2)
            dirs_to_scan.extend(args.group2)
            args.original_groups.append(args.group2)

    dirs_to_scan = sorted(list(set(dirs_to_scan)))
    print(f"Final list of directories to scan: {dirs_to_scan}")
    for i, group in enumerate(args.original_groups):
        print(f"Group {i+1}: {group}")
    
    # --- Data Loading and Pre-processing ---
    df_full = load_all_results_to_dataframe(dirs_to_scan)
    if df_full.empty:
        print("No data loaded from JSON files. Exiting.")
        return

    # --- Validate and expand metrics ---
    metrics_to_plot = []
    if args.metric:
        metrics_to_plot = validate_and_transform_metrics(df_full, args.metric)
        if not metrics_to_plot and 'all' not in args.metric:
            return

        if args.metric_type != 'all':
            print(f"Filtering metrics to only include type: '{args.metric_type}'")
            schema_keys = list(EXPECTED_CROSS_SEED_METRICS.keys()) if args.metric_type == 'cross_seed' else list(EXPECTED_PER_SEED_METRICS.keys())
            filtered_metrics = [m for m in metrics_to_plot if any(m.startswith(key.replace('-', '_')) for key in schema_keys)]
            if not filtered_metrics and metrics_to_plot:
                print(f"Warning: After filtering for type '{args.metric_type}', no metrics from your selection remained.")
            else:
                print(f"Filtered metrics to plot: {filtered_metrics}")
            metrics_to_plot = filtered_metrics

    # --- Automatic Hessian Metric Processing ---
    print("Checking for Hessian metrics to process...")
    for m_type, config in HESSIAN_CONFIG.items():
        detected_timesteps = detect_hessian_timesteps(df_full.columns, config)
        if detected_timesteps:
            print(f"Found and processing '{m_type}' type Hessian metrics with timesteps: {detected_timesteps}")
            for t in detected_timesteps:
                cond_col, uncond_col = config['cond_template'].format(t=t), config['uncond_template'].format(t=t)
                diff_col = config['diff_template'].format(t=t)
                if cond_col in df_full.columns and uncond_col in df_full.columns:
                    def subtract_lists(row, cc=cond_col, uc=uncond_col):
                        cond, uncond = row[cc], row[uc]
                        if isinstance(cond, list) and isinstance(uncond, list):
                            min_len = min(len(cond), len(uncond))
                            return [c - u for c, u in zip(cond[:min_len], uncond[:min_len])]
                        return np.nan
                    df_full[diff_col] = df_full.apply(subtract_lists, axis=1)
            
            print(f"Generating scalar sum metrics for '{m_type}' type...")
            for t in detected_timesteps:
                diff_col, scalar_diff_col = config['diff_template'].format(t=t), config['scalar_diff_template'].format(t=t)
                if diff_col in df_full.columns:
                    df_full[scalar_diff_col] = df_full[diff_col].apply(lambda x: np.sum(x) if isinstance(x, list) else np.nan)
            print(f"Finished generating scalar metrics for '{m_type}' type.")

    # --- Summary Table Generation ---
    if args.metric and 'all' in args.metric:
        all_dirs_for_summary = [d for group in args.original_groups for d in group]
        all_dirs_for_summary = sorted(list(set(all_dirs_for_summary)))
        run_summary_table(args, df_full, all_dirs_for_summary)

    # --- Plotting Orchestration ---
    if args.label_by == 'directory':
        run_directory_comparison(args, df_full, metrics_to_plot)
    else: # 'memorized_field' or 'both'
        all_dirs_for_field_comp = [d for group in args.original_groups for d in group]
        args.group1 = sorted(list(set(all_dirs_for_field_comp)))
        args.group2 = []
        run_field_comparison(args, df_full, mode=args.label_by, metrics_to_plot=metrics_to_plot)


if __name__ == "__main__":
    main()
