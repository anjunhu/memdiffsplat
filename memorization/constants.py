OUTPUT_ROOT = 'output'

ALL_METHODS = [
    'baseline',
]

# For plotting and consistent reporting
METHOD_COLORS = {
    "baseline": "black",
}

# Defines which datasets are expected for each method.
# 'default' is used for any method not explicitly listed.
EXPECTED_DATASETS = {
    'baseline': ['laion_memorized', 'laion_unmemorized', 'webvid10m'],
}

# --- Metric Structure Definitions ---

# for per-seed files like: prompt_0000_00_some_prompt_metrics.json
EXPECTED_PER_SEED_METRICS = {
    "Noise_Difference_Norm": ["noise_diff_norm_mean", "noise_diff_norm_traj"],
    "Hessian_SAIL_Metric": {
        "hessian_sail_norm": None,  # None indicates we just check for the key's existence
        "visualizations": {
            "t50": ["cond_magnitudes", "uncond_magnitudes"],
            "t1": ["cond_magnitudes", "uncond_magnitudes"],
            "t20": ["cond_magnitudes", "uncond_magnitudes"],
        }
    },
    # "HessianMetric": {
    #     "t1": ["cond_eigvals", "uncond_eigvals"],
    #     "t20": ["cond_eigvals", "uncond_eigvals"],
    #     "t50": ["cond_eigvals", "uncond_eigvals"],
    # },
    "BrightEnding_LD_Score": ["ld_score", "d_score", "be_intensity"],
    "CrossAttention_Entropy": ["entropy"]
}


# for cross-seed files like: prompt_0000_some_prompt_cross_seed.json
EXPECTED_CROSS_SEED_METRICS = {
    "Image_Diversity": ["median_sscd_similarity", "min_tiled_l2_distance", ]
}
