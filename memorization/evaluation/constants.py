OUTPUT_ROOT = 'output'

ALL_METHODS = [
    '.',
]

# For plotting and consistent reporting
METHOD_COLORS = {
    ".": "black",
}

# Defines which datasets are expected for each method.
# 'default' is used for any method not explicitly listed.
EXPECTED_DATASETS = {
    'baseline': ['laion_memorized', 'cap3d'],
}

# --- Metric Structure Definitions ---

# for per-seed files like: prompt_0000_00_some_prompt_metrics.json
# Note: Hessian timesteps are dynamic based on num_inference_steps (default 20)
# They will be t{T}, t{T//2}, t1 where T = num_inference_steps
# For 20 steps: t20, t10, t1
EXPECTED_PER_SEED_METRICS = {
    "Noise_Difference_Norm": ["noise_diff_norm_mean", "noise_diff_norm_traj"],
    "Hessian_SAIL_Metric": {
        "hessian_sail_norm": None,  # None indicates we just check for the key's existence
        "visualizations": {
            "t1": ["cond_magnitudes", "uncond_magnitudes"],
            "t10": ["cond_magnitudes", "uncond_magnitudes"],  # T//2 for 20 steps
            "t20": ["cond_magnitudes", "uncond_magnitudes"],  # T for 20 steps
        }
    },
    "BrightEnding_LD_Score": ["ld_score", "d_score", "be_intensity"],
    "CrossAttention_Entropy": ["cae-e", "cae-d"],
    "InvMM_Score": ["invmm_score", "success_rate"],
    "pLaplace_p1.0_Metric": ["mean", "max", "min", "t50", "t100", "t200", "t500"],
}


# for cross-seed files like: prompt_0000_some_prompt_cross_seed.json
EXPECTED_CROSS_SEED_METRICS = {
    "Image_Diversity": ["median_sscd_similarity", "min_tiled_l2_distance", "ssim_noise_diff"]
}
