import json
import matplotlib.pyplot as plt
import numpy as np

def filter_and_plot_clusters(
    input_path,
    output_path,
    plot_path,
    min_cluster_size=10,
    max_cluster_size=50,
    min_prompt_words=16,
    max_prompt_words=None,
    rejection_words=None,
    inclusion_words=None
):
    """
    Filters clusters from a JSON file based on size, prompt length, exclusion, and inclusion words.

    After filtering, it saves the resulting clusters to a new JSON file and plots the
    distribution of their sizes.

    Parameters:
        input_path (str): Path to the input JSON file.
        output_path (str): Path to the output JSON file for filtered data.
        plot_path (str): Path to save the output plot PNG file.
        min_cluster_size (int): Minimum number of prompts in a cluster to keep.
        max_cluster_size (int): Maximum number of prompts in a cluster to keep.
        min_prompt_words (int): Minimum word count for a prompt to be considered valid.
        max_prompt_words (int, optional): Maximum word count for a prompt. Defaults to None.
        rejection_words (list, optional): A list of words. If any of these words appear
                                          in the first 5 prompts of a cluster, the cluster
                                          is rejected. The check is case-insensitive.
        inclusion_words (list, optional): A list of words. If provided, a cluster MUST contain
                                          at least one of these words in its first 5 prompts
                                          to be kept. The check is case-insensitive.
    """
    if rejection_words is None:
        rejection_words = []
    if inclusion_words is None:
        inclusion_words = []

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_path}'")
        return

    print(f"Started with {len(data)} clusters.")

    # Helper function to check if a prompt's word count is valid.
    def is_prompt_length_valid(prompt):
        word_count = len(prompt.split())
        if max_prompt_words is None:
            return word_count >= min_prompt_words
        return min_prompt_words <= word_count <= max_prompt_words

    # Helper function to check for rejection words in the first 5 prompts.
    def contains_rejection_words(prompts, words_to_reject):
        for prompt in prompts[:5]:
            for word in words_to_reject:
                if word in prompt.lower():
                    return True
        return False

    # Helper function to check for inclusion words in the first 5 prompts.
    def contains_inclusion_words(prompts, words_to_include):
        if not words_to_include:
            return True  # If no inclusion words are specified, this condition is always met.
        for prompt in prompts[:5]:
            for word in words_to_include:
                if word in prompt.lower():
                    return True
        return False

    # Filter the data based on all criteria.
    filtered_data = {
        cluster_id: prompts for cluster_id, prompts in data.items()
        if (
            min_cluster_size <= len(prompts) <= max_cluster_size and
            any(is_prompt_length_valid(p) for p in prompts) and
            not contains_rejection_words(prompts, rejection_words) and
            contains_inclusion_words(prompts, inclusion_words)
        )
    }

    num_filtered_clusters = len(filtered_data)
    print(f"Finished with {num_filtered_clusters} clusters after filtering.")

    # Save the filtered data to the output JSON file.
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)
    print(f"Filtered data saved to '{output_path}'")

    if num_filtered_clusters == 0:
        print("No clusters remained after filtering, so no plot will be generated.")
        return

    # --- Plotting the Distribution of Cluster Sizes ---
    cluster_sizes = [len(v) for v in filtered_data.values()]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))

    max_size = max(cluster_sizes) if cluster_sizes else 0
    min_size = min(cluster_sizes) if cluster_sizes else 0
    plt.hist(cluster_sizes, bins=np.arange(min_size, max_size + 2) - 0.5, rwidth=0.8, edgecolor='black')

    plt.title('Distribution of Cluster Sizes (After Filtering)', fontsize=16)
    plt.xlabel('Cluster Size (Number of Prompts)', fontsize=12)
    plt.ylabel('Frequency (Number of Clusters)', fontsize=12)
    
    plt.xticks(np.arange(min_size, max_size + 1, step=max(1, (max_size - min_size) // 20)))
    
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Cluster size distribution plot saved to '{plot_path}'")


if __name__ == '__main__':
    exclusion_list = ['aerial', 'ink', 'fire', 'particle', 'abstract']
    inclusion_list = ['flag', 'business', 'hologram', 'doctor', 'aircraft']
    
    filter_and_plot_clusters(
        input_path='clusters-full.json',
        output_path='clusters-filtered.json',
        plot_path='cluster-filtered-distribution.png',
        min_cluster_size=10,
        max_cluster_size=500,
        min_prompt_words=16,
        max_prompt_words=None,
        rejection_words=exclusion_list,
        inclusion_words=inclusion_list,
    )