import os
import json
import objaverse
import pandas as pd
from typing import List, Dict


def uids_to_prompts(uids: List[str]) -> List[str]:
    """Converts a list of unique identifiers (uids) to tag-concat prompts based on their annotations."""
    annotations = objaverse.load_annotations(uids)
    prompts_data = []
    for uid in uids:
        if uid in annotations:
            annotation = annotations[uid]
            attributes = [annotation.get('name', '')]
            for tag in annotation.get('tags', []):
                attributes.append(tag.get('name', ''))
            attributes.append('3d asset')
            prompt = ', '.join(filter(None, attributes))
            prompts_data.append({
                "Caption": prompt,
                "uid": uid
            })
    # print(prompts_data)
    return prompts_data


def load_uids_from_clusters(clusters_file: str = 'data/objaverse-dupes/aggregated_clusters.json',
                            concept_key: str = 'teddy_bear') -> List[str]:
    try:
        with open(clusters_file, 'r') as f:
            clusters_data = json.load(f)
        
        concept_data = clusters_data.get(concept_key, {})
        
        all_uids = []
        if isinstance(concept_data, dict):
            for cluster_id, uid_list in sorted(concept_data.items(), key=lambda item: int(item[0])):
                if isinstance(uid_list, list):
                    all_uids.extend(sorted(uid_list))
        elif isinstance(concept_data, list):
            all_uids = sorted(concept_data)
        
        print(f"Loaded and sorted {len(all_uids)} UIDs for concept '{concept_key}' from {clusters_file}")
        return all_uids
        
    except Exception as e:
        print(f"Error occurred while loading UIDs: {e}")
        return []
    
# print(load_uids_from_clusters())
# print(load_uids_from_clusters())

def load_prompts_from_csv(file_path: str) -> (List[str], List[Dict]):
    """
    Args:
        file_path (str): Path to the CSV file.

    Returns:
        A tuple containing:
        - List[str]: The list of prompts.
        - List[Dict]: A corresponding list of metadata dictionaries.
    """
    if not os.path.exists(file_path):
        print(f"Warning: CSV file not found at {file_path}. Skipping.")
        return [], []

    try:
        df = pd.read_csv(file_path, sep=';')
        if "Caption" not in df.columns or "URL" not in df.columns:
            print(f"Warning: CSV {file_path} must contain 'Caption' and 'URL' columns. Skipping.")
            return [], []

        prompts = df["Caption"].tolist()
        metadata = df.apply(lambda row: {"ground_truth_url": row["URL"]}, axis=1).tolist()
        
        print(f"Loaded {len(prompts)} prompts from {file_path}")
        return prompts, metadata

    except Exception as e:
        print(f"Error loading CSV file {file_path}: {e}")
        return [], []