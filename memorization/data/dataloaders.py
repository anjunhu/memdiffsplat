"""
Data loaders for DiffSplat memorization evaluation.

Handles different types of prompt datasets including LAION memorized/unmemorized,
WebVid10M prompts, and Cap3D/Objaverse dataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
from typing import List, Dict, Optional, Tuple, Union
import os
from datasets import load_dataset
import logging
from pathlib import Path

class PromptDataset(Dataset):
    """Base dataset class for prompt-based evaluation"""
    
    def __init__(self, prompts: List[str], labels: Optional[List[str]] = None, 
                 cluster_ids: Optional[List[str]] = None):
        self.prompts = prompts
        self.labels = labels or ['unknown'] * len(prompts)
        self.cluster_ids = cluster_ids or [str(i) for i in range(len(prompts))]
        
        if not (len(self.prompts) == len(self.labels) == len(self.cluster_ids)):
            raise ValueError("Prompts, labels, and cluster_ids must have same length")
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        item = {
            'prompt': self.prompts[idx],
            'label': self.labels[idx],
            'cluster_id': self.cluster_ids[idx],
            'idx': idx
        }
        
        # Add metadata if available
        if hasattr(self, 'metadata') and idx < len(self.metadata):
            item['metadata'] = self.metadata[idx]
        
        return item

class Cap3DDataset(PromptDataset):
    """
    Dataset for Cap3D prompts from Objaverse duplicates clusters.
    
    Reads UUIDs from aggregated_clusters.json and matches them with captions
    from Cap3D_automated_Objaverse_full.csv.
    """
    
    def __init__(self, clusters_json_path: str, captions_csv_path: str, 
                 concept_key: str = None, max_prompts_per_cluster: int = 10,
                 max_clusters: Optional[int] = None):
        self.logger = logging.getLogger("Cap3DDataset")
        
        # Load cluster data
        with open(clusters_json_path, 'r') as f:
            all_clusters = json.load(f)
        
        # Select specific concept or use all
        if concept_key and concept_key in all_clusters:
            cluster_data = {concept_key: all_clusters[concept_key]}
            self.logger.info(f"Loading concept '{concept_key}' with {len(all_clusters[concept_key])} clusters")
        else:
            cluster_data = all_clusters
            available_concepts = list(all_clusters.keys())
            self.logger.info(f"Loading all concepts: {available_concepts}")
        
        # Load captions CSV - handle different possible formats
        try:
            # Try comma-separated first
            captions_df = pd.read_csv(captions_csv_path, sep=',')
        except:
            try:
                # Try semicolon-separated
                captions_df = pd.read_csv(captions_csv_path, sep=';')
            except Exception as e:
                raise ValueError(f"Could not read captions CSV {captions_csv_path}: {e}")
        
        # Create UUID to caption mapping
        uuid_to_caption = {}
        if len(captions_df.columns) >= 2:
            # Assume first column is UUID, second is caption
            uuid_col = captions_df.columns[0]
            caption_col = captions_df.columns[1]
            
            for _, row in captions_df.iterrows():
                uuid = str(row[uuid_col]).strip()
                caption = str(row[caption_col]).strip()
                if uuid and caption and caption != 'nan':
                    uuid_to_caption[uuid] = caption
        else:
            raise ValueError(f"CSV must have at least 2 columns, got {len(captions_df.columns)}")
        
        self.logger.info(f"Loaded {len(uuid_to_caption)} UUID-caption mappings")
        
        # Process clusters and build dataset
        prompts = []
        labels = []
        cluster_ids = []
        self.metadata = []
        
        processed_clusters = 0
        total_uuids = 0
        matched_uuids = 0
        
        for concept_key, concept_clusters in cluster_data.items():
            if isinstance(concept_clusters, dict):
                # Format: {"cluster_id": ["uuid1", "uuid2", ...], ...}
                sorted_cluster_items = sorted(concept_clusters.items(), 
                                            key=lambda x: int(x[0]) if x[0].isdigit() else float('inf'))
            elif isinstance(concept_clusters, list):
                # Format: ["uuid1", "uuid2", ...] - treat as single cluster
                sorted_cluster_items = [("0", concept_clusters)]
            else:
                self.logger.warning(f"Unexpected cluster format for concept {concept_key}")
                continue
            
            for cluster_id, uuid_list in sorted_cluster_items:
                if max_clusters and processed_clusters >= max_clusters:
                    break
                
                if not isinstance(uuid_list, list):
                    self.logger.warning(f"Expected list of UUIDs for cluster {cluster_id}, got {type(uuid_list)}")
                    continue
                
                # Process UUIDs in this cluster
                cluster_prompts = []
                cluster_uuids = []
                
                for uuid in uuid_list[:max_prompts_per_cluster]:
                    total_uuids += 1
                    uuid_clean = str(uuid).strip()
                    
                    if uuid_clean in uuid_to_caption:
                        caption = uuid_to_caption[uuid_clean]
                        cluster_prompts.append(caption)
                        cluster_uuids.append(uuid_clean)
                        matched_uuids += 1
                
                if cluster_prompts:  # Only add cluster if we found matching captions
                    # Add all prompts from this cluster
                    prompts.extend(cluster_prompts)
                    labels.extend(['memorized' for _ in cluster_prompts])  # Mark Cap3D as memorized by default
                    cluster_ids.extend([f'{concept_key}_{cluster_id}' for _ in cluster_prompts])
                    
                    # Add metadata for each prompt
                    for i, (prompt, uuid) in enumerate(zip(cluster_prompts, cluster_uuids)):
                        self.metadata.append({
                            'uuid': uuid,
                            'concept_key': concept_key,
                            'cluster_id': cluster_id,
                            'prompt_in_cluster': i,
                            'total_in_cluster': len(cluster_prompts),
                            'source': 'cap3d_objaverse'
                        })
                    
                    processed_clusters += 1
                    self.logger.debug(f"Added cluster {concept_key}_{cluster_id} with {len(cluster_prompts)} prompts")
        
        self.logger.info(f"Processed {processed_clusters} clusters from Cap3D dataset")
        self.logger.info(f"Total UUIDs: {total_uuids}, Matched: {matched_uuids} ({100*matched_uuids/total_uuids:.1f}%)")
        self.logger.info(f"Final dataset size: {len(prompts)} prompts")
        
        if len(prompts) == 0:
            raise ValueError("No prompts were successfully loaded. Check UUID-caption matching.")
        
        super().__init__(prompts, labels, cluster_ids)

class LaionMemorizedDataset(PromptDataset):
    """
    Dataset for LAION memorized prompts from CSV files.
    
    Expected CSV format:
    ;Caption;URL;type;mean_sscd;max_sscd;Index
    0;The No Limits Business Woman Podcast;https://...;VM;0.839;0.846;1030727993
    ...
    
    Can also handle JSON cluster format for backward compatibility.
    """
    
    def __init__(self, data_path: str, max_prompts_per_cluster: int = 10, 
                 max_clusters: Optional[int] = None, source_type: str = "auto"):
        self.logger = logging.getLogger("LaionMemorizedDataset")
        
        if source_type == "auto":
            source_type = "csv" if data_path.endswith('.csv') else "json"
        
        if source_type == "csv":
            # Load CSV data
            memorized_df = pd.read_csv(data_path, sep=';')
            print(f"Loaded CSV with {len(memorized_df)} rows"); print(memorized_df.head(10))

            if max_clusters:
                memorized_df = memorized_df.head(max_clusters)
            
            prompts = memorized_df['Caption'].tolist()
            labels = ['memorized'] * len(prompts)
            # Use Index as cluster_id if available, otherwise use row index
            if 'Index' in memorized_df.columns:
                cluster_ids = memorized_df['Index'].astype(str).tolist()
            else:
                cluster_ids = [str(i) for i in range(len(prompts))]
            
            # Store additional metadata
            self.metadata = []
            for _, row in memorized_df.iterrows():
                self.metadata.append({
                    'url': row.get('URL', ''),
                    'mean_sscd': row.get('mean_sscd', 0.0),
                    'max_sscd': row.get('max_sscd', 0.0),
                    'index': row.get('Index', 0)
                })
            
            self.logger.info(f"Loaded {len(prompts)} memorized prompts from CSV")
            
        else:
            # JSON cluster format (backward compatibility)
            with open(data_path, 'r', encoding='utf-8') as f:
                cluster_data = json.load(f)
            
            prompts = []
            labels = []
            cluster_ids = []
            self.metadata = []
            
            # Process clusters
            processed_clusters = 0
            for cluster_id, cluster_prompts in cluster_data.items():
                if max_clusters and processed_clusters >= max_clusters:
                    break
                    
                # Take only first N prompts per cluster
                selected_prompts = cluster_prompts[:max_prompts_per_cluster]
                
                prompts.extend(selected_prompts)
                labels.extend(['memorized'] * len(selected_prompts))
                cluster_ids.extend([cluster_id] * len(selected_prompts))
                
                # Add basic metadata for JSON format
                for prompt in selected_prompts:
                    self.metadata.append({
                        'cluster_id': cluster_id,
                        'source': 'json'
                    })
                
                processed_clusters += 1
            
            self.logger.info(f"Loaded {len(prompts)} memorized prompts from {processed_clusters} JSON clusters")
        
        super().__init__(prompts, labels, cluster_ids)

class LaionUnmemorizedDataset(PromptDataset):
    """
    Dataset for LAION unmemorized prompts from CSV files.
    
    Expected CSV format same as memorized dataset.
    Can load from CSV or JSON cluster format.
    """
    
    def __init__(self, data_path: str, max_prompts_per_cluster: int = 10, 
                 max_clusters: Optional[int] = None, source_type: str = "auto"):
        self.logger = logging.getLogger("LaionUnmemorizedDataset")
        
        if source_type == "auto":
            source_type = "csv" if data_path.endswith('.csv') else "json"
        
        if source_type == "csv":
            # Load CSV data
            unmemorized_df = pd.read_csv(data_path, sep=';')
            
            # Filter for unmemorized types (everything except memorized types)
            if max_clusters:
                unmemorized_df = unmemorized_df.head(max_clusters)
            
            prompts = unmemorized_df['Caption'].tolist()
            labels = ['unmemorized'] * len(prompts)
            # Use Index as cluster_id if available, otherwise use row index
            if 'Index' in unmemorized_df.columns:
                cluster_ids = unmemorized_df['Index'].astype(str).tolist()
            else:
                cluster_ids = [str(i) for i in range(len(prompts))]
            
            # Store additional metadata
            self.metadata = []
            for _, row in unmemorized_df.iterrows():
                self.metadata.append({
                    'url': row.get('URL', ''),
                    'mean_sscd': row.get('mean_sscd', 0.0),
                    'max_sscd': row.get('max_sscd', 0.0),
                    'index': row.get('Index', 0)
                })
            
            self.logger.info(f"Loaded {len(prompts)} unmemorized prompts from CSV")
            
        elif source_type == "json":
            # JSON cluster format
            with open(data_path, 'r', encoding='utf-8') as f:
                cluster_data = json.load(f)
            
            prompts = []
            labels = []
            cluster_ids = []
            self.metadata = []
            
            processed_clusters = 0
            for cluster_id, cluster_prompts in cluster_data.items():
                if max_clusters and processed_clusters >= max_clusters:
                    break
                    
                # Take only first N prompts per cluster
                selected_prompts = cluster_prompts[:max_prompts_per_cluster]
                
                prompts.extend(selected_prompts)
                labels.extend(['unmemorized'] * len(selected_prompts))
                cluster_ids.extend([cluster_id] * len(selected_prompts))
                
                # Add basic metadata for JSON format
                for prompt in selected_prompts:
                    self.metadata.append({
                        'cluster_id': cluster_id,
                        'source': 'json'
                    })
                
                processed_clusters += 1
            
            self.logger.info(f"Loaded {len(prompts)} unmemorized prompts from {processed_clusters} JSON clusters")
        
        elif source_type == "parquet":
            # Original parquet format (backward compatibility)
            if data_path.endswith('.parquet'):
                df = pd.read_parquet(data_path)
            else:
                dataset = load_dataset(data_path, data_files={'train': 'sdv1_bb_edge_groundtruth.parquet'})
                df = dataset['train'].to_pandas()
            
            # Filter for non-memorized samples (everything except MV)
            unmemorized_df = df[df['overfit_type'] != 'MV']
            
            if max_clusters:
                unmemorized_df = unmemorized_df.head(max_clusters * max_prompts_per_cluster)
            
            prompts = unmemorized_df['caption'].tolist()
            labels = ['unmemorized'] * len(prompts)
            cluster_ids = [str(i) for i in range(len(prompts))]
            
            # Basic metadata for parquet
            self.metadata = [{'source': 'parquet'} for _ in prompts]
            
            self.logger.info(f"Loaded {len(prompts)} unmemorized prompts from parquet")
        
        else:
            raise ValueError(f"Unknown source_type: {source_type}")
        
        super().__init__(prompts, labels, cluster_ids)

class WebVid10MDataset(PromptDataset):
    """
    Dataset for WebVid10M prompts from clusters-filtered.json.
    
    Expected format:
    {
        "66": ["prompt1", "prompt2", ...],
        "11": ["prompt1", "prompt2", ...],
        ...
    }
    
    Where keys are cluster ordinals (not necessarily consecutive) and values are lists of prompts.
    """
    
    def __init__(self, data_path: str = None, max_prompts_per_cluster: int = 10,
                 max_clusters: Optional[int] = None):
        self.logger = logging.getLogger("WebVid10MDataset")
        
        if data_path and os.path.exists(data_path):
            # Load from clusters-filtered.json file
            if data_path.endswith('.json'):
                with open(data_path, 'r', encoding='utf-8') as f:
                    cluster_data = json.load(f)
                
                prompts = []
                cluster_ids = []
                self.metadata = []
                
                # Sort cluster IDs to ensure consistent ordering
                sorted_cluster_ids = sorted(cluster_data.keys(), key=lambda x: int(x))
                
                processed_clusters = 0
                for cluster_id in sorted_cluster_ids:
                    if max_clusters and processed_clusters >= max_clusters:
                        break
                        
                    cluster_prompts = cluster_data[cluster_id]
                    selected_prompts = cluster_prompts[:max_prompts_per_cluster]
                    
                    prompts.extend(selected_prompts)
                    cluster_ids.extend([cluster_id] * len(selected_prompts))
                    
                    # Add metadata for each prompt
                    for i, prompt in enumerate(selected_prompts):
                        self.metadata.append({
                            'cluster_id': cluster_id,
                            'prompt_in_cluster': i,
                            'total_in_cluster': len(cluster_prompts),
                            'source': 'webvid10m_clusters'
                        })
                    
                    processed_clusters += 1
                
                self.logger.info(f"Loaded {len(prompts)} prompts from {processed_clusters} WebVid10M clusters")
                    
            else:
                # Try to load as text file (one prompt per line)
                with open(data_path, 'r', encoding='utf-8') as f:
                    prompts = [line.strip() for line in f if line.strip()]
                if max_clusters:
                    prompts = prompts[:max_clusters * max_prompts_per_cluster]
                cluster_ids = [str(i) for i in range(len(prompts))]
                
                # Basic metadata for text file
                self.metadata = []
                for i, prompt in enumerate(prompts):
                    self.metadata.append({
                        'cluster_id': str(i),
                        'source': 'webvid10m_text'
                    })
                
                self.logger.info(f"Loaded {len(prompts)} prompts from text file")
                
        labels = ['memorized'] * len(prompts)  # Mark WebVid10M as memorized by default
        
        super().__init__(prompts, labels, cluster_ids)


class DatasetManager:
    """Manages loading and combining different datasets"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger("DatasetManager")
    
    def load_datasets(self) -> Dict[str, PromptDataset]:
        """Load all configured datasets"""
        datasets = {}
        
        # Load Cap3D dataset
        if 'cap3d' in self.config:
            cfg = self.config['cap3d']
            datasets['cap3d'] = Cap3DDataset(
                clusters_json_path=cfg['clusters_json_path'],
                captions_csv_path=cfg['captions_csv_path'],
                concept_key=cfg.get('concept_key'),
                max_prompts_per_cluster=cfg.get('max_prompts_per_cluster', 10),
                max_clusters=cfg.get('max_clusters')
            )
        
        # Load LAION memorized
        if 'laion_memorized' in self.config:
            cfg = self.config['laion_memorized']
            datasets['laion_memorized'] = LaionMemorizedDataset(
                data_path=cfg['path'],
                max_prompts_per_cluster=cfg.get('max_prompts_per_cluster', 10),
                max_clusters=cfg.get('max_clusters'),
                source_type=cfg.get('source_type', 'auto')
            )
        
        # Load LAION unmemorized  
        if 'laion_unmemorized' in self.config:
            cfg = self.config['laion_unmemorized']
            datasets['laion_unmemorized'] = LaionUnmemorizedDataset(
                data_path=cfg['path'],
                max_prompts_per_cluster=cfg.get('max_prompts_per_cluster', 10),
                max_clusters=cfg.get('max_clusters'),
                source_type=cfg.get('source_type', 'auto')
            )
        
        # Load WebVid10M
        if 'webvid10m' in self.config:
            cfg = self.config['webvid10m']
            datasets['webvid10m'] = WebVid10MDataset(
                data_path=cfg.get('path'),
                max_prompts_per_cluster=cfg.get('max_prompts_per_cluster', 10),
                max_clusters=cfg.get('max_clusters')
            )
        
        return datasets
    
    def create_dataloaders(self, datasets: Dict[str, PromptDataset], 
                          batch_size: int = 1) -> Dict[str, DataLoader]:
        """Create dataloaders for datasets"""
        dataloaders = {}
        
        for name, dataset in datasets.items():
            dataloaders[name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,  # Keep order for reproducibility
                num_workers=0   # Single process for simplicity
            )
        
        return dataloaders
    
    def get_all_prompts_and_labels(self, datasets: Dict[str, PromptDataset]) -> Tuple[List[str], List[str], List[str]]:
        """Get all prompts, labels, and cluster IDs from all datasets combined"""
        all_prompts = []
        all_labels = []
        all_cluster_ids = []
        
        for name, dataset in datasets.items():
            for item in dataset:
                all_prompts.append(item['prompt'])
                all_labels.append(f"{name}_{item['label']}")
                all_cluster_ids.append(f"{name}_{item['cluster_id']}")
        
        return all_prompts, all_labels, all_cluster_ids
    
    def get_dataset_statistics(self, datasets: Dict[str, PromptDataset]) -> Dict[str, Dict]:
        """Get statistics for each dataset"""
        stats = {}
        
        for name, dataset in datasets.items():
            # Count unique clusters
            cluster_ids = set()
            prompt_lengths = []
            
            for item in dataset:
                cluster_ids.add(item['cluster_id'])
                prompt_lengths.append(len(item['prompt'].split()))
            
            stats[name] = {
                'total_prompts': len(dataset),
                'unique_clusters': len(cluster_ids),
                'avg_prompt_length': sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0,
                'min_prompt_length': min(prompt_lengths) if prompt_lengths else 0,
                'max_prompt_length': max(prompt_lengths) if prompt_lengths else 0
            }
        
        return stats

def create_default_config(base_data_path: str) -> Dict:
    """Create default configuration for datasets using CSV format"""
    return {
        'cap3d': {
            'clusters_json_path': os.path.join(base_data_path, 'objaverse-dupes', 'aggregated_clusters.json'),
            'captions_csv_path': os.path.join(base_data_path, 'objaverse-dupes', 'Cap3D_automated_Objaverse_full.csv'),
            'concept_key': 'teddy_bear',  # or None for all concepts
            'max_prompts_per_cluster': 5,
            'max_clusters': 20
        },
        'laion_memorized': {
            'path': os.path.join(base_data_path, 'nemo-prompts', 'memorized_laion_prompts.csv'),
            'max_prompts_per_cluster': 10,
            'max_clusters': 20,
            'source_type': 'csv'
        },
        'laion_unmemorized': {
            'path': os.path.join(base_data_path, 'nemo-prompts', 'unmemorized_laion_prompts.csv'), 
            'max_prompts_per_cluster': 10,
            'max_clusters': 20,
            'source_type': 'csv'
        },
        'webvid10m': {
            'path': os.path.join(base_data_path, 'webvid10m-clusters', 'clusters-filtered.json'),
            'max_prompts_per_cluster': 10,
            'max_clusters': 10
        }
    }

# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test Cap3D dataset
    try:
        dataset = Cap3DDataset(
            clusters_json_path='./data/objaverse-dupes/aggregated_clusters.json',
            captions_csv_path='./data/objaverse-dupes/Cap3D_automated_Objaverse_full.csv',
            concept_key='teddy_bear',  # Test with specific concept
            max_prompts_per_cluster=3,
            max_clusters=2
        )
        print(f"Cap3D dataset size: {len(dataset)}")
        if len(dataset) > 0:
            print(f"First Cap3D item: {dataset[0]}")
        
    except Exception as e:
        print(f"Cap3D test failed: {e}")
    
    # Test LaionMemorizedDataset with CSV
    try:
        dataset = LaionMemorizedDataset('./data/nemo-prompts/memorized_laion_prompts.csv', max_prompts_per_cluster=5, source_type='csv')
        print(f"Memorized dataset size: {len(dataset)}")
        if len(dataset) > 0:
            print(f"First memorized item: {dataset[0]}")
        
        # Test LaionUnmemorizedDataset with CSV
        unmemorized_dataset = LaionUnmemorizedDataset('./data/nemo-prompts/unmemorized_laion_prompts.csv', max_prompts_per_cluster=5, source_type='csv')
        print(f"Unmemorized dataset size: {len(unmemorized_dataset)}")
        if len(unmemorized_dataset) > 0:
            print(f"First unmemorized item: {unmemorized_dataset[0]}")
        
    except Exception as e:
        print(f"CSV test failed: {e}")
    
    # Test Cap3D dataset
    try:
        cap3d_dataset = Cap3DDataset(
            clusters_json_path='./data/objaverse-dupes/aggregated_clusters.json',
            captions_csv_path='./data/objaverse-dupes/Cap3D_automated_Objaverse_full.csv',
            concept_key='teddy_bear',  # Test with specific concept
            max_prompts_per_cluster=3,
            max_clusters=2
        )
        print(f"Cap3D dataset size: {len(cap3d_dataset)}")
        if len(cap3d_dataset) > 0:
            print(f"First Cap3D item: {cap3d_dataset[0]}")
    except Exception as e:
        print(f"Cap3D test failed: {e}")
    
    # Test dataset manager with Cap3D
    config = {
        'cap3d': {
            'clusters_json_path': './data/objaverse-dupes/aggregated_clusters.json',
            'captions_csv_path': './data/objaverse-dupes/Cap3D_automated_Objaverse_full.csv',
            'concept_key': 'teddy_bear',
            'max_prompts_per_cluster': 2,
            'max_clusters': 2
        },
        'laion_memorized': {
            'path': './data/nemo-prompts/memorized_laion_prompts.csv',
            'max_prompts_per_cluster': 2,
            'max_clusters': 2,
            'source_type': 'csv'
        }
    }
    
    manager = DatasetManager(config)
    datasets = manager.load_datasets()
    stats = manager.get_dataset_statistics(datasets)
    
    print("\nDataset statistics:")
    for name, stat in stats.items():
        print(f"{name}: {stat}")
        
    print("\nSample prompts from each dataset:")
    for name, dataset in datasets.items():
        if len(dataset) > 0:
            print(f"{name}: '{dataset[0]['prompt'][:100]}...'")