"""
Analysis and visualization tools for memorization evaluation results.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from collections import defaultdict

class ResultAnalyzer:
    """Analyzes and visualizes memorization evaluation results"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.logger = logging.getLogger("ResultAnalyzer")
        
        # Load results
        self.results = self._load_all_results()
        self.summary = self._load_summary()
        self.statistics = self._load_statistics()
    
    def _load_all_results(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Load all results from the results directory"""
        results = {}
        
        # Look for dataset subdirectories
        for dataset_dir in self.results_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
                
            dataset_name = dataset_dir.name
            if dataset_name.startswith('.'):
                continue
                
            dataset_results = {}
            
            # Load metric results
            for metric_file in dataset_dir.glob("*_results.json"):
                metric_name = metric_file.stem.replace("_results", "")
                
                try:
                    with open(metric_file, 'r') as f:
                        metric_data = json.load(f)
                    dataset_results[metric_name] = metric_data
                except Exception as e:
                    self.logger.warning(f"Failed to load {metric_file}: {e}")
            
            if dataset_results:
                results[dataset_name] = dataset_results
        
        return results
    
    def _load_summary(self) -> Optional[Dict]:
        """Load evaluation summary if available"""
        summary_file = self.results_dir / "evaluation_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                return json.load(f)
        return None
    
    def _load_statistics(self) -> Optional[Dict]:
        """Load dataset statistics if available"""
        stats_file = self.results_dir / "dataset_statistics.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                return json.load(f)
        return None
    
    def create_dataframe(self) -> pd.DataFrame:
        """Create a pandas DataFrame from all results for analysis"""
        rows = []
        
        for dataset_name, dataset_results in self.results.items():
            # Find the length of results (should be same for all metrics)
            n_samples = len(list(dataset_results.values())[0]) if dataset_results else 0
            
            for i in range(n_samples):
                row = {
                    'dataset': dataset_name,
                    'prompt_idx': i
                }
                
                # Add metric values
                for metric_name, metric_results in dataset_results.items():
                    if i < len(metric_results):
                        result = metric_results[i]
                        row[f'{metric_name}_value'] = result['value']
                        
                        # Add some metadata
                        metadata = result.get('metadata', {})
                        row[f'{metric_name}_cluster_id'] = metadata.get('cluster_id', 'unknown')
                        row[f'{metric_name}_label'] = metadata.get('label', 'unknown')
                        
                        # Add computation time if available
                        if 'computation_time' in metadata:
                            row[f'{metric_name}_time'] = metadata['computation_time']
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def plot_metric_distributions(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Plot distributions of all metrics across datasets"""
        df = self.create_dataframe()
        
        # Get metric names
        metric_names = []
        for col in df.columns:
            if col.endswith('_value'):
                metric_names.append(col.replace('_value', ''))
        
        # Create subplots
        n_metrics = len(metric_names)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, metric_name in enumerate(metric_names):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Plot distribution for each dataset
            for dataset in df['dataset'].unique():
                dataset_data = df[df['dataset'] == dataset]
                values = dataset_data[f'{metric_name}_value'].dropna()
                
                if len(values) > 0:
                    ax.hist(values, alpha=0.6, label=dataset, bins=20)
            
            ax.set_title(f'{metric_name} Distribution')
            ax.set_xlabel('Metric Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_metric_comparison(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Create box plots comparing metrics across datasets"""
        df = self.create_dataframe()
        
        # Get metric names
        metric_names = []
        for col in df.columns:
            if col.endswith('_value'):
                metric_names.append(col.replace('_value', ''))
        
        # Prepare data for plotting
        plot_data = []
        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            for metric_name in metric_names:
                values = dataset_data[f'{metric_name}_value'].dropna()
                for value in values:
                    plot_data.append({
                        'Dataset': dataset,
                        'Metric': metric_name,
                        'Value': value
                    })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create figure with subplots for each metric
        n_metrics = len(metric_names)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, metric_name in enumerate(metric_names):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            metric_data = plot_df[plot_df['Metric'] == metric_name]
            
            if len(metric_data) > 0:
                sns.boxplot(data=metric_data, x='Dataset', y='Value', ax=ax)
                ax.set_title(f'{metric_name}')
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.set_title(f'{metric_name} (No Data)')
        
        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def compute_statistical_tests(self) -> Dict[str, Dict]:
        """Compute statistical tests between memorized and unmemorized datasets"""
        from scipy import stats
        
        df = self.create_dataframe()
        
        # Get metric names
        metric_names = []
        for col in df.columns:
            if col.endswith('_value'):
                metric_names.append(col.replace('_value', ''))
        
        results = {}
        
        # Compare memorized vs unmemorized if both exist
        memorized_data = df[df['dataset'].str.contains('memorized', na=False)]
        unmemorized_data = df[df['dataset'].str.contains('unmemorized', na=False)]
        
        if len(memorized_data) > 0 and len(unmemorized_data) > 0:
            for metric_name in metric_names:
                mem_values = memorized_data[f'{metric_name}_value'].dropna()
                unmem_values = unmemorized_data[f'{metric_name}_value'].dropna()
                
                if len(mem_values) > 0 and len(unmem_values) > 0:
                    # Mann-Whitney U test (non-parametric)
                    statistic, p_value = stats.mannwhitneyu(mem_values, unmem_values, alternative='two-sided')
                    
                    # Cohen's d (effect size)
                    pooled_std = np.sqrt(((len(mem_values) - 1) * mem_values.std()**2 + 
                                        (len(unmem_values) - 1) * unmem_values.std()**2) / 
                                       (len(mem_values) + len(unmem_values) - 2))
                    cohens_d = (mem_values.mean() - unmem_values.mean()) / pooled_std if pooled_std > 0 else 0
                    
                    results[metric_name] = {
                        'mann_whitney_u': {
                            'statistic': statistic,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        },
                        'effect_size': {
                            'cohens_d': cohens_d,
                            'interpretation': self._interpret_cohens_d(abs(cohens_d))
                        },
                        'descriptive': {
                            'memorized_mean': mem_values.mean(),
                            'memorized_std': mem_values.std(),
                            'unmemorized_mean': unmem_values.mean(),
                            'unmemorized_std': unmem_values.std()
                        }
                    }
        
        return results
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate a comprehensive analysis report"""
        if output_file is None:
            output_file = self.results_dir / "analysis_report.txt"
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("LAVIE MEMORIZATION EVALUATION ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Dataset overview
        report_lines.append("DATASET OVERVIEW")
        report_lines.append("-" * 20)
        df = self.create_dataframe()
        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            report_lines.append(f"{dataset}: {len(dataset_data)} prompts")
        report_lines.append("")
        
        # Metric summaries
        report_lines.append("METRIC SUMMARIES")