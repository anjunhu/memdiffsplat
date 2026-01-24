# DiffSplat/memorization/metrics/base.py 

from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseMetric(ABC):
    """Base class for all memorization metrics."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the metric."""
        pass
    
    @property
    @abstractmethod
    def metric_type(self) -> str:
        """Returns 'per_seed' or 'per_prompt_across_seeds'."""
        pass
    
    @property
    def requires_intermediates(self) -> bool:
        """Whether this metric needs the intermediates dict from pipeline."""
        return False
    
    @property 
    def requires_model(self) -> bool:
        """Whether this metric needs direct access to the model."""
        return False
        
    @property
    def requires_contexts(self) -> bool:
        """Whether this metric needs conditioning/unconditioning contexts."""
        return False
        
    @property
    def requires_attention_maps(self) -> bool:
        """Whether this metric needs attention maps from controller."""
        return False
    
    @abstractmethod
    def measure(self, **kwargs) -> Dict[str, Any]:
        """Computes the metric. Subclasses override this method."""
        pass