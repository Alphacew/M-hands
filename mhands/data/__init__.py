"""
Data management, collection, and synthetic generation modules for M-Hands.
"""

from mhands.data.synthetic_generator import SyntheticHandGenerator
from mhands.data.dataset import GestureDataset, load_benchmark_splits
from mhands.data.collector import GestureDataCollector

__all__ = [
    "SyntheticHandGenerator",
    "GestureDataset",
    "load_benchmark_splits",
    "GestureDataCollector",
]
