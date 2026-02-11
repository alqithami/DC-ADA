"""
Seeding utility for reproducibility.
"""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """Set seeds for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_global_seed(seed: int):
    """Alias for set_seed() for compatibility."""
    return set_seed(seed)
