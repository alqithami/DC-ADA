"""
Multi-Robot Environments for DC-Ada Experiments

Pure NumPy implementation - no PyBullet dependency required.
Works on any platform (Mac, Linux, Windows).
"""

from .warehouse_env import WarehouseEnv, SearchRescueEnv, CollaborativeMappingEnv
from .base_env import BaseMultiRobotEnv, Robot

__all__ = [
    'WarehouseEnv',
    'SearchRescueEnv', 
    'CollaborativeMappingEnv',
    'BaseMultiRobotEnv',
    'Robot'
]

ENV_MAP = {
    "warehouse": WarehouseEnv,
    "search_rescue": SearchRescueEnv,
    "mapping": CollaborativeMappingEnv,
}

def make_env(env_name: str = None, name: str = None, **kwargs):
    """Factory function to create environments by name."""
    if env_name is None:
        env_name = name
    if env_name not in ENV_MAP:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(ENV_MAP.keys())}")
    return ENV_MAP[env_name](**kwargs)
