# config/base_config.py

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Any, Dict


@dataclass
class BaseConfig:
    """
    Base configuration with common parameters across training,
    inference, and deployment.
    """
    # Randomization & reproducibility
    seed: int = 42

    # Runtime
    device: str = "cpu"  # e.g. "cpu", "cuda:0"
    num_workers: int = 4
    batch_size: int = 8

    # Logging / output
    log_level: str = "INFO"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"

    # Data paths
    data_root: str = "./data"
    map_cache_dir: str = "./maps"

    # Misc
    debug: bool = False
    config_name: str = field(default_factory=lambda: os.path.basename(__file__))

    @classmethod
    def load_from_yaml(cls, path: str) -> "BaseConfig":
        """
        Load config values from a YAML file, overriding defaults.
        Unknown keys are ignored.
        """
        with open(path, "r") as f:
            cfg_dict = yaml.safe_load(f) or {}

        # Filter only known fields
        valid_keys = set(f.name for f in cls.__dataclass_fields__.values())
        filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        return cls(**filtered)

    def save_to_yaml(self, path: str) -> None:
        """
        Dump the current config to a YAML file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a plain dict for serialization or introspection.
        """
        return asdict(self)
