# src/core/base_module.py

import abc
from typing import Any
import torch.nn as nn


class BaseModule(nn.Module, abc.ABC):
    """
    Abstract base class for all model components.
    Subclasses must implement build() and forward().
    Provides common config handling and weight I/O stubs.
    """

    def __init__(self, config: Any):
        """
        Args:
            config: A dataclass or dict containing module-specific settings.
        """
        super().__init__()
        self.config = config
        self.build()

    @abc.abstractmethod
    def build(self) -> None:
        """
        Construct all sub-modules (e.g., layers, embeddings).
        Called automatically at __init__ time.
        """
        ...

    @abc.abstractmethod
    def forward(self, *inputs: Any) -> Any:
        """
        Define the forward pass.
        Args:
            *inputs: One or more inputs (tensors or tuples of tensors).
        Returns:
            Output tensor(s).
        """
        ...

    def load_weights(self, path: str) -> None:
        """
        Load module weights from a file.
        Args:
            path: Path to the checkpoint (state_dict) file.
        """
        state = nn.utils.loading.load_state_dict(self, path)
        self.load_state_dict(state)

    def save_weights(self, path: str) -> None:
        """
        Save module weights to a file.
        Args:
            path: Path where to write the checkpoint (state_dict).
        """
        torch.save(self.state_dict(), path)
