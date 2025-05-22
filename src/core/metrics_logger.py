# src/core/metrics_logger.py

import os
import csv
from typing import Any, Dict, List, Optional


class MetricsLogger:
    """
    Logger for collecting and saving metrics throughout training
    or inference runs. Accumulates metrics in memory and can export
    to CSV.
    """

    def __init__(self, output_dir: str = "./metrics", filename: str = "metrics.csv"):
        """
        Args:
            output_dir: Directory where metrics CSV will be saved.
            filename: Name of the CSV file.
        """
        self.output_dir = output_dir
        self.filename = filename
        self._rows: List[Dict[str, Any]] = []
        self._fieldnames: Optional[List[str]] = None

        os.makedirs(self.output_dir, exist_ok=True)

    def log(self, step: int, phase: str, metrics: Dict[str, Any], **kwargs: Any) -> None:
        """
        Log a set of metrics for a given step.

        Args:
            step: Training/inference step or epoch number.
            phase: Identifier for phase (e.g., "train", "val", "test", "inference").
            metrics: Dict mapping metric names to values (scalars).
            **kwargs: Additional context (e.g., learning rate, timestamp).
        """
        row = {"step": step, "phase": phase, **metrics, **kwargs}

        # Initialize fieldnames on first log
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        else:
            # Extend fieldnames if new keys appear
            for key in row.keys():
                if key not in self._fieldnames:
                    self._fieldnames.append(key)

        self._rows.append(row)

    def save_csv(self, path: Optional[str] = None) -> None:
        """
        Write all logged metrics to a CSV file.

        Args:
            path: Optional override for output path. If None, uses configured output_dir/filename.
        """
        if self._fieldnames is None or not self._rows:
            raise RuntimeError("No metrics to save.")

        output_path = path or os.path.join(self.output_dir, self.filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
            for row in self._rows:
                # Fill missing fields with None
                full_row = {k: row.get(k, None) for k in self._fieldnames}
                writer.writerow(full_row)

    def reset(self) -> None:
        """
        Clear all logged metrics.
        """
        self._rows = []
        self._fieldnames = None

    def get_latest(self) -> Dict[str, Any]:
        """
        Return the most recently logged row.
        """
        if not self._rows:
            raise RuntimeError("No metrics logged yet.")
        return self._rows[-1]

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Return all logged metrics as a list of dicts.
        """
        return list(self._rows)
