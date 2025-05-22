# src/utils/visualization.py

import os
from typing import List, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt


def show_image(
    img: np.ndarray,
    title: Optional[str] = None,
    figsize: Optional[tuple] = (6, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Display a single image (H×W×C float32 array in [0,1]).

    Args:
        img: Image array.
        title: Optional title string.
        figsize: Figure size.
        save_path: If provided, save the figure to this path.
    """
    plt.figure(figsize=figsize)
    plt.imshow(np.clip(img, 0, 1))
    plt.axis('off')
    if title:
        plt.title(title)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def show_batch(
    imgs: List[np.ndarray],
    ncols: int = 4,
    titles: Optional[List[str]] = None,
    figsize: Optional[tuple] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Display a batch of images in a grid.

    Args:
        imgs: List of image arrays.
        ncols: Number of columns.
        titles: Optional list of titles per image.
        figsize: Figure size.
        save_path: If provided, save the grid image.
    """
    n = len(imgs)
    nrows = (n + ncols - 1) // ncols
    plt.figure(figsize=figsize)
    for idx, img in enumerate(imgs):
        ax = plt.subplot(nrows, ncols, idx + 1)
        ax.imshow(np.clip(img, 0, 1))
        ax.axis('off')
        if titles and idx < len(titles):
            ax.set_title(titles[idx])
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_trajectory(
    trajectory: np.ndarray,
    current_pos: Optional[np.ndarray] = None,
    title: Optional[str] = "Planned Trajectory",
    figsize: Optional[tuple] = (6, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot a 2D trajectory with optional current position.

    Args:
        trajectory: Array of shape (T, 2) for x, y offsets.
        current_pos: Optional (x, y) start point. Defaults to origin.
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save the figure.
    """
    traj = np.asarray(trajectory)
    xs, ys = traj[:, 0], traj[:, 1]
    plt.figure(figsize=figsize)
    if current_pos is not None:
        plt.scatter(current_pos[0], current_pos[1], c='r', label='Current')
    plt.plot(xs, ys, marker='o', label='Future Path')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(title)
    plt.axis('equal')
    plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def visualize_bev_heatmap(
    bev_feats: np.ndarray,
    height: int,
    width: int,
    channel: int = 0,
    title: Optional[str] = "BEV Feature Map",
    figsize: Optional[tuple] = (6, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Visualize a single-channel heatmap from flatten BEV features.

    Args:
        bev_feats: Array of shape (H*W, C) or (H, W, C).
        height: BEV grid height.
        width: BEV grid width.
        channel: Channel index to visualize.
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save the heatmap.
    """
    arr = np.asarray(bev_feats)
    if arr.ndim == 2:
        # (H*W, C) -> (H, W)
        heat = arr[:, channel].reshape(height, width)
    elif arr.ndim == 3:
        heat = arr[:, :, channel]
    else:
        raise ValueError("bev_feats must be 2D or 3D array")
    plt.figure(figsize=figsize)
    plt.imshow(heat, origin='lower', aspect='equal')
    plt.colorbar(label='Activation')
    plt.title(title)
    plt.axis('off')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
