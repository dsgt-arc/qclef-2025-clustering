import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

def load_scm_colormap(txt_path, name="custom_cmap"):
    """
    Load a colormap from a text file.
    
    Args:
        txt_path: Path to the .txt file containing RGB values
        name: Name to give to the colormap
        
    Returns:
        A matplotlib colormap object
    """
    data = np.loadtxt(txt_path)
    return LinearSegmentedColormap.from_list(name, data)

def load_colormap(colormap_name, colormaps_dir):
    """
    Load a colormap by name, either from file or built-in matplotlib.
    
    Args:
        colormap_name: Name of the colormap to use
        colormaps_dir: Directory containing custom colormap .txt files
        
    Returns:
        A matplotlib colormap object
    """
    # First check if it's a built-in matplotlib colormap
    if colormap_name in plt.colormaps():
        return plt.get_cmap(colormap_name)
    
    # If not, try to load it from file
    try:
        cmap_path = os.path.join(colormaps_dir, f"{colormap_name}.txt")
        if os.path.exists(cmap_path):
            return load_scm_colormap(cmap_path, name=colormap_name)
        else:
            print(f"Warning: Colormap file {cmap_path} not found. Using default 'viridis' colormap.")
            return plt.get_cmap('viridis')
    except Exception as e:
        print(f"Error loading colormap {colormap_name}: {e}")
        print("Using default 'viridis' colormap instead.")
        return plt.get_cmap('viridis')

def plot_embeddings(
    reduced_embeddings,
    labels=None,
    medoids=None,
    refined_medoids=None,
    title="UMAP Plot",
    save_path=None,
    cmap=None,
    show=True
):
    """
    Plot UMAP-reduced embeddings with optional labels and medoids.
    
    Args:
        reduced_embeddings: 2D array of shape (n_samples, 2)
        labels: Optional cluster labels for each point
        medoids: Optional array of classical medoid coordinates
        refined_medoids: Optional array of quantum-refined medoid coordinates
        title: Plot title
        save_path: Path to save the plot
        cmap: Colormap to use (matplotlib colormap object)
        show: Whether to display the plot
        
    Returns:
        The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Use the provided colormap or fall back to a default
    if cmap is None:
        cmap = plt.get_cmap('viridis')
    
    default_color = cmap(0.9) if hasattr(cmap, '__call__') else "grey"

    scatter = ax.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=labels if labels is not None else [default_color] * len(reduced_embeddings),
        cmap=cmap if labels is not None else None,
        s=20,
        alpha=0.6,
        edgecolors="k",
        linewidth=0.3
    )

    if medoids is not None:
        ax.scatter(
            medoids[:, 0], medoids[:, 1],
            c='black', marker='X', s=150,
            edgecolors="white", linewidth=1.5,
            label="Classical Medoids"
        )

    if refined_medoids is not None:
        ax.scatter(
            refined_medoids[:, 0], refined_medoids[:, 1],
            c='purple', marker='X', s=150,
            edgecolors="white", linewidth=1.5,
            label="Quantum-Refined Medoids"
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)

    if medoids is not None or refined_medoids is not None:
        ax.legend(loc="upper right", fontsize=10, frameon=True)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig