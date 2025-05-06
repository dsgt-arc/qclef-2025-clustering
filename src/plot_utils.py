import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

def load_colormap(colormap_name, colormaps_dir=None):
    """
    Load a colormap by name, either from matplotlib or from a custom file.
    
    Args:
        colormap_name: Name of the colormap
        colormaps_dir: Directory containing custom colormaps
        
    Returns:
        Matplotlib colormap
    """
    if colormaps_dir and os.path.exists(os.path.join(colormaps_dir, f"{colormap_name}.npy")):
        cmap_data = np.load(os.path.join(colormaps_dir, f"{colormap_name}.npy"))
        return LinearSegmentedColormap.from_list(colormap_name, cmap_data)
    else:
        return plt.get_cmap(colormap_name)

def plot_embeddings(embeddings, labels=None, medoids=None, refined_medoids=None, 
                   title="UMAP Embedding", save_path=None, cmap='Spectral'):
    """
    Plot embeddings in 2D with optional cluster labels and medoids.
    
    Args:
        embeddings: 2D array of shape (n_samples, 2)
        labels: Optional cluster labels
        medoids: Optional array of medoid coordinates
        refined_medoids: Optional array of refined medoid coordinates
        title: Plot title
        save_path: Path to save the plot
        cmap: Colormap (as string or colormap object)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    ax.set_axisbelow(True)
        
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    
    if labels is not None:
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        scatter = ax.scatter(
            embeddings[:, 0], 
            embeddings[:, 1], 
            c=labels, 
            cmap=cmap, 
            s=20, 
            alpha=0.8,
            edgecolors='k',
            linewidth=0.3
        )
        
    else:
        ax.scatter(
            embeddings[:, 0], 
            embeddings[:, 1], 
            s=20, 
            alpha=0.8,
            edgecolors='k',
            linewidth=0.3
        )
    
    if medoids is not None:
        ax.scatter(
            medoids[:, 0], medoids[:, 1],
            c='black', marker='X', s=100,  # Made slightly smaller (was 150)
            edgecolors="white", linewidth=1.0,  # Made thinner (was 1.5)
            label="Classical Medoids"
        )
    
    if refined_medoids is not None:
        ax.scatter(
            refined_medoids[:, 0], refined_medoids[:, 1],
            c='white', marker='X', s=100,  # Made slightly smaller (was 150)
            edgecolors="#FF00FF",  # Fuchsia/magenta color
            linewidth=1.0,  # Made thinner (was 1.5)
            label="Quantum-Refined Medoids"
        )
    
    ax.set_title(title, fontsize=14, pad=15)
    
    ax.tick_params(axis='both', which='both', length=0, labelsize=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    ax.grid(True, linestyle="--", alpha=0.5, color="#b3c6ff")
    
    # Comment out the legend
    # if medoids is not None or refined_medoids is not None:
    #     ax.legend(loc="upper right", fontsize=10, frameon=True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    
    plt.close()

def plot_cluster_spectrum(
    reduced_embeddings,
    membership_probs,
    medoids=None,
    refined_medoids=None,
    title="Cluster Spectrum",
    save_path=None,
    show=False,
    cmap='Spectral'
):
    """
    Plot embeddings with colors blended across clusters based on membership probabilities.
    
    Args:
        reduced_embeddings: 2D array of shape (n_samples, 2)
        membership_probs: Membership probabilities matrix (n_samples x n_clusters)
        medoids: Optional array of classical medoid coordinates
        refined_medoids: Optional array of quantum-refined medoid coordinates
        title: Plot title
        save_path: Path to save the plot
        show: Whether to display the plot
        cmap: Colormap to use for cluster colors (default: 'Spectral')
        
    Returns:
        The matplotlib figure object
    """
    n_docs, n_clusters = membership_probs.shape
    
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    
    colors = cmap(np.linspace(0, 1, n_clusters))
    
    doc_colors = np.zeros((n_docs, 4))  # RGBA
    for cluster_idx in range(n_clusters):
        doc_colors[:, 0] += membership_probs[:, cluster_idx] * colors[cluster_idx, 0]  # R
        doc_colors[:, 1] += membership_probs[:, cluster_idx] * colors[cluster_idx, 1]  # G
        doc_colors[:, 2] += membership_probs[:, cluster_idx] * colors[cluster_idx, 2]  # B
    
    doc_colors[:, 3] = 0.8
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    ax.set_axisbelow(True)
    
    scatter = ax.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=doc_colors,
        s=20,
        edgecolors="k",
        linewidth=0.3
    )
    
    if medoids is not None:
        ax.scatter(
            medoids[:, 0], medoids[:, 1],
            c='black', marker='X', s=100,  # Made slightly smaller (was 150)
            edgecolors="white", linewidth=1.0,  # Made thinner (was 1.5)
            label="Classical Medoids"
        )

    if refined_medoids is not None:
        ax.scatter(
            refined_medoids[:, 0], refined_medoids[:, 1],
            c='white', marker='X', s=100,  # Made slightly smaller (was 150)
            edgecolors="#FF00FF",  # Fuchsia/magenta color
            linewidth=1.0,  # Made thinner (was 1.5)
            label="Quantum-Refined Medoids"
        )

    ax.set_title(title, fontsize=14, pad=15)
    
    ax.tick_params(axis='both', which='both', length=0, labelsize=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    ax.grid(True, linestyle="--", alpha=0.5, color="#b3c6ff")
    
    # Comment out the legend
    # if medoids is not None or refined_medoids is not None:
    #     ax.legend(loc="upper right", fontsize=10, frameon=True)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved cluster spectrum plot to {save_path}")

    if show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close()

    return fig