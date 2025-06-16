import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
from collections import Counter

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

def plot_merged_clusters(embeddings, initial_labels, final_labels, 
                   correspondence, save_path=None, cmap='Spectral',
                   cluster_colors=None, title="Merged Clusters Visualization"):
    """
    Create a visualization highlighting one final cluster and its merged component clusters.
    The visualization greys out irrelevant clusters and highlights the component clusters
    with distinct colors.
    
    Args:
        embeddings: 2D array of shape (n_samples, 2) - UMAP reduced embeddings
        initial_labels: Initial cluster labels before quantum refinement
        final_labels: Final cluster labels after quantum refinement
        correspondence: Dictionary mapping final cluster IDs to most corresponding initial cluster IDs
        save_path: Path to save the plot
        cmap: Colormap to use for cluster colors
        cluster_colors: Optional dictionary mapping original cluster IDs to colors
        title: Plot title
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.cm as cm
    from collections import Counter
    
    # Find a good example of a merged cluster
    # This finds a final cluster that contains points from multiple initial clusters
    merged_clusters = []
    
    # For each final cluster, count how many initial clusters contributed to it
    for final_cluster in np.unique(final_labels):
        # Get all points in this final cluster
        mask = final_labels == final_cluster
        # Count the initial cluster assignments for these points
        initial_counts = Counter(initial_labels[mask])
        # If this final cluster contains points from multiple initial clusters
        if len(initial_counts) > 1:
            # Store the final cluster ID, the number of initial clusters, and the size
            merged_clusters.append((final_cluster, len(initial_counts), sum(mask)))
    
    # Sort by number of contributing clusters (descending) and then by size (descending)
    merged_clusters.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    if not merged_clusters:
        print("No merged clusters found!")
        return
    
    # Select the best example (first in sorted list)
    target_cluster = merged_clusters[0][0]
    
    print(f"Selected final cluster {target_cluster} for visualization, "
          f"which contains points from {merged_clusters[0][1]} initial clusters "
          f"with {merged_clusters[0][2]} total points")
    
    # Find which initial clusters contributed to this final cluster
    mask = final_labels == target_cluster
    contributing_initial_clusters = np.unique(initial_labels[mask])
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Configure axes
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='both', length=0, labelsize=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, linestyle="--", alpha=0.5, color="#b3c6ff")
    
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    
    # First plot all points in grey (background)
    ax.scatter(
        embeddings[:, 0], 
        embeddings[:, 1], 
        c='#888888',  # Darker grey
        s=15, 
        alpha=0.35,
        edgecolors='none'
    )
    
    # Generate colors for contributing initial clusters
    if cluster_colors is not None:
        colors = [cluster_colors.get(cluster, cmap(i / max(1, len(contributing_initial_clusters)))) 
                 for i, cluster in enumerate(contributing_initial_clusters)]
    else:
        colors = cmap(np.linspace(0, 1, len(contributing_initial_clusters)))
    
    # Plot each contributing initial cluster with its own color
    for i, initial_cluster in enumerate(contributing_initial_clusters):
        # Points that were in this initial cluster and ended up in the target final cluster
        mask = (initial_labels == initial_cluster) & (final_labels == target_cluster)
        
        if np.sum(mask) > 0:
            ax.scatter(
                embeddings[mask, 0], 
                embeddings[mask, 1], 
                c=[colors[i]], 
                s=30, 
                alpha=0.8,
                edgecolors='k',
                linewidth=0.3
            )
    
    # Add a title
    ax.set_title(f"{title}\nFinal Cluster {target_cluster}", fontsize=14, pad=15)
    
    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved merged clusters plot to {save_path}")
    
    plt.close()

def plot_top_merged_clusters(embeddings, initial_labels, final_labels, 
                      correspondence, save_dir=None, cmap='Spectral',
                      highlight_color='#25A085', title_prefix="Merged Clusters Visualization",
                      num_clusters=10):
    """
    Create visualizations for the top merged clusters, using a highlight color.
    
    Args:
        embeddings: 2D array of shape (n_samples, 2) - UMAP reduced embeddings
        initial_labels: Initial cluster labels before quantum refinement
        final_labels: Final cluster labels after quantum refinement
        correspondence: Dictionary mapping final cluster IDs to most corresponding initial cluster IDs
        save_dir: Directory to save the plots
        cmap: Colormap to use for cluster colors (not used for points but kept for compatibility)
        highlight_color: Color to use for highlighting the merged cluster points
        title_prefix: Prefix for the plot titles
        num_clusters: Number of top merged clusters to visualize
    """
    
    # Find merged clusters
    merged_clusters = []
    
    # For each final cluster, count how many initial clusters contributed to it
    for final_cluster in np.unique(final_labels):
        # Get all points in this final cluster
        mask = final_labels == final_cluster
        # Count the initial cluster assignments for these points
        initial_counts = Counter(initial_labels[mask])
        # If this final cluster contains points from multiple initial clusters
        if len(initial_counts) > 1:
            # Store the final cluster ID, the number of initial clusters, and the size
            merged_clusters.append((final_cluster, len(initial_counts), sum(mask)))
    
    # Sort by number of contributing clusters (descending) and then by size (descending)
    merged_clusters.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    if not merged_clusters:
        print("No merged clusters found!")
        return
    
    # Limit to the top N clusters
    top_clusters = merged_clusters[:min(num_clusters, len(merged_clusters))]
    print(f"Visualizing top {len(top_clusters)} merged clusters out of {len(merged_clusters)} found")
    
    # Create a visualization for each top cluster
    for idx, (target_cluster, num_contributing, size) in enumerate(top_clusters):
        print(f"Cluster {idx+1}/{len(top_clusters)}: Final cluster {target_cluster} contains points from "
              f"{num_contributing} initial clusters with {size} total points")
        
        # Find which initial clusters contributed to this final cluster
        mask = final_labels == target_cluster
        contributing_initial_clusters = np.unique(initial_labels[mask])
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Configure axes
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='both', length=0, labelsize=0)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(True, linestyle="--", alpha=0.5, color="#b3c6ff")
        
        # First plot all points in grey (background)
        ax.scatter(
            embeddings[:, 0], 
            embeddings[:, 1], 
            c='#888888',  # Darker grey
            s=15, 
            alpha=0.35,
            edgecolors='none'
        )
        
        # Plot the merged cluster points in the highlight color
        mask = final_labels == target_cluster
        ax.scatter(
            embeddings[mask, 0], 
            embeddings[mask, 1], 
            c=highlight_color,  # Use the teal highlight color
            s=55, 
            alpha=0.75,
            edgecolors='k',
            linewidth=0.6
        )
        
        # Add a title
        ax.set_title(f"{title_prefix} #{idx+1}\nFinal Cluster {target_cluster} (merged from {num_contributing} clusters)", 
                     fontsize=14, pad=15)
        
        # Save the plot
        if save_dir:
            save_path = os.path.join(save_dir, f"merged_cluster_{idx+1}_of_{len(top_clusters)}_{target_cluster}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved merged cluster plot to {save_path}")
        
        plt.close()

def generate_cluster_colors(n_clusters, cmap='Spectral'):
    """
    Generate a fixed set of colors for clusters.
    
    Args:
        n_clusters: Number of clusters
        cmap: Colormap to use
        
    Returns:
        Array of RGBA colors for each cluster
    """
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    
    return cmap(np.linspace(0, 1, n_clusters))

def track_cluster_correspondence(initial_labels, final_labels):
    """
    Track how clusters change between initial and final clustering.
    
    Args:
        initial_labels: Cluster labels from initial clustering
        final_labels: Cluster labels after refinement
        
    Returns:
        Dictionary mapping final cluster IDs to most corresponding initial cluster IDs
    """
    correspondence = {}
    
    overlap_counts = defaultdict(lambda: defaultdict(int))
    
    for i, (init_label, final_label) in enumerate(zip(initial_labels, final_labels)):
        overlap_counts[final_label][init_label] += 1
    
    for final_cluster in np.unique(final_labels):
        if len(overlap_counts[final_cluster]) > 0:
            init_cluster = max(overlap_counts[final_cluster].items(), 
                              key=lambda x: x[1])[0]
            correspondence[final_cluster] = init_cluster
        else:
            correspondence[final_cluster] = final_cluster
    
    return correspondence

def plot_embeddings(embeddings, labels=None, medoids=None, refined_medoids=None, 
                   title="UMAP Embedding", save_path=None, cmap='Spectral',
                   cluster_colors=None, color_correspondence=None):
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
        cluster_colors: Optional dictionary mapping cluster IDs to colors
        color_correspondence: Optional dictionary mapping refined cluster IDs to original IDs
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
        
        if cluster_colors is not None and color_correspondence is not None:
            point_colors = np.zeros((len(labels), 4))
            for i, label in enumerate(labels):
                orig_label = color_correspondence.get(label, label)
                if orig_label in cluster_colors:
                    point_colors[i] = cluster_colors[orig_label]
                else:
                    normalized_label = np.where(unique_labels == label)[0][0] / max(1, n_clusters - 1)
                    point_colors[i] = cmap(normalized_label)
            
            scatter = ax.scatter(
                embeddings[:, 0], 
                embeddings[:, 1], 
                c=point_colors, 
                s=20, 
                alpha=0.8,
                edgecolors='k',
                linewidth=0.3
            )
        else:
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
            c='black', marker='X', s=100,
            edgecolors="white", linewidth=1.0,
            label="Classical Medoids"
        )
    
    if refined_medoids is not None:
        ax.scatter(
            refined_medoids[:, 0], refined_medoids[:, 1],
            c='white', marker='X', s=100,
            edgecolors="#FF00FF",
            linewidth=1.0,
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
    cmap='Spectral',
    cluster_colors=None,
    color_correspondence=None
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
        cluster_colors: Optional dictionary mapping original cluster IDs to colors
        color_correspondence: Optional dictionary mapping refined cluster IDs to original IDs
        
    Returns:
        The matplotlib figure object
    """
    n_docs, n_clusters = membership_probs.shape
    
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    
    if cluster_colors is not None and color_correspondence is not None:
        colors = np.zeros((n_clusters, 4))
        for cluster_idx in range(n_clusters):
            orig_cluster = color_correspondence.get(cluster_idx, cluster_idx)
            if orig_cluster in cluster_colors:
                colors[cluster_idx] = cluster_colors[orig_cluster]
            else:
                colors[cluster_idx] = cmap(cluster_idx / max(1, n_clusters - 1))
    else:
        colors = cmap(np.linspace(0, 1, n_clusters))
    
    doc_colors = np.zeros((n_docs, 4))
    for cluster_idx in range(n_clusters):
        doc_colors[:, 0] += membership_probs[:, cluster_idx] * colors[cluster_idx, 0]
        doc_colors[:, 1] += membership_probs[:, cluster_idx] * colors[cluster_idx, 1]
        doc_colors[:, 2] += membership_probs[:, cluster_idx] * colors[cluster_idx, 2]
    
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
            c='black', marker='X', s=100,
            edgecolors="white", linewidth=1.0,
            label="Classical Medoids"
        )

    if refined_medoids is not None:
        ax.scatter(
            refined_medoids[:, 0], refined_medoids[:, 1],
            c='white', marker='X', s=100,
            edgecolors="#FF00FF",
            linewidth=1.0,
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