import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import yaml

from sklearn.metrics import pairwise_distances
from src.models.UMAPReducer import UMAPReducer
from src.models.ClassicalClustering import ClassicalClustering
from src.models.QuantumClustering import QuantumClustering, compute_clusters
from box import ConfigBox
from src.plot_utils import plot_embeddings, load_colormap

warnings.filterwarnings("ignore")

def find_best_k_with_qubo(quantum_clustering, medoid_embeddings):
    """Iterate over k_range and find the best k using QUBO clustering."""
    best_k = None
    best_dbi = float("inf")
    best_indices = None

    for k in quantum_clustering.k_range:
        refined_medoid_indices, dbi, _ = quantum_clustering.solve_qubo(medoid_embeddings, k)

        if refined_medoid_indices is not None and dbi < best_dbi:
            best_dbi = dbi
            best_k = k
            best_indices = refined_medoid_indices

    print(f"Final chosen k after QUBO clustering: {best_k}")

    return best_k, best_indices


def run_pipeline(config, colormap_name=None):
    """
    Run the clustering pipeline with a specified colormap.
    
    Args:
        config: Configuration object with clustering parameters
        colormap_name: Name of the colormap to use. Can be:
            - Name of a file in the colormaps directory (without .txt extension)
            - Name of a built-in matplotlib colormap (e.g. "Spectral")
            - None (will use the default "batlow" colormap)
    """
    np.random.seed(config.classical_clustering.random_state)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))
    colormaps_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "colormaps"))
    output_csv = os.path.join(data_dir, "antique_train_with_embeddings.csv")

    umap_plot_path = os.path.join(data_dir, "umap_plot.png")
    kmedoids_plot_path = os.path.join(data_dir, "kmedoids_clusters.png")
    final_clusters_plot_path = os.path.join(data_dir, "final_clusters.png")

    if colormap_name is None:
        colormap_name = "Spectral"
    
    cmap = load_colormap(colormap_name, colormaps_dir)
    
    def parse_embedding(text):
        return np.array(eval(text), dtype=np.float64)

    train_df = pd.read_csv(output_csv, converters={"doc_embedding": parse_embedding})
    doc_embeddings = np.stack(train_df["doc_embedding"].values)

    umap_reducer = UMAPReducer(random_state=config.classical_clustering.random_state)
    doc_embeddings_reduced = umap_reducer.fit_transform(doc_embeddings)
    np.save(os.path.join(data_dir, "doc_embeddings_reduced.npy"), doc_embeddings_reduced)
    plot_embeddings(doc_embeddings_reduced, title="UMAP Reduction", save_path=umap_plot_path, cmap=cmap)

    clustering = ClassicalClustering(**config.classical_clustering)
    kmedoid_labels, medoid_indices = clustering.find_optimal_k(doc_embeddings_reduced)

    print(f"Classical K-Medoids Labels: {kmedoid_labels}")
    print(f"Classical Medoid Indices: {medoid_indices}")

    medoid_embeddings = clustering.extract_medoids(doc_embeddings_reduced, medoid_indices)
    np.save(os.path.join(data_dir, "medoid_embeddings.npy"), medoid_embeddings)
    np.save(os.path.join(data_dir, "medoid_indices.npy"), medoid_indices)
    plot_embeddings(doc_embeddings_reduced, labels=kmedoid_labels, medoids=medoid_embeddings,
                title=f"K-Medoids Clustering (k={clustering.best_k})", save_path=kmedoids_plot_path, cmap=cmap)

    # Creating the QuantumClustering instance remains the same
    # The internal implementation has changed, but the interface remains consistent
    quantum_clustering = QuantumClustering(config.quantum_clustering.k_range, medoid_embeddings, config)

    best_k, refined_medoid_indices = find_best_k_with_qubo(quantum_clustering, medoid_embeddings)

    if refined_medoid_indices is not None:
        print(f"Quantum-Refined Medoid Indices: {refined_medoid_indices}") 
        refined_medoid_indices_of_embeddings = medoid_indices[refined_medoid_indices]
        refined_medoid_embeddings = doc_embeddings_reduced[refined_medoid_indices_of_embeddings] 
        
        print(f"Before QUBO: Assignments to Classical Medoids: {compute_clusters(doc_embeddings_reduced, medoid_indices)}")
        print(f"Refined Medoid Indices Type: {type(refined_medoid_indices)}, Shape: {refined_medoid_indices.shape}")
        print(f"Refined Medoid Embeddings Shape: {refined_medoid_embeddings.shape}")
        print(f"Before Assigning Clusters, Medoid Indices: {refined_medoid_indices_of_embeddings}")
        final_cluster_labels = compute_clusters(doc_embeddings_reduced, refined_medoid_indices_of_embeddings)
        print(f"After QUBO: Assignments to Quantum Medoids: {final_cluster_labels}")

    else:
        raise ValueError("QUBO Solver failed to find valid medoids.")
    
    print(f"Final chosen k after QUBO clustering: {best_k}")

    np.save(os.path.join(data_dir, "final_quantum_clusters.npy"), final_cluster_labels)
    np.save(os.path.join(data_dir, "refined_medoid_embeddings.npy"), refined_medoid_embeddings)
    np.save(os.path.join(data_dir, "refined_medoid_indices.npy"), refined_medoid_indices)

    print(f"Final Quantum-Refined Medoid Embeddings:\n {refined_medoid_embeddings}")
    print(f"Unique Cluster Assignments: {np.unique(final_cluster_labels)}")

    plot_embeddings(doc_embeddings_reduced,
                labels=final_cluster_labels,
                medoids=medoid_embeddings,
                refined_medoids=refined_medoid_embeddings,
                title="Final Quantum Cluster Assignments",
                save_path=final_clusters_plot_path,
                cmap=cmap)

    print(f"Saved final quantum cluster plot at: {final_clusters_plot_path}")
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run clustering pipeline with custom colormap')
    parser.add_argument('--colormap', type=str, default='Spectral', 
                        help='Colormap to use (file in colormaps dir or matplotlib name)')
    
    args = parser.parse_args()

    with open("config/kmedoids.yml", "r") as file:
        config = ConfigBox(yaml.safe_load(file))

    run_pipeline(config, args.colormap)