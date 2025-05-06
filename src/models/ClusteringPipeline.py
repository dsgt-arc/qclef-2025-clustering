import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import yaml

from sklearn.metrics import pairwise_distances
from src.models.UMAPReducer import UMAPReducer
from src.models.ClassicalClustering import ClassicalClustering
from src.models.HDBSCANClustering import HDBSCANClustering
from src.models.GMMClustering import GMMClustering
from src.models.HDBSCANGMMClustering import HDBSCANGMMClustering
from src.models.multi_membership import create_multi_membership_assignments
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

    return best_k, best_indices, best_dbi

def run_pipeline(config, colormap_name=None, clustering_method='classical', multi_membership=False, threshold=0.2):
    """
    Run the clustering pipeline with a specified colormap and clustering method.
    
    Args:
        config: Configuration object with clustering parameters
        colormap_name: Name of the colormap to use
        clustering_method: Which clustering method to use ('classical', 'hdbscan', 'gmm', 'hdbscan-gmm')
        multi_membership: Whether to create multi-membership assignments
        threshold: Probability threshold for multi-membership (default: 0.2)
    """
    np.random.seed(config.classical_clustering.random_state)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))
    colormaps_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "colormaps"))
    output_csv = os.path.join(data_dir, "antique_doc_embeddings.csv")

    umap_plot_path = os.path.join(data_dir, "umap_plot.png")
    initial_clusters_plot_path = os.path.join(data_dir, f"{clustering_method}_clusters.png")
    final_clusters_plot_path = os.path.join(data_dir, "final_clusters.png")

    if colormap_name is None:
        colormap_name = "Spectral"
    
    cmap = load_colormap(colormap_name, colormaps_dir)
    
    def parse_embedding(text):
        return np.fromstring(text[1:-1], dtype=float, sep=',')

    train_df = pd.read_csv(output_csv, converters={"doc_embeddings": parse_embedding})
    doc_embeddings = np.stack(train_df["doc_embeddings"].values)
    doc_ids = train_df['doc_id'].tolist()

    umap_reducer = UMAPReducer(random_state=config.classical_clustering.random_state)
    doc_embeddings_reduced = umap_reducer.fit_transform(doc_embeddings)
    np.save(os.path.join(data_dir, "doc_embeddings_reduced.npy"), doc_embeddings_reduced)
    plot_embeddings(doc_embeddings_reduced, title="UMAP Reduction", save_path=umap_plot_path, cmap=cmap)

    has_probabilities = clustering_method in ['gmm', 'hdbscan-gmm']
    if multi_membership and not has_probabilities:
        print(f"Warning: Multi-membership requires 'gmm' or 'hdbscan-gmm' method. Requested method '{clustering_method}' doesn't provide probabilities.")
        multi_membership = False
    
    if clustering_method == 'hdbscan':
        print("Using HDBSCAN clustering...")
        clustering = HDBSCANClustering(**config.hdbscan_clustering)
        
        initial_labels, medoid_indices = clustering.find_optimal_k(doc_embeddings_reduced)
        
        if -1 in initial_labels:
            print("Handling noise points in HDBSCAN results...")
            initial_labels = clustering.handle_noise_points(doc_embeddings_reduced, initial_labels, medoid_indices)
        
        plot_title = f"HDBSCAN Clustering (k={clustering.best_k})"
        
    elif clustering_method == 'gmm':
        print("Using Gaussian Mixture Model clustering...")
        clustering = GMMClustering(**config.gmm_clustering)
        
        initial_labels, medoid_indices = clustering.find_optimal_k(doc_embeddings_reduced)
        
        membership_probs = clustering.get_membership_probabilities()
        np.save(os.path.join(data_dir, "gmm_membership_probs.npy"), membership_probs)
        
        top_docs = clustering.get_top_documents_per_cluster(doc_ids, n=10)
        with open(os.path.join(data_dir, "top_docs_per_cluster.txt"), 'w') as f:
            for cluster_id, docs in top_docs.items():
                f.write(f"Cluster {cluster_id}:\n")
                for doc_id, prob in docs:
                    f.write(f"  {doc_id}: {prob:.4f}\n")
                f.write("\n")
        
        clustering.save_cluster_membership(doc_ids, os.path.join(data_dir, "cluster_membership.csv"))
        
        plot_title = f"GMM Clustering (k={clustering.best_k})"
        
    elif clustering_method == 'hdbscan-gmm':
        print("Using HDBSCAN-GMM hybrid clustering...")
        clustering = HDBSCANGMMClustering(**config.hdbscan_gmm_clustering)
        
        initial_labels, medoid_indices = clustering.find_optimal_k(doc_embeddings_reduced)
        
        membership_probs = clustering.get_membership_probabilities()
        np.save(os.path.join(data_dir, "hdbscan_gmm_membership_probs.npy"), membership_probs)
        
        top_docs = clustering.get_top_documents_per_cluster(doc_ids, n=10)
        with open(os.path.join(data_dir, "hdbscan_gmm_top_docs_per_cluster.txt"), 'w') as f:
            for cluster_id, docs in top_docs.items():
                f.write(f"Cluster {cluster_id}:\n")
                for doc_id, prob in docs:
                    f.write(f"  {doc_id}: {prob:.4f}\n")
                f.write("\n")
        
        clustering.save_cluster_membership(doc_ids, os.path.join(data_dir, "hdbscan_gmm_cluster_membership.csv"))
        
        plot_title = f"HDBSCAN-GMM Clustering (k={clustering.best_k})"
        
    else:
        print("Using Classical K-Medoids clustering...")
        clustering = ClassicalClustering(**config.classical_clustering)
        initial_labels, medoid_indices = clustering.find_optimal_k(doc_embeddings_reduced)
        plot_title = f"K-Medoids Clustering (k={clustering.best_k})"

    print(f"{clustering_method.capitalize()} Clustering Labels: {initial_labels}")
    print(f"{clustering_method.capitalize()} Medoid Indices: {medoid_indices}")

    medoid_embeddings = clustering.extract_medoids(doc_embeddings_reduced, medoid_indices)
    np.save(os.path.join(data_dir, "medoid_embeddings.npy"), medoid_embeddings)
    np.save(os.path.join(data_dir, "medoid_indices.npy"), medoid_indices)
    plot_embeddings(doc_embeddings_reduced, labels=initial_labels, medoids=medoid_embeddings,
                title=plot_title, save_path=initial_clusters_plot_path, cmap=cmap)

    quantum_clustering = QuantumClustering(config.quantum_clustering.k_range, medoid_embeddings, config)
    best_k, refined_medoid_indices, best_dbi = find_best_k_with_qubo(quantum_clustering, medoid_embeddings)

    if refined_medoid_indices is not None:
        print(f"Quantum-Refined Medoid Indices: {refined_medoid_indices}") 
        refined_medoid_indices_of_embeddings = medoid_indices[refined_medoid_indices]
        refined_medoid_embeddings = doc_embeddings_reduced[refined_medoid_indices_of_embeddings] 
        
        print(f"Before QUBO: Assignments to Initial Medoids: {compute_clusters(doc_embeddings_reduced, medoid_indices)}")
        print(f"Refined Medoid Indices Type: {type(refined_medoid_indices)}, Shape: {refined_medoid_indices.shape}")
        print(f"Refined Medoid Embeddings Shape: {refined_medoid_embeddings.shape}")
        print(f"Before Assigning Clusters, Medoid Indices: {refined_medoid_indices_of_embeddings}")
        final_cluster_labels = compute_clusters(doc_embeddings_reduced, refined_medoid_indices_of_embeddings)
        print(f"After QUBO: Assignments to Quantum Medoids: {final_cluster_labels}")

    else:
        raise ValueError("QUBO Solver failed to find valid medoids.")
    
    print(f"Final chosen k after QUBO clustering: {best_k}")
    print(f"Final DBI: {best_dbi:.4f}")

    np.save(os.path.join(data_dir, "final_quantum_clusters.npy"), final_cluster_labels)
    np.save(os.path.join(data_dir, "refined_medoid_embeddings.npy"), refined_medoid_embeddings)
    np.save(os.path.join(data_dir, "refined_medoid_indices.npy"), refined_medoid_indices)
    
    cluster_mapping = pd.DataFrame({
        'doc_id': doc_ids,
        'cluster': final_cluster_labels
    })
    cluster_mapping.to_csv(os.path.join(data_dir, "doc_clusters.csv"), index=False)

    if has_probabilities:
        if hasattr(clustering, 'membership_probs'):
            if multi_membership:
                create_multi_membership_assignments(
                    doc_ids,
                    doc_embeddings_reduced,
                    clustering.membership_probs,
                    final_cluster_labels,
                    refined_medoid_indices_of_embeddings,
                    refined_medoid_embeddings,
                    threshold=threshold,
                    data_dir=data_dir,
                    prefix=clustering_method
                )
            
            create_hybrid_probabilistic_assignments(
                doc_ids,
                clustering.membership_probs,
                final_cluster_labels,
                refined_medoid_indices_of_embeddings,
                data_dir,
                prefix=clustering_method
            )

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

def create_hybrid_probabilistic_assignments(doc_ids, initial_probs, quantum_labels, quantum_medoid_indices, data_dir, prefix=''):
    """
    Create hybrid probabilistic assignments combining initial probabilities with quantum clustering.
    
    Args:
        doc_ids: List of document IDs
        initial_probs: Initial membership probabilities (from GMM or HDBSCAN-GMM)
        quantum_labels: Quantum-refined cluster labels
        quantum_medoid_indices: Quantum-refined medoid indices
        data_dir: Data directory to save results
        prefix: Prefix for output files (empty, 'gmm', or 'hdbscan-gmm')
    """
    prefix = f"{prefix}_" if prefix else ""
    print(f"Creating hybrid probabilistic assignments with {prefix}probabilities...")
    
    n_components = initial_probs.shape[1]
    
    n_quantum_clusters = len(np.unique(quantum_labels))
    
    component_to_quantum = {}
    
    for comp_idx in range(n_components):
        counts = np.zeros(n_quantum_clusters)
        for doc_idx, prob in enumerate(initial_probs[:, comp_idx]):
            if prob > 0.2:
                quantum_cluster = quantum_labels[doc_idx]
                counts[quantum_cluster] += prob
        
        if np.sum(counts) > 0:
            component_to_quantum[comp_idx] = np.argmax(counts)
    
    n_docs = len(doc_ids)
    hybrid_probs = np.zeros((n_docs, n_quantum_clusters))
    
    for doc_idx in range(n_docs):
        doc_probs = initial_probs[doc_idx, :]
        
        for comp_idx, quantum_idx in component_to_quantum.items():
            hybrid_probs[doc_idx, quantum_idx] += doc_probs[comp_idx]
    
    row_sums = hybrid_probs.sum(axis=1, keepdims=True)
    hybrid_probs = np.divide(hybrid_probs, row_sums, out=np.zeros_like(hybrid_probs), where=row_sums != 0)
    
    np.save(os.path.join(data_dir, f"{prefix}hybrid_cluster_probs.npy"), hybrid_probs)
    
    data = {
        'doc_id': doc_ids
    }
    
    for cluster_id in range(n_quantum_clusters):
        data[f'quantum_cluster_{cluster_id}_prob'] = hybrid_probs[:, cluster_id]
    
    data['most_likely_cluster'] = np.argmax(hybrid_probs, axis=1)
    data['quantum_cluster'] = quantum_labels
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(data_dir, f"{prefix}hybrid_cluster_membership.csv"), index=False)
    
    print(f"Saved hybrid probabilistic assignments to {os.path.join(data_dir, f'{prefix}hybrid_cluster_membership.csv')}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run clustering pipeline with custom colormap')
    parser.add_argument('--colormap', type=str, default='Spectral', 
                        help='Colormap to use (file in colormaps dir or matplotlib name)')
    parser.add_argument('--method', type=str, 
                        choices=['classical', 'hdbscan', 'gmm', 'hdbscan-gmm'], 
                        default='classical',
                        help='Clustering method to use')
    parser.add_argument('--multi-membership', action='store_true',
                        help='Enable multi-membership assignments (only works with gmm or hdbscan-gmm)')
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='Probability threshold for multi-membership (default: 0.2)')
    
    args = parser.parse_args()

    with open("config/kmedoids.yml", "r") as file:
        config = ConfigBox(yaml.safe_load(file))
    
    try:
        with open("config/hdbscan.yml", "r") as file:
            hdbscan_config = ConfigBox(yaml.safe_load(file))
            config.update(hdbscan_config)
    except FileNotFoundError:
        print("HDBSCAN config not found, using default parameters")
        config.hdbscan_clustering = ConfigBox({
            'min_cluster_size': 20,
            'min_samples': 25,
            'cluster_selection_method': 'leaf',
            'cluster_selection_epsilon': 0.2,
            'metric': 'euclidean',
            'random_state': config.classical_clustering.random_state
        })
    
    try:
        with open("config/gmm.yml", "r") as file:
            gmm_config = ConfigBox(yaml.safe_load(file))
            config.update(gmm_config)
    except FileNotFoundError:
        print("GMM config not found, using default parameters")
        config.gmm_clustering = ConfigBox({
            'n_components_range': [10, 25, 50, 75, 100],
            'covariance_type': 'full',
            'n_init': 10,
            'random_state': config.classical_clustering.random_state
        })
    
    try:
        with open("config/hdbscan_gmm.yml", "r") as file:
            hdbscan_gmm_config = ConfigBox(yaml.safe_load(file))
            config.update(hdbscan_gmm_config)
    except FileNotFoundError:
        print("HDBSCAN-GMM config not found, using default parameters")
        config.hdbscan_gmm_clustering = ConfigBox({
            'min_cluster_size': 20,
            'min_samples': 25,
            'cluster_selection_method': 'leaf',
            'cluster_selection_epsilon': 0.2,
            'covariance_type': 'full',
            'n_init': 10,
            'metric': 'euclidean',
            'random_state': config.classical_clustering.random_state
        })

    run_pipeline(config, args.colormap, args.method, args.multi_membership, args.threshold)