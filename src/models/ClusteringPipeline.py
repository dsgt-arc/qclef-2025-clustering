import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import yaml

# from sklearn_extra.cluster import KMedoids
from sklearn.metrics import pairwise_distances
from src.models.UMAPReducer import UMAPReducer
from src.models.ClassicalClustering import ClassicalClustering
from src.models.QuantumClustering import QuantumClustering, compute_clusters
from box import ConfigBox


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


def run_pipeline(config):

    np.random.seed(config.classical_clustering.random_state)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))
    output_csv = os.path.join(data_dir, "antique_train_with_embeddings.csv")

    umap_plot_path = os.path.join(data_dir, "umap_plot.png")
    kmedoids_plot_path = os.path.join(data_dir, "kmedoids_clusters.png")
    final_clusters_plot_path = os.path.join(data_dir, "final_clusters.png")

    def parse_embedding(text):
        return np.array(eval(text), dtype=np.float64)

    train_df = pd.read_csv(output_csv, converters={"doc_embedding": parse_embedding})
    doc_embeddings = np.stack(train_df["doc_embedding"].values)

    # AP: This part should be optional, like a pre-clustering step, so that we can compare to full clustering
    # Step 1: UMAP Reduction (optional)
    umap_reducer = UMAPReducer()
    doc_embeddings_reduced = umap_reducer.fit_transform(doc_embeddings)
    np.save(os.path.join(data_dir, "doc_embeddings_reduced.npy"), doc_embeddings_reduced)
    umap_reducer.plot_embeddings(doc_embeddings_reduced, save_path=umap_plot_path)

    # Step 2: Classical K-Medoids Clustering
    clustering = ClassicalClustering(**config.classical_clustering)
    kmedoid_labels, medoid_indices = clustering.find_optimal_k(doc_embeddings_reduced)

    print(f"Classical K-Medoids Labels: {kmedoid_labels}")
    print(f"Classical Medoid Indices: {medoid_indices}")

    medoid_embeddings = clustering.extract_medoids(doc_embeddings_reduced, medoid_indices)
    np.save(os.path.join(data_dir, "medoid_embeddings.npy"), medoid_embeddings)
    np.save(os.path.join(data_dir, "medoid_indices.npy"), medoid_indices)
    clustering.plot_clusters(doc_embeddings_reduced, kmedoid_labels, medoid_embeddings, save_path=kmedoids_plot_path)

    # Step 3: Quantum Refinement with QUBO
    # doc_embeddings_reduced = np.load(os.path.join(data_dir, "doc_embeddings_reduced.npy"))
    # medoid_embeddings = np.load(os.path.join(data_dir, "medoid_embeddings.npy"))

    # k_range = config.quantum_clustering.k_range
    # AP: we instantiate the object with the desired k_range and the embeddings obtained at the previous step (to fit on the QPU)
    quantum_clustering = QuantumClustering(config.quantum_clustering.k_range, medoid_embeddings, config)

    best_k, refined_medoid_indices = find_best_k_with_qubo(quantum_clustering, medoid_embeddings)

    if refined_medoid_indices is not None:
        print(f"Quantum-Refined Medoid Indices: {refined_medoid_indices}")
        # refined_medoid_embeddings = medoid_embeddings[refined_medoid_indices]
        refined_medoid_embeddings = doc_embeddings_reduced[refined_medoid_indices]
        
        # Assign points to refined medoids
        print(f"Before QUBO: Assignments to Classical Medoids: {compute_clusters(doc_embeddings_reduced, medoid_indices)}")
        print(f"Refined Medoid Indices Type: {type(refined_medoid_indices)}, Shape: {refined_medoid_indices.shape}")
        print(f"Refined Medoid Embeddings Shape: {refined_medoid_embeddings.shape}")
        print(f"Before Assigning Clusters, Medoid Indices: {refined_medoid_indices}")
        final_cluster_labels = compute_clusters(doc_embeddings_reduced, refined_medoid_indices)
        # final_cluster_labels = compute_clusters(doc_embeddings_reduced, refined_medoid_embeddings)
        print(f"After QUBO: Assignments to Quantum Medoids: {final_cluster_labels}")

    else:
        raise ValueError("QUBO Solver failed to find valid medoids.")
    
    print(f"Final chosen k after QUBO clustering: {best_k}")

    # final_kmedoids = KMedoids(n_clusters=best_k, metric='euclidean', random_state=42, init=medoid_embeddings[refined_medoid_indices])
    # final_kmedoids.fit(doc_embeddings_reduced)

    # print(f"Final KMedoids cluster centers: {final_kmedoids.cluster_centers_}")
    # print(f"Refined Medoid Embeddings: {refined_medoid_embeddings}")

    # final_cluster_labels = final_kmedoids.labels_

    # Save final clustering results
    np.save(os.path.join(data_dir, "final_quantum_clusters.npy"), final_cluster_labels)
    np.save(os.path.join(data_dir, "refined_medoid_embeddings.npy"), refined_medoid_embeddings)
    np.save(os.path.join(data_dir, "refined_medoid_indices.npy"), refined_medoid_indices)

    # distances = pairwise_distances(doc_embeddings_reduced, refined_medoid_embeddings, metric='euclidean')
    # closest_medoid = np.argmin(distances, axis=1)
    # final_quantum_clusters = refined_medoid_indices[closest_medoid]
    # final_quantum_clusters = np.array([refined_medoid_indices[i] for i in closest_medoid]) # something new I tried today

    # np.save(os.path.join(data_dir, "final_quantum_clusters.npy"), final_quantum_clusters)

    print(f"Final Quantum-Refined Medoid Embeddings:\n {refined_medoid_embeddings}")
    print(f"Unique Cluster Assignments: {np.unique(final_cluster_labels)}")

    # Step 4: Visualization 
    plt.figure(figsize=(8, 6))
    plt.scatter(doc_embeddings_reduced[:, 0], doc_embeddings_reduced[:, 1], c=final_cluster_labels, cmap='Spectral', s=20, alpha=0.7)
    # plt.scatter(medoid_embeddings[:, 0], medoid_embeddings[:, 1], c='green', marker='X', s=100, label='All Centroid Points in the reduced set (1st stage)')
    plt.scatter(medoid_embeddings[:, 0], medoid_embeddings[:, 1], c='green', marker='X', s=100, label='Classical Medoids (1st Stage)')
    plt.scatter(refined_medoid_embeddings[:, 0], refined_medoid_embeddings[:, 1], c='black', marker='X', s=100, label='Quantum-Refined Medoids (Final)')
    # plt.scatter(refined_medoid_embeddings[:, 0], refined_medoid_embeddings[:, 1], c='black', marker='X', s=100, label='Quantum K-medoids (2nd stage)')
    # plt.scatter(final_kmedoids.cluster_centers_[:, 0], final_kmedoids.cluster_centers_[:, 1], c='red', marker='X', s=100, label='Classical k-medoids solution for all points')
    plt.legend(loc='best')

    plt.title("Final Quantum Cluster Assignments")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.savefig(final_clusters_plot_path, dpi=300)
    print(f"Saved final quantum cluster plot at: {final_clusters_plot_path}")


if __name__ == "__main__":

    with open("config/kmedoids.yml", "r") as file:
        config = ConfigBox(yaml.safe_load(file))

    run_pipeline(config)