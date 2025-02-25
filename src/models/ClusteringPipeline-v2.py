import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models.UMAPReducer import UMAPReducer
from src.models.ClassicalClustering import ClassicalClustering
from src.models.QuantumClustering import QuantumClustering
from src.models.QuboSolver import QuboSolver

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def run_umap_reduction():
    np.random.seed(42)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))
    output_csv = os.path.join(data_dir, "antique_train_with_embeddings.csv")

    def parse_embedding(text):
        return np.array(eval(text), dtype=np.float64)

    train_df = pd.read_csv(output_csv, converters={"doc_embedding": parse_embedding})
    doc_embeddings = np.stack(train_df["doc_embedding"].values)

    reducer = UMAPReducer(n_components=2)
    reduced_data = reducer.fit_transform(doc_embeddings)

    reduced_output_path = os.path.join(data_dir, "doc_embeddings_reduced.npy")
    np.save(reduced_output_path, reduced_data)
    
    plot_path = os.path.join(data_dir, "umap_plot.png")
    fig = reducer.plot_embeddings(reduced_data, save_path=plot_path)

    print(f"Reduced document embeddings saved at: {reduced_output_path}")
    print(f"UMAP plot saved at: {plot_path}")

    return fig


def run_classical_clustering():
    np.random.seed(42)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))
    reduced_embeddings_path = os.path.join(data_dir, "doc_embeddings_reduced.npy")

    if not os.path.exists(reduced_embeddings_path):
        raise FileNotFoundError(f"Reduced embeddings file not found at: {reduced_embeddings_path}")

    doc_embeddings_reduced = np.load(reduced_embeddings_path)

    clustering = ClassicalClustering()
    labels, medoid_indices = clustering.find_optimal_k(doc_embeddings_reduced)

    medoid_embeddings = clustering.extract_medoids(doc_embeddings_reduced, medoid_indices)

    np.save(os.path.join(data_dir, "medoid_embeddings.npy"), medoid_embeddings)
    np.save(os.path.join(data_dir, "medoid_indices.npy"), medoid_indices)
    np.save(os.path.join(data_dir, "cluster_labels.npy"), labels)

    plot_path = os.path.join(data_dir, "kmedoids_clusters.png")
    fig = clustering.plot_clusters(doc_embeddings_reduced, labels, save_path=plot_path)

    print("Classical clustering complete.")
    print(f"Cluster plot saved at: {plot_path}")

    return fig


def run_quantum_clustering():
    np.random.seed(42)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))
    medoid_embeddings_path = os.path.join(data_dir, "medoid_embeddings.npy")
    medoid_indices_path = os.path.join(data_dir, "medoid_indices.npy")
    qubo_matrix_path = os.path.join(data_dir, "qubo_matrix.npy")
    clustered_output_path = os.path.join(data_dir, "quantum_cluster_labels.npy")

    if not os.path.exists(medoid_embeddings_path) or not os.path.exists(medoid_indices_path):
        raise FileNotFoundError("Required files not found.")

    medoid_embeddings = np.load(medoid_embeddings_path)
    medoid_indices = np.load(medoid_indices_path)

    n_clusters = len(medoid_indices)

    quantum_clustering = QuantumClustering(n_clusters)

    quantum_clustering.build_qubo_matrix(medoid_embeddings, medoid_indices, qubo_matrix_path)

    cluster_labels = quantum_clustering.solve_qubo(qubo_matrix_path)
    
    np.save(clustered_output_path, cluster_labels)

    print(f"QUBO matrix saved at: {qubo_matrix_path}")
    print(f"Optimized quantum cluster labels saved at: {clustered_output_path}")


def run_qubo_solver():
    np.random.seed(42)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))
    
    qubo_matrix_path = os.path.join(data_dir, "qubo_matrix.npy")
    medoid_indices_path = os.path.join(data_dir, "medoid_indices.npy")
    doc_embeddings_path = os.path.join(data_dir, "doc_embeddings_reduced.npy")
    clustered_output_path = os.path.join(data_dir, "final_quantum_clusters.npy")

    if not os.path.exists(qubo_matrix_path) or not os.path.exists(medoid_indices_path) or not os.path.exists(doc_embeddings_path):
        raise FileNotFoundError("Missing QUBO matrix, medoid indices, or document embeddings.")

    qubo_matrix = np.load(qubo_matrix_path)
    medoid_indices = np.load(medoid_indices_path)
    doc_embeddings = np.load(doc_embeddings_path)

    n_clusters = len(medoid_indices)

    solver = QuboSolver(qubo_matrix, n_clusters)
    medoid_cluster_labels = solver.run_QuboSolver()

    from sklearn.metrics.pairwise import euclidean_distances

    medoid_embeddings = doc_embeddings[medoid_indices]
    distances = euclidean_distances(doc_embeddings, medoid_embeddings)

    closest_medoid = np.argmin(distances, axis=1)
    all_labels = np.array([medoid_cluster_labels[idx] for idx in closest_medoid])

    np.save(clustered_output_path, all_labels)

    print(f"Final optimized quantum cluster assignments saved at: {clustered_output_path}")


def run_final_cluster_visualization():
    """Visualizes the final quantum cluster assignments after QUBO solving."""
    np.random.seed(42)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))

    embeddings_path = os.path.join(data_dir, "doc_embeddings_reduced.npy")
    cluster_labels_path = os.path.join(data_dir, "final_quantum_clusters.npy")

    if not os.path.exists(embeddings_path) or not os.path.exists(cluster_labels_path):
        raise FileNotFoundError("Missing embeddings or cluster labels file.")

    embeddings = np.load(embeddings_path)
    cluster_labels = np.load(cluster_labels_path)

    unique_labels = np.unique(cluster_labels)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    normalized_labels = np.array([label_mapping[label] for label in cluster_labels])

    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=normalized_labels, cmap='Spectral', s=20, alpha=0.7)
    plt.colorbar(label="Cluster Labels")
    plt.title("Final Quantum Cluster Assignments")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")

    plot_path = os.path.join(data_dir, "final_quantum_clusters_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()

    print(f"Cluster visualization saved at: {plot_path}")


class ClusteringPipeline:
    def run(self):
        print("Starting Clustering Pipeline...")
        run_umap_reduction()
        run_classical_clustering()
        run_quantum_clustering()
        run_qubo_solver()
        run_final_cluster_visualization()
        print("Clustering Pipeline Completed.")


if __name__ == "__main__":
    pipeline = ClusteringPipeline()
    pipeline.run()