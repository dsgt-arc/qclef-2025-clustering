import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import pairwise_distances
from src.models.UMAPReducer import UMAPReducer
from src.models.ClassicalClustering import ClassicalClustering
from src.models.QuantumClustering import QuantumClustering

warnings.filterwarnings("ignore")

def run_pipeline():
    np.random.seed(42)

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

    umap_reducer = UMAPReducer()
    doc_embeddings_reduced = umap_reducer.fit_transform(doc_embeddings)
    np.save(os.path.join(data_dir, "doc_embeddings_reduced.npy"), doc_embeddings_reduced)
    umap_reducer.plot_embeddings(doc_embeddings_reduced, save_path=umap_plot_path)

    clustering = ClassicalClustering()
    kmedoid_labels, medoid_indices = clustering.find_optimal_k(doc_embeddings_reduced)
    medoid_embeddings = clustering.extract_medoids(doc_embeddings_reduced, medoid_indices)
    np.save(os.path.join(data_dir, "medoid_embeddings.npy"), medoid_embeddings)
    np.save(os.path.join(data_dir, "medoid_indices.npy"), medoid_indices)
    clustering.plot_clusters(doc_embeddings_reduced, kmedoid_labels, medoid_embeddings, save_path=kmedoids_plot_path)

    quantum_clustering = QuantumClustering()
    qubo_matrix = quantum_clustering.build_qubo_matrix(medoid_embeddings)
    refined_medoid_indices = quantum_clustering.solve_qubo(qubo_matrix, doc_embeddings_reduced)

    refined_medoid_embeddings = doc_embeddings_reduced[refined_medoid_indices]
    final_kmedoids = KMedoids(n_clusters=len(refined_medoid_indices), metric='euclidean', random_state=42, init=refined_medoid_embeddings)
    final_kmedoids.fit(doc_embeddings_reduced)

    final_cluster_labels = final_kmedoids.labels_
    np.save(os.path.join(data_dir, "final_quantum_clusters.npy"), final_cluster_labels)
    np.save(os.path.join(data_dir, "refined_medoid_embeddings.npy"), refined_medoid_embeddings)
    np.save(os.path.join(data_dir, "refined_medoid_indices.npy"), refined_medoid_indices)

    distances = pairwise_distances(doc_embeddings_reduced, refined_medoid_embeddings, metric='euclidean')
    closest_medoid = np.argmin(distances, axis=1)
    final_quantum_clusters = refined_medoid_indices[closest_medoid]

    np.save(os.path.join(data_dir, "final_quantum_clusters.npy"), final_quantum_clusters)

    plt.figure(figsize=(8, 6))
    plt.scatter(doc_embeddings_reduced[:, 0], doc_embeddings_reduced[:, 1], c=final_cluster_labels, cmap='Spectral', s=20, alpha=0.7)
    plt.scatter(refined_medoid_embeddings[:, 0], refined_medoid_embeddings[:, 1], c='black', marker='X', s=100)
    plt.title("Final Quantum Cluster Assignments")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.savefig(final_clusters_plot_path, dpi=300)
    print(f"Saved final quantum cluster plot at: {final_clusters_plot_path}")
    plt.show()

if __name__ == "__main__":
    run_pipeline()