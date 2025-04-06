
from src.models.QuboSolver import QuboSolver
from sklearn.metrics import davies_bouldin_score, silhouette_score, pairwise_distances
import numpy as np
 

class QuantumClustering:
    def __init__(self, k_range, data, config):
        self.k_range = k_range
        self.data = data
        self.config = config

    def solve_qubo(self, medoid_embeddings, k):
        """Run QUBO clustering for a given k (no longer searching for best k inside this function)."""
        print(f"Solving QUBO for k={k}")

        solver = QuboSolver(n_clusters=k, config=self.config)
        refined_medoid_indices = solver.run_QuboSolver(self.data, bqm_method='kmedoids')

        if refined_medoid_indices is None or len(refined_medoid_indices) == 0:
            print(f"Warning: No valid medoids found for k={k}.")
            return None, None

        final_cluster_labels = compute_clusters(medoid_embeddings, refined_medoid_indices)
        dbi = davies_bouldin_score(medoid_embeddings, final_cluster_labels)
        silhouette = silhouette_score(medoid_embeddings, final_cluster_labels)

        return refined_medoid_indices, dbi, silhouette

def compute_clusters(data, medoid_indices):
    """Assign each point to the closest medoid."""
    if len(medoid_indices) == 0:
        raise ValueError("No medoids selected. QUBO Solver likely failed. Investigate `refined_medoid_indices` output.")

    print(f"Medoid indices: {medoid_indices}")
    print(f"Medoid embeddings shape: {data[medoid_indices].shape}")

    distances = pairwise_distances(data, data[medoid_indices], metric='euclidean')
    return np.argmin(distances, axis=1)