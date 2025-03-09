
from src.models.QuboSolver import QuboSolver
from sklearn.metrics import davies_bouldin_score, silhouette_score, pairwise_distances
import numpy as np
 

class QuantumClustering:
    def __init__(self, k_range, data, config):
        self.k_range = k_range
        self.data = data
        self.config = config

    def solve_qubo(self, doc_embeddings_reduced):
        """Select medoids using QUBO and compute silhouette score."""
        best_dbi = float("inf")
        best_k = None
        best_refined_medoid_indices = None

        silhouettes_quantum = {}

        print(f"Starting QUBO solving. Current best DBI={best_dbi}")

        for k in self.k_range: # AP: k is really the number of clusters. This means, the number of 1's in the binary vector of the qubo formulation x^T Q x
            # AP: We iterate over number of clusters that we want to pick after the initial sampling with classical kmedoids
            print(f"Solving QUBO for k={k}")

            solver = QuboSolver(n_clusters = k, config=self.config)
            refined_medoid_indices = solver.run_QuboSolver(self.data, bqm_method='kmedoids')

            print(f"Refined medoid indices from QUBO Solver: {refined_medoid_indices}")

            if refined_medoid_indices is None or len(refined_medoid_indices) == 0:
                print(f"Warning: No valid medoids found for k={k}. Skipping DBI computation.")
                continue

            final_cluster_labels = compute_clusters(doc_embeddings_reduced, refined_medoid_indices)
            
            silhouettes_quantum[k] = silhouette_score(doc_embeddings_reduced, final_cluster_labels)

            dbi = davies_bouldin_score(doc_embeddings_reduced, final_cluster_labels)

            if dbi < best_dbi:
                best_dbi = dbi
                best_k = k
                best_refined_medoid_indices = refined_medoid_indices

        print(f"Selected k={best_k} with DBI={best_dbi}")
        return np.array(best_refined_medoid_indices), silhouettes_quantum

# helper function
def compute_clusters(data, medoid_indices):
    """Assign each point to the closest medoid."""
    if len(medoid_indices) == 0:
        raise ValueError("No medoids selected. QUBO Solver likely failed. Investigate `refined_medoid_indices` output.")

    print(f"Medoid indices: {medoid_indices}")
    print(f"Medoid embeddings shape: {data[medoid_indices].shape}")

    distances = pairwise_distances(data, data[medoid_indices], metric='euclidean')
    return np.argmin(distances, axis=1)