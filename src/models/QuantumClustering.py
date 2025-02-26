import os
import numpy as np
from src.models.QuboSolver import QuboSolver
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import davies_bouldin_score

class QuantumClustering:
    def __init__(self, k_range=None):
        self.k_range = k_range if k_range else [2, 3, 4, 5, 6, 7, 8, 9]  

    def build_qubo_matrix(self, medoid_embeddings):
        norms = np.linalg.norm(medoid_embeddings, axis=1, keepdims=True)
        cosine_matrix = (medoid_embeddings @ medoid_embeddings.T) / (norms @ norms.T)
        np.fill_diagonal(cosine_matrix, 0)
        for idx in range(len(medoid_embeddings)):
            cosine_matrix[idx, idx] += 2  
        return cosine_matrix

    def solve_qubo(self, qubo_matrix, full_embeddings):
        best_dbi = float("inf")
        best_k = None
        best_refined_medoid_indices = None

        for k in self.k_range:
            solver = QuboSolver(qubo_matrix, k)
            refined_medoid_indices = solver.run_QuboSolver()
            refined_medoid_embeddings = full_embeddings[refined_medoid_indices]

            final_kmedoids = KMedoids(n_clusters=k, metric='euclidean', random_state=42, init=refined_medoid_embeddings)
            final_kmedoids.fit(full_embeddings)

            final_cluster_labels = final_kmedoids.labels_
            dbi = davies_bouldin_score(full_embeddings, final_cluster_labels)

            if dbi < best_dbi:
                best_dbi = dbi
                best_k = k
                best_refined_medoid_indices = refined_medoid_indices

        return np.array(best_refined_medoid_indices)