import os
import numpy as np
import dimod
from src.models.QuboSolver import QuboSolver

class QuantumClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
    
    def build_qubo_matrix(self, embeddings, medoid_indices, qubo_matrix_path):
        """Constructs and saves the QUBO matrix for k-medoids clustering using cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        cosine_matrix = (embeddings @ embeddings.T) / (norms @ norms.T)
        np.fill_diagonal(cosine_matrix, 0)

        mapped_indices = np.arange(len(medoid_indices))

        for idx in mapped_indices:
            cosine_matrix[idx, idx] += 2  

        np.save(qubo_matrix_path, cosine_matrix)
        # print(f"QUBO matrix saved at: {qubo_matrix_path}")

    def solve_qubo(self, qubo_matrix_path):
        """Loads and solves the QUBO problem using QuboSolver."""
        qubo_matrix = np.load(qubo_matrix_path)
        solver = QuboSolver(qubo_matrix, self.n_clusters)
        return solver.run_QuboSolver()

    # def solve_qubo(self, qubo_matrix_path):
    #     """Loads and solves the QUBO problem using QuboSolver and ensures proper index mapping."""
    #     qubo_matrix = np.load(qubo_matrix_path)
    #     solver = QuboSolver(qubo_matrix, self.n_clusters)
    #     raw_assignments = solver.run_QuboSolver()
        
    #     assignments = np.full(self.n_clusters, -1, dtype=int)
    #     for i, medoid_idx in enumerate(raw_assignments):
    #         assignments[i] = medoid_idx

    #     return assignments

    def save_results(self, cluster_assignments, save_path):
        """Saves optimized cluster assignments."""
        np.save(save_path, cluster_assignments)
        print(f"Final quantum cluster assignments saved at: {save_path}")