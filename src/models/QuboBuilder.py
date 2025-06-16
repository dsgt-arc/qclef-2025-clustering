from abc import ABC, abstractmethod
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
import numpy as np
import dimod as dmd

class QuboBuilder(ABC):

    @abstractmethod
    def build_qubo(self, data):
        pass
    
    @abstractmethod
    def decode_solution(self, sample, data=None):
        pass
    
    @staticmethod
    def to_upper_triangular(M):
        diag = np.diag(M)
        diagM = np.diag(diag)

        M1 = M - diagM
        M2 = np.triu(M1)
        M2 *= 2

        return M2 + diagM

    @staticmethod
    def matrix_to_dict(M):
        q = {}
        for i in range(len(M)):
            for j in range(i, len(M)):
                if M[i, j] != 0:
                    q[(i, j)] = M[i, j]
        return q


class KMedoidsQuboBuilder(QuboBuilder):

    def __init__(self, n_clusters, config=None):
        self.n_clusters = n_clusters
        self.config = config
    
    def build_qubo(self, data, method='auto_constraint'):

        if method == 'auto_constraint':
            return self.build_kmedoids_auto_constraint(data)
        elif method == 'dimod_constraint':
            return self.build_kmedoids_with_dimod_constraint(data)
        elif method == 'quadratic_penalty':
            return self.build_kmedoids_quadratic_penalty(data)
        else:
            raise ValueError(f"Unknown QUBO building method: {method}")
    
    def decode_solution(self, sample, data=None):

        cluster_indices = [i for i, v in sample.items() if v == 1]
        selected_count = len(cluster_indices)
        print(f"Decoded {selected_count} Medoid Indices: {cluster_indices}")

        if selected_count == 0:
            print("Warning: No valid medoid indices selected by QUBO Solver.")
        elif selected_count != self.n_clusters:
            print(f"Warning: QUBO selected {selected_count} medoids instead of the required {self.n_clusters}")
            
        return np.array(cluster_indices, dtype=int)
    
    def force_valid_k_solution(self, selected_indices, data, k):
        current_count = len(selected_indices)
        
        if current_count < k:
            print(f"Adding {k - current_count} more medoids to meet the requirement")
            
            all_indices = np.arange(len(data))
            available_indices = np.setdiff1d(all_indices, selected_indices)
            
            if current_count == 0:
                np.random.shuffle(available_indices)
                return available_indices[:k]
            
            selected_data = data[selected_indices]
            
            min_distances = []
            for idx in available_indices:
                point = data[idx].reshape(1, -1)
                dists = pairwise_distances(point, selected_data)
                min_distances.append(np.min(dists))
            
            sorted_indices = available_indices[np.argsort(-np.array(min_distances))]
            
            additional_indices = sorted_indices[:(k - current_count)]
            return np.concatenate([selected_indices, additional_indices])
            
        elif current_count > k:
            print(f"Removing {current_count - k} medoids to meet the requirement")
            
            selected_data = data[selected_indices]
            distances = pairwise_distances(selected_data)
            
            np.fill_diagonal(distances, np.inf)
            
            indices_to_keep = list(range(current_count))
            
            for _ in range(current_count - k):
                min_i, min_j = np.unravel_index(np.argmin(distances), distances.shape)
                
                avg_dist_i = np.mean(distances[min_i, :])
                avg_dist_j = np.mean(distances[min_j, :])
                
                to_remove = min_i if avg_dist_i < avg_dist_j else min_j
                
                distances[to_remove, :] = np.inf
                distances[:, to_remove] = np.inf
                
                indices_to_keep.remove(to_remove)
            
            return selected_indices[indices_to_keep]
            
        else:
            return selected_indices
    
    def _compute_corrloss(self, data):
        D = distance.squareform(distance.pdist(data, metric='euclidean')) ** 2
        W = 1 - np.exp(-D / 2)
        return W
    
    def _create_initial_clustering_bqm(self, data):
        N = len(data)
        alpha = 1 / self.n_clusters
        beta = 1 / N
        
        W = self._compute_corrloss(data)
        
        Q = {}
        for i in range(N):
            Q[(i, i)] = beta * np.sum(W[i])
            for j in range(i+1, N):
                Q[(i, j)] = -alpha * W[i, j] / 2
        
        return dmd.BinaryQuadraticModel.from_qubo(Q)
    
    def build_kmedoids_auto_constraint(self, data):
        N = len(data)
        alpha = 1 / self.n_clusters
        beta = 1 / N

        W = self._compute_corrloss(data)
        
        Q = {}
        for i in range(N):
            Q[(i, i)] = beta * np.sum(W[i])
            for j in range(i+1, N):
                Q[(i, j)] = -alpha * W[i, j] / 2
        
        initial_bqm = dmd.BinaryQuadraticModel.from_qubo(Q)
        
        penalty = initial_bqm.maximum_energy_delta()
        
        constraint_bqm = dmd.generators.combinations(
            initial_bqm.variables, 
            self.n_clusters,
            strength=penalty
        )
        
        initial_bqm.update(constraint_bqm)
        
        print(f"Penalty terms - α: {alpha}, β: {beta}, auto_penalty: {penalty}")
        
        return initial_bqm.to_qubo()[0]
    
    def build_kmedoids_with_dimod_constraint(self, data):
        N = len(data)
        alpha = 1 / self.n_clusters
        beta = 1 / N

        gamma = self.config.quantum_kmedoids.gamma
        gamma_constraint = self.config.quantum_kmedoids.gamma_constraint

        W = self._compute_corrloss(data)
        Q = {}

        for i in range(N):
            Q[(i, i)] = beta * np.sum(W[i])
            for j in range(i+1, N):
                Q[(i, j)] = gamma - alpha * W[i, j] / 2

        bqm = dmd.BinaryQuadraticModel.from_qubo(Q)

        bqm_constraint = dmd.generators.combinations(N, self.n_clusters)
        combined_bqm = bqm + gamma_constraint * bqm_constraint

        print(f"Penalty terms – α: {alpha}, β: {beta}, γ_cluster: {gamma}, γ_constraint: {gamma_constraint}")

        return combined_bqm.to_qubo()[0]
    
    def build_kmedoids_quadratic_penalty(self, data):
        N = len(data)
        alpha = 1 / self.n_clusters
        beta = 1 / N
        gamma = self.config.quantum_kmedoids.gamma

        W = self._compute_corrloss(data)
        
        Q = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    Q[i, j] -= alpha * W[i, j] / 2
        
        for i in range(N):
            Q[i, i] += beta * np.sum(W[i])
            
            Q[i, i] += gamma
            
            Q[i, i] -= 2 * gamma * self.n_clusters
            
            for j in range(i+1, N):
                Q[i, j] += 2 * gamma
        
        Q = self.to_upper_triangular(Q)
        dictQ = self.matrix_to_dict(Q)
        
        print(f"Penalty terms - α: {alpha}, β: {beta}, γ: {gamma}")
        
        return dictQ