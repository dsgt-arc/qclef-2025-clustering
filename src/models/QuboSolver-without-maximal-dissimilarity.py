from sklearn.metrics import davies_bouldin_score, silhouette_score, pairwise_distances
from scipy.spatial import distance
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.system import LeapHybridSampler
from qclef import qa_access as qa
import numpy as np
import dimod as dmd

class QuboSolver:
    def __init__(self, n_clusters, num_reads=100, config=None):
        self.n_clusters = n_clusters 
        self.num_reads = num_reads
        self.bqm = None
        self.config = config

    def run_QuboSolver(self, data, bqm_method='kmedoids'):
        if bqm_method == 'kmedoids':
            qubo_dict = self.BQM_kmedoids(data)

        bqm = dmd.BinaryQuadraticModel.from_qubo(qubo_dict)

        solver_config = self.config.quantum_kmedoids

        solver_flags = [solver_config.use_quantum, solver_config.use_hybrid]
        if sum(solver_flags) > 1:
            raise ValueError("Only one of `use_quantum` or `use_hybrid` should be True.")

        if solver_config.use_quantum:
            sampler = EmbeddingComposite(DWaveSampler())
            # Add constraint validation by filtering samples
            response = qa.submit(sampler, EmbeddingComposite.sample, bqm, num_reads=self.num_reads, label="3 - Quantum Clustering")
            # Validate k-constraint in response
            valid_sample = self._find_valid_k_sample(response, self.n_clusters)
            if valid_sample is not None:
                return self._decode_clusters(valid_sample)
            else:
                print(f"Warning: No valid solution with exactly {self.n_clusters} medoids found. Trying with increased gamma.")
                # Retry with higher gamma
                self.config.quantum_kmedoids.gamma *= 2
                return self.run_QuboSolver(data, bqm_method)
        elif solver_config.use_hybrid:
            sampler = LeapHybridSampler()
            response = qa.submit(sampler, LeapHybridSampler.sample, bqm, label="3 - Hybrid Clustering")
            valid_sample = self._find_valid_k_sample(response, self.n_clusters)
            if valid_sample is not None:
                return self._decode_clusters(valid_sample)
            else:
                print(f"Warning: No valid solution with exactly {self.n_clusters} medoids found. Trying with increased gamma.")
                self.config.quantum_kmedoids.gamma *= 2
                return self.run_QuboSolver(data, bqm_method)
        else:
            sampler = SimulatedAnnealingSampler()
            response = qa.submit(sampler, SimulatedAnnealingSampler.sample, bqm, num_reads=self.num_reads, label="3 - Simulated Clustering")

        # Print debug information on number of selected medoids
        best_sample = response.first.sample
        selected_medoids_count = sum(best_sample.values())
        print(f"Selected {selected_medoids_count} medoids out of requested {self.n_clusters}")
        
        # For QA cases that didn't get caught by the above logic
        if solver_config.use_quantum or solver_config.use_hybrid:
            if selected_medoids_count != self.n_clusters:
                print(f"WARNING: QUBO solution has {selected_medoids_count} medoids instead of {self.n_clusters}")
        
        return self._decode_clusters(best_sample)

    def _find_valid_k_sample(self, response, k):
        """Find the first sample that satisfies the k-constraint"""
        for sample, energy in response.data(['sample', 'energy']):
            if sum(sample.values()) == k:
                print(f"Found valid sample with exactly {k} medoids, energy: {energy}")
                return sample
        return None

    def BQM_kmedoids(self, data):
        """Build QUBO matrix with penalty terms (α, β, γ)."""
        N = len(data)
        alpha = 1 / self.n_clusters
        beta = 1 / N
        
        # Get gamma from config
        gamma = self.config.quantum_kmedoids.gamma
        
        # If using quantum or hybrid, increase gamma for stronger constraint enforcement
        solver_config = self.config.quantum_kmedoids
        if solver_config.use_quantum or solver_config.use_hybrid:
            # Increase constraint strength for quantum methods
            gamma *= 2
            print(f"Increased gamma to {gamma} for quantum/hybrid solver")
        
        W = self._compute_corrloss(data)

        # Initialize Q matrix with similarity costs
        Q = {}
        
        # Add the linear terms (diagonal elements)
        for i in range(N):
            # Penalty for deviating from k medoids
            Q[(i, i)] = -2 * gamma * self.n_clusters + beta * np.sum(W[i])
        
        # Add quadratic terms (non-diagonal elements)
        for i in range(N):
            for j in range(i+1, N):
                Q[(i, j)] = gamma - alpha * W[i, j] / 2
        
        # Add extra constraint to ensure exactly k medoids
        # This creates a penalty term of the form gamma * (sum_i x_i - k)^2
        lagrange_term = gamma * N  # Scale with problem size
        
        # Add additional terms to enforce the k-constraint more strictly
        for i in range(N):
            for j in range(i, N):
                if i == j:
                    Q[(i, i)] += lagrange_term
                else:
                    if (i, j) in Q:
                        Q[(i, j)] += 2 * lagrange_term
                    else:
                        Q[(i, j)] = 2 * lagrange_term

        # Print debugging info
        print(f"Penalty terms - α: {alpha}, β: {beta}, γ: {gamma}, Lagrange: {lagrange_term}")
        
        return Q

    def _compute_corrloss(self, data):
        """Compute Welsch M-estimator for measure of similiarity."""
        D = distance.squareform(distance.pdist(data, metric='euclidean'))
        W = 1 - np.exp(-D / 2)
        return W

    def _decode_clusters(self, sample):
        """Extract cluster indices from QUBO solution."""
        cluster_indices = [i for i, v in sample.items() if v == 1]
        selected_count = len(cluster_indices)
        print(f"Decoded {selected_count} Medoid Indices: {cluster_indices}")

        if selected_count == 0:
            print("Warning: No valid medoid indices selected by QUBO Solver.")
        elif selected_count != self.n_clusters:
            print(f"Warning: QUBO selected {selected_count} medoids instead of the required {self.n_clusters}")
            
        return np.array(cluster_indices, dtype=int)
    
    @staticmethod
    def to_upper_triangular(M):
        """Convert the matrix to an upper triangular form required for QUBO."""
        diag = np.diag(M)
        diagM = np.diag(diag)

        M1 = M - diagM
        M2 = np.triu(M1)
        M2 *= 2

        return M2 + diagM

    @staticmethod
    def matrix_to_dict(M):
        """Convert a QUBO matrix to a dictionary format required by D-Wave solvers."""
        q = {}
        for i in range(len(M)):
            for j in range(i, len(M)):
                if M[i, j] != 0:
                    q[(i, j)] = M[i, j]
        return q