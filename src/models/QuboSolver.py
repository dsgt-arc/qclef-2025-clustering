from sklearn.metrics import davies_bouldin_score, silhouette_score, pairwise_distances
from scipy.spatial import distance
from neal import SimulatedAnnealingSampler
import numpy as np
import dimod as dmd

class QuboSolver:
    # AP: All methods to build the Q-matrix should be here, QuboSolver should not be calling stuff from Clusterrun

    def __init__(self, n_clusters, num_reads=100, config=None):
        self.n_clusters = n_clusters # AP: this would be equal to k in your case
        self.num_reads = num_reads
        self.bqm = None
        self.config = config

    # AP: Consider abstracting this away, as this can be reused for other methods too - just needs a Qubo matrix.
    def run_QuboSolver(self, data, bqm_method='kmedoids'):
        
        # AP: build the qubo matrix. There are many ways to handle this, 
        # I suggest that you abstract the whole QUBO matrix building elsewhere, and keep this class just for the solver and utilities around it.
        # Take a look at the Instance Selection class for an example. Alternatively, you keep adding different methods here.
        if bqm_method == 'kmedoids':
            qubo_dict = self.BQM_kmedoids(data)
        # elif bqm_method = 'anothermethod1':
        #     pass    
        # elif bqm_method = 'anothermethod2':
        #     pass    

        sampler = SimulatedAnnealingSampler()
        bqm = dmd.BinaryQuadraticModel.from_qubo(qubo_dict)
        response = sampler.sample(bqm, num_reads=self.num_reads, seed=self.config.quantum_kmedoids.random_state)
        
        print("Raw QUBO Response:", response.first.sample)  # Debugging output

        best_sample = response.first.sample
        return self._decode_clusters(best_sample)

    def BQM_kmedoids(self, data): #AP: This is just really for the kmedoids approach, should be named properly
        """Build QUBO matrix with penalty terms (α, β, γ)."""
        N = len(data)
        alpha = 1 / self.n_clusters
        beta = 1 / N
        gamma = self.config.quantum_kmedoids.gamma

        W = self._compute_corrloss(data) # equal to delta in their case

        Q = gamma - alpha * W / 2

        for i in range(N):
            Q[i, i] += beta * np.sum(W[i]) - 2 * gamma * self.n_clusters

        Q = self.to_upper_triangular(Q)
        dictQ = self.matrix_to_dict(Q)

        # Print debugging info
        print(f"Penalty terms - α: {alpha}, β: {beta}, γ: {gamma}")
        print(f"QUBO Matrix Min: {np.min(Q)}, Max: {np.max(Q)}, Mean: {np.mean(Q)}")
        print(f"QUBO Matrix Sample:\n{Q[:5, :5]}")

        # AP: plotting it could be a nice debugging tool too

        return dictQ
    
    def _compute_corrloss(self, data):
        """Compute Welsch M-estimator for measure of similiarity. See Baukhage 2019. eq (8) for more details """
        D = distance.squareform(distance.pdist(data, metric='euclidean'))
        
        W = 1 - np.exp(-D / 2)
        return W

    def _decode_clusters(self, sample):
        """Extract cluster indices from QUBO solution."""
        cluster_indices = [i for i, v in sample.items() if v == 1]
        print("Decoded Medoid Indices:", cluster_indices)  # Debugging output

        if len(cluster_indices) == 0:
            print("Warning: No valid medoid indices selected by QUBO Solver.")
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