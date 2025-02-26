import numpy as np
import dimod
from neal import SimulatedAnnealingSampler

class QuboSolver:
    def __init__(self, qubo_matrix, n_clusters, num_reads=100):
        self.qubo_matrix = qubo_matrix
        self.n_clusters = n_clusters
        self.num_reads = num_reads

    def run_QuboSolver(self):
        sampler = SimulatedAnnealingSampler()
        bqm = dimod.BinaryQuadraticModel.from_qubo(self.qubo_matrix)
        response = sampler.sample(bqm, num_reads=self.num_reads)
        best_sample = response.first.sample
        return self._decode_clusters(best_sample)

    def _decode_clusters(self, sample):
        cluster_labels = np.full(self.n_clusters, -1, dtype=int)
        cluster_indices = sorted(sample.keys(), key=lambda x: sample[x], reverse=True)[:self.n_clusters]
        for idx, var in enumerate(cluster_indices):
            cluster_labels[idx] = var
        return cluster_labels
