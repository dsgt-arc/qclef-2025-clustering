import os
import numpy as np
import dimod
from neal import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite

class QuboSolver:
    def __init__(self, qubo_matrix, n_clusters, num_reads=100):
        self.qubo_matrix = qubo_matrix
        self.n_clusters = n_clusters
        self.num_reads = num_reads
    
    def run_QuboSolver(self):
        """Runs Simulated Annealing on the QUBO matrix and returns optimized cluster assignments."""
        sampler = SimulatedAnnealingSampler()
        # sampler = EmbeddingComposite(DWaveSampler())

        bqm = dimod.BinaryQuadraticModel.from_qubo(self.qubo_matrix)
        response = sampler.sample(bqm, num_reads=self.num_reads)
        # response = sampler.sample(bqm, num_reads=self.num_reads)
        
        best_sample = response.first.sample
        assignments = self._decode_clusters(best_sample)
        return assignments
    
    def _decode_clusters(self, sample):
        """Decodes QUBO solution into optimized cluster assignments, ensuring stability."""
        cluster_labels = np.full(len(sample), -1, dtype=int)
        # cluster_labels = np.full(self.n_clusters, -1, dtype=int)
        cluster_indices = sorted(sample.keys(), key=lambda x: sample[x], reverse=True)[:self.n_clusters]
        
        for idx, var in enumerate(cluster_indices):
            cluster_labels[var] = idx
            # cluster_labels[idx] = var
            # cluster_labels[idx] = idx
        
        return cluster_labels