from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.system import LeapHybridSampler
from qclef import qa_access as qa
import dimod as dmd
from src.models.QuboBuilder import QuboBuilder, KMedoidsQuboBuilder


class QuboSolver:
    """
    Class dedicated to running QUBO solvers (quantum, hybrid, or simulated annealing).
    Handles the actual solving process and fallback strategies.
    """
    def __init__(self, n_clusters=None, num_reads=100, config=None):
        self.n_clusters = n_clusters
        self.num_reads = num_reads
        self.config = config
    
    def run_QuboSolver(self, data, bqm_method='kmedoids'):
        """
        Legacy method to maintain compatibility with existing code.
        Creates a builder and runs the solver.
        """
        if self.n_clusters is None:
            raise ValueError("n_clusters must be specified for run_QuboSolver")
            
        # Create the QUBO builder
        builder = KMedoidsQuboBuilder(self.n_clusters, self.config)
        
        # Map old method names to new ones
        method_mapping = {
            'kmedoids': 'auto_constraint'
        }
        
        # Build the QUBO dictionary
        method = method_mapping.get(bqm_method, bqm_method)
        qubo_dict = builder.build_qubo(data, method=method)
        
        # Solve the QUBO problem
        return self.solve(qubo_dict, self.n_clusters, data, builder)
    
    def solve(self, qubo_dict, n_clusters, data, builder):
        """
        Solve the QUBO problem using the specified solver.
        
        Args:
            qubo_dict: The QUBO dictionary to solve
            n_clusters: Number of clusters
            data: Data points (for fallback strategies)
            builder: The QuboBuilder instance used to build the QUBO
            
        Returns:
            Cluster indices from the solution
        """
        bqm = dmd.BinaryQuadraticModel.from_qubo(qubo_dict)
        solver_config = self.config.quantum_kmedoids
        
        solver_flags = [solver_config.use_quantum, solver_config.use_hybrid]
        if sum(solver_flags) > 1:
            raise ValueError("Only one of use_quantum or use_hybrid should be True.")
        
        if solver_config.use_quantum:
            sampler = EmbeddingComposite(DWaveSampler())
            response = qa.submit(sampler, EmbeddingComposite.sample, bqm, 
                                num_reads=self.num_reads, label="3 - Quantum Clustering")
            valid_sample = self._find_valid_k_sample(response, n_clusters)
            
            if valid_sample is not None:
                return builder.decode_solution(valid_sample)
            else:
                return self._try_fallback_strategy_auto(
                    response, sampler, EmbeddingComposite.sample, 
                    data, n_clusters, builder,
                    num_reads=self.num_reads, 
                    label="3 - Quantum Clustering (Fallback)"
                )

        elif solver_config.use_hybrid:
            sampler = LeapHybridSampler()
            response = qa.submit(sampler, LeapHybridSampler.sample, bqm, label="3 - Hybrid Clustering")
            valid_sample = self._find_valid_k_sample(response, n_clusters)

            if valid_sample is not None:
                return builder.decode_solution(valid_sample)
            else:
                return self._try_fallback_strategy_auto(
                    response, sampler, LeapHybridSampler.sample, 
                    data, n_clusters, builder,
                    label="3 - Hybrid Clustering (Fallback)"
                )

        else:
            sampler = SimulatedAnnealingSampler()
            response = qa.submit(sampler, SimulatedAnnealingSampler.sample, bqm, 
                                num_reads=self.num_reads, label="3 - Simulated Clustering")
            valid_sample = self._find_valid_k_sample(response, n_clusters)
            
            if valid_sample is not None:
                return builder.decode_solution(valid_sample)
            else:
                return self._try_fallback_strategy_auto(
                    response, sampler, SimulatedAnnealingSampler.sample, 
                    data, n_clusters, builder,
                    num_reads=self.num_reads,
                    label="3 - Simulated Clustering (Fallback)"
                )
    
    def _find_valid_k_sample(self, response, k):
        """Find the first sample that satisfies the k-constraint"""
        for sample, energy in response.data(['sample', 'energy']):
            if sum(sample.values()) == k:
                print(f"Found valid sample with exactly {k} medoids, energy: {energy}")
                return sample
        return None
    
    def _try_fallback_strategy_auto(self, response, sampler, sample_method, data, n_clusters, builder, **kwargs):
        """
        Try fallback strategies with auto-calculated penalty when initial solution is invalid.
        """
        initial_bqm = builder._create_initial_clustering_bqm(data)
        base_penalty = initial_bqm.maximum_energy_delta()
        
        for penalty_multiplier in [2, 4, 8, 16, 32, 64, 128]:
            penalty = base_penalty * penalty_multiplier
            print(f"Trying with increased auto_penalty = {penalty}")
            
            constraint_bqm = dmd.generators.combinations(
                initial_bqm.variables, 
                n_clusters,
                strength=penalty
            )
            
            new_bqm = initial_bqm.copy()
            new_bqm.update(constraint_bqm)
            
            response = qa.submit(sampler, sample_method, new_bqm, **kwargs)
            valid_sample = self._find_valid_k_sample(response, n_clusters)
            if valid_sample is not None:
                return builder.decode_solution(valid_sample)
        
        print(f"Warning: All fallback strategies exhausted. Falling back to forcevalid_k_solution.")
        best_sample = response.first.sample
        selected_indices = builder.decode_solution(best_sample)
        return builder.force_valid_k_solution(selected_indices, data, n_clusters)