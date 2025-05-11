from src.models.QuboBuilder import KMedoidsQuboBuilder
from src.models.QuboSolver import QuboSolver
from sklearn.metrics import davies_bouldin_score, silhouette_score, pairwise_distances
import numpy as np
import json
import os

class QuantumClustering:
    def __init__(self, k_range, data, config):
        self.k_range = k_range
        self.data = data
        self.config = config
        self.problem_ids = []  # Add this to track problem IDs

    def solve_qubo(self, medoid_embeddings, k):
        """Run QUBO clustering for a given k (no longer searching for best k inside this function)."""
        print(f"Solving QUBO for k={k}")

        builder = KMedoidsQuboBuilder(n_clusters=k, config=self.config)
        
        qubo_dict = builder.build_qubo(self.data, method='auto_constraint')
        
        solver = QuboSolver(config=self.config)
        
        refined_medoid_indices = solver.solve(qubo_dict, k, self.data, builder)
        
        # Get problem IDs from solver if available
        if hasattr(solver, 'problem_ids'):
            self.problem_ids.extend(solver.problem_ids)

        if refined_medoid_indices is None or len(refined_medoid_indices) == 0:
            print(f"Warning: No valid medoids found for k={k}.")
            return None, None, None

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


def prepare_clustering_submission(doc_embeddings, doc_ids, final_cluster_labels, refined_medoid_indices, 
                             run_output_dir, clustering_method, config, problem_ids=None):
    """
    Prepare submission file for the quantum clustering competition.
    
    Args:
        doc_embeddings: Document embeddings
        doc_ids: List of document IDs
        final_cluster_labels: Final cluster assignments
        refined_medoid_indices: Indices of refined medoids
        run_output_dir: Directory to save results
        clustering_method: Which clustering method was used
        config: Configuration object
        problem_ids: List of problem IDs from quantum annealing submissions
    """
    print("\nPreparing submission file for quantum clustering competition...")
    
    # Create submission format
    submission = []
    
    # Get unique cluster IDs
    unique_clusters = np.unique(final_cluster_labels)
    num_centroids = len(unique_clusters)
    
    # For each cluster, create an entry
    for cluster_id in unique_clusters:
        # Get documents in this cluster
        cluster_docs_idx = np.where(final_cluster_labels == cluster_id)[0]
        cluster_doc_ids = [doc_ids[idx] for idx in cluster_docs_idx]
        
        # Find the centroid for this cluster
        if cluster_id < len(refined_medoid_indices):
            centroid_idx = refined_medoid_indices[cluster_id]
            centroid_coords = doc_embeddings[centroid_idx].tolist()
        else:
            # Alternative: calculate the centroid from cluster members
            cluster_embeddings = np.array([doc_embeddings[idx] for idx in cluster_docs_idx])
            centroid_coords = np.mean(cluster_embeddings, axis=0).tolist()
        
        # Create cluster data object
        cluster_data = {
            'centroid': centroid_coords,
            'docs': cluster_doc_ids
        }
        
        submission.append(cluster_data)
    
    # Add problem IDs as the last line
    submission_json = json.dumps(submission)
    if problem_ids and len(problem_ids) > 0:
        submission_text = submission_json + "\n" + json.dumps(problem_ids)
    else:
        # If no problem IDs provided, use the default SA IDs
        submission_text = submission_json + "\n" + json.dumps(["SA-36", "SA-37", "SA-38", "SA-39", "SA-40", "SA-41", "SA-42"])
    
    # Determine method based on config
    method = "QA" if hasattr(config, 'quantum_kmedoids') and config.quantum_kmedoids.use_quantum else "SA"
    group_name = "ds-at-gt-qclef"
    submission_id = "1"
    
    # Create proper filename format
    filename = f"{num_centroids}_{method}_{group_name}_{submission_id}.txt"
    
    # Save to run directory
    submission_file = os.path.join(run_output_dir, filename)
    with open(submission_file, 'w') as f:
        f.write(submission_text)
    
    # Also save to submissions directory
    submissions_dir = os.path.join("/config/workspace/submissions")
    os.makedirs(submissions_dir, exist_ok=True)
    submissions_file = os.path.join(submissions_dir, filename)
    with open(submissions_file, 'w') as f:
        f.write(submission_text)
    
    print(f"Saved clustering submission to {submission_file}")
    print(f"Also saved to submissions directory: {submissions_file}")
    
    return submission_file