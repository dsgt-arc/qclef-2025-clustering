import os
import numpy as np
import matplotlib.pyplot as plt
from hdbscan import HDBSCAN
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.pairwise import pairwise_distances

class HDBSCANClustering:
    def __init__(self, min_cluster_size=5, min_samples=None, cluster_selection_method='eom', 
                 cluster_selection_epsilon=0.0, metric='euclidean', random_state=42, config=None):
        """
        Initialize the HDBSCAN clustering model with overclustering.
        
        Args:
            min_cluster_size: The minimum size of clusters
            min_samples: The number of samples in a neighborhood for a point to be considered a core point
            cluster_selection_method: 'eom' for overclustering, 'leaf' for standard clustering
            cluster_selection_epsilon: Smaller values create more clusters (for overclustering)
            metric: Distance metric to use
            random_state: Random state for reproducibility
            config: Configuration object (optional)
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples is not None else min_cluster_size
        self.cluster_selection_method = cluster_selection_method
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        self.random_state = random_state
        self.config = config
        
        # Will store the fitted model
        self.model = None
        # Will store the optimal number of clusters found
        self.best_k = None

    def find_optimal_k(self, embeddings):
        """
        Run HDBSCAN clustering with overclustering and find the medoid indices.
        
        Args:
            embeddings: Input embeddings to cluster
            
        Returns:
            labels: Cluster labels for each point
            medoid_indices: Indices of the cluster medoids
        """
        # Initialize and fit HDBSCAN
        self.model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_method=self.cluster_selection_method,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            core_dist_n_jobs=-1,  # Use all available cores
            prediction_data=True   # Save prediction data for later use
        )
        
        labels = self.model.fit_predict(embeddings)
        
        # Calculate number of clusters (excluding noise points with label -1)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        self.best_k = n_clusters
        
        print(f"HDBSCAN found {n_clusters} clusters (excluding noise points)")
        
        # Find medoids for each cluster
        medoid_indices = []
        
        for cluster_id in unique_labels:
            if cluster_id != -1:  # Skip noise points
                cluster_points = np.where(labels == cluster_id)[0]
                cluster_embeddings = embeddings[cluster_points]
                
                # Calculate the centroid of the cluster
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # Find the point closest to the centroid (medoid)
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                medoid_idx_in_cluster = np.argmin(distances)
                medoid_idx = cluster_points[medoid_idx_in_cluster]
                
                medoid_indices.append(medoid_idx)
        
        # Convert to numpy array
        medoid_indices = np.array(medoid_indices)
        
        # Check if we found any clusters
        if len(medoid_indices) == 0:
            print("Warning: HDBSCAN did not find any clusters. Adjusting parameters and retrying...")
            # Fall back to simpler parameters if no clusters were found
            self.min_cluster_size = max(3, self.min_cluster_size // 2)
            self.min_samples = self.min_cluster_size
            self.cluster_selection_epsilon = 0.5
            return self.find_optimal_k(embeddings)
        
        # Calculate Davies-Bouldin score (excluding noise points)
        non_noise_indices = np.where(labels != -1)[0]
        if len(non_noise_indices) > 1 and n_clusters > 1:
            try:
                dbi = davies_bouldin_score(embeddings[non_noise_indices], labels[non_noise_indices])
                print(f"Davies-Bouldin Index: {dbi:.4f}")
            except Exception as e:
                print(f"Could not calculate Davies-Bouldin Index: {e}")
        
        # Return labels and medoid indices
        return labels, medoid_indices

    def extract_medoids(self, embeddings, medoid_indices):
        """Extract medoid embeddings from the data."""
        return embeddings[medoid_indices]
    
    def handle_noise_points(self, embeddings, labels, medoid_indices):
        """
        Assign noise points to the nearest cluster.
        
        Args:
            embeddings: Input embeddings
            labels: Cluster labels from HDBSCAN
            medoid_indices: Indices of cluster medoids
            
        Returns:
            Updated labels with noise points assigned to clusters
        """
        # Get indices of noise points
        noise_indices = np.where(labels == -1)[0]
        
        if len(noise_indices) == 0:
            return labels  # No noise points to handle
        
        print(f"Assigning {len(noise_indices)} noise points to nearest clusters...")
        
        # Get embeddings of noise points and medoids
        noise_embeddings = embeddings[noise_indices]
        medoid_embeddings = embeddings[medoid_indices]
        
        # Calculate distances from each noise point to each medoid
        distances = pairwise_distances(noise_embeddings, medoid_embeddings, metric=self.metric)
        
        # Assign each noise point to the nearest medoid
        nearest_medoid_indices = np.argmin(distances, axis=1)
        
        # Create new_labels as a copy of the original labels
        new_labels = np.copy(labels)
        
        # For each noise point, assign it to the cluster of its nearest medoid
        for i, noise_idx in enumerate(noise_indices):
            # The label is the index of the medoid in the unique_clusters list
            # We need to convert this to the actual cluster label
            new_labels[noise_idx] = nearest_medoid_indices[i]
        
        print(f"After noise point assignment: {len(np.unique(new_labels))} clusters")
        
        return new_labels
