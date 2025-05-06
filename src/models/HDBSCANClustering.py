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
        
        self.model = None
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
        self.model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_method=self.cluster_selection_method,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            core_dist_n_jobs=-1,
            prediction_data=True
        )
        
        labels = self.model.fit_predict(embeddings)
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        self.best_k = n_clusters
        
        print(f"HDBSCAN found {n_clusters} clusters (excluding noise points)")
        
        medoid_indices = []
        
        for cluster_id in unique_labels:
            if cluster_id != -1:
                cluster_points = np.where(labels == cluster_id)[0]
                cluster_embeddings = embeddings[cluster_points]
                
                centroid = np.mean(cluster_embeddings, axis=0)
                
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                medoid_idx_in_cluster = np.argmin(distances)
                medoid_idx = cluster_points[medoid_idx_in_cluster]
                
                medoid_indices.append(medoid_idx)
        
        medoid_indices = np.array(medoid_indices)
        
        if len(medoid_indices) == 0:
            print("Warning: HDBSCAN did not find any clusters. Adjusting parameters and retrying...")
            self.min_cluster_size = max(3, self.min_cluster_size // 2)
            self.min_samples = self.min_cluster_size
            self.cluster_selection_epsilon = 0.5
            return self.find_optimal_k(embeddings)
        
        non_noise_indices = np.where(labels != -1)[0]
        if len(non_noise_indices) > 1 and n_clusters > 1:
            try:
                dbi = davies_bouldin_score(embeddings[non_noise_indices], labels[non_noise_indices])
                print(f"Davies-Bouldin Index: {dbi:.4f}")
            except Exception as e:
                print(f"Could not calculate Davies-Bouldin Index: {e}")
        
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
        noise_indices = np.where(labels == -1)[0]
        
        if len(noise_indices) == 0:
            return labels
        
        print(f"Assigning {len(noise_indices)} noise points to nearest clusters...")
        
        noise_embeddings = embeddings[noise_indices]
        medoid_embeddings = embeddings[medoid_indices]
        
        distances = pairwise_distances(noise_embeddings, medoid_embeddings, metric=self.metric)
        
        nearest_medoid_indices = np.argmin(distances, axis=1)
        
        new_labels = np.copy(labels)
        
        for i, noise_idx in enumerate(noise_indices):
            new_labels[noise_idx] = nearest_medoid_indices[i]
        
        print(f"After noise point assignment: {len(np.unique(new_labels))} clusters")
        
        return new_labels