import os
import numpy as np
import matplotlib.pyplot as plt
from hdbscan import HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.pairwise import pairwise_distances

class HDBSCANGMMClustering:
    def __init__(self, min_cluster_size=5, min_samples=None, cluster_selection_method='leaf', 
                 cluster_selection_epsilon=0.2, covariance_type='full', n_init=10,
                 metric='euclidean', random_state=42, config=None):

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples is not None else min_cluster_size
        self.cluster_selection_method = cluster_selection_method
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.metric = metric
        self.random_state = random_state
        self.config = config
        
        self.hdbscan_model = None
        self.gmm_models = {}
        self.best_k = None
        self.membership_probs = None

    def find_optimal_k(self, embeddings):

        print("Step 1: Running HDBSCAN to identify natural clusters...")
        
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_method=self.cluster_selection_method,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            core_dist_n_jobs=-1,
            prediction_data=True
        )
        
        hdbscan_labels = self.hdbscan_model.fit_predict(embeddings)
        
        unique_labels = np.unique(hdbscan_labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        self.best_k = n_clusters
        
        print(f"HDBSCAN found {n_clusters} clusters (excluding noise points)")
        
        if -1 in hdbscan_labels:
            print("Handling noise points in HDBSCAN results...")
            hdbscan_labels = self._assign_noise_points(embeddings, hdbscan_labels)
            
        print("Step 2: Fitting GMM to each HDBSCAN cluster for probabilistic assignments...")
        
        n_samples = embeddings.shape[0]
        self.membership_probs = np.zeros((n_samples, n_clusters))
        
        for i, cluster_id in enumerate(unique_labels[unique_labels != -1]):
            cluster_indices = np.where(hdbscan_labels == cluster_id)[0]
            cluster_points = embeddings[cluster_indices]
            
            n_components = min(
                max(1, len(cluster_points) // 20),
                5
            )
            
            print(f"  Fitting GMM with {n_components} components to cluster {cluster_id} with {len(cluster_points)} points")
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=self.covariance_type,
                n_init=self.n_init,
                random_state=self.random_state
            )
            
            try:
                gmm.fit(cluster_points)
                self.gmm_models[cluster_id] = gmm
                
                log_probs = gmm.score_samples(embeddings)
                
                self.membership_probs[:, i] = np.exp(log_probs)
                
            except Exception as e:
                print(f"  Error fitting GMM to cluster {cluster_id}: {str(e)}")
                self._calculate_distance_based_probs(embeddings, cluster_points, cluster_id, i)
        
        row_sums = np.sum(self.membership_probs, axis=1, keepdims=True)
        self.membership_probs = self.membership_probs / row_sums
        
        labels = np.argmax(self.membership_probs, axis=1)
        
        medoid_indices = []
        
        for cluster_id in range(n_clusters):
            cluster_points = np.where(labels == cluster_id)[0]
            
            if len(cluster_points) > 0:
                centroid = np.mean(embeddings[cluster_points], axis=0)
                
                distances = np.linalg.norm(embeddings[cluster_points] - centroid, axis=1)
                medoid_idx_in_cluster = np.argmin(distances)
                medoid_idx = cluster_points[medoid_idx_in_cluster]
                
                medoid_indices.append(medoid_idx)
            else:
                print(f"Warning: No points assigned to cluster {cluster_id}. Using a random point as medoid.")
                medoid_indices.append(np.random.choice(len(embeddings)))
        
        medoid_indices = np.array(medoid_indices)
        
        try:
            dbi = davies_bouldin_score(embeddings, labels)
            print(f"Davies-Bouldin Index: {dbi:.4f}")
        except Exception as e:
            print(f"Could not calculate Davies-Bouldin Index: {e}")
        
        return labels, medoid_indices

    def _assign_noise_points(self, embeddings, labels):

        noise_indices = np.where(labels == -1)[0]
        noise_count = len(noise_indices)
        
        if noise_count == 0:
            return labels
        
        print(f"Assigning {noise_count} noise points to nearest clusters...")
        
        unique_clusters = np.unique(labels[labels != -1])
        
        if len(unique_clusters) == 0:
            print("Warning: No non-noise clusters found. All points are considered noise.")
            return np.zeros_like(labels)
        
        new_labels = np.copy(labels)
        
        for noise_idx in noise_indices:
            noise_point = embeddings[noise_idx].reshape(1, -1)
            
            min_dist = float('inf')
            closest_cluster = unique_clusters[0]
            
            for cluster_id in unique_clusters:
                cluster_points = embeddings[labels == cluster_id]
                
                cluster_center = np.mean(cluster_points, axis=0).reshape(1, -1)
                
                dist = np.linalg.norm(noise_point - cluster_center)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_cluster = cluster_id
            
            new_labels[noise_idx] = closest_cluster
        
        print(f"After noise point assignment: {len(np.unique(new_labels))} clusters")
        
        return new_labels
    
    def _calculate_distance_based_probs(self, embeddings, cluster_points, cluster_id, cluster_idx):

        print(f"  Using distance-based probabilities for cluster {cluster_id} as GMM fallback")
        
        cluster_center = np.mean(cluster_points, axis=0).reshape(1, -1)
        
        distances = np.linalg.norm(embeddings - cluster_center, axis=1)
        
        if len(cluster_points) > 1:
            sigma = np.mean(np.linalg.norm(cluster_points - cluster_center, axis=1))
        else:
            sigma = 1.0
            
        self.membership_probs[:, cluster_idx] = np.exp(-0.5 * (distances / sigma)**2)

    def extract_medoids(self, embeddings, medoid_indices):
        return embeddings[medoid_indices]
    
    def get_membership_probabilities(self):
        if self.membership_probs is None:
            raise ValueError("Model has not been fit yet. Call find_optimal_k first.")
        return self.membership_probs
    
    def get_top_documents_per_cluster(self, doc_ids, n=5):

        if self.membership_probs is None:
            raise ValueError("Model has not been fit yet. Call find_optimal_k first.")
        
        top_docs = {}
        
        for cluster_id in range(self.best_k):
            cluster_probs = self.membership_probs[:, cluster_id]
            
            top_indices = np.argsort(-cluster_probs)[:n]
            
            top_docs[cluster_id] = [
                (doc_ids[idx], cluster_probs[idx]) 
                for idx in top_indices
            ]
        
        return top_docs
    
    def save_cluster_membership(self, doc_ids, output_file):

        import pandas as pd
        
        if self.membership_probs is None:
            raise ValueError("Model has not been fit yet. Call find_optimal_k first.")
        
        data = {
            'doc_id': doc_ids
        }
        
        for cluster_id in range(self.best_k):
            data[f'cluster_{cluster_id}_prob'] = self.membership_probs[:, cluster_id]
        
        data['most_likely_cluster'] = np.argmax(self.membership_probs, axis=1)
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        print(f"Saved cluster membership probabilities to {output_file}")