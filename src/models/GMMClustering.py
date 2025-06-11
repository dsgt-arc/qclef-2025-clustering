import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.pairwise import pairwise_distances

class GMMClustering:
    def __init__(self, n_components_range=[10, 25, 50, 75, 100], 
                 covariance_type='full', n_init=10, random_state=42, config=None):

        self.n_components_range = n_components_range
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.random_state = random_state
        self.config = config
        
        self.model = None
        self.best_k = None
        self.membership_probs = None

    def find_optimal_k(self, embeddings):

        print("Finding optimal number of components for GMM...")
        
        best_bic = np.inf
        best_model = None
        best_n_components = None
        
        for n_components in self.n_components_range:
            print(f"Trying GMM with {n_components} components...")
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=self.covariance_type,
                n_init=self.n_init,
                random_state=self.random_state
            )
            
            gmm.fit(embeddings)
            
            bic = gmm.bic(embeddings)
            print(f"BIC for {n_components} components: {bic:.2f}")
            
            if bic < best_bic:
                best_bic = bic
                best_model = gmm
                best_n_components = n_components
        
        self.model = best_model
        self.best_k = best_n_components
        
        print(f"Best model has {self.best_k} components with BIC: {best_bic:.2f}")
        
        labels = self.model.predict(embeddings)
        
        self.membership_probs = self.model.predict_proba(embeddings)
        
        medoid_indices = []
        
        for component_idx in range(self.best_k):
            component_mean = self.model.means_[component_idx].reshape(1, -1)
            
            cluster_points = np.where(labels == component_idx)[0]
            
            if len(cluster_points) > 0:
                cluster_embeddings = embeddings[cluster_points]
                distances = pairwise_distances(cluster_embeddings, component_mean)
                
                medoid_idx_in_cluster = np.argmin(distances.flatten())
                medoid_idx = cluster_points[medoid_idx_in_cluster]
                
                medoid_indices.append(medoid_idx)
            else:
                distances = pairwise_distances(embeddings, component_mean)
                medoid_idx = np.argmin(distances.flatten())
                medoid_indices.append(medoid_idx)
        
        medoid_indices = np.array(medoid_indices)
        
        try:
            dbi = davies_bouldin_score(embeddings, labels)
            print(f"Davies-Bouldin Index: {dbi:.4f}")
        except Exception as e:
            print(f"Could not calculate Davies-Bouldin Index: {e}")
        
        return labels, medoid_indices

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