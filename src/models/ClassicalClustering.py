import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import pairwise_distances

class ClassicalClustering:
    def __init__(self, k_range=[10, 25, 50, 75, 100], metric='euclidean', random_state=42, config=None, is_second_stage=False):

        self.k_range = k_range
        self.metric = metric
        self.random_state = random_state
        self.is_second_stage = is_second_stage
        
        if is_second_stage and config and hasattr(config, 'second_stage_clustering'):
            self.k_range = config.second_stage_clustering.k_range
            self.metric = config.second_stage_clustering.metric
            self.random_state = config.second_stage_clustering.random_state
    
        self.best_k = None
        self.model = None
        self.best_score = float("inf")

    def find_optimal_k(self, embeddings):

        best_score = float("inf")
        best_k = None
        best_labels = None
        best_medoid_indices = None

        for k in self.k_range:
            model = KMedoids(n_clusters=k, metric=self.metric, random_state=self.random_state)
            labels = model.fit_predict(embeddings)
            score = davies_bouldin_score(embeddings, labels)

            if score < best_score:
                best_score = score
                best_k = k
                best_labels = labels
                best_medoid_indices = model.medoid_indices_
                self.model = model

        self.best_k = best_k
        self.best_score = best_score
        return best_labels, best_medoid_indices

    def extract_medoids(self, embeddings, medoid_indices):

        return embeddings[medoid_indices]
    
    def assign_to_medoids(self, embeddings, medoid_embeddings, medoid_indices=None):

        distances = pairwise_distances(embeddings, medoid_embeddings, metric=self.metric)
        return np.argmin(distances, axis=1)
    
    def cluster_medoids(self, embeddings, first_stage_medoids, first_stage_medoid_indices, config=None):

        second_stage = ClassicalClustering(
            k_range=config.second_stage_clustering.k_range if config and hasattr(config, 'second_stage_clustering') else [10, 25, 50],
            metric=config.second_stage_clustering.metric if config and hasattr(config, 'second_stage_clustering') else self.metric,
            random_state=config.second_stage_clustering.random_state if config and hasattr(config, 'second_stage_clustering') else self.random_state,
            is_second_stage=True
        )
        
        second_stage_labels, second_stage_medoid_indices = second_stage.find_optimal_k(first_stage_medoids)
        
        refined_medoid_indices = first_stage_medoid_indices[second_stage_medoid_indices]
        
        refined_medoid_embeddings = embeddings[refined_medoid_indices]
        
        final_labels = second_stage.assign_to_medoids(embeddings, refined_medoid_embeddings)
        
        return final_labels, refined_medoid_indices, second_stage.best_k, second_stage.best_score