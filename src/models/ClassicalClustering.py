import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import davies_bouldin_score

class ClassicalClustering:
    def __init__(self, k_range=[10, 25, 50, 75, 100], metric='euclidean', random_state=42, config=None):
        self.k_range = k_range
        self.metric = metric
        self.random_state = random_state
    
        self.best_k = None
        self.model = None

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
        return best_labels, best_medoid_indices

    def extract_medoids(self, embeddings, medoid_indices):
        return embeddings[medoid_indices]

    def plot_clusters(self, embeddings, labels, medoid_embeddings, save_path=None, showplot=False):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap="Spectral", s=10, alpha=0.7)
        ax.scatter(medoid_embeddings[:, 0], medoid_embeddings[:, 1], c="black", marker="X", s=100)
        ax.set_title(f"K-Medoids Clustering (k={self.best_k})")

        if save_path:
            fig.savefig(save_path, dpi=300)
            print(f"Saved clustering plot at: {save_path}")
        if showplot:
            plt.show()
        else:
            plt.close()
        
        return fig