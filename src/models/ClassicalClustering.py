import os
import numpy as np
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt

class ClassicalClustering:
    def __init__(self, n_clusters=5, metric='euclidean', random_state=42):
        self.n_clusters = n_clusters
        self.metric = metric
        self.random_state = random_state
        self.model = KMedoids(n_clusters=self.n_clusters, metric=self.metric, random_state=self.random_state)
    
    def fit_predict(self, embeddings):
        """Fits k-medoids clustering and returns cluster labels."""
        return self.model.fit_predict(embeddings)
    
    def plot_clusters(self, embeddings, labels, save_path=None):
        """Visualizes clustered embeddings in a scatter plot and saves the figure if save_path is provided."""
        plt.figure(figsize=(8, 6))
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='Spectral', s=10)
        plt.colorbar()
        plt.title('K-Medoids Clustering')
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))
    reduced_embeddings_path = os.path.join(data_dir, "doc_embeddings_reduced.npy")
    
    if not os.path.exists(reduced_embeddings_path):
        raise FileNotFoundError(f"Reduced embeddings file not found at: {reduced_embeddings_path}")
    
    doc_embeddings_reduced = np.load(reduced_embeddings_path)
    
    clustering = ClassicalClustering(n_clusters=5)
    labels = clustering.fit_predict(doc_embeddings_reduced)
    
    clustered_output_path = os.path.join(data_dir, "cluster_labels.npy")
    np.save(clustered_output_path, labels)
    
    plot_path = os.path.join(data_dir, "kmedoids_clusters.png")
    clustering.plot_clusters(doc_embeddings_reduced, labels, save_path=plot_path)
    
    print(f"Cluster labels saved at: {clustered_output_path}")
    print(f"Cluster plot saved at: {plot_path}")