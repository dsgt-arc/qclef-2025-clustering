import os
import numpy as np
import pandas as pd
import umap.umap_ as umap
import matplotlib.pyplot as plt

class UMAPReducer:
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean'):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.reducer = umap.UMAP(
            n_components=self.n_components, 
            n_neighbors=self.n_neighbors, 
            min_dist=self.min_dist, 
            metric=self.metric
        )
    
    def fit_transform(self, embeddings):
        """Reduces the dimensionality of the input embeddings."""
        return self.reducer.fit_transform(embeddings)
    
    def plot_embeddings(self, reduced_embeddings, labels=None, save_path=None):
        """Visualizes the reduced embeddings in a scatter plot and saves the figure if save_path is provided."""
        fig, ax = plt.subplots(figsize=(8, 6))
        if labels is not None:
            scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='Spectral', s=20, alpha=0.7)
            plt.colorbar(scatter)
        else:
            ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=20)
        ax.set_title('UMAP Reduced Embeddings')
        
        if save_path:
            fig.savefig(save_path, dpi=300)
        
        plt.show()
        return fig