import os
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt

class UMAPReducer:
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42):
        self.reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state)

    def fit_transform(self, embeddings):
        return self.reducer.fit_transform(embeddings)

    def plot_embeddings(self, reduced_embeddings, labels=None, save_path=None, showplot=False):
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=20, alpha=0.7)
        ax.set_title('UMAP Reduced Embeddings')

        if save_path:
            fig.savefig(save_path, dpi=300)
            print(f"Saved UMAP plot at: {save_path}")
        if showplot:
            plt.show()
        else:
            plt.close()

        return fig