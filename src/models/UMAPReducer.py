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
        plt.figure(figsize=(8, 6))
        if labels is not None:
            plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='Spectral', s=5)
            plt.colorbar()
        else:
            plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=5)
        plt.title('UMAP Reduced Embeddings')
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        plt.show()

if __name__ == "__main__":
    np.random.seed(42)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))
    output_csv = os.path.join(data_dir, "antique_train_with_embeddings.csv")

    def parse_embedding(text):
        return np.array(eval(text), dtype=np.float64)

    train_df = pd.read_csv(output_csv, converters={"doc_embedding": parse_embedding})
    doc_embeddings = np.stack(train_df["doc_embedding"].values)

    reducer = UMAPReducer(n_components=2)
    reduced_data = reducer.fit_transform(doc_embeddings)

    reduced_output_path = os.path.join(data_dir, "doc_embeddings_reduced.npy")
    np.save(reduced_output_path, reduced_data)
    
    plot_path = os.path.join(data_dir, "umap_plot.png")
    reducer.plot_embeddings(reduced_data, save_path=plot_path)
    
    print(f"Reduced document embeddings saved at: {reduced_output_path}")
    print(f"UMAP plot saved at: {plot_path}")