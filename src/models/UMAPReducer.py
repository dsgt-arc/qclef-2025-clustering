import os
import numpy as np
import umap.umap_ as umap

class UMAPReducer:
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42):
        self.reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state)

    def fit_transform(self, embeddings):
        return self.reducer.fit_transform(embeddings)