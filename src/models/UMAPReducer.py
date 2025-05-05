import os
import numpy as np
import umap.umap_ as umap

class UMAPReducer:
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42):
        self.reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state)
        self.is_fitted = False

    def fit(self, embeddings):
        """Fit the UMAP reducer to the input embeddings."""
        self.reducer.fit(embeddings)
        self.is_fitted = True
        return self

    def transform(self, embeddings):
        """Transform the embeddings using the fitted UMAP reducer."""
        if not self.is_fitted:
            raise ValueError("UMAPReducer is not fitted yet. Call fit() before transform().")
        return self.reducer.transform(embeddings)

    def fit_transform(self, embeddings):
        """Fit the UMAP reducer and transform the embeddings in one step."""
        self.is_fitted = True
        return self.reducer.fit_transform(embeddings)