import os
import numpy as np
import pandas as pd
import pytest
from src.models.ClassicalClustering import ClassicalClustering

@pytest.fixture
def test_embeddings():
    data_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "data"))
    test_data_path = os.path.join(data_dir, "antique_test_with_embeddings.csv")

    test_df = pd.read_csv(test_data_path, converters={"doc_embedding": lambda x: np.array(eval(x))})
    return np.stack(test_df["doc_embedding"].values)

def test_classical_clustering(test_embeddings):
    clustering = ClassicalClustering(k_range=[5, 10, 15])
    
    labels, medoid_indices = clustering.find_optimal_k(test_embeddings)

    assert labels is not None
    assert medoid_indices is not None
    assert len(labels) == len(test_embeddings)
    assert len(medoid_indices) <= max(clustering.k_range)