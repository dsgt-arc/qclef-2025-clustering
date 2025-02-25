import os
import numpy as np
import pytest
from src.models.QuantumClustering import QuantumClustering

@pytest.fixture
def test_medoids():
    data_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "data"))
    medoid_embeddings_path = os.path.join(data_dir, "medoid_embeddings.npy")
    medoid_indices_path = os.path.join(data_dir, "medoid_indices.npy")

    medoid_embeddings = np.load(medoid_embeddings_path)
    medoid_indices = np.load(medoid_indices_path)

    return medoid_embeddings, medoid_indices

def test_quantum_clustering(test_medoids):
    medoid_embeddings, medoid_indices = test_medoids
    n_clusters = len(medoid_indices)

    quantum_clustering = QuantumClustering(n_clusters)

    qubo_matrix_path = os.path.abspath(os.path.join(os.getcwd(), "..", "data", "qubo_matrix_test.npy"))
    
    quantum_clustering.build_qubo_matrix(medoid_embeddings, medoid_indices, qubo_matrix_path)
    cluster_labels = quantum_clustering.solve_qubo(qubo_matrix_path)

    assert cluster_labels is not None
    assert len(cluster_labels) == len(medoid_indices)