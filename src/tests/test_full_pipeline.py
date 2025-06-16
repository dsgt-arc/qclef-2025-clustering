import os
import pytest
from src.models.ClusteringPipeline import ClusteringPipeline

def test_full_pipeline():
    pipeline = ClusteringPipeline()
    pipeline.run()

    data_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "data"))
    expected_outputs = [
        "doc_embeddings_reduced.npy",
        "medoid_embeddings.npy",
        "medoid_indices.npy",
        "cluster_labels.npy",
        "qubo_matrix.npy",
        "quantum_cluster_labels.npy",
        "final_quantum_clusters.npy"
    ]

    for file in expected_outputs:
        assert os.path.exists(os.path.join(data_dir, file)), f"Missing output file: {file}"