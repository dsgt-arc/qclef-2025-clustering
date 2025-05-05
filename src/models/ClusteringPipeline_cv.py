import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import yaml
from sklearn.model_selection import KFold
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import normalize
from tqdm import tqdm

from sklearn.metrics import pairwise_distances
from src.models.UMAPReducer import UMAPReducer
from src.models.ClassicalClustering import ClassicalClustering
from src.models.QuantumClustering import QuantumClustering, compute_clusters
from box import ConfigBox
from src.plot_utils import plot_embeddings, load_colormap

warnings.filterwarnings("ignore")

def find_best_k_with_qubo(quantum_clustering, medoid_embeddings):
    """Iterate over k_range and find the best k using QUBO clustering."""
    best_k = None
    best_dbi = float("inf")
    best_indices = None

    for k in quantum_clustering.k_range:
        refined_medoid_indices, dbi, _ = quantum_clustering.solve_qubo(medoid_embeddings, k)

        if refined_medoid_indices is not None and dbi < best_dbi:
            best_dbi = dbi
            best_k = k
            best_indices = refined_medoid_indices

    print(f"Final chosen k after QUBO clustering: {best_k}")

    return best_k, best_indices, best_dbi

def compute_dbi(embeddings, cluster_labels):
    """
    Compute Davies-Bouldin Index for the clustering result.
    Lower values indicate better clustering.
    """
    return davies_bouldin_score(embeddings, cluster_labels)

def dcg_at_k(r, k):
    """
    Compute Discounted Cumulative Gain at rank k.
    r: List of relevance scores in rank order
    k: Rank position to calculate DCG at
    """
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0

def ndcg_at_k(r, k):
    """
    Compute Normalized Discounted Cumulative Gain at rank k.
    r: List of relevance scores in rank order
    k: Rank position to calculate NDCG at
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max

def evaluate_retrieval(query_embeddings, doc_embeddings, centroids, cluster_assignments, 
                        qrels_df, doc_ids, k=10, umap_reducer=None):
    """
    Evaluate retrieval performance using nDCG@k.
    
    Args:
        query_embeddings: Embeddings of queries
        doc_embeddings: Embeddings of documents
        centroids: Cluster centroids in reduced space
        cluster_assignments: Cluster assignments for each document
        qrels_df: DataFrame with relevance judgments (query_id, doc_id, relevance)
        doc_ids: List of document IDs corresponding to doc_embeddings
        k: Cutoff for nDCG calculation (default: 10)
        umap_reducer: UMAP reducer to reduce query embeddings to same dimension as centroids
    
    Returns:
        Average nDCG@k across all queries
    """
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    
    if umap_reducer is not None:
        query_embeddings_reduced = umap_reducer.transform(query_embeddings)
        query_embeddings_norm = normalize(query_embeddings_reduced)
    else:
        query_embeddings_norm = normalize(query_embeddings)
    
    centroids_norm = normalize(centroids)
    doc_embeddings_norm = normalize(doc_embeddings)
    
    ndcg_scores = []
    
    query_ids = qrels_df['query_id'].unique()
    
    for q_idx, query_id in enumerate(tqdm(query_ids, desc="Evaluating Queries")):
        query_qrels = qrels_df[qrels_df['query_id'] == query_id]
        
        if len(query_qrels) == 0:
            continue
        
        # Get all document IDs that have relevance judgments for this query
        judged_doc_ids = set(query_qrels['doc_id'].values)
        
        query_embedding = query_embeddings_norm[q_idx].reshape(1, -1)
        
        centroid_similarities = np.dot(query_embedding, centroids_norm.T)[0]
        closest_centroid_idx = np.argmax(centroid_similarities)
        
        cluster_docs_idx = np.where(cluster_assignments == closest_centroid_idx)[0]
        
        if len(cluster_docs_idx) == 0:
            ndcg_scores.append(0.0)
            continue
        
        # Get the document IDs in this cluster
        cluster_doc_ids = [doc_ids[idx] for idx in cluster_docs_idx]
        
        # Find the intersection of cluster documents and judged documents
        # Only consider documents that have relevance judgments for this query
        judged_cluster_docs = set(cluster_doc_ids).intersection(judged_doc_ids)
        
        # If no judged documents in this cluster, skip
        if not judged_cluster_docs:
            ndcg_scores.append(0.0)
            continue
        
        # Get indices of judged documents in the cluster
        judged_cluster_indices = [idx for idx in cluster_docs_idx if doc_ids[idx] in judged_cluster_docs]
        
        if not judged_cluster_indices:
            ndcg_scores.append(0.0)
            continue
        
        cluster_doc_embeddings = doc_embeddings_norm[judged_cluster_indices]
        
        if umap_reducer is not None:
            doc_similarities = np.dot(query_embedding, cluster_doc_embeddings.T)[0]
        else:
            orig_query = normalize(query_embeddings[q_idx].reshape(1, -1))
            orig_docs = normalize(doc_embeddings[judged_cluster_indices])
            doc_similarities = np.dot(orig_query, orig_docs.T)[0]
        
        sorted_indices = np.argsort(-doc_similarities)
        judged_ranked_doc_indices = [judged_cluster_indices[idx] for idx in sorted_indices]
        ranked_doc_ids = [doc_ids[idx] for idx in judged_ranked_doc_indices]
        
        # Compute relevance scores for top-k ranked documents
        relevance_scores = []
        for doc_id in ranked_doc_ids[:k]:
            rel = query_qrels[query_qrels['doc_id'] == doc_id]['relevance'].values
            relevance_scores.append(float(rel[0]) if len(rel) > 0 else 0.0)
        
        ndcg = ndcg_at_k(relevance_scores, k)
        ndcg_scores.append(ndcg)
        
        # For debugging/logging
        if q_idx < 5:  # Print details for the first 5 queries
            judged_docs_found = len(judged_cluster_docs)
            total_judged = len(judged_doc_ids)
            print(f"Query {query_id}: Found {judged_docs_found}/{total_judged} judged docs in cluster {closest_centroid_idx}, nDCG@{k}: {ndcg:.4f}")
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0

def run_cv_evaluation(doc_embeddings, doc_embeddings_reduced, config, query_df, doc_ids, cv_folds=5):
    """
    Run cross-validation to evaluate clustering performance.
    
    Args:
        doc_embeddings: Original document embeddings
        doc_embeddings_reduced: Reduced document embeddings
        config: Configuration object
        query_df: DataFrame with query information
        doc_ids: List of document IDs
        cv_folds: Number of CV folds
        
    Returns:
        cv_results: Dictionary with CV results
    """
    print("\n=== Starting Cross-Validation Evaluation ===")
    
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    dbi_scores = []
    ndcg_scores = []
    
    query_df['query_embeddings'] = [np.fromstring(vec[1:-1], dtype=float, sep=',') for vec in query_df['query_embeddings']]
    query_embeddings = np.stack(query_df["query_embeddings"].values)
    qrels_df = query_df[['query_id', 'doc_id', 'relevance']]
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(doc_embeddings_reduced)):
        print(f"\nFold {fold+1}/{cv_folds}")
        
        X_train = doc_embeddings_reduced[train_idx]
        X_test = doc_embeddings_reduced[test_idx]
        train_doc_ids = [doc_ids[i] for i in train_idx]
        test_doc_ids = [doc_ids[i] for i in test_idx]
        
        clustering = ClassicalClustering(**config.classical_clustering)
        train_labels, medoid_indices = clustering.find_optimal_k(X_train)
        medoid_embeddings = clustering.extract_medoids(X_train, medoid_indices)
        
        quantum_clustering = QuantumClustering(config.quantum_clustering.k_range, medoid_embeddings, config)
        _, refined_medoid_indices, train_dbi = find_best_k_with_qubo(quantum_clustering, medoid_embeddings)
        
        if refined_medoid_indices is not None:
            refined_medoid_embeddings = medoid_embeddings[refined_medoid_indices]
            
            distances = pairwise_distances(X_test, refined_medoid_embeddings)
            test_labels = np.argmin(distances, axis=1)
            
            test_dbi = compute_dbi(X_test, test_labels)
            dbi_scores.append(test_dbi)
            
            umap_reducer = UMAPReducer(random_state=config.classical_clustering.random_state)
            umap_reducer.fit(doc_embeddings[train_idx])
            
            test_qrels = qrels_df[qrels_df['doc_id'].isin(test_doc_ids)]
            
            if not test_qrels.empty:
                test_ndcg = evaluate_retrieval(
                    query_embeddings,
                    X_test,
                    refined_medoid_embeddings,
                    test_labels,
                    test_qrels,
                    test_doc_ids,
                    k=10,
                    umap_reducer=umap_reducer
                )
                ndcg_scores.append(test_ndcg)
                print(f"Test DBI: {test_dbi:.4f}, Test nDCG@10: {test_ndcg:.4f}")
            else:
                print(f"Test DBI: {test_dbi:.4f}, No relevant queries found for test documents")
        else:
            print(f"Fold {fold+1}: Failed to find valid medoids")
    
    if dbi_scores:
        avg_dbi = np.mean(dbi_scores)
        std_dbi = np.std(dbi_scores)
        print(f"\nAverage DBI across {cv_folds} folds: {avg_dbi:.4f} ± {std_dbi:.4f}")
    else:
        avg_dbi = float('inf')
        std_dbi = 0
        print("\nWarning: No valid DBI scores computed")
    
    if ndcg_scores:
        avg_ndcg = np.mean(ndcg_scores)
        std_ndcg = np.std(ndcg_scores)
        print(f"Average nDCG@10 across {cv_folds} folds: {avg_ndcg:.4f} ± {std_ndcg:.4f}")
    else:
        avg_ndcg = 0
        std_ndcg = 0
        print("Warning: No valid nDCG scores computed")
    
    cv_results = {
        'dbi_scores': dbi_scores,
        'ndcg_scores': ndcg_scores,
        'avg_dbi': avg_dbi,
        'std_dbi': std_dbi,
        'avg_ndcg': avg_ndcg,
        'std_ndcg': std_ndcg
    }
    
    print("=== Cross-Validation Evaluation Complete ===")
    
    return cv_results

def run_pipeline(config, colormap_name=None, run_cv=True, cv_folds=5):
    """
    Run the clustering pipeline with a specified colormap and optional cross-validation.
    
    Args:
        config: Configuration object with clustering parameters
        colormap_name: Name of the colormap to use
        run_cv: Whether to run cross-validation evaluation
        cv_folds: Number of CV folds
    """
    np.random.seed(config.classical_clustering.random_state)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))
    colormaps_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "colormaps"))
    output_csv = os.path.join(data_dir, "antique_doc_embeddings.csv")
    query_csv = os.path.join(data_dir, "antique_train_queries.csv")

    umap_plot_path = os.path.join(data_dir, "umap_plot.png")
    kmedoids_plot_path = os.path.join(data_dir, "kmedoids_clusters.png")
    final_clusters_plot_path = os.path.join(data_dir, "final_clusters.png")

    if colormap_name is None:
        colormap_name = "Spectral"
    
    cmap = load_colormap(colormap_name, colormaps_dir)
    
    def parse_embedding(text):
        return np.fromstring(text[1:-1], dtype=float, sep=',')

    # Load document embeddings
    train_df = pd.read_csv(output_csv, converters={"doc_embeddings": parse_embedding})
    doc_embeddings = np.stack(train_df["doc_embeddings"].values)
    doc_ids = train_df['doc_id'].tolist()

    # Load query data for evaluation
    query_df = pd.read_csv(query_csv)
    
    # Dimensionality reduction with UMAP
    umap_reducer = UMAPReducer(random_state=config.classical_clustering.random_state)
    doc_embeddings_reduced = umap_reducer.fit_transform(doc_embeddings)
    np.save(os.path.join(data_dir, "doc_embeddings_reduced.npy"), doc_embeddings_reduced)
    plot_embeddings(doc_embeddings_reduced, title="UMAP Reduction", save_path=umap_plot_path, cmap=cmap)

    # If using cross-validation, evaluate clustering performance
    cv_results = None
    if run_cv:
        cv_results = run_cv_evaluation(
            doc_embeddings, 
            doc_embeddings_reduced, 
            config,
            query_df,
            doc_ids,
            cv_folds=cv_folds
        )
        
        # Save CV results
        np.save(os.path.join(data_dir, "cv_results.npy"), cv_results)
    
    # Run classical k-medoids clustering on full dataset
    clustering = ClassicalClustering(**config.classical_clustering)
    kmedoid_labels, medoid_indices = clustering.find_optimal_k(doc_embeddings_reduced)

    print(f"Classical K-Medoids Labels: {kmedoid_labels}")
    print(f"Classical Medoid Indices: {medoid_indices}")

    medoid_embeddings = clustering.extract_medoids(doc_embeddings_reduced, medoid_indices)
    np.save(os.path.join(data_dir, "medoid_embeddings.npy"), medoid_embeddings)
    np.save(os.path.join(data_dir, "medoid_indices.npy"), medoid_indices)
    plot_embeddings(doc_embeddings_reduced, labels=kmedoid_labels, medoids=medoid_embeddings,
                title=f"K-Medoids Clustering (k={clustering.best_k})", save_path=kmedoids_plot_path, cmap=cmap)

    # Run quantum clustering with fixed parameters from config
    quantum_clustering = QuantumClustering(config.quantum_clustering.k_range, medoid_embeddings, config)
    best_k, refined_medoid_indices, best_dbi = find_best_k_with_qubo(quantum_clustering, medoid_embeddings)

    if refined_medoid_indices is not None:
        print(f"Quantum-Refined Medoid Indices: {refined_medoid_indices}") 
        refined_medoid_indices_of_embeddings = medoid_indices[refined_medoid_indices]
        refined_medoid_embeddings = doc_embeddings_reduced[refined_medoid_indices_of_embeddings] 
        
        print(f"Before QUBO: Assignments to Classical Medoids: {compute_clusters(doc_embeddings_reduced, medoid_indices)}")
        print(f"Refined Medoid Indices Type: {type(refined_medoid_indices)}, Shape: {refined_medoid_indices.shape}")
        print(f"Refined Medoid Embeddings Shape: {refined_medoid_embeddings.shape}")
        print(f"Before Assigning Clusters, Medoid Indices: {refined_medoid_indices_of_embeddings}")
        final_cluster_labels = compute_clusters(doc_embeddings_reduced, refined_medoid_indices_of_embeddings)
        print(f"After QUBO: Assignments to Quantum Medoids: {final_cluster_labels}")

    else:
        raise ValueError("QUBO Solver failed to find valid medoids.")
    
    print(f"Final chosen k after QUBO clustering: {best_k}")
    print(f"Final DBI: {best_dbi:.4f}")

    # Save results
    results = {
        'final_k': best_k,
        'final_dbi': best_dbi,
        'doc_ids': doc_ids,
        'cluster_labels': final_cluster_labels.tolist(),
        'config': config,
        'cv_results': cv_results
    }
    
    # Save as numpy and json for different use cases
    np.save(os.path.join(data_dir, "final_quantum_clusters.npy"), final_cluster_labels)
    np.save(os.path.join(data_dir, "refined_medoid_embeddings.npy"), refined_medoid_embeddings)
    np.save(os.path.join(data_dir, "refined_medoid_indices.npy"), refined_medoid_indices)
    np.save(os.path.join(data_dir, "clustering_results.npy"), results)
    
    # Save a mapping of doc_id to cluster for easy reference
    cluster_mapping = pd.DataFrame({
        'doc_id': doc_ids,
        'cluster': final_cluster_labels
    })
    cluster_mapping.to_csv(os.path.join(data_dir, "doc_clusters.csv"), index=False)

    print(f"Final Quantum-Refined Medoid Embeddings:\n {refined_medoid_embeddings}")
    print(f"Unique Cluster Assignments: {np.unique(final_cluster_labels)}")

    # Evaluate nDCG@10 on the full dataset
    print("\nEvaluating nDCG@10 on full dataset...")
    try:
        # Load query data with error handling
        query_df['query_embeddings'] = query_df['query_embeddings'].apply(
            lambda x: np.fromstring(x[1:-1], dtype=float, sep=',') if isinstance(x, str) else x
        )
        
        # Filter out any rows where query embeddings couldn't be parsed
        valid_queries = query_df[query_df['query_embeddings'].apply(lambda x: isinstance(x, np.ndarray) and len(x) > 0)]
        
        if len(valid_queries) > 0:
            # Check if all embeddings have the same shape
            first_shape = len(valid_queries['query_embeddings'].iloc[0])
            valid_queries = valid_queries[valid_queries['query_embeddings'].apply(lambda x: len(x) == first_shape)]
            
            if len(valid_queries) > 0:
                query_embeddings = np.stack(valid_queries["query_embeddings"].values)
                qrels_df = valid_queries[['query_id', 'doc_id', 'relevance']]
                
                ndcg_10 = evaluate_retrieval(
                    query_embeddings,
                    doc_embeddings_reduced,
                    refined_medoid_embeddings,
                    final_cluster_labels,
                    qrels_df,
                    doc_ids,
                    k=10,
                    umap_reducer=umap_reducer
                )
                print(f"Full dataset nDCG@10: {ndcg_10:.4f}")
            else:
                print("No valid queries with consistent embedding dimensions found.")
                ndcg_10 = 0.0
        else:
            print("No valid query embeddings found.")
            ndcg_10 = 0.0
    except Exception as e:
        print(f"Error evaluating nDCG@10 on full dataset: {str(e)}")
        ndcg_10 = 0.0

    # Plot final clusters
    plot_embeddings(doc_embeddings_reduced,
                labels=final_cluster_labels,
                medoids=medoid_embeddings,
                refined_medoids=refined_medoid_embeddings,
                title="Final Quantum Cluster Assignments",
                save_path=final_clusters_plot_path,
                cmap=cmap)

    print(f"Saved final quantum cluster plot at: {final_clusters_plot_path}")
    plt.show()

    # Print evaluation summary
    print("\n=== Clustering Evaluation Summary ===")
    print(f"Number of clusters: {best_k}")
    print(f"Davies-Bouldin Index: {best_dbi:.4f}")
    print(f"Full dataset nDCG@10: {ndcg_10:.4f}")
    if cv_results:
        print(f"Cross-validation DBI: {cv_results['avg_dbi']:.4f} ± {cv_results['std_dbi']:.4f}")
        print(f"Cross-validation nDCG@10: {cv_results['avg_ndcg']:.4f} ± {cv_results['std_ndcg']:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run clustering pipeline with cross-validation')
    parser.add_argument('--colormap', type=str, default='Spectral', 
                        help='Colormap to use (file in colormaps dir or matplotlib name)')
    parser.add_argument('--no_cv', action='store_true',
                        help='Disable cross-validation evaluation')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Number of CV folds')
    
    args = parser.parse_args()

    with open("config/kmedoids.yml", "r") as file:
        config = ConfigBox(yaml.safe_load(file))

    run_pipeline(
        config, 
        colormap_name=args.colormap,
        run_cv=not args.no_cv,
        cv_folds=args.cv_folds
    )