import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import yaml
import datetime
import json
from sklearn.model_selection import KFold
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import normalize
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

from sklearn.metrics import pairwise_distances
from src.models.UMAPReducer import UMAPReducer
from src.models.ClassicalClustering import ClassicalClustering
from src.models.HDBSCANClustering import HDBSCANClustering
from src.models.GMMClustering import GMMClustering
from src.models.HDBSCANGMMClustering import HDBSCANGMMClustering
from src.models.multi_membership import create_multi_membership_assignments
from src.models.QuantumClustering import QuantumClustering, compute_clusters
from src.plot_utils import plot_embeddings, load_colormap, plot_cluster_spectrum
from src.plot_utils import plot_merged_clusters, plot_top_merged_clusters
from box import ConfigBox

warnings.filterwarnings("ignore")

def save_initial_clustering_results(initial_labels, doc_ids, run_output_dir, method_name, timestamp):
    """
    Save information about initial clustering results.
    
    Args:
        initial_labels: Cluster labels from initial clustering
        doc_ids: List of document IDs
        run_output_dir: Directory to save results
        method_name: Name of the clustering method used
        timestamp: Timestamp string for filenames
    """
    # Save initial cluster assignments
    initial_cluster_mapping = pd.DataFrame({
        'doc_id': doc_ids,
        'initial_cluster': initial_labels
    })
    initial_cluster_mapping.to_csv(os.path.join(run_output_dir, f"initial_doc_clusters.csv"), index=False)
    
    # Compute and save initial cluster size distribution
    initial_cluster_sizes = pd.Series(initial_labels).value_counts().sort_index()
    initial_cluster_sizes_df = pd.DataFrame({
        'cluster_id': initial_cluster_sizes.index,
        'size': initial_cluster_sizes.values
    })
    initial_cluster_sizes_df.to_csv(os.path.join(run_output_dir, f"initial_cluster_sizes.csv"), index=False)
    
    # Print cluster distribution summary
    print(f"\nInitial {method_name.upper()} cluster distribution summary:")
    print(f"Number of clusters: {len(initial_cluster_sizes)}")
    print(f"Min cluster size: {initial_cluster_sizes.min()}")
    print(f"Max cluster size: {initial_cluster_sizes.max()}")
    print(f"Mean cluster size: {initial_cluster_sizes.mean():.2f}")
    print(f"Median cluster size: {initial_cluster_sizes.median():.2f}")
    
    # Save visual representation of initial distribution
    plt.figure(figsize=(12, 6))
    
    # Sort initial clusters by size for better visualization
    sorted_indices = np.argsort(-initial_cluster_sizes.values)
    sorted_clusters = initial_cluster_sizes.index[sorted_indices]
    sorted_sizes = initial_cluster_sizes.values[sorted_indices]
    
    plt.bar(range(len(sorted_clusters)), sorted_sizes)
    plt.title(f"Initial {method_name.upper()} Cluster Size Distribution")
    plt.xlabel("Cluster ID (sorted by size)")
    plt.ylabel("Number of Documents")
    
    # Add a reference line for mean size
    plt.axhline(y=initial_cluster_sizes.mean(), color='r', linestyle='-', 
                label=f'Mean Size: {initial_cluster_sizes.mean():.1f}')
    plt.legend()
    
    plt.tight_layout()
    initial_dist_plot_path = os.path.join(run_output_dir, f"initial_cluster_distribution_{timestamp}.png")
    plt.savefig(initial_dist_plot_path)
    plt.close()
    print(f"Saved initial cluster distribution plot at: {initial_dist_plot_path}")
    
    # Calculate percentage of documents in top clusters
    top_clusters = initial_cluster_sizes.nlargest(10)
    total_docs = sum(initial_cluster_sizes)
    top_percentage = sum(top_clusters) / total_docs * 100
    
    print(f"Top 10 clusters contain {top_percentage:.2f}% of all documents")
    
    # Save top clusters to file
    top_clusters_df = pd.DataFrame({
        'cluster_id': top_clusters.index,
        'size': top_clusters.values,
        'percentage': (top_clusters.values / total_docs * 100).round(2)
    })
    top_clusters_df.to_csv(os.path.join(run_output_dir, f"initial_top_clusters.csv"), index=False)
    
    return initial_cluster_sizes

def compare_cluster_distributions(initial_labels, final_labels, run_output_dir, method_name, timestamp):
    """
    Create comparison visualizations between initial and final cluster distributions.
    
    Args:
        initial_labels: Initial cluster labels
        final_labels: Final cluster labels after refinement
        run_output_dir: Directory to save results
        method_name: Name of the clustering method
        timestamp: Timestamp string for filenames
    """
    try:
        initial_sizes = pd.Series(initial_labels).value_counts()
        final_sizes = pd.Series(final_labels).value_counts()
        
        plt.figure(figsize=(12, 10))
        
        # Sort initial clusters by size for better visualization
        sorted_init_indices = np.argsort(-initial_sizes.values)
        sorted_init_clusters = initial_sizes.index[sorted_init_indices]
        sorted_init_sizes = initial_sizes.values[sorted_init_indices]
        
        plt.subplot(2, 1, 1)
        plt.bar(range(len(sorted_init_clusters)), sorted_init_sizes)
        plt.title(f"Initial {len(initial_sizes)} Clusters Size Distribution")
        plt.xlabel("Cluster ID (sorted by size)")
        plt.ylabel("Number of Documents")
        plt.yscale('log')
        
        # Final clusters
        plt.subplot(2, 1, 2)
        plt.bar(final_sizes.index, final_sizes.values)
        plt.title(f"Final {len(final_sizes)} Clusters Size Distribution")
        plt.xlabel("Cluster ID")
        plt.ylabel("Number of Documents")
        plt.yscale('log')
        
        plt.tight_layout()
        comparison_plot_path = os.path.join(run_output_dir, f"cluster_size_comparison_{timestamp}.png")
        plt.savefig(comparison_plot_path)
        plt.close()
        print(f"Saved cluster size comparison plot at: {comparison_plot_path}")
        
        # Create statistics comparison table
        initial_stats = {
            'num_clusters': len(initial_sizes),
            'min_size': initial_sizes.min(),
            'max_size': initial_sizes.max(),
            'mean_size': initial_sizes.mean(),
            'median_size': initial_sizes.median(),
            'top10_pct': (initial_sizes.nlargest(10).sum() / initial_sizes.sum() * 100),
            'top3_pct': (initial_sizes.nlargest(3).sum() / initial_sizes.sum() * 100)
        }
        
        final_stats = {
            'num_clusters': len(final_sizes),
            'min_size': final_sizes.min(),
            'max_size': final_sizes.max(),
            'mean_size': final_sizes.mean(),
            'median_size': final_sizes.median(),
            'top10_pct': (final_sizes.nlargest(10).sum() / final_sizes.sum() * 100),
            'top3_pct': (final_sizes.nlargest(3).sum() / final_sizes.sum() * 100)
        }
        
        stats_df = pd.DataFrame({
            'Initial': initial_stats,
            'Final': final_stats
        })
        
        stats_df.to_csv(os.path.join(run_output_dir, f"cluster_stats_comparison.csv"))
        
        print("\n=== Cluster Distribution Comparison ===")
        print(f"Initial clusters: {initial_stats['num_clusters']}, Final clusters: {final_stats['num_clusters']}")
        print(f"Initial top 3 clusters: {initial_stats['top3_pct']:.2f}% of documents")
        print(f"Final top 3 clusters: {final_stats['top3_pct']:.2f}% of documents")
        
    except Exception as e:
        print(f"Error creating cluster distribution comparison: {str(e)}")

def track_cluster_correspondence(initial_labels, final_labels):
    """
    Track how clusters change between initial and final clustering.
    
    Args:
        initial_labels: Cluster labels from initial clustering
        final_labels: Cluster labels after refinement
        
    Returns:
        Dictionary mapping final cluster IDs to most corresponding initial cluster IDs
    """
    correspondence = {}
    
    overlap_counts = defaultdict(lambda: defaultdict(int))
    
    for i, (init_label, final_label) in enumerate(zip(initial_labels, final_labels)):
        overlap_counts[final_label][init_label] += 1
    
    for final_cluster in np.unique(final_labels):
        if len(overlap_counts[final_cluster]) > 0:
            init_cluster = max(overlap_counts[final_cluster].items(), 
                              key=lambda x: x[1])[0]
            correspondence[final_cluster] = init_cluster
        else:
            correspondence[final_cluster] = final_cluster
    
    return correspondence

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
                        qrels_df, doc_ids, k=10, umap_reducer=None, multi_cluster_assignments=None):
    """
    Evaluate retrieval performance using nDCG@k and relevant coverage, with multi-membership support.
    
    Args:
        query_embeddings: Embeddings of queries
        doc_embeddings: Embeddings of documents
        centroids: Cluster centroids in reduced space
        cluster_assignments: Cluster assignments for each document
        qrels_df: DataFrame with relevance judgments (query_id, doc_id, relevance)
        doc_ids: List of document IDs corresponding to doc_embeddings
        k: Cutoff for nDCG calculation (default: 10)
        umap_reducer: UMAP reducer to reduce query embeddings to same dimension as centroids
        multi_cluster_assignments: DataFrame with multi-membership assignments (if using multi-membership)
    
    Returns:
        Dictionary with:
        - 'ndcg': Average nDCG@k across all queries (hard assignment)
        - 'coverage': Average relevant coverage (hard assignment)
        - 'ndcg_multi': Average nDCG@k with multi-membership assignments (if applicable)
        - 'coverage_multi': Average relevant coverage with multi-membership (if applicable)
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
    coverage_scores = []
    
    ndcg_multi_scores = []
    coverage_multi_scores = []
    has_multi = multi_cluster_assignments is not None
    
    query_ids = qrels_df['query_id'].unique()
    
    for q_idx, query_id in enumerate(tqdm(query_ids, desc="Evaluating Queries")):
        query_qrels = qrels_df[qrels_df['query_id'] == query_id]
        
        if len(query_qrels) == 0:
            continue
        
        judged_doc_ids = set(query_qrels['doc_id'].values)
        relevant_doc_ids = set(query_qrels[query_qrels['relevance'] > 0]['doc_id'].values)
        
        if not relevant_doc_ids:
            continue
            
        query_embedding = query_embeddings_norm[q_idx].reshape(1, -1)
        
        centroid_similarities = np.dot(query_embedding, centroids_norm.T)[0]
        closest_centroid_idx = np.argmax(centroid_similarities)
        
        cluster_docs_idx = np.where(cluster_assignments == closest_centroid_idx)[0]
        
        if len(cluster_docs_idx) == 0:
            ndcg_scores.append(0.0)
            coverage_scores.append(0.0)
            if has_multi:
                ndcg_multi_scores.append(0.0)
                coverage_multi_scores.append(0.0)
            continue
        
        cluster_doc_ids = [doc_ids[idx] for idx in cluster_docs_idx]
        
        retrieved_relevant = relevant_doc_ids.intersection(set(cluster_doc_ids))
        coverage = len(retrieved_relevant) / len(relevant_doc_ids) if relevant_doc_ids else 0.0
        coverage_scores.append(coverage)
        
        cluster_doc_embeddings = doc_embeddings_norm[cluster_docs_idx]
        
        if umap_reducer is not None:
            doc_similarities = np.dot(query_embedding, cluster_doc_embeddings.T)[0]
        else:
            orig_query = normalize(query_embeddings[q_idx].reshape(1, -1))
            orig_docs = normalize(doc_embeddings[cluster_docs_idx])
            doc_similarities = np.dot(orig_query, orig_docs.T)[0]
        
        sorted_indices = np.argsort(-doc_similarities)
        ranked_cluster_indices = [cluster_docs_idx[idx] for idx in sorted_indices]
        ranked_doc_ids = [doc_ids[idx] for idx in ranked_cluster_indices]
        
        relevance_scores = []
        for doc_id in ranked_doc_ids[:k]:
            rel_values = query_qrels[query_qrels['doc_id'] == doc_id]['relevance'].values
            if len(rel_values) > 0:
                relevance_scores.append(float(rel_values[0]))
            else:
                relevance_scores.append(0.0)
        
        ndcg = ndcg_at_k(relevance_scores, k)
        ndcg_scores.append(ndcg)
        
        if has_multi:
            multi_docs_idx = []
            for doc_idx, row in enumerate(multi_cluster_assignments.iloc):
                if closest_centroid_idx in row['multi_membership']:
                    multi_docs_idx.append(doc_idx)
            
            if len(multi_docs_idx) == 0:
                ndcg_multi_scores.append(0.0)
                coverage_multi_scores.append(0.0)
                continue
            
            multi_doc_ids = [doc_ids[idx] for idx in multi_docs_idx]
            
            multi_retrieved_relevant = relevant_doc_ids.intersection(set(multi_doc_ids))
            multi_coverage = len(multi_retrieved_relevant) / len(relevant_doc_ids) if relevant_doc_ids else 0.0
            coverage_multi_scores.append(multi_coverage)
            
            multi_doc_embeddings = doc_embeddings_norm[multi_docs_idx]
            
            if umap_reducer is not None:
                multi_doc_similarities = np.dot(query_embedding, multi_doc_embeddings.T)[0]
            else:
                orig_query = normalize(query_embeddings[q_idx].reshape(1, -1))
                orig_multi_docs = normalize(doc_embeddings[multi_docs_idx])
                multi_doc_similarities = np.dot(orig_query, orig_multi_docs.T)[0]
            
            multi_sorted_indices = np.argsort(-multi_doc_similarities)
            multi_ranked_indices = [multi_docs_idx[idx] for idx in multi_sorted_indices]
            multi_ranked_doc_ids = [doc_ids[idx] for idx in multi_ranked_indices]
            
            multi_relevance_scores = []
            for doc_id in multi_ranked_doc_ids[:k]:
                rel_values = query_qrels[query_qrels['doc_id'] == doc_id]['relevance'].values
                if len(rel_values) > 0:
                    multi_relevance_scores.append(float(rel_values[0]))
                else:
                    multi_relevance_scores.append(0.0)
            
            ndcg_multi = ndcg_at_k(multi_relevance_scores, k)
            ndcg_multi_scores.append(ndcg_multi)
        
        if q_idx < 5:
            judged_docs_found = len(set(cluster_doc_ids).intersection(judged_doc_ids))
            total_judged = len(judged_doc_ids)
            print(f"Query {query_id}: Found {judged_docs_found}/{total_judged} judged docs, "
                  f"{len(retrieved_relevant)}/{len(relevant_doc_ids)} relevant docs in cluster {closest_centroid_idx}, "
                  f"nDCG@{k}: {ndcg:.4f}, Coverage: {coverage:.4f}")
            
            if has_multi:
                multi_judged_found = len(set(multi_doc_ids).intersection(judged_doc_ids))
                print(f"  Multi-Membership: Found {multi_judged_found}/{total_judged} judged docs, "
                      f"{len(multi_retrieved_relevant)}/{len(relevant_doc_ids)} relevant docs, "
                      f"nDCG@{k}: {ndcg_multi:.4f}, Coverage: {multi_coverage:.4f}")
    
    results = {
        'ndcg': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        'coverage': np.mean(coverage_scores) if coverage_scores else 0.0
    }
    
    if has_multi:
        results['ndcg_multi'] = np.mean(ndcg_multi_scores) if ndcg_multi_scores else 0.0
        results['coverage_multi'] = np.mean(coverage_multi_scores) if coverage_multi_scores else 0.0
    
    return results


def run_cv_evaluation(doc_embeddings, doc_embeddings_reduced, config, query_df, doc_ids, 
                      cv_folds=5, clustering_method='classical'):
    """
    Run cross-validation to evaluate clustering performance.
    
    Args:
        doc_embeddings: Original document embeddings
        doc_embeddings_reduced: Reduced document embeddings
        config: Configuration object
        query_df: DataFrame with query information
        doc_ids: List of document IDs
        cv_folds: Number of CV folds
        clustering_method: Which clustering method to use ('classical', 'hdbscan', 'gmm', or 'hdbscan-gmm')
        
    Returns:
        cv_results: Dictionary with CV results
    """
    print(f"\n=== Starting Cross-Validation Evaluation with {clustering_method.upper()} clustering ===")
    
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    dbi_scores = []
    ndcg_scores = []
    coverage_scores = []
    ndcg_multi_scores = []
    coverage_multi_scores = []
    
    query_df['query_embeddings'] = query_df['query_embeddings'].apply(
        lambda x: np.fromstring(x[1:-1], dtype=float, sep=',') if isinstance(x, str) else x
    )
    
    valid_queries = query_df[query_df['query_embeddings'].apply(lambda x: isinstance(x, np.ndarray) and len(x) > 0)]
    
    if len(valid_queries) > 0:
        first_shape = len(valid_queries['query_embeddings'].iloc[0])
        valid_queries = valid_queries[valid_queries['query_embeddings'].apply(lambda x: len(x) == first_shape)]
        
        if len(valid_queries) > 0:
            query_embeddings = np.stack(valid_queries["query_embeddings"].values)
            qrels_df = valid_queries[['query_id', 'doc_id', 'relevance']]
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(doc_embeddings_reduced)):
        print(f"\nFold {fold+1}/{cv_folds}")
        
        X_train = doc_embeddings_reduced[train_idx]
        X_test = doc_embeddings_reduced[test_idx]
        train_doc_ids = [doc_ids[i] for i in train_idx]
        test_doc_ids = [doc_ids[i] for i in test_idx]
        
        has_probabilities = clustering_method in ['gmm', 'hdbscan-gmm']
        multi_membership_df = None
        
        if clustering_method == 'hdbscan':
            clustering = HDBSCANClustering(**config.hdbscan_clustering)
            train_labels, medoid_indices = clustering.find_optimal_k(X_train)
            
            if -1 in train_labels:
                train_labels = clustering.handle_noise_points(X_train, train_labels, medoid_indices)
        elif clustering_method == 'gmm':
            clustering = GMMClustering(**config.gmm_clustering)
            train_labels, medoid_indices = clustering.find_optimal_k(X_train)
        elif clustering_method == 'hdbscan-gmm':
            clustering = HDBSCANGMMClustering(**config.hdbscan_gmm_clustering)
            train_labels, medoid_indices = clustering.find_optimal_k(X_train)
        else:
            clustering = ClassicalClustering(**config.classical_clustering)
            train_labels, medoid_indices = clustering.find_optimal_k(X_train)
            
        medoid_embeddings = clustering.extract_medoids(X_train, medoid_indices)
        
        # Determine refinement method based on config
        is_annealing = getattr(config.general, 'annealing', True)
        is_second_stage = (not is_annealing and 
                         clustering_method == 'classical' and 
                         hasattr(config, 'second_stage_clustering') and 
                         getattr(config.second_stage_clustering, 'enabled', False))
        
        if is_annealing:
            # Use quantum annealing
            quantum_clustering = QuantumClustering(config.quantum_clustering.k_range, medoid_embeddings, config)
            _, refined_medoid_indices, train_dbi = find_best_k_with_qubo(quantum_clustering, medoid_embeddings)
            refinement_method = "quantum"
            
            if refined_medoid_indices is not None:
                refined_medoid_indices_of_embeddings = medoid_indices[refined_medoid_indices]
                refined_medoid_embeddings = X_train[refined_medoid_indices_of_embeddings]
            else:
                print(f"Fold {fold+1}: Failed to find valid medoids")
                continue
                
        elif is_second_stage:
            # Use second-stage clustering
            final_cluster_labels, refined_medoid_indices_of_embeddings, best_k, train_dbi = clustering.cluster_medoids(
                X_train,
                medoid_embeddings,
                medoid_indices,
                config
            )
            refinement_method = "second_stage"
            refined_medoid_embeddings = X_train[refined_medoid_indices_of_embeddings]
            
        else:
            # No refinement
            refined_medoid_indices_of_embeddings = medoid_indices
            refined_medoid_embeddings = X_train[refined_medoid_indices_of_embeddings]
            train_dbi = compute_dbi(X_train, train_labels)
            refinement_method = "none"
        
        distances = pairwise_distances(X_test, refined_medoid_embeddings)
        test_labels = np.argmin(distances, axis=1)
        
        color_correspondence = track_cluster_correspondence(train_labels, test_labels)
        
        test_dbi = compute_dbi(X_test, test_labels)
        dbi_scores.append(test_dbi)
        
        umap_reducer = UMAPReducer(random_state=config.classical_clustering.random_state)
        umap_reducer.fit(doc_embeddings[train_idx])
        
        test_qrels = qrels_df[qrels_df['doc_id'].isin(test_doc_ids)]
        
        if has_probabilities and hasattr(clustering, 'membership_probs'):
            threshold = 0.2
            
            n_quantum_clusters = len(np.unique(test_labels))
            n_test_docs = len(test_doc_ids)
            
            quantum_probs = np.zeros((n_test_docs, n_quantum_clusters))
            
            component_to_quantum = {}
            for comp_idx in range(clustering.membership_probs.shape[1]):
                counts = np.zeros(n_quantum_clusters)
                for doc_idx, prob in enumerate(clustering.membership_probs[train_idx], 0):
                    if prob > 0.1:
                        quantum_cluster = test_labels[doc_idx]
                        counts[quantum_cluster] += prob
                
                if np.sum(counts) > 0:
                    component_to_quantum[comp_idx] = np.argmax(counts)
            
            for doc_idx in range(n_test_docs):
                doc_probs = clustering.membership_probs[test_idx[doc_idx], :]
                
                for comp_idx, quantum_idx in component_to_quantum.items():
                    quantum_probs[doc_idx, quantum_idx] += doc_probs[comp_idx]
            
            row_sums = quantum_probs.sum(axis=1, keepdims=True)
            quantum_probs = np.divide(quantum_probs, row_sums, 
                                     out=np.zeros_like(quantum_probs), 
                                     where=row_sums != 0)
            
            data = {
                'doc_id': test_doc_ids,
                'primary_cluster': test_labels
            }
            
            multi_memberships = []
            for doc_idx in range(n_test_docs):
                doc_clusters = np.where(quantum_probs[doc_idx, :] >= threshold)[0]
                if len(doc_clusters) == 0:
                    doc_clusters = [np.argmax(quantum_probs[doc_idx, :])]
                multi_memberships.append(doc_clusters.tolist())
            
            data['multi_membership'] = multi_memberships
            data['membership_count'] = [len(clusters) for clusters in multi_memberships]
            
            multi_membership_df = pd.DataFrame(data)
        
        if not test_qrels.empty:
            eval_results = evaluate_retrieval(
                query_embeddings,
                X_test,
                refined_medoid_embeddings,
                test_labels,
                test_qrels,
                test_doc_ids,
                k=10,
                umap_reducer=umap_reducer,
                multi_cluster_assignments=multi_membership_df
            )
            
            ndcg_scores.append(eval_results['ndcg'])
            coverage_scores.append(eval_results['coverage'])
            
            if 'ndcg_multi' in eval_results:
                ndcg_multi_scores.append(eval_results['ndcg_multi'])
                coverage_multi_scores.append(eval_results['coverage_multi'])
                print(f"Test DBI: {test_dbi:.4f}, Test nDCG@10: {eval_results['ndcg']:.4f}, Coverage: {eval_results['coverage']:.4f}, Multi-nDCG@10: {eval_results['ndcg_multi']:.4f}, Multi-Coverage: {eval_results['coverage_multi']:.4f}")
            else:
                print(f"Test DBI: {test_dbi:.4f}, Test nDCG@10: {eval_results['ndcg']:.4f}, Coverage: {eval_results['coverage']:.4f}")
        else:
            print(f"Test DBI: {test_dbi:.4f}, No relevant queries found for test documents")
    
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
    
    if coverage_scores:
        avg_cov = np.mean(coverage_scores)
        std_cov = np.std(coverage_scores)
        print(f"Average Coverage across {cv_folds} folds: {avg_cov:.4f} ± {std_cov:.4f}")
    else:
        avg_cov = 0
        std_cov = 0
    
    if ndcg_multi_scores:
        avg_ndcg_multi = np.mean(ndcg_multi_scores)
        std_ndcg_multi = np.std(ndcg_multi_scores)
        print(f"Average Multi-nDCG@10 across {cv_folds} folds: {avg_ndcg_multi:.4f} ± {std_ndcg_multi:.4f}")
        
        avg_cov_multi = np.mean(coverage_multi_scores)
        std_cov_multi = np.std(coverage_multi_scores)
        print(f"Average Multi-Coverage across {cv_folds} folds: {avg_cov_multi:.4f} ± {std_cov_multi:.4f}")
    else:
        avg_ndcg_multi = 0
        std_ndcg_multi = 0
        avg_cov_multi = 0
        std_cov_multi = 0
    
    cv_results = {
        'dbi_scores': dbi_scores,
        'ndcg_scores': ndcg_scores,
        'coverage_scores': coverage_scores,
        'ndcg_multi_scores': ndcg_multi_scores,
        'coverage_multi_scores': coverage_multi_scores,
        'avg_dbi': avg_dbi,
        'std_dbi': std_dbi,
        'avg_ndcg': avg_ndcg,
        'std_ndcg': std_ndcg,
        'avg_coverage': avg_cov,
        'std_coverage': std_cov,
        'avg_ndcg_multi': avg_ndcg_multi,
        'std_ndcg_multi': std_ndcg_multi,
        'avg_coverage_multi': avg_cov_multi,
        'std_coverage_multi': std_cov_multi
    }
    
    print("=== Cross-Validation Evaluation Complete ===")
    
    return cv_results


def create_hybrid_probabilistic_assignments(doc_ids, initial_probs, quantum_labels, quantum_medoid_indices, data_dir, prefix=''):
    """
    Create hybrid probabilistic assignments combining initial probabilities with quantum clustering.
    
    Args:
        doc_ids: List of document IDs
        initial_probs: Initial membership probabilities (from GMM or HDBSCAN-GMM)
        quantum_labels: Quantum-refined cluster labels
        quantum_medoid_indices: Quantum-refined medoid indices
        data_dir: Data directory to save results
        prefix: Prefix for output files (empty, 'gmm', or 'hdbscan-gmm')
    """
    prefix = f"{prefix}_" if prefix else ""
    print(f"Creating hybrid probabilistic assignments with {prefix}probabilities...")
    
    n_components = initial_probs.shape[1]
    
    n_quantum_clusters = len(np.unique(quantum_labels))
    
    component_to_quantum = {}
    
    for comp_idx in range(n_components):
        counts = np.zeros(n_quantum_clusters)
        for doc_idx, prob in enumerate(initial_probs[:, comp_idx]):
            if prob > 0.2:
                quantum_cluster = quantum_labels[doc_idx]
                counts[quantum_cluster] += prob
        
        if np.sum(counts) > 0:
            component_to_quantum[comp_idx] = np.argmax(counts)
    
    n_docs = len(doc_ids)
    hybrid_probs = np.zeros((n_docs, n_quantum_clusters))
    
    for doc_idx in range(n_docs):
        doc_probs = initial_probs[doc_idx, :]
        
        for comp_idx, quantum_idx in component_to_quantum.items():
            hybrid_probs[doc_idx, quantum_idx] += doc_probs[comp_idx]
    
    row_sums = hybrid_probs.sum(axis=1, keepdims=True)
    hybrid_probs = np.divide(hybrid_probs, row_sums, out=np.zeros_like(hybrid_probs), where=row_sums != 0)
    
    np.save(os.path.join(data_dir, f"{prefix}hybrid_cluster_probs.npy"), hybrid_probs)
    
    data = {
        'doc_id': doc_ids
    }
    
    for cluster_id in range(n_quantum_clusters):
        data[f'quantum_cluster_{cluster_id}_prob'] = hybrid_probs[:, cluster_id]
    
    data['most_likely_cluster'] = np.argmax(hybrid_probs, axis=1)
    data['quantum_cluster'] = quantum_labels
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(data_dir, f"{prefix}hybrid_cluster_membership.csv"), index=False)
    
    print(f"Saved hybrid probabilistic assignments to {os.path.join(data_dir, f'{prefix}hybrid_cluster_membership.csv')}")


def run_pipeline(config, colormap_name=None, run_cv=True, cv_folds=5, clustering_method='classical', multi_membership=False, threshold=0.2):
    """
    Run the clustering pipeline with a specified colormap and optional cross-validation.
    
    Args:
        config: Configuration object with clustering parameters
        colormap_name: Name of the colormap to use
        run_cv: Whether to run cross-validation evaluation
        cv_folds: Number of CV folds
        clustering_method: Which clustering method to use ('classical', 'hdbscan', 'gmm', 'hdbscan-gmm')
        multi_membership: Whether to create multi-membership assignments
        threshold: Probability threshold for multi-membership (default: 0.2)
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine refinement method based on config
    is_annealing = getattr(config.general, 'annealing', True)
    is_second_stage = (not is_annealing and 
                     clustering_method == 'classical' and 
                     hasattr(config, 'second_stage_clustering') and 
                     getattr(config.second_stage_clustering, 'enabled', False))
    
    if is_annealing:
        refinement_method = "quantum"
    elif is_second_stage:
        refinement_method = "second_stage"
    else:
        refinement_method = "none"
    
    run_info = {
        'timestamp': timestamp,
        'clustering_method': clustering_method,
        'colormap': colormap_name if colormap_name else "Spectral",
        'cv_enabled': run_cv,
        'cv_folds': cv_folds if run_cv else None,
        'multi_membership': multi_membership,
        'threshold': threshold if multi_membership else None,
        'annealing': is_annealing,
        'second_stage': is_second_stage,
        'hyperparameters': {},
        'results': {}
    }
    
    # Add hyperparameters to run_info based on clustering method
    if clustering_method == 'classical':
        run_info['hyperparameters']['classical'] = dict(config.classical_clustering)
        if run_info['second_stage'] and hasattr(config, 'second_stage_clustering'):
            run_info['hyperparameters']['second_stage'] = dict(config.second_stage_clustering)
    elif clustering_method == 'hdbscan':
        run_info['hyperparameters']['hdbscan'] = dict(config.hdbscan_clustering)
    elif clustering_method == 'gmm':
        run_info['hyperparameters']['gmm'] = dict(config.gmm_clustering)
    elif clustering_method == 'hdbscan-gmm':
        run_info['hyperparameters']['hdbscan_gmm'] = dict(config.hdbscan_gmm_clustering)
    
    # Add quantum parameters only if using annealing
    if run_info['annealing']:
        run_info['hyperparameters']['quantum'] = dict(config.quantum_clustering)
    
    np.random.seed(config.classical_clustering.random_state)

    # Setup directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))
    colormaps_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "colormaps"))
    output_csv = os.path.join(data_dir, "antique_doc_embeddings.csv")
    query_csv = os.path.join(data_dir, "antique_train_queries.csv")
    
    # Create output directory with method and timestamp
    run_output_dir = os.path.join(data_dir, f"run_{timestamp}_{clustering_method}_{refinement_method}")
    if run_cv:
        run_output_dir += "_cv"
    os.makedirs(run_output_dir, exist_ok=True)
    
    # Define plot paths
    umap_plot_path = os.path.join(run_output_dir, f"umap_plot_{timestamp}.png")
    initial_clusters_plot_path = os.path.join(run_output_dir, f"{clustering_method}_clusters_{timestamp}.png")
    final_clusters_plot_path = os.path.join(run_output_dir, f"final_clusters_{timestamp}.png")
    spectrum_plot_path = os.path.join(run_output_dir, f"cluster_spectrum_{timestamp}.png")

    # Load colormap
    if colormap_name is None:
        colormap_name = "Spectral"
    
    cmap = load_colormap(colormap_name, colormaps_dir)
    
    # Parse embeddings function
    def parse_embedding(text):
        return np.fromstring(text[1:-1], dtype=float, sep=',')

    # Load data
    train_df = pd.read_csv(output_csv, converters={"doc_embeddings": parse_embedding})
    doc_embeddings = np.stack(train_df["doc_embeddings"].values)
    doc_ids = train_df['doc_id'].tolist()

    query_df = pd.read_csv(query_csv)

    # Initialize UMAP reducer
    umap_reducer = UMAPReducer(random_state=config.classical_clustering.random_state)
    
    # Apply dimension reduction if specified
    if config.general.reduction:
        doc_embeddings_reduced = umap_reducer.fit_transform(doc_embeddings)
    else: 
        doc_embeddings_reduced = doc_embeddings
    np.save(os.path.join(run_output_dir, "doc_embeddings_reduced.npy"), doc_embeddings_reduced)
    
    # Plot UMAP reduced dimensions
    umap_title = f"UMAP Reduction\nMethod: {clustering_method.upper()}, {timestamp}"
    plot_embeddings(doc_embeddings_reduced, title=umap_title, save_path=umap_plot_path, cmap=cmap)

    # Check if method can provide probabilities (for multi-membership)
    has_probabilities = clustering_method in ['gmm', 'hdbscan-gmm']
    if multi_membership and not has_probabilities:
        print(f"Warning: Multi-membership requires 'gmm' or 'hdbscan-gmm' method. Requested method '{clustering_method}' doesn't provide probabilities.")
        multi_membership = False
        run_info['multi_membership'] = False
    
    # Run cross-validation if enabled
    cv_results = None
    if run_cv:
        cv_results = run_cv_evaluation(
            doc_embeddings, 
            doc_embeddings_reduced, 
            config,
            query_df,
            doc_ids,
            cv_folds=cv_folds,
            clustering_method=clustering_method
        )
        
        np.save(os.path.join(run_output_dir, "cv_results.npy"), cv_results)
        run_info['cv_results'] = {
            'avg_dbi': float(cv_results['avg_dbi']),
            'std_dbi': float(cv_results['std_dbi']),
            'avg_ndcg': float(cv_results['avg_ndcg']),
            'std_ndcg': float(cv_results['std_ndcg']),
            'avg_coverage': float(cv_results['avg_coverage']),
            'std_coverage': float(cv_results['std_coverage'])
        }
        
        if 'avg_ndcg_multi' in cv_results and cv_results['avg_ndcg_multi'] > 0:
            run_info['cv_results']['avg_ndcg_multi'] = float(cv_results['avg_ndcg_multi'])
            run_info['cv_results']['std_ndcg_multi'] = float(cv_results['std_ndcg_multi'])
            run_info['cv_results']['avg_coverage_multi'] = float(cv_results['avg_coverage_multi'])
            run_info['cv_results']['std_coverage_multi'] = float(cv_results['std_coverage_multi'])
    
    # Perform initial clustering based on method
    if clustering_method == 'hdbscan':
        print("Using HDBSCAN clustering...")
        clustering = HDBSCANClustering(**config.hdbscan_clustering)
        
        initial_labels, medoid_indices = clustering.find_optimal_k(doc_embeddings_reduced)
        
        if -1 in initial_labels:
            print("Handling noise points in HDBSCAN results...")
            initial_labels = clustering.handle_noise_points(doc_embeddings_reduced, initial_labels, medoid_indices)
        
        # Save initial clustering results
        save_initial_clustering_results(initial_labels, doc_ids, run_output_dir, "hdbscan", timestamp)
        
        initial_dbi = davies_bouldin_score(doc_embeddings_reduced, initial_labels) if len(np.unique(initial_labels)) > 1 else float('inf')
        run_info['results']['initial_clusters'] = clustering.best_k
        run_info['results']['initial_dbi'] = float(initial_dbi)
        
        plot_title = f"HDBSCAN Clustering\nk={clustering.best_k}, DBI={initial_dbi:.4f}, {timestamp}"
        
    elif clustering_method == 'gmm':
        print("Using Gaussian Mixture Model clustering...")
        clustering = GMMClustering(**config.gmm_clustering)
        
        initial_labels, medoid_indices = clustering.find_optimal_k(doc_embeddings_reduced)
        
        # Save initial clustering results
        save_initial_clustering_results(initial_labels, doc_ids, run_output_dir, "gmm", timestamp)
        
        membership_probs = clustering.get_membership_probabilities()
        np.save(os.path.join(run_output_dir, "gmm_membership_probs.npy"), membership_probs)
        
        top_docs = clustering.get_top_documents_per_cluster(doc_ids, n=10)
        with open(os.path.join(run_output_dir, "top_docs_per_cluster.txt"), 'w') as f:
            for cluster_id, docs in top_docs.items():
                f.write(f"Cluster {cluster_id}:\n")
                for doc_id, prob in docs:
                    f.write(f"  {doc_id}: {prob:.4f}\n")
                f.write("\n")
        
        clustering.save_cluster_membership(doc_ids, os.path.join(run_output_dir, "cluster_membership.csv"))
        
        initial_dbi = davies_bouldin_score(doc_embeddings_reduced, initial_labels) if len(np.unique(initial_labels)) > 1 else float('inf')
        run_info['results']['initial_clusters'] = clustering.best_k
        run_info['results']['initial_dbi'] = float(initial_dbi)
        
        plot_title = f"GMM Clustering\nk={clustering.best_k}, DBI={initial_dbi:.4f}, {timestamp}"
        
    elif clustering_method == 'hdbscan-gmm':
        print("Using HDBSCAN-GMM hybrid clustering...")
        clustering = HDBSCANGMMClustering(**config.hdbscan_gmm_clustering)
        
        initial_labels, medoid_indices = clustering.find_optimal_k(doc_embeddings_reduced)
        
        # Save initial clustering results
        save_initial_clustering_results(initial_labels, doc_ids, run_output_dir, "hdbscan-gmm", timestamp)
        
        membership_probs = clustering.get_membership_probabilities()
        np.save(os.path.join(run_output_dir, "hdbscan_gmm_membership_probs.npy"), membership_probs)
        
        top_docs = clustering.get_top_documents_per_cluster(doc_ids, n=10)
        with open(os.path.join(run_output_dir, "hdbscan_gmm_top_docs_per_cluster.txt"), 'w') as f:
            for cluster_id, docs in top_docs.items():
                f.write(f"Cluster {cluster_id}:\n")
                for doc_id, prob in docs:
                    f.write(f"  {doc_id}: {prob:.4f}\n")
                f.write("\n")
        
        clustering.save_cluster_membership(doc_ids, os.path.join(run_output_dir, "hdbscan_gmm_cluster_membership.csv"))
        
        initial_dbi = davies_bouldin_score(doc_embeddings_reduced, initial_labels) if len(np.unique(initial_labels)) > 1 else float('inf')
        run_info['results']['initial_clusters'] = clustering.best_k
        run_info['results']['initial_dbi'] = float(initial_dbi)
        
        plot_title = f"HDBSCAN-GMM Clustering\nk={clustering.best_k}, DBI={initial_dbi:.4f}, {timestamp}"
        
    else:
        print("Using Classical K-Medoids clustering...")
        clustering = ClassicalClustering(**config.classical_clustering)
        initial_labels, medoid_indices = clustering.find_optimal_k(doc_embeddings_reduced)
        
        # Save initial clustering results
        save_initial_clustering_results(initial_labels, doc_ids, run_output_dir, "classical", timestamp)
 
        initial_dbi = davies_bouldin_score(doc_embeddings_reduced, initial_labels) if len(np.unique(initial_labels)) > 1 else float('inf')
        run_info['results']['initial_clusters'] = clustering.best_k
        run_info['results']['initial_dbi'] = float(initial_dbi)
        
        plot_title = f"K-Medoids Clustering\nk={clustering.best_k}, DBI={initial_dbi:.4f}, {timestamp}"

    print(f"{clustering_method.capitalize()} Clustering Labels: {initial_labels}")
    print(f"{clustering_method.capitalize()} Medoid Indices: {medoid_indices}")

    # Set up visualization colors
    unique_initial_labels = np.unique(initial_labels)
    n_initial_clusters = len(unique_initial_labels)
    
    initial_colors = {}
    for i, label in enumerate(unique_initial_labels):
        if isinstance(cmap, str):
            color_value = plt.get_cmap(cmap)(i / max(1, n_initial_clusters - 1))
        else:
            color_value = cmap(i / max(1, n_initial_clusters - 1))
        initial_colors[label] = color_value

    # Extract medoid embeddings
    medoid_embeddings = clustering.extract_medoids(doc_embeddings_reduced, medoid_indices)
    np.save(os.path.join(run_output_dir, "medoid_embeddings.npy"), medoid_embeddings)
    np.save(os.path.join(run_output_dir, "medoid_indices.npy"), medoid_indices)
    
    # Plot initial clusters
    plot_embeddings(doc_embeddings_reduced, labels=initial_labels, medoids=medoid_embeddings,
                title=plot_title, save_path=initial_clusters_plot_path, cmap=cmap,
                cluster_colors=initial_colors)

    # Refinement step - use either quantum annealing or second-stage clustering
    if is_annealing:
        # Quantum annealing approach (original)
        print("Using quantum annealing for cluster refinement...")
        quantum_clustering = QuantumClustering(config.quantum_clustering.k_range, medoid_embeddings, config)
        best_k, refined_medoid_indices, best_dbi = find_best_k_with_qubo(quantum_clustering, medoid_embeddings)

        if refined_medoid_indices is not None:
            print(f"Quantum-Refined Medoid Indices: {refined_medoid_indices}") 
            refined_medoid_indices_of_embeddings = medoid_indices[refined_medoid_indices]
            refined_medoid_embeddings = doc_embeddings_reduced[refined_medoid_indices_of_embeddings] 
            
            print(f"Before QUBO: Assignments to Initial Medoids: {compute_clusters(doc_embeddings_reduced, medoid_indices)}")
            print(f"Refined Medoid Indices Type: {type(refined_medoid_indices)}, Shape: {refined_medoid_indices.shape}")
            print(f"Refined Medoid Embeddings Shape: {refined_medoid_embeddings.shape}")
            print(f"Before Assigning Clusters, Medoid Indices: {refined_medoid_indices_of_embeddings}")
            final_cluster_labels = compute_clusters(doc_embeddings_reduced, refined_medoid_indices_of_embeddings)
            print(f"After QUBO: Assignments to Quantum Medoids: {final_cluster_labels}")

            run_info['results']['final_clusters'] = int(best_k)
            run_info['results']['final_dbi'] = float(best_dbi)
        else:
            raise ValueError("QUBO Solver failed to find valid medoids.")
        
    elif is_second_stage:
        # Second-stage k-medoids clustering
        print("Using second-stage classical clustering for refinement...")
        # Perform second-stage clustering on the medoids
        final_cluster_labels, refined_medoid_indices, best_k, best_dbi = clustering.cluster_medoids(
            doc_embeddings_reduced,
            medoid_embeddings,
            medoid_indices,
            config
        )
        
        # Get the embeddings of the refined medoids
        refined_medoid_indices_of_embeddings = refined_medoid_indices
        refined_medoid_embeddings = doc_embeddings_reduced[refined_medoid_indices_of_embeddings]
        
        print(f"Second-Stage Clustering: k={best_k}, DBI={best_dbi:.4f}")
        
        run_info['results']['final_clusters'] = int(best_k)
        run_info['results']['final_dbi'] = float(best_dbi)
        
    else:
        # No refinement - just use initial clustering
        print("No refinement applied, using initial clustering results...")
        final_cluster_labels = initial_labels
        refined_medoid_indices = medoid_indices
        refined_medoid_embeddings = medoid_embeddings
        best_k = len(np.unique(initial_labels))
        best_dbi = initial_dbi
        
        run_info['results']['final_clusters'] = int(best_k)
        run_info['results']['final_dbi'] = float(best_dbi)
    
    print(f"Final chosen k after {refinement_method} refinement: {best_k}")
    print(f"Final DBI: {best_dbi:.4f}")

    # Save final clustering results
    np.save(os.path.join(run_output_dir, f"final_{refinement_method}_clusters.npy"), final_cluster_labels)
    np.save(os.path.join(run_output_dir, "refined_medoid_embeddings.npy"), refined_medoid_embeddings)
    np.save(os.path.join(run_output_dir, "refined_medoid_indices.npy"), refined_medoid_indices)
    
    cluster_mapping = pd.DataFrame({
        'doc_id': doc_ids,
        'cluster': final_cluster_labels
    })
    cluster_mapping.to_csv(os.path.join(run_output_dir, "doc_clusters.csv"), index=False)

    compare_cluster_distributions(initial_labels, final_cluster_labels, run_output_dir, clustering_method, timestamp)

    color_correspondence = track_cluster_correspondence(initial_labels, final_cluster_labels)
    
    correspondence_df = pd.DataFrame({
        'final_cluster': list(color_correspondence.keys()),
        'initial_cluster': list(color_correspondence.values())
    })
    correspondence_df.to_csv(os.path.join(run_output_dir, "cluster_correspondence.csv"), index=False)

    # Handle probabilities for multi-membership
    n_quantum_clusters = len(np.unique(final_cluster_labels))
    n_docs = len(doc_ids)
    
    quantum_probs = np.zeros((n_docs, n_quantum_clusters))

    if has_probabilities:
        if hasattr(clustering, 'membership_probs'):
            component_to_quantum = {}
            
            for comp_idx in range(clustering.membership_probs.shape[1]):
                counts = np.zeros(n_quantum_clusters)
                for doc_idx, prob in enumerate(clustering.membership_probs[:, comp_idx]):
                    if prob > 0.1:
                        quantum_cluster = final_cluster_labels[doc_idx]
                        counts[quantum_cluster] += prob
                
                if np.sum(counts) > 0:
                    component_to_quantum[comp_idx] = np.argmax(counts)
            
            for doc_idx in range(n_docs):
                doc_probs = clustering.membership_probs[doc_idx, :]
                
                for comp_idx, quantum_idx in component_to_quantum.items():
                    quantum_probs[doc_idx, quantum_idx] += doc_probs[comp_idx]
                    
            row_sums = quantum_probs.sum(axis=1, keepdims=True)
            quantum_probs = np.divide(quantum_probs, row_sums, 
                                        out=np.zeros_like(quantum_probs), 
                                        where=row_sums != 0)
            
            np.save(os.path.join(run_output_dir, "final_probabilities.npy"), quantum_probs)
            
            create_hybrid_probabilistic_assignments(
                doc_ids,
                clustering.membership_probs,
                final_cluster_labels,
                refined_medoid_indices_of_embeddings,
                run_output_dir,
                prefix=clustering_method
            )
    
    # Multi-membership assignments
    multi_membership_df = None
    if has_probabilities and multi_membership:
        print(f"Creating multi-membership assignments with threshold={threshold}...")
        multi_membership_df = create_multi_membership_assignments(
            doc_ids,
            doc_embeddings_reduced,
            clustering.membership_probs,
            final_cluster_labels,
            refined_medoid_indices_of_embeddings,
            refined_medoid_embeddings,
            threshold=threshold,
            data_dir=run_output_dir,
            prefix=clustering_method
        )
        
        membership_counts = multi_membership_df['membership_count'].values
        run_info['results']['multi_membership'] = {
            'avg_memberships': float(np.mean(membership_counts)),
            'max_memberships': int(np.max(membership_counts)),
            'docs_with_multiple': int(np.sum(membership_counts > 1)),
            'percent_multi': float((np.sum(membership_counts > 1) / len(doc_ids)) * 100)
        }

    print(f"Final Refined Medoid Embeddings:\n {refined_medoid_embeddings}")
    print(f"Unique Cluster Assignments: {np.unique(final_cluster_labels)}")

    # Evaluate retrieval metrics on full dataset
    print("\nEvaluating retrieval metrics on full dataset...")
    try:
        # Parse query embeddings from CSV
        query_df['query_embeddings'] = query_df['query_embeddings'].apply(
            lambda x: np.fromstring(x[1:-1], dtype=float, sep=',') if isinstance(x, str) else x
        )
        
        # Filter to queries with valid embeddings
        valid_queries = query_df[query_df['query_embeddings'].apply(lambda x: isinstance(x, np.ndarray) and len(x) > 0)]
        
        if len(valid_queries) > 0:
            first_shape = len(valid_queries['query_embeddings'].iloc[0])
            valid_queries = valid_queries[valid_queries['query_embeddings'].apply(lambda x: len(x) == first_shape)]
            
            if len(valid_queries) > 0:
                # Get unique query IDs
                unique_query_ids = valid_queries['query_id'].unique()
                print(f"Found {len(unique_query_ids)} unique queries out of {len(valid_queries)} query-document pairs")
                
                # Create deduplicated query dataframe with unique embeddings
                dedup_queries = []
                for qid in unique_query_ids:
                    # Get all rows for this query ID
                    query_rows = valid_queries[valid_queries['query_id'] == qid]
                    # Just take the first one - they should all have the same embedding
                    dedup_queries.append(query_rows.iloc[0])
                
                # Create deduplicated dataframe
                dedup_query_df = pd.DataFrame(dedup_queries)
                print(f"Created deduplicated query dataframe with {len(dedup_query_df)} unique queries")
                
                # Extract embeddings from deduplicated queries
                unique_query_embeddings = np.stack(dedup_query_df["query_embeddings"].values)
                
                # Keep all relevance judgments from the original dataframe
                qrels_df = valid_queries[['query_id', 'doc_id', 'relevance']]
                
                # Verify we have unique embeddings after deduplication
                if len(unique_query_embeddings) >= 5:
                    print("Checking first 5 query embeddings after deduplication:")
                    for i in range(5):
                        print(f"Query {dedup_query_df['query_id'].iloc[i]} first 5 values: {unique_query_embeddings[i][:5]}")
                
                if config.general.reduction:
                    umap_reducer_plot = umap_reducer
                else:
                    umap_reducer_plot = None
                    
                # Call the evaluation function with deduplicated queries but complete relevance judgments
                evaluation_results = evaluate_retrieval(
                    unique_query_embeddings,
                    doc_embeddings_reduced,
                    refined_medoid_embeddings,
                    final_cluster_labels,
                    qrels_df,
                    doc_ids,
                    k=10,
                    umap_reducer=umap_reducer_plot,
                    multi_cluster_assignments=multi_membership_df if multi_membership else None
                )
                
                ndcg_10 = evaluation_results['ndcg']
                coverage = evaluation_results['coverage']
                
                print(f"Full dataset nDCG@10: {ndcg_10:.4f}")
                print(f"Full dataset Relevant Coverage: {coverage:.4f}")
                
                run_info['results']['ndcg_10'] = float(ndcg_10)
                run_info['results']['relevant_coverage'] = float(coverage)
                
                if 'ndcg_multi' in evaluation_results:
                    ndcg_multi = evaluation_results['ndcg_multi']
                    coverage_multi = evaluation_results['coverage_multi']
                    print(f"Full dataset Multi-Membership nDCG@10: {ndcg_multi:.4f}")
                    print(f"Full dataset Multi-Membership Coverage: {coverage_multi:.4f}")
                    run_info['results']['ndcg_multi_10'] = float(ndcg_multi)
                    run_info['results']['coverage_multi'] = float(coverage_multi)
            else:
                print("No valid queries with consistent embedding dimensions found.")
                run_info['results']['ndcg_10'] = 0.0
                run_info['results']['relevant_coverage'] = 0.0
        else:
            print("No valid query embeddings found.")
            run_info['results']['ndcg_10'] = 0.0
            run_info['results']['relevant_coverage'] = 0.0
    except Exception as e:
        print(f"Error evaluating retrieval metrics on full dataset: {str(e)}")
        run_info['results']['ndcg_10'] = 0.0
        run_info['results']['relevant_coverage'] = 0.0

    # Create plots
    final_plot_title = f"Final {refinement_method.capitalize()} Cluster Assignments\n" + \
                     f"Method: {clustering_method.upper()}, k={best_k}, DBI={best_dbi:.4f}, {timestamp}"
     
    plot_embeddings(doc_embeddings_reduced,
                  labels=final_cluster_labels,
                  medoids=medoid_embeddings,
                  refined_medoids=refined_medoid_embeddings,
                  title=final_plot_title, 
                  save_path=final_clusters_plot_path, 
                  cmap=cmap,
                  cluster_colors=initial_colors,
                  color_correspondence=color_correspondence)

    # Create merged clusters visualization
    merged_clusters_dir = os.path.join(run_output_dir, f"merged_clusters_{timestamp}")
    os.makedirs(merged_clusters_dir, exist_ok=True)

    # Create visualizations for top 10 merged clusters
    plot_top_merged_clusters(
        doc_embeddings_reduced,
        initial_labels,
        final_cluster_labels,
        color_correspondence,
        save_dir=merged_clusters_dir,
        cmap=cmap,
        highlight_color='#25A085',
        title_prefix=f"Cluster Merging Visualization\nMethod: {clustering_method.upper()}, k={best_k}",
        num_clusters=10
    )
    print(f"Saved top merged clusters visualizations to {merged_clusters_dir}")

    # Create spectrum plot for probabilistic methods
    if has_probabilities:
        spectrum_title = "Cluster Color Spectrum\n" + \
                       f"Method: {clustering_method.upper()}, k={best_k}, DBI={best_dbi:.4f}, {timestamp}"
        
        plot_cluster_spectrum(
            doc_embeddings_reduced,
            quantum_probs,
            medoids=medoid_embeddings,
            refined_medoids=refined_medoid_embeddings,
            title=spectrum_title,
            save_path=spectrum_plot_path,
            cmap=cmap,
            cluster_colors=initial_colors,
            color_correspondence=color_correspondence
        )
        print(f"Saved cluster spectrum plot at: {spectrum_plot_path}")

    print(f"Saved final {refinement_method} cluster plot at: {final_clusters_plot_path}")
    
    # Save run information
    run_info_path = os.path.join(run_output_dir, f"run_info_{timestamp}.json")
    with open(run_info_path, 'w') as f:
        json.dump(run_info, f, indent=2)
    print(f"Saved run information to {run_info_path}")
    
    summary_path = os.path.join(data_dir, f"run_summary_{clustering_method}_{refinement_method}_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(run_info, f, indent=2)
    
    # Print summary
    print("\n=== Clustering Run Summary ===")
    print(f"Timestamp: {timestamp}")
    print(f"Method: {clustering_method}" + (" with Multi-Membership" if multi_membership else ""))
    print(f"Refinement: {refinement_method}")
    print(f"Initial Clusters: {run_info['results']['initial_clusters']}, DBI: {run_info['results']['initial_dbi']:.4f}")
    print(f"Final Clusters: {run_info['results']['final_clusters']}, DBI: {run_info['results']['final_dbi']:.4f}")
    print(f"nDCG@10: {run_info['results']['ndcg_10']:.4f}, Relevant Coverage: {run_info['results']['relevant_coverage']:.4f}")
    if multi_membership and has_probabilities:
        if 'ndcg_multi_10' in run_info['results'] and 'coverage_multi' in run_info['results']:
            print(f"Multi-Membership nDCG@10: {run_info['results']['ndcg_multi_10']:.4f}")
            print(f"Multi-Membership Coverage: {run_info['results']['coverage_multi']:.4f}")
        mm_stats = run_info['results']['multi_membership']
        print(f"Multi-Membership: {mm_stats['percent_multi']:.1f}% of documents belong to multiple clusters")
        print(f"Average memberships per document: {mm_stats['avg_memberships']:.2f}")
    
    if cv_results:
        print(f"Cross-validation DBI: {cv_results['avg_dbi']:.4f} ± {cv_results['std_dbi']:.4f}")
        print(f"Cross-validation nDCG@10: {cv_results['avg_ndcg']:.4f} ± {cv_results['std_ndcg']:.4f}")
        print(f"Cross-validation Coverage: {cv_results.get('avg_coverage', 0.0):.4f} ± {cv_results.get('std_coverage', 0.0):.4f}")
        
        if 'avg_ndcg_multi' in cv_results and cv_results['avg_ndcg_multi'] > 0:
            print(f"Cross-validation Multi-nDCG@10: {cv_results['avg_ndcg_multi']:.4f} ± {cv_results['std_ndcg_multi']:.4f}")
            print(f"Cross-validation Multi-Coverage: {cv_results['avg_coverage_multi']:.4f} ± {cv_results['std_coverage_multi']:.4f}")
    
    print(f"All results saved to: {run_output_dir}")
    print("===============================")
    
    plt.show()
    
    return run_info


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run clustering pipeline with cross-validation')
    parser.add_argument('--colormap', type=str, default='Spectral', 
                        help='Colormap to use (file in colormaps dir or matplotlib name)')
    parser.add_argument('--no_cv', action='store_true',
                        help='Disable cross-validation evaluation')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--method', type=str, 
                        choices=['classical', 'hdbscan', 'gmm', 'hdbscan-gmm'], 
                        default='classical',
                        help='Clustering method to use')
    parser.add_argument('--multi-membership', action='store_true',
                        help='Enable multi-membership assignments (only works with gmm or hdbscan-gmm)')
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='Probability threshold for multi-membership (default: 0.2)')
    parser.add_argument('--no-annealing', action='store_true',
                        help='Disable quantum annealing and use second-stage clustering instead')
    
    args = parser.parse_args()

    with open("config/general.yml", "r") as file:
        config = ConfigBox(yaml.safe_load(file))
        # Set annealing flag based on command line argument
        if args.no_annealing:
            config.general.annealing = False

    with open("config/kmedoids.yml", "r") as file:
        kmedoids_config = ConfigBox(yaml.safe_load(file))
        config.update(kmedoids_config)
    
    try:
        with open("config/hdbscan.yml", "r") as file:
            hdbscan_config = ConfigBox(yaml.safe_load(file))
            config.update(hdbscan_config)
    except FileNotFoundError:
        print("HDBSCAN config not found, using default parameters")
        config.hdbscan_clustering = ConfigBox({
            'min_cluster_size': 20,
            'min_samples': 25,
            'cluster_selection_method': 'leaf',
            'cluster_selection_epsilon': 0.2,
            'metric': 'euclidean',
            'random_state': config.classical_clustering.random_state
        })
    
    try:
        with open("config/gmm.yml", "r") as file:
            gmm_config = ConfigBox(yaml.safe_load(file))
            config.update(gmm_config)
    except FileNotFoundError:
        print("GMM config not found, using default parameters")
        config.gmm_clustering = ConfigBox({
            'n_components_range': [10, 25, 50, 75, 100],
            'covariance_type': 'full',
            'n_init': 10,
            'random_state': config.classical_clustering.random_state
        })
    
    try:
        with open("config/hdbscan_gmm.yml", "r") as file:
            hdbscan_gmm_config = ConfigBox(yaml.safe_load(file))
            config.update(hdbscan_gmm_config)
    except FileNotFoundError:
        print("HDBSCAN-GMM config not found, using default parameters")
        config.hdbscan_gmm_clustering = ConfigBox({
            'min_cluster_size': 20,
            'min_samples': 25,
            'cluster_selection_method': 'leaf',
            'cluster_selection_epsilon': 0.2,
            'covariance_type': 'full',
            'n_init': 10,
            'metric': 'euclidean',
            'random_state': config.classical_clustering.random_state
        })

    run_pipeline(
        config, 
        colormap_name=args.colormap,
        run_cv=not args.no_cv,
        cv_folds=args.cv_folds,
        clustering_method=args.method,
        multi_membership=args.multi_membership,
        threshold=args.threshold
    )