import os
import json
import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import normalize

from src.models.UMAPReducer import UMAPReducer
from src.models.ClassicalClustering import ClassicalClustering
from src.models.HDBSCANClustering import HDBSCANClustering
from src.models.GMMClustering import GMMClustering
from src.models.HDBSCANGMMClustering import HDBSCANGMMClustering
from src.models.QuantumClustering import QuantumClustering, compute_clusters
from src.models.QuantumClustering import prepare_clustering_submission
from src.models.multi_membership import create_multi_membership_assignments
from src.plot_utils import (plot_embeddings, load_colormap, plot_cluster_spectrum, 
                          plot_merged_clusters, plot_top_merged_clusters)

warnings.filterwarnings("ignore")

@dataclass
class ClusteringResults:
    labels: np.ndarray
    medoid_indices: np.ndarray
    medoid_embeddings: np.ndarray
    medoid_embeddings_original: np.ndarray
    medoid_embeddings_plot: np.ndarray
    n_clusters: int
    dbi_score: float
    membership_probs: Optional[np.ndarray] = None
    clustering_object: Optional[Any] = None


@dataclass
class EvaluationMetrics:
    ndcg_10: float
    relevant_coverage: float
    ndcg_multi_10: Optional[float] = None
    coverage_multi: Optional[float] = None


@dataclass
class RunConfiguration:
    method: str
    colormap: str
    multi_membership: bool
    threshold: float
    annealing: bool
    second_stage: bool
    cv_folds: int = 5


class DataManager:
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.embeddings_file = self.data_dir / "antique_doc_embeddings.csv"
        self.queries_file = self.data_dir / "antique_train_queries.csv"
    
    @staticmethod
    def parse_embedding(text: str) -> np.ndarray:
        return np.fromstring(text[1:-1], dtype=float, sep=',')
    
    def load_document_data(self) -> Tuple[np.ndarray, List[str]]:
        df = pd.read_csv(self.embeddings_file, 
                        converters={"doc_embeddings": self.parse_embedding})
        embeddings = np.stack(df["doc_embeddings"].values)
        doc_ids = df['doc_id'].tolist()
        return embeddings, doc_ids
    
    def load_query_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.queries_file)
        df['query_embeddings'] = df['query_embeddings'].apply(
            lambda x: self.parse_embedding(x) if isinstance(x, str) else x
        )
        return self._validate_and_deduplicate_queries(df)
    
    def _validate_and_deduplicate_queries(self, df: pd.DataFrame) -> pd.DataFrame:
        valid_queries = df[df['query_embeddings'].apply(lambda x: isinstance(x, np.ndarray) and len(x) > 0)]
        
        if len(valid_queries) > 0:
            first_shape = len(valid_queries['query_embeddings'].iloc[0])
            valid_queries = valid_queries[valid_queries['query_embeddings'].apply(lambda x: len(x) == first_shape)]
            
            if len(valid_queries) > 0:
                unique_query_ids = valid_queries['query_id'].unique()
                print(f"Found {len(unique_query_ids)} unique queries out of {len(valid_queries)} query-document pairs")
                
                dedup_queries = []
                for qid in unique_query_ids:
                    query_rows = valid_queries[valid_queries['query_id'] == qid]
                    dedup_queries.append(query_rows.iloc[0])
                
                dedup_query_df = pd.DataFrame(dedup_queries)
                print(f"Created deduplicated query dataframe with {len(dedup_query_df)} unique queries")
                
                return dedup_query_df
        
        return pd.DataFrame()


class ClusteringStrategy(ABC):
    
    @abstractmethod
    def cluster(self, embeddings: np.ndarray, original_embeddings: np.ndarray, 
               plot_embeddings: np.ndarray, config: Any) -> ClusteringResults:
        pass
    
    @abstractmethod
    def supports_probabilities(self) -> bool:
        pass


class ClassicalClusteringStrategy(ClusteringStrategy):
    
    def cluster(self, embeddings: np.ndarray, original_embeddings: np.ndarray, 
               plot_embeddings: np.ndarray, config: Any) -> ClusteringResults:
        clustering = ClassicalClustering(**config.classical_clustering)
        labels, medoid_indices = clustering.find_optimal_k(embeddings)
        
        medoid_embeddings = clustering.extract_medoids(embeddings, medoid_indices)
        medoid_embeddings_original = clustering.extract_medoids(original_embeddings, medoid_indices)
        medoid_embeddings_plot = clustering.extract_medoids(plot_embeddings, medoid_indices)
        
        dbi_score = (davies_bouldin_score(embeddings, labels) 
                    if len(np.unique(labels)) > 1 else float('inf'))
        
        return ClusteringResults(
            labels=labels,
            medoid_indices=medoid_indices,
            medoid_embeddings=medoid_embeddings,
            medoid_embeddings_original=medoid_embeddings_original,
            medoid_embeddings_plot=medoid_embeddings_plot,
            n_clusters=len(np.unique(labels)),
            dbi_score=dbi_score,
            clustering_object=clustering
        )
    
    def supports_probabilities(self) -> bool:
        return False


class HDBSCANClusteringStrategy(ClusteringStrategy):
    
    def cluster(self, embeddings: np.ndarray, original_embeddings: np.ndarray, 
               plot_embeddings: np.ndarray, config: Any) -> ClusteringResults:
        clustering = HDBSCANClustering(**config.hdbscan_clustering)
        labels, medoid_indices = clustering.find_optimal_k(embeddings)
        
        if -1 in labels:
            print("Handling noise points in HDBSCAN results...")
            labels = clustering.handle_noise_points(embeddings, labels, medoid_indices)
        
        medoid_embeddings = clustering.extract_medoids(embeddings, medoid_indices)
        medoid_embeddings_original = clustering.extract_medoids(original_embeddings, medoid_indices)
        medoid_embeddings_plot = clustering.extract_medoids(plot_embeddings, medoid_indices)
        
        dbi_score = (davies_bouldin_score(embeddings, labels) 
                    if len(np.unique(labels)) > 1 else float('inf'))
        
        return ClusteringResults(
            labels=labels,
            medoid_indices=medoid_indices,
            medoid_embeddings=medoid_embeddings,
            medoid_embeddings_original=medoid_embeddings_original,
            medoid_embeddings_plot=medoid_embeddings_plot,
            n_clusters=len(np.unique(labels)),
            dbi_score=dbi_score,
            clustering_object=clustering
        )
    
    def supports_probabilities(self) -> bool:
        return False


class GMMClusteringStrategy(ClusteringStrategy):
    
    def cluster(self, embeddings: np.ndarray, original_embeddings: np.ndarray, 
               plot_embeddings: np.ndarray, config: Any) -> ClusteringResults:
        clustering = GMMClustering(**config.gmm_clustering)
        labels, medoid_indices = clustering.find_optimal_k(embeddings)
        
        medoid_embeddings = clustering.extract_medoids(embeddings, medoid_indices)
        medoid_embeddings_original = clustering.extract_medoids(original_embeddings, medoid_indices)
        medoid_embeddings_plot = clustering.extract_medoids(plot_embeddings, medoid_indices)
        
        dbi_score = (davies_bouldin_score(embeddings, labels) 
                    if len(np.unique(labels)) > 1 else float('inf'))
        
        return ClusteringResults(
            labels=labels,
            medoid_indices=medoid_indices,
            medoid_embeddings=medoid_embeddings,
            medoid_embeddings_original=medoid_embeddings_original,
            medoid_embeddings_plot=medoid_embeddings_plot,
            n_clusters=len(np.unique(labels)),
            dbi_score=dbi_score,
            membership_probs=clustering.get_membership_probabilities(),
            clustering_object=clustering
        )
    
    def supports_probabilities(self) -> bool:
        return True


class HDBSCANGMMClusteringStrategy(ClusteringStrategy):
    
    def cluster(self, embeddings: np.ndarray, original_embeddings: np.ndarray, 
               plot_embeddings: np.ndarray, config: Any) -> ClusteringResults:
        clustering = HDBSCANGMMClustering(**config.hdbscan_gmm_clustering)
        labels, medoid_indices = clustering.find_optimal_k(embeddings)
        
        medoid_embeddings = clustering.extract_medoids(embeddings, medoid_indices)
        medoid_embeddings_original = clustering.extract_medoids(original_embeddings, medoid_indices)
        medoid_embeddings_plot = clustering.extract_medoids(plot_embeddings, medoid_indices)
        
        dbi_score = (davies_bouldin_score(embeddings, labels) 
                    if len(np.unique(labels)) > 1 else float('inf'))
        
        return ClusteringResults(
            labels=labels,
            medoid_indices=medoid_indices,
            medoid_embeddings=medoid_embeddings,
            medoid_embeddings_original=medoid_embeddings_original,
            medoid_embeddings_plot=medoid_embeddings_plot,
            n_clusters=len(np.unique(labels)),
            dbi_score=dbi_score,
            membership_probs=clustering.get_membership_probabilities(),
            clustering_object=clustering
        )
    
    def supports_probabilities(self) -> bool:
        return True


class ClusteringStrategyFactory:
    
    _strategies = {
        'classical': ClassicalClusteringStrategy,
        'hdbscan': HDBSCANClusteringStrategy,
        'gmm': GMMClusteringStrategy,
        'hdbscan-gmm': HDBSCANGMMClusteringStrategy
    }
    
    @classmethod
    def create(cls, method: str) -> ClusteringStrategy:
        if method not in cls._strategies:
            raise ValueError(f"Unknown clustering method: {method}")
        return cls._strategies[method]()


class ClusterRefinement:
    
    def __init__(self, config: Any):
        self.config = config
    
    def refine_clusters(self, 
                       initial_results: ClusteringResults,
                       embeddings: np.ndarray,
                       original_embeddings: np.ndarray,
                       plot_embeddings: np.ndarray,
                       method: str,
                       use_annealing: bool,
                       use_second_stage: bool,
                       doc_ids: List[str],
                       output_dir: str) -> ClusteringResults:
        
        if use_annealing:
            return self._quantum_refinement(initial_results, embeddings, original_embeddings, 
                                           plot_embeddings, doc_ids, output_dir, method)
        elif use_second_stage and method == 'classical':
            return self._second_stage_refinement(initial_results, embeddings, original_embeddings, 
                                                plot_embeddings)
        else:
            return initial_results
    
    def _quantum_refinement(self, initial_results: ClusteringResults, embeddings: np.ndarray,
                           original_embeddings: np.ndarray, plot_embeddings: np.ndarray,
                           doc_ids: List[str], output_dir: str, method: str) -> ClusteringResults:
        print("Using quantum annealing for cluster refinement...")
        quantum_clustering = QuantumClustering(
            self.config.quantum_clustering.k_range,
            initial_results.medoid_embeddings,
            self.config
        )
        
        best_k, refined_indices, best_dbi = self._find_best_k_with_qubo(
            quantum_clustering, initial_results.medoid_embeddings
        )
        
        if refined_indices is None:
            raise ValueError("QUBO Solver failed to find valid medoids.")
        
        print(f"Quantum-Refined Medoid Indices: {refined_indices}")
        refined_medoid_indices_of_embeddings = initial_results.medoid_indices[refined_indices]
        refined_medoid_embeddings = embeddings[refined_medoid_indices_of_embeddings]
        refined_medoid_embeddings_original = original_embeddings[refined_medoid_indices_of_embeddings]
        refined_medoid_embeddings_plot = plot_embeddings[refined_medoid_indices_of_embeddings]
        
        print(f"Before QUBO: Assignments to Initial Medoids: {compute_clusters(embeddings, initial_results.medoid_indices)}")
        print(f"Refined Medoid Indices Type: {type(refined_indices)}, Shape: {refined_indices.shape}")
        print(f"Refined Medoid Embeddings Shape: {refined_medoid_embeddings.shape}")
        print(f"Before Assigning Clusters, Medoid Indices: {refined_medoid_indices_of_embeddings}")
        
        final_cluster_labels = compute_clusters(embeddings, refined_medoid_indices_of_embeddings)
        print(f"After QUBO: Assignments to Quantum Medoids: {final_cluster_labels}")
        
        problem_ids = []
        if hasattr(quantum_clustering, 'problem_ids'):
            problem_ids = quantum_clustering.problem_ids
        
        prepare_clustering_submission(
            original_embeddings,
            doc_ids,
            final_cluster_labels,
            refined_medoid_indices_of_embeddings,
            output_dir,
            method,
            self.config,
            problem_ids
        )
        
        final_dbi_original = (davies_bouldin_score(original_embeddings, final_cluster_labels)
                             if len(np.unique(final_cluster_labels)) > 1 else float('inf'))
        
        return ClusteringResults(
            labels=final_cluster_labels,
            medoid_indices=refined_medoid_indices_of_embeddings,
            medoid_embeddings=refined_medoid_embeddings,
            medoid_embeddings_original=refined_medoid_embeddings_original,
            medoid_embeddings_plot=refined_medoid_embeddings_plot,
            n_clusters=best_k,
            dbi_score=final_dbi_original
        )
    
    def _second_stage_refinement(self, initial_results: ClusteringResults,
                               embeddings: np.ndarray, original_embeddings: np.ndarray,
                               plot_embeddings: np.ndarray) -> ClusteringResults:
        print("Using second-stage classical clustering for refinement...")
        clustering = ClassicalClustering(**self.config.classical_clustering)
        
        final_labels, refined_indices, best_k, best_dbi = clustering.cluster_medoids(
            embeddings,
            initial_results.medoid_embeddings,
            initial_results.medoid_indices,
            self.config
        )
        
        refined_medoid_embeddings = embeddings[refined_indices]
        refined_medoid_embeddings_original = original_embeddings[refined_indices]
        refined_medoid_embeddings_plot = plot_embeddings[refined_indices]
        
        print(f"Second-Stage Clustering: k={best_k}")
        
        return ClusteringResults(
            labels=final_labels,
            medoid_indices=refined_indices,
            medoid_embeddings=refined_medoid_embeddings,
            medoid_embeddings_original=refined_medoid_embeddings_original,
            medoid_embeddings_plot=refined_medoid_embeddings_plot,
            n_clusters=best_k,
            dbi_score=best_dbi
        )
    
    def _find_best_k_with_qubo(self, quantum_clustering, medoid_embeddings):
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


class EvaluationEngine:
    
    @staticmethod
    def dcg_at_k(r: np.ndarray, k: int) -> float:
        r = np.asfarray(r)[:k]
        if r.size:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(3, r.size + 2)))
        return 0.0
    
    @staticmethod
    def ndcg_at_k(r: List[float], k: int) -> float:
        dcg_max = EvaluationEngine.dcg_at_k(sorted(r, reverse=True), k)
        if not dcg_max:
            return 0.0
        return EvaluationEngine.dcg_at_k(r, k) / dcg_max
    
    def evaluate_retrieval(self,
                         query_embeddings: np.ndarray,
                         doc_embeddings: np.ndarray,
                         centroids: np.ndarray,
                         cluster_assignments: np.ndarray,
                         qrels_df: pd.DataFrame,
                         doc_ids: List[str],
                         k: int = 10,
                         multi_cluster_assignments: Optional[pd.DataFrame] = None) -> EvaluationMetrics:

        doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        
        query_embeddings_norm = normalize(query_embeddings)
        centroids_norm = normalize(centroids)
        doc_embeddings_norm = normalize(doc_embeddings)
        
        print(f"Evaluation embedding dimensions:")
        print(f"  Query embeddings: {query_embeddings_norm.shape}")
        print(f"  Document embeddings: {doc_embeddings_norm.shape}")
        print(f"  Centroids: {centroids_norm.shape}")
        
        ndcg_scores = []
        coverage_scores = []
        
        ndcg_multi_scores = []
        coverage_multi_scores = []
        has_multi = multi_cluster_assignments is not None
        
        unique_query_ids = qrels_df['query_id'].unique()
        
        for q_idx, query_id in enumerate(unique_query_ids):
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

            if q_idx < 5:
                print(f"Query {query_id}: Closest centroid is {closest_centroid_idx} with similarity {centroid_similarities[closest_centroid_idx]:.4f}")
                top_centroids = np.argsort(-centroid_similarities)[:3]
                print(f"  Top-3 centroids: {top_centroids} with similarities: {centroid_similarities[top_centroids]}")

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
            doc_similarities = np.dot(query_embedding, cluster_doc_embeddings.T)[0]
            
            sorted_indices = np.argsort(-doc_similarities)
            ranked_cluster_indices = [cluster_docs_idx[idx] for idx in sorted_indices]
            ranked_doc_ids = [doc_ids[idx] for idx in ranked_cluster_indices]

            relevance_scores = []
            for doc_id in ranked_doc_ids:
                rel_values = query_qrels[query_qrels['doc_id'] == doc_id]['relevance'].values
                if len(rel_values) > 0:
                    relevance_scores.append(float(rel_values[0]))
                    if len(relevance_scores) >= k:
                        break
            
            ndcg = self.ndcg_at_k(relevance_scores, k)
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
                multi_doc_similarities = np.dot(query_embedding, multi_doc_embeddings.T)[0]
                
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
                
                ndcg_multi = self.ndcg_at_k(multi_relevance_scores, k)
                ndcg_multi_scores.append(ndcg_multi)
            
            if q_idx < 5:
                print(f"  Relevance scores for top {k}: {relevance_scores}")
                print(f"  DCG: {self.dcg_at_k(relevance_scores, k):.4f}, Ideal DCG: {self.dcg_at_k(sorted(relevance_scores, reverse=True), k):.4f}")
                if ndcg == 0.0 and any(score > 0 for score in relevance_scores):
                    print(f"  WARNING: nDCG=0 despite having relevant documents in top {k}!")
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
        
        return EvaluationMetrics(
            ndcg_10=results['ndcg'],
            relevant_coverage=results['coverage'],
            ndcg_multi_10=results.get('ndcg_multi'),
            coverage_multi=results.get('coverage_multi')
        )


class VisualizationManager:
    
    def __init__(self, colormap_name: str, colormaps_dir: str):
        self.cmap = load_colormap(colormap_name, colormaps_dir)
        self.colormap_name = colormap_name
    
    def create_initial_colors(self, labels: np.ndarray) -> Dict[int, Any]:
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        colors = {}
        for i, label in enumerate(unique_labels):
            if isinstance(self.cmap, str):
                color_value = plt.get_cmap(self.cmap)(i / max(1, n_clusters - 1))
            else:
                color_value = self.cmap(i / max(1, n_clusters - 1))
            colors[label] = color_value
        
        return colors
    
    def plot_umap_reduction(self, umap_embeddings: np.ndarray, 
                          save_path: str, method: str, timestamp: str):
        title = f"UMAP Reduction\nMethod: {method.upper()}, {timestamp}"
        plot_embeddings(umap_embeddings, title=title, save_path=save_path, cmap=self.cmap)
    
    def plot_initial_clustering(self, umap_embeddings: np.ndarray,
                              results: ClusteringResults,
                              save_path: str, method: str, timestamp: str,
                              initial_colors: Dict[int, Any]):
        title = f"{method.upper()} Clustering\nk={results.n_clusters}, DBI={results.dbi_score:.4f}, {timestamp}"
        plot_embeddings(umap_embeddings, labels=results.labels, 
                       medoids=results.medoid_embeddings_plot, title=title,
                       save_path=save_path, cmap=self.cmap, 
                       cluster_colors=initial_colors)
    
    def plot_final_clustering(self, umap_embeddings: np.ndarray,
                            final_results: ClusteringResults,
                            initial_medoid_embeddings_plot: np.ndarray,
                            save_path: str, method: str, refinement: str,
                            timestamp: str, initial_colors: Dict[int, Any],
                            color_correspondence: Dict[int, int]):
        title = (f"Final {refinement.capitalize()} Cluster Assignments\n"
                f"Method: {method.upper()}, k={final_results.n_clusters}, "
                f"DBI={final_results.dbi_score:.4f}, {timestamp}")
        
        plot_embeddings(umap_embeddings, labels=final_results.labels,
                       medoids=initial_medoid_embeddings_plot,
                       refined_medoids=final_results.medoid_embeddings_plot,
                       title=title, save_path=save_path, cmap=self.cmap,
                       cluster_colors=initial_colors,
                       color_correspondence=color_correspondence)
    
    def plot_merged_clusters_visualization(self, umap_embeddings: np.ndarray,
                                         initial_labels: np.ndarray,
                                         final_labels: np.ndarray,
                                         color_correspondence: Dict[int, int],
                                         save_dir: str, method: str,
                                         final_results: ClusteringResults,
                                         timestamp: str):
        plot_top_merged_clusters(
            umap_embeddings,
            initial_labels,
            final_labels,
            color_correspondence,
            save_dir=save_dir,
            cmap=self.cmap,
            highlight_color='#25A085',
            title_prefix=f"Cluster Merging Visualization\nMethod: {method.upper()}, k={final_results.n_clusters}",
            num_clusters=10
        )
    
    def plot_cluster_spectrum_visualization(self, umap_embeddings: np.ndarray,
                                          quantum_probs: np.ndarray,
                                          initial_medoid_embeddings_plot: np.ndarray,
                                          final_results: ClusteringResults,
                                          save_path: str, method: str, timestamp: str,
                                          initial_colors: Dict[int, Any],
                                          color_correspondence: Dict[int, int]):
        spectrum_title = ("Cluster Color Spectrum\n" + 
                         f"Method: {method.upper()}, k={final_results.n_clusters}, "
                         f"DBI={final_results.dbi_score:.4f}, {timestamp}")
        
        plot_cluster_spectrum(
            umap_embeddings,
            quantum_probs,
            medoids=initial_medoid_embeddings_plot,
            refined_medoids=final_results.medoid_embeddings_plot,
            title=spectrum_title,
            save_path=save_path,
            cmap=self.cmap,
            cluster_colors=initial_colors,
            color_correspondence=color_correspondence
        )


class ResultsManager:
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_embeddings(self, **kwargs):
        for name, data in kwargs.items():
            if data is not None:
                np.save(self.output_dir / f"{name}.npy", data)
    
    def save_cluster_mapping(self, doc_ids: List[str], labels: np.ndarray, 
                           filename: str = "doc_clusters.csv"):
        df = pd.DataFrame({'doc_id': doc_ids, 'cluster': labels})
        df.to_csv(self.output_dir / filename, index=False)
    
    def save_initial_clustering_results(self, initial_labels: np.ndarray, doc_ids: List[str], 
                                      method_name: str, timestamp: str):
        initial_cluster_mapping = pd.DataFrame({
            'doc_id': doc_ids,
            'initial_cluster': initial_labels
        })
        initial_cluster_mapping.to_csv(self.output_dir / "initial_doc_clusters.csv", index=False)
        
        initial_cluster_sizes = pd.Series(initial_labels).value_counts().sort_index()
        initial_cluster_sizes_df = pd.DataFrame({
            'cluster_id': initial_cluster_sizes.index,
            'size': initial_cluster_sizes.values
        })
        initial_cluster_sizes_df.to_csv(self.output_dir / "initial_cluster_sizes.csv", index=False)
        
        print(f"\nInitial {method_name.upper()} cluster distribution summary:")
        print(f"Number of clusters: {len(initial_cluster_sizes)}")
        print(f"Min cluster size: {initial_cluster_sizes.min()}")
        print(f"Max cluster size: {initial_cluster_sizes.max()}")
        print(f"Mean cluster size: {initial_cluster_sizes.mean():.2f}")
        print(f"Median cluster size: {initial_cluster_sizes.median():.2f}")
        
        plt.figure(figsize=(12, 6))
        
        sorted_indices = np.argsort(-initial_cluster_sizes.values)
        sorted_clusters = initial_cluster_sizes.index[sorted_indices]
        sorted_sizes = initial_cluster_sizes.values[sorted_indices]
        
        plt.bar(range(len(sorted_clusters)), sorted_sizes)
        plt.title(f"Initial {method_name.upper()} Cluster Size Distribution")
        plt.xlabel("Cluster ID (sorted by size)")
        plt.ylabel("Number of Documents")
        
        plt.axhline(y=initial_cluster_sizes.mean(), color='r', linestyle='-', 
                    label=f'Mean Size: {initial_cluster_sizes.mean():.1f}')
        plt.legend()
        
        plt.tight_layout()
        initial_dist_plot_path = self.output_dir / f"initial_cluster_distribution_{timestamp}.png"
        plt.savefig(initial_dist_plot_path)
        plt.close()
        print(f"Saved initial cluster distribution plot at: {initial_dist_plot_path}")
        
        top_clusters = initial_cluster_sizes.nlargest(10)
        total_docs = sum(initial_cluster_sizes)
        top_percentage = sum(top_clusters) / total_docs * 100
        
        print(f"Top 10 clusters contain {top_percentage:.2f}% of all documents")
        
        top_clusters_df = pd.DataFrame({
            'cluster_id': top_clusters.index,
            'size': top_clusters.values,
            'percentage': (top_clusters.values / total_docs * 100).round(2)
        })
        top_clusters_df.to_csv(self.output_dir / "initial_top_clusters.csv", index=False)
        
        return initial_cluster_sizes
    
    def compare_cluster_distributions(self, initial_labels: np.ndarray, 
                                    final_labels: np.ndarray, method: str, timestamp: str):
        try:
            initial_sizes = pd.Series(initial_labels).value_counts()
            final_sizes = pd.Series(final_labels).value_counts()
            
            plt.figure(figsize=(12, 10))
            
            sorted_init_indices = np.argsort(-initial_sizes.values)
            sorted_init_clusters = initial_sizes.index[sorted_init_indices]
            sorted_init_sizes = initial_sizes.values[sorted_init_indices]
            
            plt.subplot(2, 1, 1)
            plt.bar(range(len(sorted_init_clusters)), sorted_init_sizes)
            plt.title(f"Initial {len(initial_sizes)} Clusters Size Distribution")
            plt.xlabel("Cluster ID (sorted by size)")
            plt.ylabel("Number of Documents")
            plt.yscale('log')
            
            plt.subplot(2, 1, 2)
            plt.bar(final_sizes.index, final_sizes.values)
            plt.title(f"Final {len(final_sizes)} Clusters Size Distribution")
            plt.xlabel("Cluster ID")
            plt.ylabel("Number of Documents")
            plt.yscale('log')
            
            plt.tight_layout()
            comparison_plot_path = self.output_dir / f"cluster_size_comparison_{timestamp}.png"
            plt.savefig(comparison_plot_path)
            plt.close()
            print(f"Saved cluster size comparison plot at: {comparison_plot_path}")
            
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
            
            stats_df.to_csv(self.output_dir / "cluster_stats_comparison.csv")
            
            print("\n=== Cluster Distribution Comparison ===")
            print(f"Initial clusters: {initial_stats['num_clusters']}, Final clusters: {final_stats['num_clusters']}")
            print(f"Initial top 3 clusters: {initial_stats['top3_pct']:.2f}% of documents")
            print(f"Final top 3 clusters: {final_stats['top3_pct']:.2f}% of documents")
            
        except Exception as e:
            print(f"Error creating cluster distribution comparison: {str(e)}")
    
    def save_run_info(self, run_info: Dict[str, Any], timestamp: str):
        with open(self.output_dir / f"run_info_{timestamp}.json", 'w') as f:
            json.dump(run_info, f, indent=2)
    
    def track_cluster_correspondence(self, initial_labels: np.ndarray, 
                                   final_labels: np.ndarray) -> Dict[int, int]:
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
        
        df = pd.DataFrame({
            'final_cluster': list(correspondence.keys()),
            'initial_cluster': list(correspondence.values())
        })
        df.to_csv(self.output_dir / "cluster_correspondence.csv", index=False)
        
        return correspondence
    
    def save_probabilistic_results(self, clustering_object, doc_ids: List[str], 
                                 method: str, output_dir: str):
        if hasattr(clustering_object, 'get_membership_probabilities'):
            membership_probs = clustering_object.get_membership_probabilities()
            np.save(Path(output_dir) / f"{method}_membership_probs.npy", membership_probs)
            
            if hasattr(clustering_object, 'get_top_documents_per_cluster'):
                top_docs = clustering_object.get_top_documents_per_cluster(doc_ids, n=10)
                with open(Path(output_dir) / f"{method}_top_docs_per_cluster.txt", 'w') as f:
                    for cluster_id, docs in top_docs.items():
                        f.write(f"Cluster {cluster_id}:\n")
                        for doc_id, prob in docs:
                            f.write(f"  {doc_id}: {prob:.4f}\n")
                        f.write("\n")
            
            if hasattr(clustering_object, 'save_cluster_membership'):
                clustering_object.save_cluster_membership(
                    doc_ids, Path(output_dir) / f"{method}_cluster_membership.csv"
                )


class ClusteringPipeline:
    
    def __init__(self, config: Any, data_dir: str, colormaps_dir: str):
        self.config = config
        self.data_manager = DataManager(data_dir)
        self.colormaps_dir = colormaps_dir
        self.refinement = ClusterRefinement(config)
        self.evaluator = EvaluationEngine()
    
    def run(self, run_config: RunConfiguration) -> Dict[str, Any]:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        np.random.seed(self.config.classical_clustering.random_state)
        
        refinement_method = self._get_refinement_method(run_config)
        output_dir = (f"{self.data_manager.data_dir}/run_{timestamp}_"
                     f"{run_config.method}_{refinement_method}")
        
        results_manager = ResultsManager(output_dir)
        viz_manager = VisualizationManager(run_config.colormap, self.colormaps_dir)
        
        print("Loading data...")
        doc_embeddings, doc_ids = self.data_manager.load_document_data()
        query_df = self.data_manager.load_query_data()
        
        print("Performing dimensionality reduction...")
        umap_reducer = UMAPReducer(random_state=self.config.classical_clustering.random_state)
        umap_reduced_dimensions = umap_reducer.fit_transform(doc_embeddings)
        
        if self.config.general.reduction:
            doc_embeddings_reduced = umap_reducer.fit_transform(doc_embeddings)
        else:
            doc_embeddings_reduced = doc_embeddings
        
        results_manager.save_embeddings(
            doc_embeddings_reduced=doc_embeddings_reduced,
            umap_embeddings=umap_reduced_dimensions
        )
        
        umap_plot_path = str(results_manager.output_dir / f"umap_plot_{timestamp}.png")
        viz_manager.plot_umap_reduction(umap_reduced_dimensions, umap_plot_path, 
                                       run_config.method, timestamp)
        
        print(f"Performing {run_config.method} clustering...")
        strategy = ClusteringStrategyFactory.create(run_config.method)
        initial_results = strategy.cluster(doc_embeddings_reduced, doc_embeddings, 
                                         umap_reduced_dimensions, self.config)
        
        has_probabilities = run_config.method in ['gmm', 'hdbscan-gmm']
        if run_config.multi_membership and not has_probabilities:
            print(f"Warning: Multi-membership requires 'gmm' or 'hdbscan-gmm' method. "
                  f"Requested method '{run_config.method}' doesn't provide probabilities.")
            run_config.multi_membership = False
        
        results_manager.save_initial_clustering_results(
            initial_results.labels, doc_ids, run_config.method, timestamp
        )
        results_manager.save_embeddings(
            medoid_embeddings_reduced=initial_results.medoid_embeddings,
            medoid_embeddings_original=initial_results.medoid_embeddings_original,
            medoid_indices=initial_results.medoid_indices
        )
        
        if initial_results.clustering_object and has_probabilities:
            results_manager.save_probabilistic_results(
                initial_results.clustering_object, doc_ids, run_config.method, str(results_manager.output_dir)
            )
        
        print(f"{run_config.method.capitalize()} Clustering Labels: {initial_results.labels}")
        print(f"{run_config.method.capitalize()} Medoid Indices: {initial_results.medoid_indices}")
        
        initial_colors = viz_manager.create_initial_colors(initial_results.labels)
        
        initial_plot_path = str(results_manager.output_dir / f"{run_config.method}_clusters_{timestamp}.png")
        viz_manager.plot_initial_clustering(
            umap_reduced_dimensions, initial_results,
            initial_plot_path, run_config.method, timestamp, initial_colors
        )
        
        print(f"Applying refinement method: {refinement_method}")
        final_results = self.refinement.refine_clusters(
            initial_results, doc_embeddings_reduced, doc_embeddings, umap_reduced_dimensions,
            run_config.method, run_config.annealing, run_config.second_stage,
            doc_ids, str(results_manager.output_dir)
        )
        
        print(f"Final chosen k after {refinement_method} refinement: {final_results.n_clusters}")
        
        results_manager.save_embeddings(
            refined_medoid_embeddings_original=final_results.medoid_embeddings_original,
            refined_medoid_embeddings=final_results.medoid_embeddings,
            refined_medoid_indices=final_results.medoid_indices
        )
        
        np.save(results_manager.output_dir / f"final_{refinement_method}_clusters.npy", 
                final_results.labels)
        results_manager.save_cluster_mapping(doc_ids, final_results.labels)
        
        color_correspondence = results_manager.track_cluster_correspondence(
            initial_results.labels, final_results.labels
        )
        
        results_manager.compare_cluster_distributions(
            initial_results.labels, final_results.labels, 
            run_config.method, timestamp
        )
        
        multi_membership_df = None
        quantum_probs = None
        
        if has_probabilities:
            quantum_probs = self._create_quantum_probabilities(
                initial_results, final_results, len(doc_ids), results_manager.output_dir
            )
            
            if run_config.multi_membership:
                print(f"Creating multi-membership assignments with threshold={run_config.threshold}")
                multi_membership_df = create_multi_membership_assignments(
                    doc_ids, doc_embeddings_reduced, initial_results.membership_probs,
                    final_results.labels, final_results.medoid_indices,
                    final_results.medoid_embeddings, threshold=run_config.threshold,
                    data_dir=str(results_manager.output_dir), prefix=run_config.method
                )
        
        print(f"Final Refined Medoid Embeddings:\n {final_results.medoid_embeddings}")
        print(f"Unique Cluster Assignments: {np.unique(final_results.labels)}")
        
        print("\nEvaluating retrieval metrics on full dataset...")
        evaluation_metrics = self._evaluate_pipeline(
            query_df, doc_embeddings, final_results, doc_ids, multi_membership_df
        )
        
        final_plot_path = str(results_manager.output_dir / f"final_clusters_{timestamp}.png")
        
        viz_manager.plot_final_clustering(
            umap_reduced_dimensions, final_results, initial_results.medoid_embeddings_plot,
            final_plot_path, run_config.method, refinement_method, timestamp, 
            initial_colors, color_correspondence
        )
        
        merged_clusters_dir = str(results_manager.output_dir / f"merged_clusters_{timestamp}")
        os.makedirs(merged_clusters_dir, exist_ok=True)
        
        viz_manager.plot_merged_clusters_visualization(
            umap_reduced_dimensions, initial_results.labels, final_results.labels,
            color_correspondence, merged_clusters_dir, run_config.method,
            final_results, timestamp
        )
        print(f"Saved top merged clusters visualizations to {merged_clusters_dir}")
        
        if has_probabilities and quantum_probs is not None:
            spectrum_plot_path = str(results_manager.output_dir / f"cluster_spectrum_{timestamp}.png")
            viz_manager.plot_cluster_spectrum_visualization(
                umap_reduced_dimensions, quantum_probs, initial_results.medoid_embeddings_plot,
                final_results, spectrum_plot_path, run_config.method, timestamp,
                initial_colors, color_correspondence
            )
            print(f"Saved cluster spectrum plot at: {spectrum_plot_path}")
        
        print(f"Saved final {refinement_method} cluster plot at: {final_plot_path}")
        
        run_info = self._create_run_info(
            run_config, timestamp, initial_results, final_results,
            evaluation_metrics, multi_membership_df, refinement_method
        )
        
        results_manager.save_run_info(run_info, timestamp)
        
        summary_path = (Path(self.data_manager.data_dir) / 
                       f"run_summary_{run_config.method}_{refinement_method}_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(run_info, f, indent=2)
        
        self._print_summary(run_info, str(results_manager.output_dir))
        
        plt.show()
        
        return run_info
    
    def _get_refinement_method(self, run_config: RunConfiguration) -> str:
        if run_config.annealing:
            return "quantum"
        elif run_config.second_stage and run_config.method == 'classical':
            return "second_stage"
        else:
            return "none"
    
    def _create_quantum_probabilities(self, initial_results: ClusteringResults,
                                    final_results: ClusteringResults, n_docs: int,
                                    output_dir: Path) -> np.ndarray:
        n_quantum_clusters = len(np.unique(final_results.labels))
        quantum_probs = np.zeros((n_docs, n_quantum_clusters))

        if hasattr(initial_results.clustering_object, 'membership_probs'):
            component_to_quantum = {}
            
            for comp_idx in range(initial_results.membership_probs.shape[1]):
                counts = np.zeros(n_quantum_clusters)
                for doc_idx, prob in enumerate(initial_results.membership_probs[:, comp_idx]):
                    if prob > 0.1:
                        quantum_cluster = final_results.labels[doc_idx]
                        counts[quantum_cluster] += prob
                
                if np.sum(counts) > 0:
                    component_to_quantum[comp_idx] = np.argmax(counts)
            
            for doc_idx in range(n_docs):
                doc_probs = initial_results.membership_probs[doc_idx, :]
                
                for comp_idx, quantum_idx in component_to_quantum.items():
                    quantum_probs[doc_idx, quantum_idx] += doc_probs[comp_idx]
                    
            row_sums = quantum_probs.sum(axis=1, keepdims=True)
            quantum_probs = np.divide(quantum_probs, row_sums, 
                                        out=np.zeros_like(quantum_probs), 
                                        where=row_sums != 0)
            
            np.save(output_dir / "final_probabilities.npy", quantum_probs)
        
        return quantum_probs
    
    def _evaluate_pipeline(self, query_df: pd.DataFrame, doc_embeddings: np.ndarray,
                          final_results: ClusteringResults, doc_ids: List[str],
                          multi_membership_df: Optional[pd.DataFrame]) -> EvaluationMetrics:
        if len(query_df) == 0:
            print("No valid queries found for evaluation.")
            return EvaluationMetrics(ndcg_10=0.0, relevant_coverage=0.0)
        
        try:
            unique_query_embeddings = np.stack(query_df["query_embeddings"].values)
            qrels_df = query_df[['query_id', 'doc_id', 'relevance']]
            
            if len(unique_query_embeddings) >= 5:
                print("Checking first 5 query embeddings after deduplication:")
                for i in range(5):
                    print(f"Query {query_df['query_id'].iloc[i]} first 5 values: {unique_query_embeddings[i][:5]}")
            
            from src.models.QuantumClustering import compute_clusters
            original_space_cluster_labels = compute_clusters(doc_embeddings, final_results.medoid_indices)
            
            return self.evaluator.evaluate_retrieval(
                unique_query_embeddings,
                doc_embeddings,
                final_results.medoid_embeddings_original,
                original_space_cluster_labels,
                qrels_df,
                doc_ids,
                k=10,
                multi_cluster_assignments=multi_membership_df
            )
        
        except Exception as e:
            print(f"Error evaluating retrieval metrics on full dataset: {str(e)}")
            return EvaluationMetrics(ndcg_10=0.0, relevant_coverage=0.0)
    
    def _create_run_info(self, run_config: RunConfiguration, timestamp: str,
                        initial_results: ClusteringResults, final_results: ClusteringResults,
                        evaluation_metrics: EvaluationMetrics,
                        multi_membership_df: Optional[pd.DataFrame],
                        refinement_method: str) -> Dict[str, Any]:
        
        run_info = {
            'timestamp': timestamp,
            'clustering_method': run_config.method,
            'colormap': run_config.colormap,
            'cv_enabled': False,
            'cv_folds': None,
            'multi_membership': run_config.multi_membership,
            'threshold': run_config.threshold if run_config.multi_membership else None,
            'annealing': run_config.annealing,
            'second_stage': run_config.second_stage,
            'hyperparameters': self._extract_hyperparameters(run_config),
            'results': {
                'initial_clusters': initial_results.n_clusters,
                'initial_dbi': float(initial_results.dbi_score),
                'final_clusters': final_results.n_clusters,
                'final_dbi': float(final_results.dbi_score),
                'ndcg_10': float(evaluation_metrics.ndcg_10),
                'relevant_coverage': float(evaluation_metrics.relevant_coverage)
            }
        }
        
        if evaluation_metrics.ndcg_multi_10 is not None:
            run_info['results']['ndcg_multi_10'] = float(evaluation_metrics.ndcg_multi_10)
            run_info['results']['coverage_multi'] = float(evaluation_metrics.coverage_multi)
        
        if multi_membership_df is not None:
            membership_counts = multi_membership_df['membership_count'].values
            run_info['results']['multi_membership'] = {
                'avg_memberships': float(np.mean(membership_counts)),
                'max_memberships': int(np.max(membership_counts)),
                'docs_with_multiple': int(np.sum(membership_counts > 1)),
                'percent_multi': float((np.sum(membership_counts > 1) / len(membership_counts)) * 100)
            }
        
        return run_info
    
    def _extract_hyperparameters(self, run_config: RunConfiguration) -> Dict[str, Any]:
        hyperparams = {}
        
        if run_config.method == 'classical':
            hyperparams['classical'] = dict(self.config.classical_clustering)
            if run_config.second_stage and hasattr(self.config, 'second_stage_clustering'):
                hyperparams['second_stage'] = dict(self.config.second_stage_clustering)
        elif run_config.method == 'hdbscan':
            hyperparams['hdbscan'] = dict(self.config.hdbscan_clustering)
        elif run_config.method == 'gmm':
            hyperparams['gmm'] = dict(self.config.gmm_clustering)
        elif run_config.method == 'hdbscan-gmm':
            hyperparams['hdbscan_gmm'] = dict(self.config.hdbscan_gmm_clustering)
        
        if run_config.annealing:
            hyperparams['quantum'] = dict(self.config.quantum_clustering)
        
        return hyperparams
    
    def _print_summary(self, run_info: Dict[str, Any], output_dir: str):
        print("\n=== Clustering Run Summary ===")
        print(f"Timestamp: {run_info['timestamp']}")
        print(f"Method: {run_info['clustering_method']}" + 
              (" with Multi-Membership" if run_info['multi_membership'] else ""))
        print(f"Refinement: {run_info.get('refinement_method', 'none')}")
        print(f"Initial Clusters: {run_info['results']['initial_clusters']}, "
              f"DBI: {run_info['results']['initial_dbi']:.4f}")
        print(f"Final Clusters: {run_info['results']['final_clusters']}, "
              f"DBI: {run_info['results']['final_dbi']:.4f}")
        print(f"nDCG@10: {run_info['results']['ndcg_10']:.4f}, "
              f"Relevant Coverage: {run_info['results']['relevant_coverage']:.4f}")
        
        if run_info['multi_membership'] and 'multi_membership' in run_info['results']:
            if 'ndcg_multi_10' in run_info['results'] and 'coverage_multi' in run_info['results']:
                print(f"Multi-Membership nDCG@10: {run_info['results']['ndcg_multi_10']:.4f}")
                print(f"Multi-Membership Coverage: {run_info['results']['coverage_multi']:.4f}")
            mm_stats = run_info['results']['multi_membership']
            print(f"Multi-Membership: {mm_stats['percent_multi']:.1f}% of documents belong to multiple clusters")
            print(f"Average memberships per document: {mm_stats['avg_memberships']:.2f}")
        
        print(f"All results saved to: {output_dir}")
        print("===============================")


def create_pipeline_from_config(config_path: str, data_dir: str, colormaps_dir: str) -> ClusteringPipeline:
    import yaml
    from box import ConfigBox
    
    with open(config_path, "r") as file:
        config = ConfigBox(yaml.safe_load(file))
    
    config_dir = Path(config_path).parent
    
    kmedoids_config_path = config_dir / "kmedoids.yml"
    if kmedoids_config_path.exists():
        with open(kmedoids_config_path, "r") as file:
            kmedoids_config = ConfigBox(yaml.safe_load(file))
            config.update(kmedoids_config)
    
    method_configs = {
        'hdbscan.yml': {
            'hdbscan_clustering': {
                'min_cluster_size': 20,
                'min_samples': 25,
                'cluster_selection_method': 'leaf',
                'cluster_selection_epsilon': 0.2,
                'metric': 'euclidean',
                'random_state': getattr(config.classical_clustering, 'random_state', 42)
            }
        },
        'gmm.yml': {
            'gmm_clustering': {
                'n_components_range': [10, 25, 50, 75, 100],
                'covariance_type': 'full',
                'n_init': 10,
                'random_state': getattr(config.classical_clustering, 'random_state', 42)
            }
        },
        'hdbscan_gmm.yml': {
            'hdbscan_gmm_clustering': {
                'min_cluster_size': 20,
                'min_samples': 25,
                'cluster_selection_method': 'leaf',
                'cluster_selection_epsilon': 0.2,
                'covariance_type': 'full',
                'n_init': 10,
                'metric': 'euclidean',
                'random_state': getattr(config.classical_clustering, 'random_state', 42)
            }
        }
    }
    
    for config_file, default_config in method_configs.items():
        config_file_path = config_dir / config_file
        if config_file_path.exists():
            try:
                with open(config_file_path, "r") as file:
                    method_config = ConfigBox(yaml.safe_load(file))
                    config.update(method_config)
            except FileNotFoundError:
                print(f"Config file {config_file} not found, using defaults")
                config.update(ConfigBox(default_config))
        else:
            print(f"Config file {config_file} not found, using defaults")
            config.update(ConfigBox(default_config))
    
    return ClusteringPipeline(config, data_dir, colormaps_dir)


def main():
    import argparse
    import yaml
    from box import ConfigBox
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))
    default_colormaps_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "colormaps"))
    
    parser = argparse.ArgumentParser(description='Run clustering pipeline with clean architecture')
    parser.add_argument('--colormap', type=str, default='Spectral', 
                        help='Colormap to use (file in colormaps dir or matplotlib name)')
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
    
    run_config = RunConfiguration(
        method=args.method,
        colormap=args.colormap,
        multi_membership=args.multi_membership,
        threshold=args.threshold,
        annealing=not args.no_annealing,
        second_stage=args.no_annealing and args.method == 'classical'
    )
    
    pipeline = ClusteringPipeline(config, default_data_dir, default_colormaps_dir)
    results = pipeline.run(run_config)
    
    return results


if __name__ == "__main__":
    main()