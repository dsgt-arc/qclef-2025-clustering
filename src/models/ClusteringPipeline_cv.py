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
from sklearn.model_selection import KFold, StratifiedKFold

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
class CVFoldResults:
    fold_idx: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_results: ClusteringResults
    test_metrics: EvaluationMetrics
    test_labels: np.ndarray
    test_medoid_indices: np.ndarray


@dataclass
class CVAggregateResults:
    mean_ndcg: float
    std_ndcg: float
    mean_coverage: float
    std_coverage: float
    mean_clusters: float
    std_clusters: float
    mean_dbi: float
    std_dbi: float
    fold_results: List[CVFoldResults]
    best_fold_idx: int
    mean_ndcg_multi: Optional[float] = None
    std_ndcg_multi: Optional[float] = None
    mean_coverage_multi: Optional[float] = None
    std_coverage_multi: Optional[float] = None


@dataclass
class RunConfiguration:
    method: str
    colormap: str
    multi_membership: bool
    threshold: float
    annealing: bool
    second_stage: bool
    cv_folds: int = 5
    cv_strategy: str = 'kfold'
    cv_random_state: int = 42


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


class CVSplitter:
    def __init__(self, cv_strategy: str = 'kfold', n_splits: int = 5, random_state: int = 42):
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.random_state = random_state
    
    def create_splits(self, query_df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        n_queries = len(query_df)
        
        if self.cv_strategy == 'kfold':
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            return list(kf.split(range(n_queries)))
        elif self.cv_strategy == 'stratified':
            relevance_labels = self._create_relevance_labels(query_df)
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            return list(skf.split(range(n_queries), relevance_labels))
        else:
            raise ValueError(f"Unknown CV strategy: {self.cv_strategy}")
    
    def _create_relevance_labels(self, query_df: pd.DataFrame) -> np.ndarray:
        labels = []
        for _, row in query_df.iterrows():
            if row['relevance'] > 2:
                labels.append('high')
            elif row['relevance'] > 0:
                labels.append('medium')
            else:
                labels.append('low')
        return np.array(labels)


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
            
            all_doc_similarities = np.dot(query_embedding, doc_embeddings_norm.T)[0]
            top_k_all_indices = np.argsort(-all_doc_similarities)[:k]
            top_k_all_doc_ids = [doc_ids[idx] for idx in top_k_all_indices]

            all_retrieved_relevant = relevant_doc_ids.intersection(set(top_k_all_doc_ids))
            coverage = len(all_retrieved_relevant) / len(relevant_doc_ids) if relevant_doc_ids else 0.0
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


class TestAssigner:
    def __init__(self, config: Any):
        self.config = config
    
    def assign_test_documents(self, 
                            train_results: ClusteringResults,
                            test_embeddings: np.ndarray,
                            test_embeddings_original: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        test_labels = compute_clusters(test_embeddings, train_results.medoid_indices)
        test_medoid_indices = train_results.medoid_indices
        
        return test_labels, test_medoid_indices


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
    
    def plot_cv_fold_results(self, fold_results: List[CVFoldResults], 
                           output_dir: str, method: str, timestamp: str):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Cross-Validation Results - {method.upper()}', fontsize=16)
        
        ndcg_scores = [fold.test_metrics.ndcg_10 for fold in fold_results]
        coverage_scores = [fold.test_metrics.relevant_coverage for fold in fold_results]
        cluster_counts = [fold.train_results.n_clusters for fold in fold_results]
        dbi_scores = [fold.train_results.dbi_score for fold in fold_results]
        
        fold_indices = range(1, len(fold_results) + 1)
        
        axes[0, 0].bar(fold_indices, ndcg_scores)
        axes[0, 0].set_title('nDCG@10 by Fold')
        axes[0, 0].set_xlabel('Fold')
        axes[0, 0].set_ylabel('nDCG@10')
        axes[0, 0].axhline(y=np.mean(ndcg_scores), color='r', linestyle='--', label=f'Mean: {np.mean(ndcg_scores):.3f}')
        axes[0, 0].legend()
        
        axes[0, 1].bar(fold_indices, coverage_scores)
        axes[0, 1].set_title('Coverage by Fold')
        axes[0, 1].set_xlabel('Fold')
        axes[0, 1].set_ylabel('Coverage')
        axes[0, 1].axhline(y=np.mean(coverage_scores), color='r', linestyle='--', label=f'Mean: {np.mean(coverage_scores):.3f}')
        axes[0, 1].legend()
        
        axes[0, 2].bar(fold_indices, cluster_counts)
        axes[0, 2].set_title('Number of Clusters by Fold')
        axes[0, 2].set_xlabel('Fold')
        axes[0, 2].set_ylabel('Clusters')
        axes[0, 2].axhline(y=np.mean(cluster_counts), color='r', linestyle='--', label=f'Mean: {np.mean(cluster_counts):.1f}')
        axes[0, 2].legend()
        
        axes[1, 0].bar(fold_indices, dbi_scores)
        axes[1, 0].set_title('Davies-Bouldin Index by Fold')
        axes[1, 0].set_xlabel('Fold')
        axes[1, 0].set_ylabel('DBI Score')
        axes[1, 0].axhline(y=np.mean(dbi_scores), color='r', linestyle='--', label=f'Mean: {np.mean(dbi_scores):.3f}')
        axes[1, 0].legend()
        
        axes[1, 1].boxplot([ndcg_scores, coverage_scores], labels=['nDCG@10', 'Coverage'])
        axes[1, 1].set_title('Score Distributions')
        axes[1, 1].set_ylabel('Score')
        
        correlation = np.corrcoef(ndcg_scores, coverage_scores)[0, 1]
        axes[1, 2].scatter(ndcg_scores, coverage_scores)
        axes[1, 2].set_xlabel('nDCG@10')
        axes[1, 2].set_ylabel('Coverage')
        axes[1, 2].set_title(f'nDCG vs Coverage (r={correlation:.3f})')
        
        plt.tight_layout()
        cv_plot_path = Path(output_dir) / f"cv_results_{method}_{timestamp}.png"
        plt.savefig(cv_plot_path)
        plt.close()
        
        return str(cv_plot_path)


class ResultsManager:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_cv_results(self, cv_results: CVAggregateResults, timestamp: str):
        cv_summary = {
            'mean_ndcg': cv_results.mean_ndcg,
            'std_ndcg': cv_results.std_ndcg,
            'mean_coverage': cv_results.mean_coverage,
            'std_coverage': cv_results.std_coverage,
            'mean_clusters': cv_results.mean_clusters,
            'std_clusters': cv_results.std_clusters,
            'mean_dbi': cv_results.mean_dbi,
            'std_dbi': cv_results.std_dbi,
            'best_fold': cv_results.best_fold_idx,
            'num_folds': len(cv_results.fold_results)
        }
        
        if cv_results.mean_ndcg_multi is not None:
            cv_summary['mean_ndcg_multi'] = cv_results.mean_ndcg_multi
            cv_summary['std_ndcg_multi'] = cv_results.std_ndcg_multi
            cv_summary['mean_coverage_multi'] = cv_results.mean_coverage_multi
            cv_summary['std_coverage_multi'] = cv_results.std_coverage_multi
        
        with open(self.output_dir / f"cv_summary_{timestamp}.json", 'w') as f:
            json.dump(cv_summary, f, indent=2)
        
        fold_details = []
        for fold in cv_results.fold_results:
            fold_detail = {
                'fold_idx': fold.fold_idx,
                'train_clusters': fold.train_results.n_clusters,
                'train_dbi': fold.train_results.dbi_score,
                'test_ndcg': fold.test_metrics.ndcg_10,
                'test_coverage': fold.test_metrics.relevant_coverage,
                'train_size': len(fold.train_indices),
                'test_size': len(fold.test_indices)
            }
            
            if fold.test_metrics.ndcg_multi_10 is not None:
                fold_detail['test_ndcg_multi'] = fold.test_metrics.ndcg_multi_10
                fold_detail['test_coverage_multi'] = fold.test_metrics.coverage_multi
            
            fold_details.append(fold_detail)
        
        fold_df = pd.DataFrame(fold_details)
        fold_df.to_csv(self.output_dir / f"cv_fold_details_{timestamp}.csv", index=False)
        
        return cv_summary
    
    def save_embeddings(self, **kwargs):
        for name, data in kwargs.items():
            if data is not None:
                np.save(self.output_dir / f"{name}.npy", data)
    
    def save_cluster_mapping(self, doc_ids: List[str], labels: np.ndarray, 
                           filename: str = "doc_clusters.csv"):
        df = pd.DataFrame({'doc_id': doc_ids, 'cluster': labels})
        df.to_csv(self.output_dir / filename, index=False)
    
    def save_run_info(self, run_info: Dict[str, Any], timestamp: str):
        with open(self.output_dir / f"run_info_{timestamp}.json", 'w') as f:
            json.dump(run_info, f, indent=2)


class CrossValidationPipeline:
    def __init__(self, config: Any, data_dir: str, colormaps_dir: str):
        self.config = config
        self.data_manager = DataManager(data_dir)
        self.colormaps_dir = colormaps_dir
        self.refinement = ClusterRefinement(config)
        self.evaluator = EvaluationEngine()
        self.test_assigner = TestAssigner(config)
    
    def run(self, run_config: RunConfiguration) -> Dict[str, Any]:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        np.random.seed(self.config.classical_clustering.random_state)
        
        refinement_method = self._get_refinement_method(run_config)
        output_dir = (f"{self.data_manager.data_dir}/cv_run_{timestamp}_"
                     f"{run_config.method}_{refinement_method}_{run_config.cv_folds}fold")
        
        results_manager = ResultsManager(output_dir)
        viz_manager = VisualizationManager(run_config.colormap, self.colormaps_dir)
        
        print("Loading data...")
        doc_embeddings, doc_ids = self.data_manager.load_document_data()
        query_df = self.data_manager.load_query_data()
        
        if len(query_df) == 0:
            raise ValueError("No valid queries found for cross-validation")
        
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
        
        print(f"Setting up {run_config.cv_folds}-fold cross-validation...")
        cv_splitter = CVSplitter(
            cv_strategy=run_config.cv_strategy,
            n_splits=run_config.cv_folds,
            random_state=run_config.cv_random_state
        )
        
        splits = cv_splitter.create_splits(query_df)
        strategy = ClusteringStrategyFactory.create(run_config.method)
        
        has_probabilities = run_config.method in ['gmm', 'hdbscan-gmm']
        if run_config.multi_membership and not has_probabilities:
            print(f"Warning: Multi-membership requires 'gmm' or 'hdbscan-gmm' method. "
                  f"Requested method '{run_config.method}' doesn't provide probabilities.")
            run_config.multi_membership = False
        
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            print(f"\n=== Processing Fold {fold_idx + 1}/{run_config.cv_folds} ===")
            
            train_queries = query_df.iloc[train_idx]
            test_queries = query_df.iloc[test_idx]
            
            print(f"Train queries: {len(train_queries)}, Test queries: {len(test_queries)}")
            
            print(f"Performing {run_config.method} clustering on training data...")
            train_results = strategy.cluster(
                doc_embeddings_reduced, doc_embeddings, umap_reduced_dimensions, self.config
            )
            
            print(f"Applying refinement method: {refinement_method}")
            fold_output_dir = str(Path(output_dir) / f"fold_{fold_idx + 1}")
            os.makedirs(fold_output_dir, exist_ok=True)
            
            refined_train_results = self.refinement.refine_clusters(
                train_results, doc_embeddings_reduced, doc_embeddings, umap_reduced_dimensions,
                run_config.method, run_config.annealing, run_config.second_stage,
                doc_ids, fold_output_dir
            )
            
            print(f"Assigning test documents to clusters...")
            test_labels, test_medoid_indices = self.test_assigner.assign_test_documents(
                refined_train_results, doc_embeddings_reduced, doc_embeddings
            )
            
            multi_membership_df = None
            if has_probabilities and run_config.multi_membership:
                print(f"Creating multi-membership assignments for fold {fold_idx + 1}")
                multi_membership_df = create_multi_membership_assignments(
                    doc_ids, doc_embeddings_reduced, train_results.membership_probs,
                    refined_train_results.labels, refined_train_results.medoid_indices,
                    refined_train_results.medoid_embeddings, threshold=run_config.threshold,
                    data_dir=fold_output_dir, prefix=f"{run_config.method}_fold{fold_idx + 1}"
                )
            
            print(f"Evaluating on test queries...")
            test_metrics = self._evaluate_fold(
                test_queries, doc_embeddings, refined_train_results, doc_ids, multi_membership_df
            )
            
            fold_result = CVFoldResults(
                fold_idx=fold_idx + 1,
                train_indices=train_idx,
                test_indices=test_idx,
                train_results=refined_train_results,
                test_metrics=test_metrics,
                test_labels=test_labels,
                test_medoid_indices=test_medoid_indices
            )
            
            fold_results.append(fold_result)
            
            print(f"Fold {fold_idx + 1} Results:")
            print(f"  Train clusters: {refined_train_results.n_clusters}, DBI: {refined_train_results.dbi_score:.4f}")
            print(f"  Test nDCG@10: {test_metrics.ndcg_10:.4f}, Coverage: {test_metrics.relevant_coverage:.4f}")
            if test_metrics.ndcg_multi_10 is not None:
                print(f"  Test Multi nDCG@10: {test_metrics.ndcg_multi_10:.4f}, Multi Coverage: {test_metrics.coverage_multi:.4f}")
        
        print("\n=== Aggregating Cross-Validation Results ===")
        cv_aggregate = self._aggregate_cv_results(fold_results)
        
        cv_summary = results_manager.save_cv_results(cv_aggregate, timestamp)
        
        cv_plot_path = viz_manager.plot_cv_fold_results(
            fold_results, output_dir, run_config.method, timestamp
        )
        print(f"Saved CV results plot at: {cv_plot_path}")
        
        best_fold = fold_results[cv_aggregate.best_fold_idx]
        print(f"Best fold: {cv_aggregate.best_fold_idx + 1} (nDCG@10: {best_fold.test_metrics.ndcg_10:.4f})")
        
        results_manager.save_embeddings(
            best_fold_medoid_embeddings=best_fold.train_results.medoid_embeddings_original,
            best_fold_medoid_indices=best_fold.train_results.medoid_indices
        )
        
        np.save(Path(output_dir) / "best_fold_labels.npy", best_fold.train_results.labels)
        results_manager.save_cluster_mapping(doc_ids, best_fold.train_results.labels, "best_fold_clusters.csv")
        
        run_info = self._create_cv_run_info(
            run_config, timestamp, cv_aggregate, refinement_method
        )
        
        results_manager.save_run_info(run_info, timestamp)
        
        summary_path = (Path(self.data_manager.data_dir) / 
                       f"cv_summary_{run_config.method}_{refinement_method}_{run_config.cv_folds}fold_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(run_info, f, indent=2)
        
        self._print_cv_summary(run_info, str(results_manager.output_dir))
        
        return run_info
    
    def _get_refinement_method(self, run_config: RunConfiguration) -> str:
        if run_config.annealing:
            return "quantum"
        elif run_config.second_stage and run_config.method == 'classical':
            return "second_stage"
        else:
            return "none"
    
    def _evaluate_fold(self, test_queries: pd.DataFrame, doc_embeddings: np.ndarray,
                      train_results: ClusteringResults, doc_ids: List[str],
                      multi_membership_df: Optional[pd.DataFrame]) -> EvaluationMetrics:
        
        try:
            test_query_embeddings = np.stack(test_queries["query_embeddings"].values)
            test_qrels_df = test_queries[['query_id', 'doc_id', 'relevance']]
            
            original_space_cluster_labels = compute_clusters(doc_embeddings, train_results.medoid_indices)
            
            return self.evaluator.evaluate_retrieval(
                test_query_embeddings,
                doc_embeddings,
                train_results.medoid_embeddings_original,
                original_space_cluster_labels,
                test_qrels_df,
                doc_ids,
                k=10,
                multi_cluster_assignments=multi_membership_df
            )
        
        except Exception as e:
            print(f"Error evaluating fold: {str(e)}")
            return EvaluationMetrics(ndcg_10=0.0, relevant_coverage=0.0)
    
    def _aggregate_cv_results(self, fold_results: List[CVFoldResults]) -> CVAggregateResults:
        ndcg_scores = [fold.test_metrics.ndcg_10 for fold in fold_results]
        coverage_scores = [fold.test_metrics.relevant_coverage for fold in fold_results]
        cluster_counts = [fold.train_results.n_clusters for fold in fold_results]
        dbi_scores = [fold.train_results.dbi_score for fold in fold_results]
        
        best_fold_idx = np.argmax(ndcg_scores)
        
        ndcg_multi_scores = [fold.test_metrics.ndcg_multi_10 for fold in fold_results 
                           if fold.test_metrics.ndcg_multi_10 is not None]
        coverage_multi_scores = [fold.test_metrics.coverage_multi for fold in fold_results 
                               if fold.test_metrics.coverage_multi is not None]
        
        cv_results = CVAggregateResults(
            mean_ndcg=np.mean(ndcg_scores),
            std_ndcg=np.std(ndcg_scores),
            mean_coverage=np.mean(coverage_scores),
            std_coverage=np.std(coverage_scores),
            mean_clusters=np.mean(cluster_counts),
            std_clusters=np.std(cluster_counts),
            mean_dbi=np.mean(dbi_scores),
            std_dbi=np.std(dbi_scores),
            fold_results=fold_results,
            best_fold_idx=best_fold_idx
        )
        
        if ndcg_multi_scores:
            cv_results.mean_ndcg_multi = np.mean(ndcg_multi_scores)
            cv_results.std_ndcg_multi = np.std(ndcg_multi_scores)
            cv_results.mean_coverage_multi = np.mean(coverage_multi_scores)
            cv_results.std_coverage_multi = np.std(coverage_multi_scores)
        
        return cv_results
    
    def _create_cv_run_info(self, run_config: RunConfiguration, timestamp: str,
                          cv_results: CVAggregateResults, refinement_method: str) -> Dict[str, Any]:
        
        run_info = {
            'timestamp': timestamp,
            'clustering_method': run_config.method,
            'colormap': run_config.colormap,
            'cv_enabled': True,
            'cv_folds': run_config.cv_folds,
            'cv_strategy': run_config.cv_strategy,
            'cv_random_state': run_config.cv_random_state,
            'multi_membership': run_config.multi_membership,
            'threshold': run_config.threshold if run_config.multi_membership else None,
            'annealing': run_config.annealing,
            'second_stage': run_config.second_stage,
            'refinement_method': refinement_method,
            'hyperparameters': self._extract_hyperparameters(run_config),
            'cv_results': {
                'mean_ndcg': float(cv_results.mean_ndcg),
                'std_ndcg': float(cv_results.std_ndcg),
                'mean_coverage': float(cv_results.mean_coverage),
                'std_coverage': float(cv_results.std_coverage),
                'mean_clusters': float(cv_results.mean_clusters),
                'std_clusters': float(cv_results.std_clusters),
                'mean_dbi': float(cv_results.mean_dbi),
                'std_dbi': float(cv_results.std_dbi),
                'best_fold': cv_results.best_fold_idx + 1,
                'best_fold_ndcg': float(cv_results.fold_results[cv_results.best_fold_idx].test_metrics.ndcg_10)
            }
        }
        
        if cv_results.mean_ndcg_multi is not None:
            run_info['cv_results']['mean_ndcg_multi'] = float(cv_results.mean_ndcg_multi)
            run_info['cv_results']['std_ndcg_multi'] = float(cv_results.std_ndcg_multi)
            run_info['cv_results']['mean_coverage_multi'] = float(cv_results.mean_coverage_multi)
            run_info['cv_results']['std_coverage_multi'] = float(cv_results.std_coverage_multi)
        
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
    
    def _print_cv_summary(self, run_info: Dict[str, Any], output_dir: str):
        print("\n=== Cross-Validation Run Summary ===")
        print(f"Timestamp: {run_info['timestamp']}")
        print(f"Method: {run_info['clustering_method']}" + 
              (" with Multi-Membership" if run_info['multi_membership'] else ""))
        print(f"Refinement: {run_info['refinement_method']}")
        print(f"CV Folds: {run_info['cv_folds']} ({run_info['cv_strategy']})")
        
        cv_res = run_info['cv_results']
        print(f"Mean nDCG@10: {cv_res['mean_ndcg']:.4f} ± {cv_res['std_ndcg']:.4f}")
        print(f"Mean Coverage: {cv_res['mean_coverage']:.4f} ± {cv_res['std_coverage']:.4f}")
        print(f"Mean Clusters: {cv_res['mean_clusters']:.1f} ± {cv_res['std_clusters']:.1f}")
        print(f"Mean DBI: {cv_res['mean_dbi']:.4f} ± {cv_res['std_dbi']:.4f}")
        print(f"Best Fold: {cv_res['best_fold']} (nDCG@10: {cv_res['best_fold_ndcg']:.4f})")
        
        if run_info['multi_membership'] and 'mean_ndcg_multi' in cv_res:
            print(f"Multi-Membership Mean nDCG@10: {cv_res['mean_ndcg_multi']:.4f} ± {cv_res['std_ndcg_multi']:.4f}")
            print(f"Multi-Membership Mean Coverage: {cv_res['mean_coverage_multi']:.4f} ± {cv_res['std_coverage_multi']:.4f}")
        
        print(f"All results saved to: {output_dir}")
        print("===============================")


def create_cv_pipeline_from_config(config_path: str, data_dir: str, colormaps_dir: str) -> CrossValidationPipeline:
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
    
    return CrossValidationPipeline(config, data_dir, colormaps_dir)


def main():
    import argparse
    import yaml
    from box import ConfigBox
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "data"))
    default_colormaps_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "colormaps"))
    
    parser = argparse.ArgumentParser(description='Run cross-validation clustering pipeline')
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
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--cv-strategy', type=str, choices=['kfold', 'stratified'], default='kfold',
                        help='Cross-validation strategy (default: kfold)')
    parser.add_argument('--cv-random-state', type=int, default=42,
                        help='Random state for CV splitting (default: 42)')

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
        second_stage=args.no_annealing and args.method == 'classical',
        cv_folds=args.cv_folds,
        cv_strategy=args.cv_strategy,
        cv_random_state=args.cv_random_state
    )
    
    pipeline = CrossValidationPipeline(config, default_data_dir, default_colormaps_dir)
    results = pipeline.run(run_config)
    
    return results


if __name__ == "__main__":
    main()