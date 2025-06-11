def create_multi_membership_assignments(doc_ids, embeddings, initial_probs, quantum_labels, 
                                  quantum_medoid_indices, refined_medoid_embeddings, 
                                  threshold=0.2, data_dir=None, prefix=''):

    import numpy as np
    import pandas as pd
    import os
    from sklearn.metrics.pairwise import pairwise_distances
    
    prefix = f"{prefix}_" if prefix else ""
    print(f"Creating multi-membership assignments with {prefix}probabilities (threshold: {threshold})...")
    
    n_components = initial_probs.shape[1]
    
    n_quantum_clusters = len(np.unique(quantum_labels))
    
    n_docs = len(doc_ids)
    
    component_to_quantum = {}
    
    for comp_idx in range(n_components):
        counts = np.zeros(n_quantum_clusters)
        for doc_idx, prob in enumerate(initial_probs[:, comp_idx]):
            if prob > 0.1:
                quantum_cluster = quantum_labels[doc_idx]
                counts[quantum_cluster] += prob
        
        if np.sum(counts) > 0:
            component_to_quantum[comp_idx] = np.argmax(counts)
    
    quantum_probs = np.zeros((n_docs, n_quantum_clusters))
    
    for doc_idx in range(n_docs):
        doc_probs = initial_probs[doc_idx, :]
        
        for comp_idx, quantum_idx in component_to_quantum.items():
            quantum_probs[doc_idx, quantum_idx] += doc_probs[comp_idx]
    
    row_sums = quantum_probs.sum(axis=1, keepdims=True)
    quantum_probs = np.divide(quantum_probs, row_sums, 
                             out=np.zeros_like(quantum_probs), 
                             where=row_sums != 0)
    
    data = {
        'doc_id': doc_ids,
        'primary_cluster': quantum_labels
    }
    
    for cluster_id in range(n_quantum_clusters):
        data[f'cluster_{cluster_id}_prob'] = quantum_probs[:, cluster_id]
    
    data['most_likely_cluster'] = np.argmax(quantum_probs, axis=1)
    
    multi_memberships = []
    for doc_idx in range(n_docs):
        doc_clusters = np.where(quantum_probs[doc_idx, :] >= threshold)[0]
        if len(doc_clusters) == 0:
            doc_clusters = [np.argmax(quantum_probs[doc_idx, :])]
        multi_memberships.append(doc_clusters.tolist())
    
    data['multi_membership'] = multi_memberships
    
    data['membership_count'] = [len(clusters) for clusters in multi_memberships]
    
    membership_counts = np.array([len(clusters) for clusters in multi_memberships])
    avg_memberships = np.mean(membership_counts)
    max_memberships = np.max(membership_counts)
    docs_with_multiple = np.sum(membership_counts > 1)
    percent_multi = (docs_with_multiple / n_docs) * 100
    
    print(f"Multi-membership statistics (threshold: {threshold}):")
    print(f"  Average memberships per document: {avg_memberships:.2f}")
    print(f"  Maximum memberships for a document: {max_memberships}")
    print(f"  Documents with multiple memberships: {docs_with_multiple}/{n_docs} ({percent_multi:.1f}%)")
    
    df = pd.DataFrame(data)
    
    if data_dir:
        output_file = os.path.join(data_dir, f"{prefix}multi_membership_t{int(threshold*100)}.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved multi-membership assignments to {output_file}")
        
        rows = []
        for doc_idx, doc_id in enumerate(doc_ids):
            for cluster_id in multi_memberships[doc_idx]:
                prob = quantum_probs[doc_idx, cluster_id]
                rows.append({
                    'doc_id': doc_id,
                    'cluster_id': cluster_id,
                    'probability': prob,
                    'is_primary': (quantum_labels[doc_idx] == cluster_id)
                })
        
        expanded_df = pd.DataFrame(rows)
        expanded_file = os.path.join(data_dir, f"{prefix}multi_membership_expanded_t{int(threshold*100)}.csv")
        expanded_df.to_csv(expanded_file, index=False)
        print(f"Saved expanded multi-membership format to {expanded_file}")
    
    return df