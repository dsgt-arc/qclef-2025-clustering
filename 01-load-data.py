import os
import shutil
from pathlib import Path
import builtins

##############################################################################
# 1) Ensure Python uses UTF-8 by default
##############################################################################
os.environ["PYTHONUTF8"] = "1"

##############################################################################
# 2) Monkey-patch open() to force UTF-8 for text, leaving binary mode as-is
##############################################################################
_real_open = builtins.open

def _open_utf8(*args, **kwargs):
    if len(args) >= 2:
        mode = args[1]
    else:
        mode = kwargs.get('mode', 'r')
    # Only force UTF-8 if it's text mode
    if 'b' not in mode:
        kwargs.setdefault('encoding', 'utf-8')
        kwargs.setdefault('errors', 'replace')
    return _real_open(*args, **kwargs)

builtins.open = _open_utf8

##############################################################################
# 3) Delete cached ANTIQUE dataset so it re-downloads fresh under UTF-8
##############################################################################
ir_datasets_home = Path.home() / ".ir_datasets"
antique_dir = ir_datasets_home / "antique"
if antique_dir.exists():
    print(f"Deleting cached ANTIQUE data at: {antique_dir}")
    shutil.rmtree(antique_dir)

##############################################################################
# 4) Load ANTIQUE and create a pandas DataFrame
##############################################################################
# import ir_datasets
# import pandas as pd

# print("Loading ANTIQUE dataset...")
# dataset = ir_datasets.load("antique")

# documents = []
# for doc in dataset.docs_iter():
#     documents.append({"doc_id": doc.doc_id, "text": doc.text})

# df = pd.DataFrame(documents)
# print("Initial DataFrame created.")
# print(df.head())

# ##############################################################################
# # 5) Shuffle the DataFrame, then split into train/validation CSVs
# ##############################################################################
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# # Create the splits
# train_df = df.iloc[:6500]          # first 6,500 documents
# valid_df = df.iloc[6500:6500+2200] # next 2,200 documents

# # Save to CSV
# train_df.to_csv("antique_train_split.csv", index=False)
# valid_df.to_csv("antique_valid_split.csv", index=False)

# print(f"Train set: {len(train_df)} documents -> antique_train_split.csv")
# print(f"Valid set: {len(valid_df)} documents -> antique_valid_split.csv")

import ir_datasets
import pandas as pd

# Ensure 'data/' directory exists
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Define datasets to download
datasets = {
    "antique": "antique",
    "antique_test": "antique/test",
    "antique_test_non_offensive": "antique/test/non-offensive",
    "antique_train": "antique/train",
    "antique_train_split200_train": "antique/train/split200-train",
    "antique_train_split200_valid": "antique/train/split200-valid"
}

# Function to fetch and save documents
def save_documents(dataset_name, dataset):
    documents = []
    for doc in dataset.docs_iter():
        documents.append({"doc_id": doc.doc_id, "text": doc.text})

    df = pd.DataFrame(documents)
    file_path = os.path.join(data_dir, f"{dataset_name}_docs.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved {file_path} with {len(df)} records.")

# Function to fetch and save queries
def save_queries(dataset_name, dataset):
    queries = []
    for query in dataset.queries_iter():
        queries.append({"query_id": query.query_id, "text": query.text})

    df = pd.DataFrame(queries)
    file_path = os.path.join(data_dir, f"{dataset_name}_queries.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved {file_path} with {len(df)} records.")

# Function to fetch and save qrels (query relevance judgments)
def save_qrels(dataset_name, dataset):
    qrels = []
    for qrel in dataset.qrels_iter():
        qrels.append({
            "query_id": qrel.query_id,
            "doc_id": qrel.doc_id,
            "relevance": qrel.relevance,
            "iteration": getattr(qrel, "iteration", None)  # Some datasets have 'iteration', some don't
        })

    df = pd.DataFrame(qrels)
    file_path = os.path.join(data_dir, f"{dataset_name}_qrels.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved {file_path} with {len(df)} records.")

# Iterate through datasets and save each component
for name, path in datasets.items():
    dataset = ir_datasets.load(path)
    
    # Save documents
    save_documents(name, dataset)

    # Save queries if available
    if hasattr(dataset, "queries_iter"):
        save_queries(name, dataset)

    # Save qrels if available
    if hasattr(dataset, "qrels_iter"):
        save_qrels(name, dataset)

print("All datasets have been saved in the 'data/' folder!")
