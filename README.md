# qclef-clustering

Repository for the Clustering Task (Task 3) of Quantum CLEF 2025.

## Installation

Clone the repository and set up a Python environment:

```bash
git clone https://github.com/dsgt-arc/qclef-2025-clustering.git
cd qclef-2025-clustering

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

## Repository Structure

* `src/` — main source code for clustering, quantum solvers, and evaluation pipeline
* `config/` — YAML configuration files for various clustering methods
* `colormaps/` — colormaps used in visualizations
* `data/` — local test data

## Getting Started

Run the full pipeline using:

```bash
python src/models/ClusteringPipeline_cv.py
```

You can customize clustering method, colormap, and CV settings via CLI or by editing `config/*.yml`.

## Contributors

This clustering task was developed by Karishma Thakrar and Alex Pramov as key contributors through Data Science @ Georgia Tech (ARC).