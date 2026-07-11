# Inflow and Outflow Centrality

Python implementation of **inflow** and **outflow centrality** — two novel, node-feature-aware centrality metrics inspired by the aggregation approach used in graph convolutional networks (GCNs). 

The package also includes two baseline node-weighted metrics — weighted degree centrality and weighted closeness centrality — used in the accompanying paper for comparison.

📄 Paper: *Inflow and outflow centrality: novel centrality metrics inspired by graph convolution*, published in *Applied Network Science*.
DOI: [10.1007/s41109-026-00782-7](https://doi.org/10.1007/s41109-026-00782-7)

---

## Installation

Requires Python ≥ 3.9.

```bash
git clone https://github.com/AramPapaz/Inflow-and-Outflow-Centrality.git
cd Inflow-and-Outflow-Centrality
pip install .
```

This installs the `inflow_outflow_centrality` package along with its dependencies (`pandas`, `numpy`, `networkx`).

For development (editable install):

```bash
pip install -e .
```

---

## Quickstart

### As a Python library

```python
from inflow_outflow_centrality import utilities
import pickle

# 1. Load your network
interactions = utilities.read_interactions_csv("example/AirportNetwork.csv")
# or for .ea/.ea.gz files:
# interactions = utilities.read_interactions_ea("example/4efm_intsc.ea")

# 2. Provide node features as a dict {node_id: value}
# (or load real features from your own data)
with open("example/population.pkl", "rb") as f:
    features = pickle.load(f)


# 3. Compute inflow/outflow centrality
result = utilities.flow_metric(interactions, features, flow_type="both")
print(result.head())

# Other available metrics for comparison:
deg_centrality = utilities.node_weighted_degree_centrality(interactions, features)
close_centrality = utilities.node_weighted_closeness_centrality(interactions, features)
```

### From the command line

```bash
python -m inflow_outflow_centrality.utilities <flow_type> <interactions_file> [features_csv]
```

- `flow_type` — `in`, `out`, or `both`
- `interactions_file` — network file; `.ea`/`.ea.gz` files are parsed with `read_interactions_ea`, everything else with `read_interactions_csv`
- `features_csv` *(optional)* — 2-column CSV of `node_id, feature_value`; if omitted, every node defaults to a feature value of `1`

The result is written to `<flow_type>_output.csv` in the same directory as `interactions_file`.

**Example**, using the sample data included in this repo:

```bash
python -m inflow_outflow_centrality.utilities both example/AirportNetwork.csv
```

This reproduces `example/both_output.csv` (comparable to the provided `example/inflow_output.csv` / `example/outflow_output.csv`).

---

## API Reference

| Function | Description |
|---|---|
| `flow_metric(interactions, features, flow_type)` | Core inflow/outflow centrality computation. Auto-detects an optional float weight column in `interactions` and, if present, also returns weight-based versions of the metric (`WeightSum`, `WeightedInflowValue`, `WeightedOutflowValue`). |
| `node_weighted_degree_centrality(interactions, features, f=lambda x: x)` | Node-weighted degree centrality baseline. `f` can transform feature values before aggregation. |
| `node_weighted_closeness_centrality(interactions, features, f=lambda x: x)` | Node-weighted closeness centrality baseline. |
| `read_interactions_csv(csv_path)` | Loads a CSV edge list into the DataFrame format expected by `flow_metric`. |
| `read_interactions_ea(ea_path)` | Loads a `.ea` or `.ea.gz` interaction file, keeping only rows with `combi:all_all` in the description and non-negative scores. |

### Input format

- **`interactions`**: a `pandas.DataFrame` with two node-id columns (source, target) and an optional third float column for edge weight. Duplicate and self-interactions are automatically removed.
- **`features`**: a `dict` mapping `node_id → feature_value`. Values should roughly follow a normal distribution for the metric to behave well. In the example data this is provided as a pickled dict (`example/population.pkl`), loaded with `pickle.load`.

### Output format

`flow_metric` returns a `pandas.DataFrame`:

- `flow_type="in"` → `Node, InflowValue, Degree` (+ `WeightSum, WeightedInflowValue` if weights were provided)
- `flow_type="out"` → `Node, OutflowValue, Degree` (+ `WeightSum, WeightedOutflowValue` if weights were provided)
- `flow_type="both"` → `Node, InflowValue, OutflowValue, Degree` (+ weighted columns if weights were provided)

---

## Example data

The `example/` folder contains ready-to-use sample data:

- `AirportNetwork.csv` — an airport interaction network
- `4efm_intsc.ea` — a residue interaction network in `.ea` format
- `population.pkl` — node features (population) for use with the airport network
- `inflow_output.csv`, `outflow_output.csv` — precomputed results for reference/comparison



