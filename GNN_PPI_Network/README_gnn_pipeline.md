# Team 3 PPI-GNN Pipeline README

## Overview

This module is the Part 2B biological network branch of the broader **Evidence-Aware Validation of AI-Generated Drug Candidates** pipeline described in the Team3 pipeline document. In the full system, this branch estimates **biological/network risk** for a drug target neighborhood by:

1. building a protein-protein interaction graph around a target such as `HMGCR`
2. attaching node features to each protein
3. assigning binary disease-risk labels for supervised learning
4. training a graph neural network for node classification
5. aggregating node-level predictions into a network-level risk signal

The current codebase includes the core building blocks for steps 1 through 4:

- `ppi_graph_builder.py`
- `node_feature_builder.py`
- `node_label.py`
- `gnn_model.py`
- `graph_trainer.py`

The code is structurally aligned with the intended Part 2B design in the Team3 pipeline document, but it is **not yet a complete runnable pipeline**. It is better understood as a set of connected components that still need an orchestration layer and a few bug fixes.

---

## Intended data flow

The intended execution order is:

```text
PPI network sources (STRING + BioGRID)
        |
        v
PPIGraphBuilder
  -> combined edge table
  -> proteins list
  -> protein-to-index mapping
  -> edge_index
  -> edge_weights
        |
        +-------------------------+
        |                         |
        v                         v
feature_extracter             node_labeler
  -> UniProt GO features        -> DisGeNET scores
  -> x tensor                   -> y tensor
        |                         |
        +------------+------------+
                     |
                     v
        PyTorch Geometric Data object
     (x, edge_index, y, masks, optionally edge_attr)
                     |
                     v
                GNNModel
                     |
                     v
                GNNtrainer
                     |
                     v
       node-level disease-risk logits/predictions
                     |
                     v
       downstream risk aggregation around HMGCR
```

This flow is consistent with the Team3 pipeline description, where Part 2B takes a target-centered PPI network and predicts biologically risky proteins based on graph structure and node annotations.

---

## File-by-file explanation

## 1. `ppi_graph_builder.py`

### Purpose
Builds the graph structure for the GNN from STRING DB and BioGRID.

### What it does

`PPIGraphBuilder` handles:

- downloading a STRING network with `stringdb.get_network(...)`
- loading a BioGRID tab-delimited file and filtering to human interactions
- restricting the BioGRID subset to interactions involving a target protein, default `HMGCR`
- merging STRING and BioGRID edges into one table
- extracting a sorted protein list
- creating an integer mapping from protein name to node index
- building `edge_index` in PyTorch Geometric COO format
- building normalized edge weights from confidence scores

### Key outputs

- `proteins`: ordered list of node names
- `protein_mapping`: dictionary mapping protein name to node id
- `edge_index`: tensor of shape `[2, num_edges*2]` for an undirected graph
- `edge_weights`: tensor aligned with the duplicated undirected edges

### Why it matters
This file defines the graph topology. Every downstream object depends on the protein ordering created here.

### Important notes

- `protein_extraction()` writes `proteins.txt`. That file becomes the shared alignment key used by the feature and label builders.
- `build_edges()` duplicates edges in both directions, which is correct for undirected message passing.
- `edge_weights()` is computed, but the current GNN does not use edge weights yet.

### Current limitations

- `get_biogrid_network()` only pulls interactions where one endpoint is exactly the target protein. That means BioGRID contributes only **first-hop target interactions**, while STRING may contribute a wider neighborhood depending on `add_nodes`.
- `merge_networks()` removes duplicates only by exact `(preferredName_A, preferredName_B)` ordering. If one source stores `(A, B)` and another stores `(B, A)`, those could survive as duplicates in an undirected setting.
- There is no explicit removal of self-loops before `build_edges()`, although the GNN layer later adds self-loops anyway.

---

## 2. `node_feature_builder.py`

### Purpose
Creates node features from UniProt annotations.

### What it does

`feature_extracter`:

- loads the same ordered protein list used by the graph builder
- queries UniProt one gene at a time
- requests selected fields including GO annotations
- normalizes the JSON response into a DataFrame
- reorders rows to match `proteins.txt`
- extracts GO IDs from `uniProtKBCrossReferences`
- one-hot encodes GO terms with `MultiLabelBinarizer`
- converts the encoded matrix to a float tensor `x`

### Key outputs

- `uniprot_features.csv`: cached raw UniProt output
- `x_tensor`: node feature matrix of shape `[num_nodes, num_features]`

### Why it matters
This provides the node attributes the GNN uses for message passing and classification.

### Current logic issues

1. **Bug in `get_uniprot()`**
   - It calls `self.fetch_gene`, but no such method exists.
   - It almost certainly should call `self.get_protein_gene`.

2. **Feature scope is narrower than the Team3 design**
   - The Team3 document says node features should include UniProt, DisGeNET, and possibly KEGG/pathway information.
   - The current implementation only encodes GO terms from UniProt.
   - That is usable, but it is more limited than the intended design.

3. **Potential one-to-many ambiguity in UniProt results**
   - A gene query can return multiple reviewed entries.
   - `pd.json_normalize(uniprot_results)` may therefore create multiple rows for the same gene.
   - The later `set_index('gene_name').reindex(...)` step assumes a one-row-per-gene layout and may behave unpredictably when duplicates exist.

4. **No persistence of the fitted encoder**
   - `MultiLabelBinarizer()` is fit ad hoc inside `node_features_encoder()`.
   - That is fine for one-shot training, but if you want reproducible inference on a new graph, the vocabulary should be saved.

### Intended role in the full pipeline
This file corresponds to the Team3 “NodeFeatureBuilder” concept, but in the current code it is really a **UniProt GO feature encoder**, not a complete multi-source feature builder.

---

## 3. `node_label.py`

### Purpose
Builds supervised node labels from DisGeNET disease association scores.

### What it does

`node_labeler`:

- loads the shared protein list
- reads `disgenet_features.csv`
- coerces the `score` column to numeric
- groups by `geneSymbol` and keeps the max score per gene
- converts each score to a binary label using a threshold, default `0.3`
- aligns the labels to the graph protein order
- returns a `torch.long` tensor `y`

### Key outputs

- `y_tensor`: node label vector of shape `[num_nodes]`

### Why it matters
This is the supervision signal for node classification. In your framing, it labels proteins as higher-risk or lower-risk disease-associated nodes.

### Current limitations

- The file assumes `disgenet_features.csv` already exists, but the extraction step for generating that file is not included here.
- Unmatched proteins default to label `0`. That is simple and practical, but it treats missing annotation as low risk rather than unknown.
- The label is based only on a disease-association threshold, which may be too broad if your real goal is off-target toxicity or mechanistic adverse-effect risk rather than general disease involvement.

### Conceptual note
The Team3 document describes these labels as “high risk” nodes. In practice, this code is learning **disease association**, which is only a proxy for biological risk. That is a reasonable starting point, but the README should distinguish between:

- what the code literally predicts now: disease-associated vs not
- what the project wants to infer later: off-target biological risk around the screened candidate

---

## 4. `gnn_model.py`

### Purpose
Defines the graph neural network architecture used for node classification.

### What it does

This file contains two classes:

### `GNNLayer`
A custom message-passing layer that:

- adds self-loops
- linearly transforms node features
- computes symmetric degree normalization
- propagates normalized neighbor messages
- adds a learnable bias

This is essentially a hand-built GCN-style layer.

### `GNNModel`
Stacks three `GNNLayer`s with:

- ReLU after each layer
- dropout after each layer
- a final linear classifier

Output:

- raw logits for each node
- shape `[num_nodes, out_classes]`

### Why it matters
This is the core learning component that combines node features and graph topology.

### Strengths

- The normalization logic is appropriate for a GCN-like model.
- The code is readable and educational.
- The model is compatible with the `edge_index` representation built by `PPIGraphBuilder`.

### Current limitations

1. **Edge weights are ignored**
   - `ppi_graph_builder.py` computes confidence-based edge weights.
   - `GNNLayer.forward()` does not accept `edge_weight` or `edge_attr`.
   - So STRING/BioGRID confidence information is currently discarded.

2. **Three layers may oversmooth on small/local PPI graphs**
   - This is not a bug, but it is a modeling risk.
   - Depending on graph size and label density, two layers might be more stable.

3. **No batch norm, residuals, or output probabilities**
   - That is acceptable for a first version.
   - Softmax should be applied outside the model only when probabilities are needed.

---

## 5. `graph_trainer.py`

### Purpose
Wraps training logic for node classification.

### What it does

`GNNtrainer`:

- stores model, optimizer, loss function, and device
- moves the model to the device
- in `train(data)`:
  - switches model to train mode
  - zeroes gradients
  - moves `x`, `edge_index`, `y`, and `train_mask` to device
  - runs a forward pass
  - computes masked loss
  - backpropagates
  - updates parameters
  - returns scalar loss

### Why it matters
This file is the training loop abstraction for the GNN branch.

### Current limitations

1. **No evaluation method**
   - There is no `evaluate()` or `predict()` method.
   - The Team3 design implies the need to inspect node scores after training.

2. **Assumes `data.train_mask` already exists**
   - No code here creates `train_mask`, `val_mask`, or `test_mask`.
   - That orchestration step is currently missing.

3. **Default device may fail on CPU-only machines**
   - `device=torch.device('cuda')` will break if CUDA is unavailable.
   - A safer default is conditional device selection.

4. **No checkpointing or metric reporting**
   - There is no accuracy, F1, AUROC, class balance check, or early stopping.

---

## Connectivity across files

## What already connects correctly

These files are conceptually connected in a coherent order:

1. `PPIGraphBuilder.protein_extraction()` creates a sorted protein list.
2. `feature_extracter` and `node_labeler` both rely on that same ordered protein list.
3. `feature_extracter.node_features_encoder()` produces `x`.
4. `node_labeler.node_labels()` produces `y`.
5. `PPIGraphBuilder.build_edges()` produces `edge_index`.
6. `GNNModel.forward(x, edge_index)` consumes the graph tensors.
7. `GNNtrainer.train(data)` expects those tensors inside a PyG `Data` object.

That means the intended **index alignment contract** is clear and mostly sound:

- node 17 in `edge_index`
- row 17 in `x`
- element 17 in `y`

all refer to the same protein, provided `proteins.txt` is generated once and reused consistently.

## What is still missing to make it runnable end-to-end

The following integration layer is not present yet:

1. **A PyTorch Geometric `Data` object builder**
   - Something like:
   - `Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_weights)`

2. **Mask generation**
   - `train_mask`, `val_mask`, and optionally `test_mask`

3. **A top-level orchestration script or class**
   - analogous to the Team3 document’s `gnn_agent.py`

4. **Inference and aggregation logic**
   - node probabilities around HMGCR
   - ranking or flagging risky neighbors
   - collapsing those node-level outputs into a network risk score

5. **Dependency generation for CSV inputs**
   - the code expects `disgenet_features.csv` and likely a BioGRID file to already be available

---

## Best interpretation of how the code is intended to work

A practical reading of the current system is:

1. Start with `HMGCR` and query STRING for a local PPI neighborhood.
2. Add experimentally supported BioGRID edges involving HMGCR.
3. Convert proteins to indexed nodes and produce `edge_index`.
4. Use UniProt GO annotations to describe each node functionally.
5. Use DisGeNET to assign binary disease-association labels.
6. Train a GCN-like model to classify proteins as disease-associated or not.
7. Use predicted high-risk proteins near the target neighborhood as a proxy for biological/off-target risk.

That is consistent with your project theme of assessing risk through **topological proximity to biologically risky proteins**. The missing piece is that the current code still predicts risk at the **protein node level**, not directly at the **virtual-screened compound level**. The compound-to-protein bridge would need to be supplied by the larger pipeline.

---

## Major issues to fix before running

## Required code fixes

### In `node_feature_builder.py`
Replace:

```python
dfs = list(executor.map(self.fetch_gene, self.protein_genes))
```

with:

```python
dfs = list(executor.map(self.get_protein_gene, self.protein_genes))
```

### In `graph_trainer.py`
Consider replacing the device default with:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Add a graph assembly step
You need a script that does something like:

```python
from torch_geometric.data import Data

data = Data(
    x=x,
    edge_index=edge_index,
    y=y,
    edge_attr=edge_weights
)
```

### Add masks
Example:

```python
num_nodes = y.size(0)
perm = torch.randperm(num_nodes)
train_end = int(0.7 * num_nodes)
val_end = int(0.85 * num_nodes)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[perm[:train_end]] = True
val_mask[perm[train_end:val_end]] = True
test_mask[perm[val_end:]] = True

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask
```

## Recommended modeling fixes

- normalize or use edge weights in `GNNLayer`
- add an evaluation loop with accuracy and F1
- inspect class imbalance in `y`
- distinguish unknown labels from true negatives
- make BioGRID deduplication undirected-aware
- save the fitted feature vocabulary if inference will be done later
- consider extending node features with pathway membership or network statistics

---

## Minimal orchestration example

Below is the missing glue the current files imply.

```python
import torch
import torch.nn as nn
from torch_geometric.data import Data

from ppi_graph_builder import PPIGraphBuilder
from node_feature_builder import feature_extracter
from node_label import node_labeler
from gnn_model import GNNModel
from graph_trainer import GNNtrainer

# 1. Build graph
builder = PPIGraphBuilder(species=9606, score_threshold=400, add_nodes=100)
string_df = builder.get_stringdb_network(['HMGCR'])
biogrid_df = builder.get_biogrid_network('BIOGRID-ALL.tab3.txt', target_protein='HMGCR')
combined_df = builder.merge_networks(string_df, biogrid_df)
proteins, protein_mapping = builder.protein_extraction(combined_df)
edge_index = builder.build_edges(combined_df, protein_mapping)
edge_weights = builder.edge_weights(combined_df)

# 2. Build node features
feature_builder = feature_extracter(proteins=proteins)
uniprot_df = feature_builder.get_uniprot()
x = feature_builder.node_features_encoder(uniprot_df)

# 3. Build labels
label_builder = node_labeler(proteins=proteins)
y = label_builder.node_labels(disease_score_threshold=0.3)

# 4. Assemble PyG data object
data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_weights)

# 5. Create masks
num_nodes = y.size(0)
perm = torch.randperm(num_nodes)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[perm[:int(0.8 * num_nodes)]] = True
data.train_mask = train_mask

# 6. Initialize model and trainer
model = GNNModel(in_channels=x.size(1), hidden_channels=64, out_classes=2, dropout=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
loss_fn = nn.CrossEntropyLoss()
trainer = GNNtrainer(model, optimizer, loss_fn, device=torch.device('cpu'))

# 7. Train
for epoch in range(100):
    loss = trainer.train(data)
    print(f'Epoch {epoch+1}: loss={loss:.4f}')
```

This is not the complete production design, but it matches how the current files are meant to connect.

---

## Alignment with the Team3 pipeline document

The Team3 pipeline document describes this branch as:

- `C1` Graph construction
- `C2` Node features
- `C3` Node labels
- `C4` GNN model
- `C5` Graph trainer
- `C6` Neighbor risk aggregation
- `C7` GNN orchestrator

The current code covers **C1 through C5 only partially**.

### Covered now

- `C1` `PPIGraphBuilder`
- `C2` partial `NodeFeatureBuilder` functionality through UniProt GO encoding
- `C3` `node_labeler`
- `C4` `GNNModel`
- `C5` `GNNtrainer`

### Still absent

- `C6` neighbor risk aggregation
- `C7` orchestrator class
- full multi-source feature assembly promised in the document
- explicit inference and flagged protein outputs

So the codebase is **directionally consistent** with the Team3 architecture, but the implementation is still in a prototype state rather than a finished agent.

---

## Practical conclusion

The attached `.py` files form a sensible backbone for a PPI-based GNN node-classification workflow. The logic is coherent at a high level:

- graph topology comes from STRING/BioGRID
- features come from UniProt
- labels come from DisGeNET
- the GNN learns on the resulting graph

The most important positive point is that all components are built around the same shared protein ordering, which is the main requirement for correct graph learning.

The most important weaknesses are:

- one direct bug in `node_feature_builder.py`
- no orchestration script to produce the PyG `Data` object and masks
- no evaluation or inference stage
- no current use of edge weights
- a gap between “disease association” labels and the broader notion of “off-target risk” in the project narrative

In other words, the code is best described as a **working architectural skeleton** for the Team3 GNN branch, with a few solid components already implemented and a few crucial integration steps still missing.

---

## Suggested next files to add

To complete this branch cleanly, I would add:

- `gnn_agent.py`
  - orchestrates graph build, features, labels, training, and inference
- `graph_data_builder.py`
  - constructs the PyG `Data` object and masks
- `neighbor_risk_aggregator.py`
  - converts node probabilities near HMGCR into a network risk score
- `evaluate.py`
  - accuracy, F1, AUROC, confusion matrix
- `disgenet_feature_builder.py`
  - creates `disgenet_features.csv` instead of assuming it already exists

---

## Input files expected by the current code

At minimum, the current code expects access to:

- an internet connection for UniProt queries
- the `stringdb` Python package configured correctly
- a BioGRID tab-delimited file such as `BIOGRID-ALL-*.tab3.txt`
- a `disgenet_features.csv` file with at least:
  - `geneSymbol`
  - `score`
- optionally a previously saved `uniprot_features.csv`

---

## Output objects produced or implied

The current modules produce or imply:

- `proteins.txt`
- `uniprot_features.csv`
- `edge_index`
- `edge_weights`
- `x` node feature tensor
- `y` node label tensor
- node-level logits from `GNNModel`
- training loss from `GNNtrainer`

The following downstream outputs are implied by the Team3 design but not yet implemented here:

- node-level probabilities
- risky neighbor ranking
- flagged off-target proteins
- aggregated network risk score

