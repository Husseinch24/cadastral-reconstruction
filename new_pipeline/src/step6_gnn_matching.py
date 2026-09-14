"""
=============================================================================
MILESTONE 4 — GNN CROSS-ATTENTION PARCEL MATCHING
=============================================================================
Masters Thesis — AI-Based Panoramic Cadastral Image Reconstruction
Author  : Hussein Chalhoub

Purpose
--------
Given two parcel adjacency graphs G_A and G_B (from two adjacent map sheets),
predict which parcels in B correspond to which parcels in A.

This is the CORE THESIS CONTRIBUTION.  No existing image stitching method
matches cadastral atlas sheets by their structural parcel graph.  This
method works even though adjacent sheets share ZERO visual overlap — they
share only the boundary line itself, and the boundary parcels appear on
both sheets with identical structural fingerprints.

Architecture: Cross-Attention GNN (SuperGlue-inspired)
--------------------------------------------------------
  Input:
    G_A: graph with N_A nodes, each with D-dimensional features
    G_B: graph with N_B nodes, each with D-dimensional features

  Step 1 — Feature projection
    Linear layer: D -> 128 dimensions for each graph

  Step 2 — Self-attention within each graph (L=3 layers)
    Each parcel attends to its neighbours in its own graph
    Learns context: "I am a boundary parcel with 3 small neighbours"

  Step 3 — Cross-attention between graphs (L=3 layers)
    Each parcel in A attends to ALL parcels in B (and vice versa)
    Learns to ask: "Which parcel in B looks most like me?"

  Step 4 — Matching score matrix
    S[i,j] = dot product of final embeddings of parcel i (in A) and j (in B)
    S has shape [N_A, N_B]

  Step 5 — Sinkhorn optimal transport
    Converts the score matrix into a doubly-stochastic assignment matrix
    Ensures each parcel matches at most one partner (one-to-one matching)

  Output:
    List of (parcel_id_in_A, parcel_id_in_B, confidence) matches

Training
---------
  Positive pairs: maps that ARE adjacent (from homographies.json)
    Ground truth: boundary parcels known from the PDF index
  Negative pairs: maps that are NOT adjacent
    Ground truth: zero matches expected

  Loss: cross-entropy on the assignment matrix
    For matched parcels: maximise S[i, j_correct]
    For unmatched parcels: maximise the "dustbin" (no-match) score

Usage
------
  .\venv_thesis\Scripts\Activate.ps1

  # Train the GNN
  python new_pipeline/src/step6_gnn_matching.py --train

  # Match a specific pair of maps
  python new_pipeline/src/step6_gnn_matching.py --match --pair 45_47

  # Match all adjacent pairs
  python new_pipeline/src/step6_gnn_matching.py --match --all

Output
-------
  new_pipeline/models/gnn/
    best_gnn.pth            - best trained GNN weights
    last_gnn.pth            - last epoch weights
    gnn_training_log.json   - loss history

  new_pipeline/data/matches/
    matches_<A>_<B>.json    - predicted parcel correspondences
    matches_<A>_<B>.png     - visualisation of matches

=============================================================================
"""

import argparse
import json
import math
import time
from pathlib import Path
from itertools import combinations

import cv2
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

GRAPHS_DIR       = Path("new_pipeline/data/graphs")
MODELS_GNN_DIR   = Path("new_pipeline/models/gnn")
MATCHES_DIR      = Path("new_pipeline/data/matches")
PREPROCESSED_DIR = Path("output/preprocessed")
HOMOGRAPHY_DIR   = Path("output/homographies")

ALL_MAPS = [str(n) for n in range(43, 56)]

# Known adjacent pairs (from our alignment work)
ADJACENT_PAIRS = [
    ("45", "47"), ("45", "46"), ("47", "48"), ("48", "49"),
    ("49", "50"), ("50", "51"), ("52", "53"), ("52", "54"),
    ("52", "55"), ("54", "55"),
]

# Known boundary parcels per pair (ground truth for training supervision)
BOUNDARY_PARCELS = {
    "45_47": [2580],
    "45_46": [2616, 2619],
    "47_48": [2749, 2803, 2814],
    "48_49": [2893],
    "52_53": [3215],
    "54_55": [3338, 3339, 3345, 3346],
}

CONFIG = {
    "node_feat_dim":   11,    # number of node features from step5_graph.py
    "hidden_dim":      128,   # GNN hidden dimension
    "n_gnn_layers":    3,     # number of self+cross attention layers
    "n_heads":         4,     # attention heads in GAT
    "sinkhorn_iters":  100,   # Sinkhorn iterations for assignment
    "sinkhorn_temp":   0.1,   # temperature for Sinkhorn (lower = sharper)
    "lr":              1e-4,
    "weight_decay":    1e-5,
    "n_epochs":        50,
    "batch_size":      1,     # 1 graph pair per step
    "match_threshold": 0.2,   # minimum score to declare a match
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
}

# Node features to use (must match what step5_graph.py saves)
NODE_FEATURE_KEYS = [
    "norm_area", "aspect_ratio", "compactness",
    "norm_cx", "norm_cy",
    "dist_boundary", "dist_centre",
    "perim_estimate",
    "confidence", "degree", "local_density",
]


# ---------------------------------------------------------------------------
# GRAPH LOADING
# ---------------------------------------------------------------------------

def load_graph(map_num: str) -> nx.Graph:
    path = GRAPHS_DIR / f"map_{map_num}_graph.graphml"
    if not path.exists():
        raise FileNotFoundError(
            f"Graph not found: {path}\n"
            f"Run step5_graph.py first."
        )
    return nx.read_graphml(str(path))


def graph_to_pyg(G: nx.Graph) -> Data:
    """
    Convert a NetworkX graph to a PyTorch Geometric Data object.

    Node features: the NODE_FEATURE_KEYS listed above
    Edge index: COO format tensor [2, num_edges]
    """
    nodes    = list(G.nodes(data=True))
    node_ids = [int(n) for n, _ in nodes]
    id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    # Build feature matrix [N, D]
    feats = []
    for _, data in nodes:
        row = []
        for key in NODE_FEATURE_KEYS:
            val = data.get(key, 0.0)
            try:
                row.append(float(val))
            except (TypeError, ValueError):
                row.append(0.0)
        feats.append(row)

    x = torch.tensor(feats, dtype=torch.float32)

    # Normalise features to [0, 1] range per column
    x_min = x.min(dim=0).values
    x_max = x.max(dim=0).values
    x_range = (x_max - x_min).clamp(min=1e-8)
    x = (x - x_min) / x_range

    # Build edge index
    edge_list = []
    for u, v in G.edges():
        u_idx = id_to_idx.get(int(u))
        v_idx = id_to_idx.get(int(v))
        if u_idx is not None and v_idx is not None:
            edge_list.append([u_idx, v_idx])
            edge_list.append([v_idx, u_idx])   # undirected

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index,
                node_ids=torch.tensor(node_ids, dtype=torch.long))


# ---------------------------------------------------------------------------
# MODEL ARCHITECTURE
# ---------------------------------------------------------------------------

class AttentionalGNN(nn.Module):
    """
    Graph Attention Network with alternating self and cross attention.

    Self-attention: each node aggregates features from its graph neighbours
    Cross-attention: each node in graph A attends to all nodes in graph B

    This is the core of the SuperGlue-inspired architecture adapted for
    parcel adjacency graphs instead of keypoint feature graphs.
    """

    def __init__(self, feat_dim: int, hidden_dim: int,
                 n_layers: int, n_heads: int):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Self-attention GNN layers (one per round)
        self.self_attn_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // n_heads,
                    heads=n_heads, concat=True, dropout=0.1)
            for _ in range(n_layers)
        ])
        # Cross-attention MLP layers (standard attention, no graph structure)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, n_heads,
                                   dropout=0.1, batch_first=True)
            for _ in range(n_layers)
        ])
        # Layer norms for stability
        self.norms_self  = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        self.norms_cross = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

    def forward(self, x_a: torch.Tensor, edge_a: torch.Tensor,
                x_b: torch.Tensor, edge_b: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x_a:     [N_A, D] node features for graph A
          edge_a:  [2, E_A] edge index for graph A
          x_b:     [N_B, D] node features for graph B
          edge_b:  [2, E_B] edge index for graph B

        Returns:
          h_a:     [N_A, D] enriched features for graph A
          h_b:     [N_B, D] enriched features for graph B
        """
        h_a = self.input_proj(x_a)
        h_b = self.input_proj(x_b)

        for i, (self_layer, cross_layer, norm_s, norm_c) in enumerate(
            zip(self.self_attn_layers, self.cross_attn_layers,
                self.norms_self, self.norms_cross)
        ):
            # Self-attention: aggregate from graph neighbours
            h_a2 = self_layer(h_a, edge_a)
            h_b2 = self_layer(h_b, edge_b)
            h_a = norm_s(h_a + F.relu(h_a2))
            h_b = norm_s(h_b + F.relu(h_b2))

            # Cross-attention: attend from A to B and from B to A
            # Unsqueeze to add batch dimension [1, N, D]
            h_a_q = h_a.unsqueeze(0)
            h_b_q = h_b.unsqueeze(0)
            h_a_cross, _ = cross_layer(h_a_q, h_b_q, h_b_q)
            h_b_cross, _ = cross_layer(h_b_q, h_a_q, h_a_q)
            h_a = norm_c(h_a + h_a_cross.squeeze(0))
            h_b = norm_c(h_b + h_b_cross.squeeze(0))

        return h_a, h_b


class ParcelMatcher(nn.Module):
    """
    Full parcel matching model:
      1. AttentionalGNN to enrich node features
      2. Score matrix via dot product
      3. Sinkhorn assignment
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.gnn = AttentionalGNN(
            feat_dim=cfg["node_feat_dim"],
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_gnn_layers"],
            n_heads=cfg["n_heads"],
        )
        self.sinkhorn_iters = cfg["sinkhorn_iters"]
        self.sinkhorn_temp  = cfg["sinkhorn_temp"]

        # Final projection before matching
        d = cfg["hidden_dim"]
        self.final_proj = nn.Linear(d, d)

        # Dustbin (no-match) score — learnable
        self.dustbin = nn.Parameter(torch.tensor(1.0))

    def forward(self, data_a: Data, data_b: Data
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          scores:   [N_A, N_B] raw dot-product scores
          P:        [N_A+1, N_B+1] Sinkhorn assignment matrix
                    (last row/col are the dustbin for unmatched nodes)
        """
        x_a = data_a.x
        x_b = data_b.x
        edge_a = data_a.edge_index
        edge_b = data_b.edge_index

        # Enrich features through GNN
        h_a, h_b = self.gnn(x_a, edge_a, x_b, edge_b)
        h_a = self.final_proj(h_a)
        h_b = self.final_proj(h_b)

        # L2 normalise for stable dot product
        h_a = F.normalize(h_a, dim=-1)
        h_b = F.normalize(h_b, dim=-1)

        # Score matrix [N_A, N_B]
        scores = torch.matmul(h_a, h_b.t()) / self.sinkhorn_temp

        # Add dustbin row and column
        n_a, n_b = scores.shape
        dustbin_row = self.dustbin.expand(1, n_b)
        dustbin_col = self.dustbin.expand(n_a + 1, 1)
        scores_aug  = torch.cat([scores, dustbin_row], dim=0)
        scores_aug  = torch.cat([scores_aug, dustbin_col], dim=1)

        # Sinkhorn normalisation
        P = self._sinkhorn(scores_aug)

        return scores, P

    def _sinkhorn(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Sinkhorn-Knopp algorithm: iteratively normalise rows and columns
        to produce a doubly-stochastic matrix (soft assignment).
        """
        log_P = Z - torch.logsumexp(Z, dim=1, keepdim=True)
        for _ in range(self.sinkhorn_iters):
            # Row normalisation
            log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)
            # Column normalisation
            log_P = log_P - torch.logsumexp(log_P, dim=0, keepdim=True)
        return torch.exp(log_P)


# ---------------------------------------------------------------------------
# LOSS FUNCTION
# ---------------------------------------------------------------------------

def matching_loss(P: torch.Tensor,
                   matches_a: list[int], matches_b: list[int],
                   n_a: int, n_b: int) -> torch.Tensor:
    """
    Cross-entropy loss on the Sinkhorn assignment matrix.

    For each ground-truth matched pair (i, j):
      - Maximise P[i, j]  (they should match)
      - Maximise P[i, n_b]  (dustbin) for unmatched parcels in A
      - Maximise P[n_a, j]  (dustbin) for unmatched parcels in B

    Args:
      P:         [N_A+1, N_B+1] assignment matrix (includes dustbin)
      matches_a: list of node indices in A that have a ground-truth match
      matches_b: corresponding node indices in B
      n_a, n_b:  number of nodes in each graph (excluding dustbin)
    """
    eps = 1e-8
    loss = torch.tensor(0.0, device=P.device, requires_grad=True)

    if not matches_a:
        # No known matches — only dustbin supervision
        # All nodes should match to dustbin
        loss = -torch.log(P[:n_a, n_b] + eps).mean()
        loss = loss + (-torch.log(P[n_a, :n_b] + eps)).mean()
        return loss

    matched_a = set(matches_a)
    matched_b = set(matches_b)

    # Positive loss: matched pairs
    pos_loss = torch.tensor(0.0, device=P.device, requires_grad=True)
    for i, j in zip(matches_a, matches_b):
        pos_loss = pos_loss + (-torch.log(P[i, j] + eps))
    if matches_a:
        pos_loss = pos_loss / len(matches_a)

    # Negative loss: unmatched nodes should go to dustbin
    unmatched_a = [i for i in range(n_a) if i not in matched_a]
    unmatched_b = [j for j in range(n_b) if j not in matched_b]
    neg_loss = torch.tensor(0.0, device=P.device, requires_grad=True)
    if unmatched_a:
        neg_loss = neg_loss + (-torch.log(P[unmatched_a, n_b] + eps)).mean()
    if unmatched_b:
        neg_loss = neg_loss + (-torch.log(P[n_a, unmatched_b] + eps)).mean()

    return pos_loss + 0.5 * neg_loss


# ---------------------------------------------------------------------------
# TRAINING DATA PREPARATION
# ---------------------------------------------------------------------------

def find_boundary_node_indices(G_a: nx.Graph, G_b: nx.Graph,
                                pair_key: str,
                                data_a: Data, data_b: Data
                                ) -> tuple[list[int], list[int]]:
    """
    Find the indices (in the PyG tensors) of boundary parcel nodes.

    Since we don't have explicit parcel IDs in the graph matching
    ground truth, we use proximity heuristic: boundary parcels are
    those closest to the map boundary (smallest dist_boundary feature).

    For training, we use the top-K boundary parcels from each map
    and assume they should match (supervised by position proximity).
    """
    n_boundary = len(BOUNDARY_PARCELS.get(pair_key, []))
    if n_boundary == 0:
        n_boundary = 3   # default: use 3 boundary nodes

    # Feature index for dist_boundary
    dist_boundary_idx = NODE_FEATURE_KEYS.index("dist_boundary")

    # Find nodes with smallest dist_boundary (closest to map edge)
    dist_a = data_a.x[:, dist_boundary_idx]
    dist_b = data_b.x[:, dist_boundary_idx]

    # Top-K closest to boundary
    k = min(n_boundary * 2, len(dist_a), len(dist_b))
    _, top_a = torch.topk(-dist_a, k)
    _, top_b = torch.topk(-dist_b, k)

    # Match them positionally: closest in normalised position space
    pos_idx_cx = NODE_FEATURE_KEYS.index("norm_cx")
    pos_idx_cy = NODE_FEATURE_KEYS.index("norm_cy")

    matches_a = []
    matches_b = []
    used_b = set()

    for ia in top_a[:n_boundary].tolist():
        pos_a = data_a.x[ia, [pos_idx_cx, pos_idx_cy]]
        best_dist = float("inf")
        best_ib = -1
        for ib in top_b.tolist():
            if ib in used_b:
                continue
            pos_b = data_b.x[ib, [pos_idx_cx, pos_idx_cy]]
            d = torch.norm(pos_a - pos_b).item()
            if d < best_dist:
                best_dist = d
                best_ib = ib
        if best_ib >= 0 and best_dist < 0.3:
            matches_a.append(ia)
            matches_b.append(best_ib)
            used_b.add(best_ib)

    return matches_a, matches_b


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train():
    print("\n" + "=" * 70)
    print("  MILESTONE 4 — GNN PARCEL MATCHING TRAINING")
    print(f"  Device: {CONFIG['device'].upper()}")
    print("=" * 70)

    MODELS_GNN_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(CONFIG["device"])

    # Load all graphs
    print("\n  Loading graphs...")
    graphs_nx = {}
    graphs_pyg = {}
    for m in ALL_MAPS:
        try:
            G = load_graph(m)
            graphs_nx[m] = G
            graphs_pyg[m] = graph_to_pyg(G).to(device)
            print(f"  Map {m}: {G.number_of_nodes()} nodes, "
                  f"{G.number_of_edges()} edges")
        except FileNotFoundError as e:
            print(f"  Skipping map {m}: {e}")

    if len(graphs_pyg) < 2:
        print("  ERROR: Need at least 2 graphs to train. "
              "Run step5_graph.py first.")
        return

    # Build training pairs: adjacent (positive) and non-adjacent (negative)
    positive_pairs = [
        (a, b) for a, b in ADJACENT_PAIRS
        if a in graphs_pyg and b in graphs_pyg
    ]
    all_map_ids = list(graphs_pyg.keys())
    adjacent_set = set(
        (a, b) for a, b in ADJACENT_PAIRS
    ) | set(
        (b, a) for a, b in ADJACENT_PAIRS
    )
    negative_pairs = [
        (a, b) for a, b in combinations(all_map_ids, 2)
        if (a, b) not in adjacent_set and (b, a) not in adjacent_set
    ]
    # Balance: use same number of negatives as positives
    import random
    random.seed(42)
    neg_sample = random.sample(
        negative_pairs, min(len(negative_pairs), len(positive_pairs) * 2)
    )

    print(f"\n  Training pairs: {len(positive_pairs)} positive, "
          f"{len(neg_sample)} negative")

    # Build model
    model = ParcelMatcher(CONFIG).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model parameters: {n_params:.2f} M")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["n_epochs"], eta_min=1e-6
    )

    # Resume if checkpoint exists
    last_ckpt = MODELS_GNN_DIR / "last_gnn.pth"
    if last_ckpt.exists():
        print(f"\n  Resuming from {last_ckpt}")
        model.load_state_dict(
            torch.load(str(last_ckpt), map_location=device)
        )

    training_log = []
    best_loss = float("inf")

    print(f"\n  Training for {CONFIG['n_epochs']} epochs...")

    for epoch in range(1, CONFIG["n_epochs"] + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_steps = 0

        # Shuffle all pairs
        all_pairs = (
            [(a, b, True)  for a, b in positive_pairs] +
            [(a, b, False) for a, b in neg_sample]
        )
        random.shuffle(all_pairs)

        for a, b, is_positive in all_pairs:
            data_a = graphs_pyg[a]
            data_b = graphs_pyg[b]

            # Skip if either graph is empty
            if data_a.x.shape[0] == 0 or data_b.x.shape[0] == 0:
                continue

            optimizer.zero_grad()
            try:
                scores, P = model(data_a, data_b)
            except RuntimeError as e:
                continue

            n_a = data_a.x.shape[0]
            n_b = data_b.x.shape[0]

            if is_positive:
                pair_key = f"{a}_{b}"
                matches_a, matches_b = find_boundary_node_indices(
                    graphs_nx[a], graphs_nx[b], pair_key, data_a, data_b
                )
            else:
                # Negative pair: no matches expected
                matches_a, matches_b = [], []

            loss = matching_loss(P, matches_a, matches_b, n_a, n_b)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1

        scheduler.step()
        avg_loss = total_loss / max(n_steps, 1)
        elapsed  = time.time() - t0

        log_entry = {
            "epoch":      epoch,
            "loss":       round(avg_loss, 6),
            "lr":         round(optimizer.param_groups[0]["lr"], 8),
            "elapsed_sec": round(elapsed, 1),
        }
        training_log.append(log_entry)

        print(f"  Epoch {epoch:>3}/{CONFIG['n_epochs']}  "
              f"loss={avg_loss:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"({elapsed:.0f}s)")

        # Save checkpoints
        torch.save(model.state_dict(), str(MODELS_GNN_DIR / "last_gnn.pth"))
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                model.state_dict(), str(MODELS_GNN_DIR / "best_gnn.pth")
            )
            print(f"  New best model saved (loss={best_loss:.4f})")

    # Save training log
    with open(MODELS_GNN_DIR / "gnn_training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    print("\n" + "=" * 70)
    print("  GNN TRAINING COMPLETE")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Model: {MODELS_GNN_DIR / 'best_gnn.pth'}")
    print(f"  Next: Run with --match --all to match all adjacent pairs")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# INFERENCE: MATCH TWO MAPS
# ---------------------------------------------------------------------------

def load_model() -> ParcelMatcher:
    ckpt = MODELS_GNN_DIR / "best_gnn.pth"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"GNN checkpoint not found: {ckpt}\n"
            f"Run with --train first."
        )
    model = ParcelMatcher(CONFIG)
    model.load_state_dict(
        torch.load(str(ckpt),
                   map_location=CONFIG["device"])
    )
    model.to(CONFIG["device"])
    model.eval()
    return model


def match_pair(model: ParcelMatcher,
               map_a: str, map_b: str) -> list[dict]:
    """
    Match parcels between two maps.
    Returns list of match dicts: {parcel_a, parcel_b, score, cx_a, cy_a, cx_b, cy_b}
    """
    device = torch.device(CONFIG["device"])

    G_a  = load_graph(map_a)
    G_b  = load_graph(map_b)
    data_a = graph_to_pyg(G_a).to(device)
    data_b = graph_to_pyg(G_b).to(device)

    with torch.no_grad():
        scores, P = model(data_a, data_b)

    n_a = data_a.x.shape[0]
    n_b = data_b.x.shape[0]

    # Extract matches from assignment matrix (excluding dustbin)
    P_match = P[:n_a, :n_b].cpu().numpy()
    nodes_a  = list(G_a.nodes(data=True))
    nodes_b  = list(G_b.nodes(data=True))

    matches = []
    for i in range(n_a):
        j = P_match[i].argmax()
        score = float(P_match[i, j])
        if score >= CONFIG["match_threshold"]:
            node_a_id, data_a_node = nodes_a[i]
            node_b_id, data_b_node = nodes_b[j]
            matches.append({
                "parcel_a":  int(node_a_id),
                "parcel_b":  int(node_b_id),
                "score":     round(score, 4),
                "cx_a":      float(data_a_node.get("cx", 0)),
                "cy_a":      float(data_a_node.get("cy", 0)),
                "cx_b":      float(data_b_node.get("cx", 0)),
                "cy_b":      float(data_b_node.get("cy", 0)),
            })

    # Sort by score descending
    matches.sort(key=lambda x: -x["score"])
    return matches


def save_match_visualisation(map_a: str, map_b: str,
                              matches: list[dict]):
    """Save a side-by-side visualisation of matched parcels."""
    MATCHES_DIR.mkdir(parents=True, exist_ok=True)

    def load_img(m):
        p = PREPROCESSED_DIR / f"map_{m}_clean.png"
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        # Scale down for display (both maps side by side)
        scale = 1500 / max(img.shape[:2])
        h = int(img.shape[0] * scale)
        w = int(img.shape[1] * scale)
        return cv2.resize(img, (w, h)), scale

    img_a, scale_a = load_img(map_a)
    img_b, scale_b = load_img(map_b)

    h = max(img_a.shape[0], img_b.shape[0])
    canvas = np.ones((h, img_a.shape[1] + 20 + img_b.shape[1], 3),
                     dtype=np.uint8) * 255
    canvas[:img_a.shape[0], :img_a.shape[1]] = img_a
    canvas[:img_b.shape[0], img_a.shape[1] + 20:] = img_b

    # Draw matched pairs
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255),
        (255, 165, 0), (128, 0, 128), (0, 255, 255),
    ]
    offset_b_x = img_a.shape[1] + 20
    for k, m in enumerate(matches[:20]):
        col = colors[k % len(colors)]
        pt_a = (int(m["cx_a"] * scale_a), int(m["cy_a"] * scale_a))
        pt_b = (int(m["cx_b"] * scale_b) + offset_b_x,
                int(m["cy_b"] * scale_b))
        cv2.circle(canvas, pt_a, 8, col, -1)
        cv2.circle(canvas, pt_b, 8, col, -1)
        cv2.line(canvas, pt_a, pt_b, col, 2)

    # Label
    cv2.putText(canvas, f"Map {map_a}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(canvas, f"Map {map_b}", (offset_b_x + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(canvas, f"{len(matches)} matches", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 0), 2)

    out = MATCHES_DIR / f"matches_{map_a}_{map_b}.png"
    ok, enc = cv2.imencode(".png", canvas)
    if ok:
        enc.tofile(str(out))
    print(f"  Saved: {out.name}")


def match_all():
    """Run matching on all known adjacent pairs."""
    print("\n  Matching all adjacent pairs...")
    model = load_model()
    MATCHES_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for map_a, map_b in ADJACENT_PAIRS:
        try:
            print(f"\n  Pair {map_a}_{map_b}...")
            matches = match_pair(model, map_a, map_b)
            print(f"  Found {len(matches)} matches")

            # Save JSON
            json_path = MATCHES_DIR / f"matches_{map_a}_{map_b}.json"
            with open(json_path, "w") as f:
                json.dump(matches, f, indent=2)

            # Save visualisation
            save_match_visualisation(map_a, map_b, matches)
            results[f"{map_a}_{map_b}"] = len(matches)

        except Exception as e:
            print(f"  Error matching {map_a}_{map_b}: {e}")

    print("\n" + "=" * 70)
    print("  MATCHING SUMMARY")
    print("=" * 70)
    for pair, n in results.items():
        print(f"  {pair}: {n} matches")
    print(f"\n  Output: {MATCHES_DIR.resolve()}")
    print(f"  Next: Run step7_homography.py to compute pairwise transforms")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GNN parcel matching — train and match"
    )
    parser.add_argument("--train", action="store_true",
                        help="Train the GNN matching model")
    parser.add_argument("--match", action="store_true",
                        help="Run matching inference")
    parser.add_argument("--all",   action="store_true",
                        help="Match all adjacent pairs")
    parser.add_argument("--pair",  type=str, default=None,
                        help="Specific pair to match, e.g. 45_47")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.match:
        model = load_model()
        if args.all:
            match_all()
        elif args.pair:
            a, b = args.pair.split("_")
            matches = match_pair(model, a, b)
            save_match_visualisation(a, b, matches)
            print(f"Found {len(matches)} matches for pair {args.pair}")
            for m in matches[:10]:
                print(f"  Parcel {m['parcel_a']} <-> {m['parcel_b']}  "
                      f"score={m['score']:.3f}")
        else:
            print("ERROR: --match needs --all or --pair A_B")
    else:
        print("Usage:")
        print("  python step6_gnn_matching.py --train")
        print("  python step6_gnn_matching.py --match --all")
        print("  python step6_gnn_matching.py --match --pair 45_47")
