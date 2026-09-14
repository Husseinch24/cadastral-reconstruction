"""
=============================================================================
MILESTONE 3 — PARCEL ADJACENCY GRAPH BUILDER
=============================================================================
Masters Thesis — AI-Based Panoramic Cadastral Image Reconstruction
Author  : Hussein Chalhoub

Purpose
--------
Take the parcel detections from Milestone 2 (Mask R-CNN) and build a
structural graph for each map where:

  Nodes = detected parcels
  Edges = spatial proximity (parcels that are likely neighbours)

Each node carries a rich feature vector:
  - Geometric: area, perimeter estimate, aspect ratio, compactness
  - Positional: centroid (normalised to [0,1] relative to map size)
  - Structural: degree (number of neighbours), local density

These graphs are the input to Milestone 4 (GNN matching).
The key insight: two adjacent atlas sheets share boundary parcels that
have IDENTICAL structural fingerprints in both maps — same area, same
neighbours, same position relative to the sheet boundary.  The GNN
learns to find these matches automatically.

Algorithm
----------
  For each map M:
    1. Load parcel predictions (JSON from Milestone 2)
    2. Build spatial index for fast proximity queries
    3. For each parcel pair (i, j):
         if distance(centroid_i, centroid_j) < PROXIMITY_THRESHOLD:
           add edge (i, j) with features [distance, overlap_ratio]
    4. Compute node features from geometry + local graph structure
    5. Save as GraphML file (readable by NetworkX and PyTorch Geometric)

Output
-------
  new_pipeline/data/graphs/
    map_<N>_graph.graphml     - full parcel graph
    map_<N>_graph.json        - same graph in JSON (for inspection)
    graph_stats.json          - summary statistics

Usage
------
  .\venv_thesis\Scripts\Activate.ps1
  python new_pipeline/src/step5_graph.py

=============================================================================
"""

import json
import time
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PREDICTIONS_DIR  = Path("new_pipeline/data/predictions")
GRAPHS_DIR       = Path("new_pipeline/data/graphs")
PREPROCESSED_DIR = Path("output/preprocessed")

ALL_MAPS = [str(n) for n in range(43, 56)]

# Proximity threshold: parcels whose centroids are within this many pixels
# are considered neighbours and get a graph edge.
# Set to roughly 2x the average parcel boundary width.
PROXIMITY_THRESHOLD_PX = 150

# Maximum neighbours per parcel (prevents huge dense graphs in crowded areas)
MAX_NEIGHBOURS = 12

# Minimum confidence for a parcel to be included in the graph
MIN_CONFIDENCE = 0.35


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def read_image_shape(map_num: str) -> tuple[int, int]:
    """Return (height, width) of the preprocessed map image."""
    path = PREPROCESSED_DIR / f"map_{map_num}_clean.png"
    data = np.fromfile(str(path), dtype=np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (6000, 8000)   # fallback default
    return img.shape


def load_parcels(map_num: str) -> list[dict]:
    """Load parcel predictions from Milestone 2 JSON output."""
    path = PREDICTIONS_DIR / f"map_{map_num}_parcels.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions not found: {path}\n"
            f"Run step4_segmentation.py --infer --all first."
        )
    with open(path, "r", encoding="utf-8") as f:
        parcels = json.load(f)

    # Filter by confidence
    parcels = [p for p in parcels if p.get("confidence", 1.0) >= MIN_CONFIDENCE]
    return parcels


# ---------------------------------------------------------------------------
# FEATURE COMPUTATION
# ---------------------------------------------------------------------------

def compute_node_features(parcel: dict,
                            map_h: int, map_w: int) -> dict:
    """
    Compute a rich feature vector for a single parcel node.

    Features are designed to be:
    - Scale-invariant (normalised by map size or parcel size)
    - Rotation-sensitive (position relative to map centre matters)
    - Distinctive (boundary parcels have different profiles than interior ones)
    """
    cx, cy   = parcel["cx"], parcel["cy"]
    x, y, bw, bh = parcel["bbox"]
    area     = parcel.get("area_px", bw * bh)

    # Normalised centroid position [0, 1] relative to map
    norm_cx = cx / max(map_w, 1)
    norm_cy = cy / max(map_h, 1)

    # Aspect ratio (1.0 = square, >1 = wide, <1 = tall)
    aspect = bw / max(bh, 1)

    # Compactness: how close to a square is the bounding box
    # 1.0 = perfectly square, 0 = infinitely elongated
    compactness = min(bw, bh) / max(max(bw, bh), 1)

    # Normalised area (relative to map area)
    norm_area = area / max(map_h * map_w, 1)

    # Perimeter estimate from bounding box (2*(w+h))
    perim_estimate = 2 * (bw + bh)

    # Distance from map boundary (useful for identifying boundary parcels)
    dist_left   = cx / max(map_w, 1)
    dist_right  = (map_w - cx) / max(map_w, 1)
    dist_top    = cy / max(map_h, 1)
    dist_bottom = (map_h - cy) / max(map_h, 1)
    dist_boundary = min(dist_left, dist_right, dist_top, dist_bottom)

    # Distance from map centre
    dist_centre = np.sqrt(
        (norm_cx - 0.5) ** 2 + (norm_cy - 0.5) ** 2
    )

    return {
        # Geometric features
        "area_px":          float(area),
        "norm_area":        float(norm_area),
        "bbox_w":           float(bw),
        "bbox_h":           float(bh),
        "aspect_ratio":     float(aspect),
        "compactness":      float(compactness),
        "perim_estimate":   float(perim_estimate),
        # Positional features
        "norm_cx":          float(norm_cx),
        "norm_cy":          float(norm_cy),
        "dist_boundary":    float(dist_boundary),
        "dist_centre":      float(dist_centre),
        # Raw (for edge computation, not GNN features)
        "cx":               float(cx),
        "cy":               float(cy),
        # Detection quality
        "confidence":       float(parcel.get("confidence", 1.0)),
    }


def compute_edge_features(p_i: dict, p_j: dict,
                            dist: float) -> dict:
    """Compute features for an edge between two parcels."""
    # Bounding box overlap ratio (IoU-style)
    xi1, yi1, wi1, hi1 = p_i["bbox"]
    xi2, yi2, wi2, hi2 = p_j["bbox"]

    ix0 = max(xi1, xi2); iy0 = max(yi1, yi2)
    ix1 = min(xi1 + wi1, xi2 + wi2)
    iy1 = min(yi1 + hi1, yi2 + hi2)

    inter_area = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union_area = (wi1 * hi1 + wi2 * hi2 - inter_area)
    iou = inter_area / max(union_area, 1)

    # Relative direction (angle from i to j)
    dx = p_j["cx"] - p_i["cx"]
    dy = p_j["cy"] - p_i["cy"]
    angle = float(np.arctan2(dy, dx))

    # Relative size ratio
    area_i = p_i.get("area_px", wi1 * hi1)
    area_j = p_j.get("area_px", wi2 * hi2)
    size_ratio = min(area_i, area_j) / max(max(area_i, area_j), 1)

    return {
        "distance":   float(dist),
        "iou":        float(iou),
        "angle":      float(angle),
        "size_ratio": float(size_ratio),
    }


# ---------------------------------------------------------------------------
# GRAPH BUILDER
# ---------------------------------------------------------------------------

def build_graph_for_map(map_num: str) -> nx.Graph:
    """
    Build the parcel adjacency graph for a single map.

    Returns a NetworkX Graph where:
      - Each node has all the features from compute_node_features()
      - Each edge has all the features from compute_edge_features()
      - Node IDs are integers matching the parcel_id from predictions
    """
    parcels = load_parcels(map_num)
    map_h, map_w = read_image_shape(map_num)

    G = nx.Graph()
    G.graph["map_num"] = map_num
    G.graph["map_h"]   = map_h
    G.graph["map_w"]   = map_w
    G.graph["n_parcels_raw"] = len(parcels)

    if not parcels:
        return G

    # Add nodes with features
    centroids = []
    for p in parcels:
        features = compute_node_features(p, map_h, map_w)
        node_id  = int(p["parcel_id"])
        G.add_node(node_id, **features,
                   # Store bbox separately for edge computation
                   bbox=p["bbox"])
        centroids.append([p["cx"], p["cy"]])

    centroids_arr = np.array(centroids)

    # Build spatial index for fast nearest-neighbour queries
    kdtree = cKDTree(centroids_arr)

    # Find all pairs within PROXIMITY_THRESHOLD_PX
    pairs = kdtree.query_pairs(r=PROXIMITY_THRESHOLD_PX)

    # Add edges
    for i_idx, j_idx in pairs:
        p_i = parcels[i_idx]
        p_j = parcels[j_idx]
        dist = float(np.linalg.norm(
            centroids_arr[i_idx] - centroids_arr[j_idx]
        ))
        edge_feats = compute_edge_features(p_i, p_j, dist)
        G.add_edge(
            int(p_i["parcel_id"]),
            int(p_j["parcel_id"]),
            **edge_feats
        )

    # Prune: if a node has too many neighbours, keep only the closest ones
    for node in list(G.nodes()):
        neighbours = list(G.neighbors(node))
        if len(neighbours) > MAX_NEIGHBOURS:
            # Sort by distance, keep closest MAX_NEIGHBOURS
            neighbour_dists = [
                (nb, G[node][nb]["distance"]) for nb in neighbours
            ]
            neighbour_dists.sort(key=lambda x: x[1])
            to_remove = [nb for nb, _ in neighbour_dists[MAX_NEIGHBOURS:]]
            for nb in to_remove:
                G.remove_edge(node, nb)

    # Add structural features AFTER graph is built (requires knowing degrees)
    for node in G.nodes():
        G.nodes[node]["degree"] = G.degree(node)

        # Local density: edges among neighbours / possible edges
        neighbours = list(G.neighbors(node))
        if len(neighbours) >= 2:
            possible = len(neighbours) * (len(neighbours) - 1) / 2
            actual   = sum(
                1 for i in range(len(neighbours))
                for j in range(i + 1, len(neighbours))
                if G.has_edge(neighbours[i], neighbours[j])
            )
            local_density = actual / max(possible, 1)
        else:
            local_density = 0.0
        G.nodes[node]["local_density"] = float(local_density)

    G.graph["n_nodes"] = G.number_of_nodes()
    G.graph["n_edges"] = G.number_of_edges()
    return G


def graph_to_json(G: nx.Graph) -> dict:
    """Convert NetworkX graph to a JSON-serializable dict."""
    return {
        "graph":  dict(G.graph),
        "nodes":  [
            {"id": n, **{k: v for k, v in d.items() if k != "bbox"}}
            for n, d in G.nodes(data=True)
        ],
        "edges":  [
            {"source": u, "target": v, **d}
            for u, v, d in G.edges(data=True)
        ],
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run():
    print("\n" + "=" * 70)
    print("  MILESTONE 3 — PARCEL ADJACENCY GRAPH BUILDER")
    print("  Cadastral Panoramic Reconstruction — Masters Thesis")
    print("=" * 70)

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    t0_total = time.time()
    all_stats = {}

    for map_num in tqdm(ALL_MAPS, desc="  Building graphs"):
        pred_path = PREDICTIONS_DIR / f"map_{map_num}_parcels.json"
        if not pred_path.exists():
            print(f"\n  Skipping map {map_num} (no predictions found)")
            continue

        t0 = time.time()
        G = build_graph_for_map(map_num)

        # Save GraphML (for PyTorch Geometric loading)
        # Strip bbox attribute first (not GraphML-compatible as list)
        G_save = G.copy()
        for node in G_save.nodes():
            if "bbox" in G_save.nodes[node]:
                del G_save.nodes[node]["bbox"]
        graphml_path = GRAPHS_DIR / f"map_{map_num}_graph.graphml"
        nx.write_graphml(G_save, str(graphml_path))

        # Save JSON (for inspection)
        json_path = GRAPHS_DIR / f"map_{map_num}_graph.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(graph_to_json(G), f, indent=2)

        elapsed = time.time() - t0
        stats = {
            "map_num":        map_num,
            "n_parcels_raw":  G.graph.get("n_parcels_raw", 0),
            "n_nodes":        G.number_of_nodes(),
            "n_edges":        G.number_of_edges(),
            "avg_degree":     round(
                sum(d for _, d in G.degree()) / max(G.number_of_nodes(), 1), 2
            ),
            "elapsed_sec":    round(elapsed, 2),
        }
        all_stats[map_num] = stats
        tqdm.write(
            f"  Map {map_num}: {stats['n_nodes']} nodes, "
            f"{stats['n_edges']} edges, "
            f"avg_degree={stats['avg_degree']:.1f}  "
            f"({elapsed:.1f}s)"
        )

    # Save summary
    summary = {
        "total_maps":    len(all_stats),
        "total_nodes":   sum(s["n_nodes"] for s in all_stats.values()),
        "total_edges":   sum(s["n_edges"] for s in all_stats.values()),
        "total_time_sec": round(time.time() - t0_total, 1),
        "maps":          all_stats,
    }
    stats_path = GRAPHS_DIR / "graph_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("  MILESTONE 3 SUMMARY")
    print("=" * 70)
    print(f"  {'Map':<6} {'Nodes':>7} {'Edges':>7} {'Avg deg':>8}")
    print(f"  {'-'*32}")
    for m, s in all_stats.items():
        print(f"  {m:<6} {s['n_nodes']:>7} {s['n_edges']:>7} "
              f"{s['avg_degree']:>8.1f}")
    print(f"  {'-'*32}")
    print(f"  {'Total':<6} "
          f"{summary['total_nodes']:>7} "
          f"{summary['total_edges']:>7}")
    print(f"\n  Output: {GRAPHS_DIR.resolve()}")
    print(f"  Time:   {summary['total_time_sec']:.1f}s")
    print(f"\n  Next: Run step6_gnn_matching.py to train the GNN matcher")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run()
