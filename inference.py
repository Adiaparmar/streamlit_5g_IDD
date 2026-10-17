"""
inference.py — CEM-RAG offline inference wrapper.

Loads trained model + artifacts from netslab_episode_kb_v3/ and exposes
a single .predict(kpi_array) call.  No database access required.

Usage:
    from inference import CEMRAGInference
    eng = CEMRAGInference()
    result = eng.predict(my_array, col_names=[...])
"""

import os
import json
import numpy as np
import torch
import faiss

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict
from scipy.stats import linregress
from scipy.stats import entropy as sp_entropy
from scipy.signal import welch


# ---------------------------------------------------------------------------
# Two-tower model — exact replica of complete_pipeline.py architecture
# ---------------------------------------------------------------------------

class GraphConvLayer(torch.nn.Module):
    """Single GCN layer: H' = ReLU(D^{-0.5} A D^{-0.5} H W)"""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.bn = torch.nn.BatchNorm1d(out_dim)

    def forward(self, x, adj_norm):
        x_prop = torch.bmm(adj_norm, x)
        h = self.linear(x_prop)
        h = self.bn(h.transpose(1, 2)).transpose(1, 2)
        return self.dropout(torch.nn.functional.relu(h))


class GraphEncoder(torch.nn.Module):
    """Tower 2: GCN — identical to complete_pipeline.py GraphEncoder."""
    def __init__(self, n_nodes: int, node_feat_dim: int, hidden_dim: int,
                 output_dim: int, n_layers: int = 2):
        super().__init__()
        self.n_nodes = n_nodes
        self.node_encoder = torch.nn.Sequential(
            torch.nn.Linear(node_feat_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_dim),
        )
        self.convs = torch.nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, 4), torch.nn.ReLU(),
            torch.nn.Linear(4, 1), torch.nn.Sigmoid(),
        )
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_dim * 2),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim * 2, output_dim),
        )

    def forward(self, node_feats, adj_mats, edge_feats=None):
        if edge_feats is not None:
            edge_w = self.edge_encoder(edge_feats).squeeze(-1)
            adj_mats = adj_mats * edge_w
        d = adj_mats.sum(dim=-1) + 1e-10
        d_inv_sqrt = d.pow(-0.5)
        adj_norm = adj_mats * d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(-2)
        b, n, f = node_feats.shape
        h_flat = node_feats.reshape(-1, f)
        h = self.node_encoder(h_flat).reshape(b, n, -1)
        for conv in self.convs:
            h = conv(h, adj_norm)
        emb = self.projection(h.mean(dim=1))
        return torch.nn.functional.normalize(emb, p=2, dim=1)


class TemporalEncoder(torch.nn.Module):
    """Tower 1: Conv1D + Self-Attention — identical to complete_pipeline.py TemporalEncoder."""
    def __init__(self, n_kpis: int, window_size: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(n_kpis, hidden_dim // 2, kernel_size=3, padding=1)
        self.bn1   = torch.nn.BatchNorm1d(hidden_dim // 2)
        self.conv2 = torch.nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=5, padding=2)
        self.bn2   = torch.nn.BatchNorm1d(hidden_dim)
        self.attention = torch.nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.attn_norm = torch.nn.LayerNorm(hidden_dim)
        self.pos_encoding = torch.nn.Parameter(torch.randn(1, window_size, hidden_dim) * 0.02)
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_dim * 2),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim * 2, output_dim),
        )

    def forward(self, x):
        x_conv = x.transpose(1, 2)
        x_conv = torch.nn.functional.relu(self.bn1(self.conv1(x_conv)))
        x_conv = torch.nn.functional.relu(self.bn2(self.conv2(x_conv)))
        x_attn = x_conv.transpose(1, 2) + self.pos_encoding[:, :x_conv.shape[2], :]
        attn_out, _ = self.attention(x_attn, x_attn, x_attn)
        x_attn = self.attn_norm(x_attn + attn_out)
        x_pooled = x_attn.mean(dim=1)
        emb = self.projection(x_pooled)
        return torch.nn.functional.normalize(emb, p=2, dim=1)


class TwoTowerIDS(torch.nn.Module):
    """Two-Tower IDS — identical to complete_pipeline.py."""
    def __init__(self, n_kpis: int, window_size: int = 24, n_nodes: int = 16,
                 node_feat_dim: int = 8,
                 temporal_hidden: int = 64, temporal_out: int = 64,
                 graph_hidden: int = 64, graph_out: int = 64,
                 fusion_out: int = 128, gcn_layers: int = 2):
        super().__init__()
        self.temporal   = TemporalEncoder(n_kpis, window_size,
                                           temporal_hidden, temporal_out)
        self.structural = GraphEncoder(n_nodes, node_feat_dim,
                                        graph_hidden, graph_out, gcn_layers)
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(temporal_out + graph_out, fusion_out),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(fusion_out),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(fusion_out, fusion_out),
        )

    def forward(self, kpi_seq, node_feats, adj_mats, edge_feats=None):
        t_emb = self.temporal(kpi_seq)
        s_emb = self.structural(node_feats, adj_mats, edge_feats)
        combined = torch.cat([t_emb, s_emb], dim=1)
        u_emb = self.fusion(combined)
        return t_emb, s_emb, torch.nn.functional.normalize(u_emb, p=2, dim=1)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    pred_category:     str
    pred_subcategory:  str
    category_conf:     float
    subcategory_conf:  float
    top_categories:    Dict[str, float]
    top_subcategories: Dict[str, float]
    retrieved_labels:  List[Dict]            # [{category, subcategory, score}]
    anomalous_kpis:    List[Tuple[str, float]]  # top anomalous (kpi, z)
    graph_edges:       List[Tuple[str, str, float]]  # (kpi_a, kpi_b, r)
    kpi_z_scores:      Dict[str, float]      # full z-score map
    kpi_trends:        Dict[str, str]        # rising / falling / flat
    raw_kpi_array:     np.ndarray            # (24, K) z-scored window


# ---------------------------------------------------------------------------
# Main inference engine
# ---------------------------------------------------------------------------

class CEMRAGInference:
    """Offline inference wrapper for a trained CEM-RAG pipeline."""

    CORR_THRESHOLD = 0.3
    Z_CLIP         = 5.0
    GRAPH_ALPHA    = 0.75
    GRAPH_BETA     = 0.25

    def __init__(self, kb_dir: str = "netslab_episode_kb_v3", device: str = "cpu"):
        self.kb_dir = kb_dir
        self.device = device
        self._load_artifacts()

    def _load_artifacts(self):
        d = self.kb_dir

        with open(os.path.join(d, "kpi_names.json")) as f:
            self.kpi_names: List[str] = json.load(f)
        self.n_kpis = len(self.kpi_names)

        with open(os.path.join(d, "kpi_norm_stats.json")) as f:
            self.kpi_stats: Dict = json.load(f)

        with open(os.path.join(d, "nf_norm_stats.json")) as f:
            nf = json.load(f)
        self.nf_mean = np.array(nf["mean"], dtype=np.float32)
        self.nf_std  = np.array(nf["std"],  dtype=np.float32)

        self.faiss_index = faiss.read_index(os.path.join(d, "faiss_index.bin"))

        with open(os.path.join(d, "window_labels.json")) as f:
            self.window_labels: List[Dict] = json.load(f)

        model = TwoTowerIDS(
            n_kpis=self.n_kpis,
            window_size=24,
            n_nodes=self.n_kpis,   # one node per KPI
            node_feat_dim=8,
            temporal_hidden=64, temporal_out=64,
            graph_hidden=64, graph_out=64,
            fusion_out=128, gcn_layers=2,
        )
        model.load_state_dict(
            torch.load(os.path.join(d, "twotower_model.pt"), map_location="cpu")
        )
        model.eval()
        self.model = model.to(self.device)
        print(f"[CEMRAGInference] Loaded: {self.n_kpis} KPIs, "
              f"{self.faiss_index.ntotal} KB vectors.")

    # ------------------------------------------------------------------
    def predict(self, kpi_array: np.ndarray,
                col_names: Optional[List[str]] = None,
                k: int = 10) -> InferenceResult:
        """Run inference on a raw KPI array.

        kpi_array : (T, K_user) — any number of rows, any subset of KPI columns.
        col_names : column names for kpi_array. None = assume full 16-KPI order.
        """
        arr = self._align_columns(kpi_array, col_names)
        arr = self._pad_or_crop(arr, 24)
        kpi_z, z_means = self._zscore(arr)
        adj, ef = self._build_corr_graph(kpi_z)
        nf = self._compute_node_features(kpi_z, z_means)
        nf = self._normalize_node_features(nf)
        emb = self._embed(kpi_z, nf, adj, ef)
        retrieved, raw_scores = self._retrieve(emb, adj, k)

        pred_cat, cat_conf, top_cats, \
        pred_sub, sub_conf, top_subs = self._vote(retrieved)

        anomalous = sorted(
            [(self.kpi_names[i], float(z_means[i])) for i in range(self.n_kpis)],
            key=lambda x: abs(x[1]), reverse=True
        )[:6]

        return InferenceResult(
            pred_category=pred_cat,
            pred_subcategory=pred_sub,
            category_conf=float(cat_conf),
            subcategory_conf=float(sub_conf),
            top_categories=top_cats,
            top_subcategories=top_subs,
            retrieved_labels=[
                {
                    "category":    self.window_labels[idx]["category"],
                    "subcategory": self.window_labels[idx]["subcategory"],
                    "score":       float(s),
                }
                for idx, s in retrieved
            ],
            anomalous_kpis=anomalous,
            graph_edges=self._top_edges(adj),
            kpi_z_scores={self.kpi_names[i]: float(z_means[i])
                          for i in range(self.n_kpis)},
            kpi_trends=self._compute_trends(kpi_z),
            raw_kpi_array=kpi_z,
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _align_columns(self, arr, col_names):
        T = arr.shape[0]
        out = np.zeros((T, self.n_kpis), dtype=np.float32)
        if col_names is None:
            n = min(arr.shape[1], self.n_kpis)
            out[:, :n] = arr[:, :n].astype(np.float32)
        else:
            lut = {n.lower().strip(): i for i, n in enumerate(self.kpi_names)}
            for j, name in enumerate(col_names):
                idx = lut.get(name.lower().strip())
                if idx is not None and j < arr.shape[1]:
                    out[:, idx] = arr[:, j].astype(np.float32)
        return out

    def _pad_or_crop(self, arr, W=24):
        T = arr.shape[0]
        if T == W:  return arr
        if T > W:   return arr[-W:]
        pad = np.tile(arr[0:1], (W - T, 1))
        return np.vstack([pad, arr])

    def _zscore(self, arr):
        out = np.zeros_like(arr, dtype=np.float32)
        z_means = np.zeros(self.n_kpis, dtype=np.float32)
        for i, kpi in enumerate(self.kpi_names):
            s = self.kpi_stats.get(kpi, {"mean": 0.0, "std": 1.0})
            col = (arr[:, i] - s["mean"]) / max(s["std"], 1e-8)
            col = np.clip(col, -self.Z_CLIP, self.Z_CLIP)
            out[:, i] = col
            z_means[i] = float(np.mean(col))
        return out, z_means

    def _build_corr_graph(self, kpi_z):
        n = kpi_z.shape[1]
        adj = np.zeros((n, n), dtype=np.float32)
        ef  = np.zeros((n, n, 1), dtype=np.float32)
        if kpi_z.shape[0] > 2:
            try:
                corr = np.nan_to_num(np.corrcoef(kpi_z.T), nan=0.0)
                np.fill_diagonal(corr, 0.0)
                mask = np.abs(corr) > self.CORR_THRESHOLD
                adj[mask] = np.abs(corr[mask])
                ef[mask, 0] = corr[mask]
            except Exception:
                pass
        return adj, ef

    def _compute_node_features(self, kpi_z, z_means):
        n = kpi_z.shape[1]
        nf = np.zeros((n, 8), dtype=np.float32)
        for i in range(n):
            vals = kpi_z[:, i]
            std = float(np.std(vals))
            if std < 1e-10:
                nf[i, 0] = z_means[i]; continue
            m = float(np.mean(vals))
            try:   slope = float(linregress(np.arange(len(vals)), vals).slope)
            except: slope = 0.0
            var  = float(np.var(vals))
            curr = float(vals[-1])
            z    = float((curr - m) / (std + 1e-8))
            try:
                h, _ = np.histogram(vals, bins=5)
                ent = float(sp_entropy(h / (h.sum() + 1e-10)))
            except: ent = 0.0
            try:
                _, psd = welch(vals, nperseg=min(8, len(vals)))
                se = float(np.sum(psd))
            except: se = 0.0
            ma = float(np.mean(vals[-max(2, len(vals)//4):]))
            nf[i] = [m, slope, var, curr, z, ent, se, ma]
        return nf

    def _normalize_node_features(self, nf):
        flat = nf.reshape(-1, nf.shape[-1])
        flat = np.clip((flat - self.nf_mean) / self.nf_std, -5.0, 5.0)
        return flat.reshape(nf.shape).astype(np.float32)

    def _embed(self, kpi_z, nf, adj, ef):
        with torch.no_grad():
            k  = torch.from_numpy(kpi_z).float().unsqueeze(0).to(self.device)
            n  = torch.from_numpy(nf).float().unsqueeze(0).to(self.device)
            a  = torch.from_numpy(adj).float().unsqueeze(0).to(self.device)
            e  = torch.from_numpy(ef).float().unsqueeze(0).to(self.device)
            _, _, u = self.model(k, n, a, e)
        return u.squeeze(0).cpu().numpy().astype(np.float32)

    def _retrieve(self, emb, adj, k=10):
        q = emb.reshape(1, -1)
        q = q / (np.linalg.norm(q) + 1e-8)
        sk = min(k * 5, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(q, sk)
        candidates = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            cos = float(scores[0][i])
            sa  = np.array(self.window_labels[idx]["graph_adj"], dtype=np.float32)
            jac = self._jaccard(adj, sa)
            candidates.append((idx, self.GRAPH_ALPHA * cos + self.GRAPH_BETA * jac))
        candidates.sort(key=lambda x: -x[1])
        top = candidates[:k]
        return top, [s for _, s in top]

    def _jaccard(self, a, b):
        ma = (a > self.CORR_THRESHOLD)
        mb = (b > self.CORR_THRESHOLD)
        i  = np.logical_and(ma, mb).sum()
        u  = np.logical_or(ma, mb).sum()
        return float(i / u) if u > 0 else 0.0

    def _vote(self, retrieved):
        cat_s: Dict[str, float] = defaultdict(float)
        sub_s: Dict[str, float] = defaultdict(float)
        for (idx, s) in retrieved:
            lbl = self.window_labels[idx]
            cat_s[lbl["category"]]    += s
            sub_s[lbl["subcategory"]] += s

        pred_cat = max(cat_s, key=cat_s.get)
        cat_conf = cat_s[pred_cat] / max(sum(cat_s.values()), 1e-8)

        cat_sub = {
            sub: v for sub, v in sub_s.items()
            if any(
                self.window_labels[idx]["category"] == pred_cat
                and self.window_labels[idx]["subcategory"] == sub
                for (idx, _) in retrieved
            )
        }
        if not cat_sub:
            cat_sub = dict(sub_s)

        pred_sub = max(cat_sub, key=cat_sub.get)
        sub_conf = cat_sub[pred_sub] / max(sum(cat_sub.values()), 1e-8)

        return (pred_cat, cat_conf, dict(cat_s),
                pred_sub, sub_conf, dict(cat_sub))

    def _top_edges(self, adj, top_n=5):
        edges = []
        for i in range(adj.shape[0]):
            for j in range(i + 1, adj.shape[0]):
                if adj[i, j] > self.CORR_THRESHOLD:
                    edges.append((self.kpi_names[i], self.kpi_names[j], float(adj[i, j])))
        return sorted(edges, key=lambda x: -x[2])[:top_n]

    def _compute_trends(self, kpi_z):
        out = {}
        for i, kpi in enumerate(self.kpi_names):
            vals = kpi_z[:, i]
            try:
                sl = linregress(np.arange(len(vals)), vals).slope
                out[kpi] = "rising" if sl > 0.05 else ("falling" if sl < -0.05 else "flat")
            except:
                out[kpi] = "flat"
        return out

