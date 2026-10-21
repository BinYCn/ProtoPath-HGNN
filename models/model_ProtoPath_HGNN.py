"""ProtoPath-HGNN for whole-slide survival prediction.

The input is a PyG Data object with node features ``x``, integer ``node_type``,
``edge_index``, and four-column ``edge_attr``. The last two edge columns are
used by GAT. If present, ``rrwp`` contains eight node random-walk features;
the model does not compute RRWP or consume edge RRWP features. Each forward
call represents one whole-slide graph and updates ``data.x`` in place.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, LayerNorm

from models.gcn_layer import GraphConvolution
from models.model_utils import Attn_Net_Gated


class RRWPNodeEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.fc = nn.Linear(8, out_dim, bias=False)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, rrwp):
        return self.fc(rrwp)


class PositionAwareAttention(nn.Module):
    def __init__(self, feature_dim, position_dim, hidden_dim=64):
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.position_proj = nn.Linear(position_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, feature_dim)

    def forward(self, features, positions):
        features = self.feature_proj(features).unsqueeze(0)
        positions = self.position_proj(positions).unsqueeze(0)
        attended, _ = self.attention(positions, features, features)
        return self.output_proj(self.norm(attended.squeeze(0) + features.squeeze(0)))


class EdgeAwareGNNBlock(nn.Module):
    """GAT with node RRWP attention, residual connections and a feed-forward block."""

    def __init__(self, feature_dim, dropout=0.25):
        super().__init__()
        self.rrwp_node_encoder = RRWPNodeEncoder(feature_dim // 4)
        self.position_attention = PositionAwareAttention(feature_dim, feature_dim // 4)
        self.edge_encoder = nn.Sequential(
            nn.Linear(2, feature_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.conv1 = GATConv(feature_dim, feature_dim, edge_dim=feature_dim)
        self.norm = LayerNorm(feature_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm1_local = LayerNorm(feature_dim)
        self.ff_linear1 = nn.Linear(feature_dim, feature_dim * 2)
        self.ff_linear2 = nn.Linear(feature_dim * 2, feature_dim)
        self.act_fn_ff = nn.ReLU()
        self.norm2 = LayerNorm(feature_dim)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, data):
        x_in = data.x
        x = x_in
        if getattr(data, 'rrwp', None) is not None:
            x = self.position_attention(x, self.rrwp_node_encoder(data.rrwp))
        edge_attr = self.edge_encoder(data.edge_attr[:, 2:])
        x = self.conv1(x, data.edge_index, edge_attr=edge_attr)
        x = self.dropout(x_in + self.act(self.norm(x)))
        x = self.norm1_local(x)
        ff = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.norm2(x + self.ff_dropout2(self.ff_linear2(ff)))


class LightweightPrototypeEnhancedCosineGraph(nn.Module):
    def __init__(self, feature_dim, num_cell_types):
        super().__init__()
        self.num_cell_types = num_cell_types
        self.prototypes = nn.ParameterList([
            nn.Parameter(torch.randn(3, feature_dim)) for _ in range(num_cell_types)
        ])
        for prototypes in self.prototypes:
            nn.init.xavier_uniform_(prototypes)
        self.prototype_proj = nn.Linear(feature_dim, feature_dim // 4)
        self.fusion_weight = nn.Parameter(torch.tensor([0.6, 0.4]))

    def compute_prototype_aware_similarity(self, features, node_types):
        normalized = F.normalize(features, dim=1)
        original_sim = torch.mm(normalized, normalized.t())
        enhanced = features.clone()
        for cell_type in range(self.num_cell_types):
            mask = node_types == cell_type
            if not mask.any():
                continue
            type_features = features[mask]
            projected = F.normalize(self.prototype_proj(type_features), dim=1)
            prototypes = F.normalize(self.prototype_proj(self.prototypes[cell_type]), dim=1)
            weights = F.softmax(torch.mm(projected, prototypes.t()) / 0.1, dim=1)
            enhanced[mask] = type_features + 0.1 * torch.mm(weights, self.prototypes[cell_type])
        normalized = F.normalize(enhanced, dim=1)
        enhanced_sim = torch.mm(normalized, normalized.t())
        weights = F.softmax(self.fusion_weight, dim=0)
        similarity = (weights[0] * original_sim + weights[1] * enhanced_sim >= 0.9).float()
        row_sums = similarity.sum(dim=1, keepdim=True)
        similarity = similarity / torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
        return similarity, enhanced


class ProtoPath_HGNN(nn.Module):
    """Fixed GAT/RRWP/prototype configuration with add_1 fusion.

    With survival=True, returns hazards, survival probabilities, predicted
    interval, raw attention scores, and prototype diversity loss. With
    survival=False, returns logits, class probabilities, predicted class,
    raw attention scores, and prototype diversity loss. Node type 0 skips
    the intra-type GAT.
    """

    def __init__(self, in_feats=1024, out_feats=128, num_classes=4,
                 dropout=0.25, num_node_types=6, survival=True):
        super().__init__()
        self.num_node_types = num_node_types
        self.survival = survival
        self.fc = nn.Sequential(nn.Linear(in_feats, out_feats), nn.ReLU(), nn.Dropout(dropout))
        self.fc1 = nn.Sequential(nn.Linear(in_feats, out_feats), nn.ReLU(), nn.Dropout(dropout))
        self.node_gcns = nn.ModuleList([
            EdgeAwareGNNBlock(out_feats, dropout) for _ in range(num_node_types - 1)
        ])
        self.tissue_gcn = EdgeAwareGNNBlock(out_feats, dropout)
        self.graph_norm = LayerNorm(out_feats)
        self.prototype_cosine_graph = LightweightPrototypeEnhancedCosineGraph(out_feats, num_node_types)
        self.path_phi = nn.Sequential(nn.Linear(out_feats, out_feats), nn.ReLU(), nn.Dropout(dropout))
        self.path_attention_head = Attn_Net_Gated(L=out_feats, D=out_feats, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(nn.Linear(out_feats, out_feats), nn.ReLU())
        self.classifier = nn.Linear(out_feats, num_classes)
        self.gcn_cos_0 = GraphConvolution(out_feats, out_feats)
        self.gcn_cos_norm_0 = nn.Sequential(nn.LayerNorm(out_feats), nn.ReLU(inplace=True))
        self.gcn_cos_1 = GraphConvolution(out_feats, out_feats)
        self.gcn_cos_norm_1 = nn.Sequential(nn.LayerNorm(out_feats), nn.ReLU(inplace=True))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(out_feats)

    def compute_lightweight_prototype_loss(self):
        loss = 0.0
        for prototypes in self.prototype_cosine_graph.prototypes:
            normalized = F.normalize(prototypes, dim=1)
            similarities = torch.mm(normalized, normalized.t())
            mask = ~torch.eye(prototypes.size(0), dtype=torch.bool, device=prototypes.device)
            loss = loss + similarities[mask].mean()
        return loss * 0.1

    def forward(self, **kwargs):
        data = kwargs['x_path']
        x = self.fc(data.x)
        cos_x = self.fc1(data.x)
        data.x = x
        cos_sim, cos_features = self.prototype_cosine_graph.compute_prototype_aware_similarity(
            cos_x, data.node_type
        )
        for node_type in range(1, self.num_node_types):
            mask = data.node_type == node_type
            if mask.sum() > 1:
                data.x[mask] = self.node_gcns[node_type - 1](data.subgraph(mask))
        x_1 = self.graph_norm(self.tissue_gcn(data))
        x_2 = self.gcn_cos_norm_0(self.gcn_cos_0(cos_features, cos_sim))
        x_2 = self.gcn_cos_norm_1(self.gcn_cos_1(x_2, cos_sim))
        fused_features = self.norm(self.gamma * x_1 + (1 - self.gamma) * x_2)
        attention, features = self.path_attention_head(self.path_phi(fused_features))
        attention = attention.transpose(1, 0)
        features = torch.mm(F.softmax(attention, dim=1), features)
        h = self.path_rho(features)
        logits = self.classifier(h)
        prediction = torch.topk(logits, 1, dim=1)[1]
        prototype_loss = self.compute_lightweight_prototype_loss()
        if self.survival:
            hazards = torch.sigmoid(logits)
            survival = torch.cumprod(1 - hazards, dim=1)
            return hazards, survival, prediction, attention, prototype_loss
        probabilities = F.softmax(logits, dim=1)
        return logits, probabilities, prediction, attention, prototype_loss
