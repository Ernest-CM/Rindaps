"""GAT (v2) graph classifier with per-package embeddings."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool


class GATClassifier(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int = 64,
        embed_dim: int = 32,
        heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.embed_dim = embed_dim
        if embed_dim > 0:
            self.embedding = torch.nn.Embedding(num_nodes, embed_dim)
            torch.nn.init.normal_(self.embedding.weight, std=0.1)
            prev = in_channels + embed_dim
        else:
            self.embedding = None
            prev = in_channels

        self.convs = torch.nn.ModuleList()
        for layer_idx in range(num_layers):
            is_last = layer_idx == num_layers - 1
            out_per_head = hidden_channels if is_last else hidden_channels // heads
            concat = not is_last
            self.convs.append(GATv2Conv(prev, out_per_head, heads=heads, concat=concat, dropout=dropout))
            prev = out_per_head * heads if concat else out_per_head
        self.head = torch.nn.Sequential(
            torch.nn.Linear(2 * prev, prev),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(prev, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        node_id: torch.Tensor,
    ) -> torch.Tensor:
        if self.embedding is not None:
            embeds = self.embedding(node_id)
            x = torch.cat([x, embeds], dim=1)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        return self.head(x).squeeze(-1)
