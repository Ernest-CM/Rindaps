"""GCN graph classifier with per-package embeddings."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool


class GCNClassifier(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int = 64,
        embed_dim: int = 32,
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
        for _ in range(num_layers):
            self.convs.append(GCNConv(prev, hidden_channels))
            prev = hidden_channels
        self.head = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, 1),
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
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        return self.head(x).squeeze(-1)
