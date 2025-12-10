"""Temporal graph neural network layers and components."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class TimeEncoding(nn.Module):
    """Sinusoidal time encoding for temporal features."""
    
    def __init__(self, dim: int):
        """Initialize time encoding.
        
        Args:
            dim: Dimension of time encoding
        """
        super().__init__()
        self.dim = dim
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Encode time values.
        
        Args:
            time: Time tensor of shape (N,)
            
        Returns:
            torch.Tensor: Encoded time features of shape (N, dim)
        """
        device = time.device
        time = time.float()
        
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        return embeddings


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for dynamic graphs."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        """Initialize temporal attention.
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        time: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply temporal attention.
        
        Args:
            x: Node features of shape (N, hidden_dim)
            time: Time features of shape (N, time_dim)
            mask: Optional attention mask
            
        Returns:
            torch.Tensor: Attended features
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Combine features and time
        combined = x + time
        
        # Multi-head attention
        q = self.query(combined).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(combined).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(combined).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)
        
        return out


class TemporalGCNLayer(MessagePassing):
    """Temporal Graph Convolutional Layer."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.1,
        aggr: str = "add"
    ):
        """Initialize temporal GCN layer.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            time_dim: Time encoding dimension
            dropout: Dropout rate
            aggr: Aggregation method
        """
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        
        self.linear = nn.Linear(in_channels + time_dim, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        time_encoding: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            time_encoding: Time encodings
            edge_weight: Optional edge weights
            
        Returns:
            torch.Tensor: Updated node features
        """
        # Combine features with time encoding
        x_time = torch.cat([x, time_encoding], dim=-1)
        
        # Propagate messages
        out = self.propagate(edge_index, x=x_time, edge_weight=edge_weight)
        
        # Apply linear transformation and normalization
        out = self.linear(out)
        out = self.dropout(out)
        out = self.norm(out)
        
        return out
    
    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor:
        """Message function.
        
        Args:
            x_j: Source node features
            edge_weight: Optional edge weights
            
        Returns:
            torch.Tensor: Messages
        """
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        return x_j


class TemporalNeighborSampler:
    """Temporal neighbor sampling for dynamic graphs."""
    
    def __init__(
        self,
        edge_index: torch.Tensor,
        timestamps: torch.Tensor,
        num_neighbors: int = 10
    ):
        """Initialize temporal neighbor sampler.
        
        Args:
            edge_index: Edge indices
            timestamps: Edge timestamps
            num_neighbors: Number of neighbors to sample
        """
        self.edge_index = edge_index
        self.timestamps = timestamps
        self.num_neighbors = num_neighbors
        
        # Build temporal adjacency lists
        self._build_temporal_adj()
    
    def _build_temporal_adj(self) -> None:
        """Build temporal adjacency lists."""
        self.temporal_adj = {}
        
        for i, (src, dst) in enumerate(self.edge_index.t()):
            timestamp = self.timestamps[i].item()
            
            if src.item() not in self.temporal_adj:
                self.temporal_adj[src.item()] = []
            
            self.temporal_adj[src.item()].append((dst.item(), timestamp))
    
    def sample_neighbors(
        self, 
        node: int, 
        current_time: float,
        max_time_diff: float = float('inf')
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample temporal neighbors for a node.
        
        Args:
            node: Node ID
            current_time: Current timestamp
            max_time_diff: Maximum time difference for neighbors
            
        Returns:
            Tuple of (neighbor_ids, timestamps)
        """
        if node not in self.temporal_adj:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float)
        
        # Filter neighbors by time
        valid_neighbors = [
            (neighbor, timestamp) 
            for neighbor, timestamp in self.temporal_adj[node]
            if current_time - timestamp <= max_time_diff
        ]
        
        if not valid_neighbors:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float)
        
        # Sort by timestamp (most recent first)
        valid_neighbors.sort(key=lambda x: x[1], reverse=True)
        
        # Sample neighbors
        sampled = valid_neighbors[:self.num_neighbors]
        
        neighbor_ids = torch.tensor([n[0] for n in sampled], dtype=torch.long)
        timestamps = torch.tensor([n[1] for n in sampled], dtype=torch.float)
        
        return neighbor_ids, timestamps
