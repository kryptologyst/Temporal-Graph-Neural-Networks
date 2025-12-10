"""Advanced temporal graph neural network models."""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric_temporal.nn.recurrent import EvolveGCNH, EvolveGCNO

from .layers import TimeEncoding, TemporalAttention, TemporalGCNLayer, TemporalNeighborSampler


class EvolveGCN(nn.Module):
    """EvolveGCN model for temporal graph learning."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        variant: str = "H"
    ):
        """Initialize EvolveGCN model.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            hidden_channels: Hidden layer dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
            variant: EvolveGCN variant ('H' or 'O')
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.variant = variant
        
        # EvolveGCN layers
        if variant == "H":
            self.evolve_layers = nn.ModuleList([
                EvolveGCNH(in_channels if i == 0 else hidden_channels, hidden_channels)
                for i in range(num_layers)
            ])
        else:
            self.evolve_layers = nn.ModuleList([
                EvolveGCNO(in_channels if i == 0 else hidden_channels, hidden_channels)
                for i in range(num_layers)
            ])
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        h: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_weight: Optional edge weights
            h: Optional hidden states
            
        Returns:
            Tuple of (output, hidden_states)
        """
        if h is None:
            h = [None] * self.num_layers
        
        # Forward through evolve layers
        new_h = []
        for i, layer in enumerate(self.evolve_layers):
            x, hidden = layer(x, edge_index, edge_weight, h[i])
            x = self.dropout(x)
            new_h.append(hidden)
        
        # Classification
        out = self.classifier(x)
        
        return out, new_h


class TGAT(nn.Module):
    """Temporal Graph Attention Network."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        time_dim: int = 16
    ):
        """Initialize TGAT model.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            hidden_channels: Hidden layer dimension
            num_heads: Number of attention heads
            num_layers: Number of layers
            dropout: Dropout rate
            time_dim: Time encoding dimension
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.time_dim = time_dim
        
        # Time encoding
        self.time_encoding = TimeEncoding(time_dim)
        
        # Input projection
        self.input_proj = nn.Linear(in_channels + time_dim, hidden_channels)
        
        # Temporal attention layers
        self.temporal_layers = nn.ModuleList([
            TemporalAttention(hidden_channels, num_heads)
            for _ in range(num_layers)
        ])
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GATConv(
                hidden_channels,
                hidden_channels // num_heads,
                heads=num_heads,
                dropout=dropout,
                concat=(i < num_layers - 1)
            )
            for i in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        timestamps: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            timestamps: Edge timestamps
            edge_weight: Optional edge weights
            
        Returns:
            torch.Tensor: Output predictions
        """
        # Encode timestamps
        time_enc = self.time_encoding(timestamps)
        
        # Combine features with time encoding
        x_time = torch.cat([x, time_enc], dim=-1)
        x = self.input_proj(x_time)
        
        # Apply temporal and graph attention layers
        for temporal_layer, gat_layer in zip(self.temporal_layers, self.gat_layers):
            # Temporal attention
            x_temp = temporal_layer(x.unsqueeze(1), time_enc.unsqueeze(1)).squeeze(1)
            
            # Graph attention
            x = gat_layer(x_temp, edge_index)
            x = self.dropout(x)
        
        # Classification
        out = self.classifier(x)
        
        return out


class TGN(nn.Module):
    """Temporal Graph Network."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        time_dim: int = 16
    ):
        """Initialize TGN model.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            hidden_channels: Hidden layer dimension
            num_layers: Number of layers
            dropout: Dropout rate
            time_dim: Time encoding dimension
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.time_dim = time_dim
        
        # Time encoding
        self.time_encoding = TimeEncoding(time_dim)
        
        # Memory modules
        self.node_memory = nn.Parameter(torch.randn(1000, hidden_channels))  # Fixed size for now
        self.edge_memory = nn.Parameter(torch.randn(1000, hidden_channels))
        
        # Message passing layers
        self.temporal_layers = nn.ModuleList([
            TemporalGCNLayer(
                in_channels if i == 0 else hidden_channels,
                hidden_channels,
                time_dim,
                dropout
            )
            for i in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        timestamps: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            timestamps: Edge timestamps
            edge_weight: Optional edge weights
            
        Returns:
            torch.Tensor: Output predictions
        """
        # Encode timestamps
        time_enc = self.time_encoding(timestamps)
        
        # Apply temporal GCN layers
        for layer in self.temporal_layers:
            x = layer(x, edge_index, time_enc, edge_weight)
            x = self.dropout(x)
        
        # Classification
        out = self.classifier(x)
        
        return out


class DyRep(nn.Module):
    """Dynamic Representation Learning for Temporal Graphs."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        time_dim: int = 16
    ):
        """Initialize DyRep model.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            hidden_channels: Hidden layer dimension
            num_layers: Number of layers
            dropout: Dropout rate
            time_dim: Time encoding dimension
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.time_dim = time_dim
        
        # Time encoding
        self.time_encoding = TimeEncoding(time_dim)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_channels + time_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal layers
        self.temporal_layers = nn.ModuleList([
            nn.LSTM(hidden_channels, hidden_channels, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        timestamps: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            timestamps: Edge timestamps
            edge_weight: Optional edge weights
            
        Returns:
            torch.Tensor: Output predictions
        """
        # Encode timestamps
        time_enc = self.time_encoding(timestamps)
        
        # Combine features with time encoding
        x_time = torch.cat([x, time_enc], dim=-1)
        x = self.encoder(x_time)
        
        # Apply temporal layers
        for lstm in self.temporal_layers:
            x, _ = lstm(x.unsqueeze(1))
            x = x.squeeze(1)
        
        # Decode
        out = self.decoder(x)
        
        return out


class TemporalGNNEnsemble(nn.Module):
    """Ensemble of temporal GNN models."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        dropout: float = 0.1,
        ensemble_size: int = 3
    ):
        """Initialize ensemble model.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            hidden_channels: Hidden layer dimension
            dropout: Dropout rate
            ensemble_size: Number of models in ensemble
        """
        super().__init__()
        self.ensemble_size = ensemble_size
        
        # Create ensemble of different models
        self.models = nn.ModuleList([
            EvolveGCN(in_channels, out_channels, hidden_channels, dropout=dropout),
            TGAT(in_channels, out_channels, hidden_channels, dropout=dropout),
            TGN(in_channels, out_channels, hidden_channels, dropout=dropout)
        ])
        
        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(ensemble_size) / ensemble_size)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        h: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            timestamps: Optional timestamps
            edge_weight: Optional edge weights
            h: Optional hidden states
            
        Returns:
            torch.Tensor: Ensemble predictions
        """
        outputs = []
        
        for i, model in enumerate(self.models):
            if isinstance(model, EvolveGCN):
                out, _ = model(x, edge_index, edge_weight, h)
            else:
                out = model(x, edge_index, timestamps, edge_weight)
            outputs.append(out)
        
        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_out = sum(w * out for w, out in zip(weights, outputs))
        
        return ensemble_out
