"""Unit tests for temporal graph neural networks."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

# Add src to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import EvolveGCN, TGAT, TGN, DyRep, TemporalGNNEnsemble
from src.layers import TimeEncoding, TemporalAttention, TemporalGCNLayer
from src.data import TemporalGraphDataset, create_temporal_graph_data
from src.utils import set_seed, get_device, EarlyStopping
from src.eval import TemporalMetrics, ModelLeaderboard


class TestModels:
    """Test temporal GNN models."""
    
    def test_evolve_gcn(self):
        """Test EvolveGCN model."""
        model = EvolveGCN(in_channels=10, out_channels=2, hidden_channels=32)
        
        # Test forward pass
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        edge_weight = torch.randn(200)
        
        out, h = model(x, edge_index, edge_weight)
        
        assert out.shape == (100, 2)
        assert len(h) == 2  # num_layers
        assert h[0] is not None
    
    def test_tgat(self):
        """Test TGAT model."""
        model = TGAT(in_channels=10, out_channels=2, hidden_channels=32)
        
        # Test forward pass
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        timestamps = torch.randn(200)
        
        out = model(x, edge_index, timestamps)
        
        assert out.shape == (100, 2)
    
    def test_tgn(self):
        """Test TGN model."""
        model = TGN(in_channels=10, out_channels=2, hidden_channels=32)
        
        # Test forward pass
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        timestamps = torch.randn(200)
        
        out = model(x, edge_index, timestamps)
        
        assert out.shape == (100, 2)
    
    def test_dyrep(self):
        """Test DyRep model."""
        model = DyRep(in_channels=10, out_channels=2, hidden_channels=32)
        
        # Test forward pass
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        timestamps = torch.randn(200)
        
        out = model(x, edge_index, timestamps)
        
        assert out.shape == (100, 2)
    
    def test_ensemble(self):
        """Test ensemble model."""
        model = TemporalGNNEnsemble(in_channels=10, out_channels=2, hidden_channels=32)
        
        # Test forward pass
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        timestamps = torch.randn(200)
        
        out = model(x, edge_index, timestamps)
        
        assert out.shape == (100, 2)


class TestLayers:
    """Test custom layers."""
    
    def test_time_encoding(self):
        """Test time encoding layer."""
        encoder = TimeEncoding(dim=16)
        
        time = torch.tensor([0.0, 1.0, 2.0])
        encoded = encoder(time)
        
        assert encoded.shape == (3, 16)
        assert torch.allclose(encoded[0, 0], torch.sin(0.0), atol=1e-6)
    
    def test_temporal_attention(self):
        """Test temporal attention layer."""
        attention = TemporalAttention(hidden_dim=64, num_heads=4)
        
        x = torch.randn(10, 5, 64)  # batch, seq, hidden
        time = torch.randn(10, 5, 16)  # batch, seq, time_dim
        
        out = attention(x, time)
        
        assert out.shape == (10, 5, 64)
    
    def test_temporal_gcn_layer(self):
        """Test temporal GCN layer."""
        layer = TemporalGCNLayer(in_channels=10, out_channels=32, time_dim=16)
        
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        time_encoding = torch.randn(100, 16)
        
        out = layer(x, edge_index, time_encoding)
        
        assert out.shape == (100, 32)


class TestData:
    """Test data loading and processing."""
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        snapshots = create_temporal_graph_data(
            num_nodes=100,
            num_edges=200,
            num_timesteps=10
        )
        
        assert len(snapshots) == 10
        assert snapshots[0].num_nodes == 100
        assert snapshots[0].num_edges == 200
        assert snapshots[0].x.shape == (100, 10)
        assert snapshots[0].y.shape == (100,)
    
    @patch('src.data.EllipticBitcoinDataset')
    def test_temporal_graph_dataset(self, mock_dataset):
        """Test temporal graph dataset."""
        # Mock the dataset
        mock_snapshot = Mock()
        mock_snapshot.num_nodes = 100
        mock_snapshot.num_edges = 200
        mock_snapshot.x = torch.randn(100, 10)
        mock_snapshot.y = torch.randint(0, 2, (100,))
        mock_snapshot.edge_index = torch.randint(0, 100, (2, 200))
        
        mock_loader = [mock_snapshot for _ in range(5)]
        mock_dataset.return_value.get_dataset.return_value = mock_loader
        
        dataset = TemporalGraphDataset(root="test_data", name="elliptic")
        
        assert len(dataset) == 5
        snapshot = dataset[0]
        assert snapshot.num_nodes == 100


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test that seed is set
        torch.manual_seed(42)
        a = torch.randn(10)
        
        set_seed(42)
        torch.manual_seed(42)
        b = torch.randn(10)
        
        assert torch.allclose(a, b)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ['cuda', 'mps', 'cpu']
    
    def test_early_stopping(self):
        """Test early stopping."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)
        
        model = Mock()
        
        # Test improvement
        assert not early_stopping(0.8, model)
        assert not early_stopping(0.9, model)
        
        # Test no improvement
        assert not early_stopping(0.9, model)
        assert not early_stopping(0.9, model)
        assert not early_stopping(0.9, model)
        assert early_stopping(0.9, model)  # Should stop


class TestEvaluation:
    """Test evaluation metrics."""
    
    def test_temporal_metrics(self):
        """Test temporal metrics."""
        metrics = TemporalMetrics(num_classes=2)
        
        # Test update
        predictions = torch.tensor([0, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 1])
        timestamps = torch.tensor([0.0, 1.0, 2.0, 3.0])
        
        metrics.update(predictions, targets, timestamps)
        
        # Test compute
        result = metrics.compute()
        
        assert "accuracy" in result
        assert "f1" in result
        assert "auroc" in result
        assert result["accuracy"] == 1.0  # Perfect predictions
    
    def test_model_leaderboard(self):
        """Test model leaderboard."""
        leaderboard = ModelLeaderboard()
        
        # Add results
        metrics1 = {"accuracy": 0.8, "f1": 0.75, "auroc": 0.85}
        metrics2 = {"accuracy": 0.9, "f1": 0.88, "auroc": 0.92}
        
        leaderboard.add_result("Model1", metrics1)
        leaderboard.add_result("Model2", metrics2)
        
        # Test leaderboard
        df = leaderboard.get_leaderboard()
        
        assert len(df) == 2
        assert df.iloc[0]["model_name"] == "Model2"  # Higher score
        assert df.iloc[0]["score"] > df.iloc[1]["score"]


if __name__ == "__main__":
    pytest.main([__file__])
