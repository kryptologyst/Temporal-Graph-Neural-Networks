#!/usr/bin/env python3
"""Evaluation script for temporal graph neural networks."""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data import TemporalGraphDataset, TemporalDataLoader
from src.models import EvolveGCN, TGAT, TGN, DyRep, TemporalGNNEnsemble
from src.eval import TemporalEvaluator
from src.utils import get_device, set_seed


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file."""
    return OmegaConf.load(config_path)


def create_model(config: DictConfig) -> torch.nn.Module:
    """Create model based on configuration."""
    model_name = config.model.name.lower()
    
    if model_name == "evolve_gcn":
        return EvolveGCN(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            hidden_channels=config.model.hidden_channels,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            variant=config.model.get("evolve_gcn", {}).get("variant", "H")
        )
    elif model_name == "tgat":
        return TGAT(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            hidden_channels=config.model.hidden_channels,
            num_heads=config.model.get("tgat", {}).get("num_heads", 4),
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            time_dim=config.model.get("tgat", {}).get("time_dim", 16)
        )
    elif model_name == "tgn":
        return TGN(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            hidden_channels=config.model.hidden_channels,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            time_dim=config.model.get("tgn", {}).get("time_dim", 16)
        )
    elif model_name == "dyrep":
        return DyRep(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            hidden_channels=config.model.hidden_channels,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            time_dim=config.model.get("dyrep", {}).get("time_dim", 16)
        )
    elif model_name == "ensemble":
        return TemporalGNNEnsemble(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            hidden_channels=config.model.hidden_channels,
            dropout=config.model.dropout,
            ensemble_size=config.model.get("ensemble", {}).get("ensemble_size", 3)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate temporal graph neural networks")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--data", type=str, default="data",
                       help="Path to data directory")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Output file for results")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override device if specified
    if args.device:
        config.system.device = args.device
    
    # Set random seed
    set_seed(config.system.seed)
    
    # Create directories
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Prepare data
    print("Preparing data...")
    dataset = TemporalGraphDataset(root=args.data, name=config.data.name)
    data_loader = TemporalDataLoader(
        dataset=dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio
    )
    
    test_loader = data_loader.get_test_loader()
    
    # Create model
    print(f"Creating {config.model.name} model...")
    model = create_model(config)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=get_device())
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(get_device())
    
    # Create evaluator
    evaluator = TemporalEvaluator(
        num_classes=config.evaluation.num_classes,
        task=config.evaluation.task,
        save_path=os.path.dirname(args.output)
    )
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluator.evaluate_model(
        model=model,
        data_loader=test_loader,
        model_name=config.model.name,
        config=OmegaConf.to_container(config)
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in metrics.items():
        if metric != "confusion_matrix":
            print(f"{metric}: {value:.4f}")
    
    # Save results
    import json
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
