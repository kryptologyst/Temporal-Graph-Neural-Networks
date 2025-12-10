#!/usr/bin/env python3
"""Main training script for temporal graph neural networks."""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data import TemporalGraphDataset, TemporalDataLoader, TemporalGraphPreprocessor
from src.models import EvolveGCN, TGAT, TGN, DyRep, TemporalGNNEnsemble
from src.train import TemporalTrainer
from src.utils import get_device, set_seed


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration
    """
    return OmegaConf.load(config_path)


def create_model(config: DictConfig) -> torch.nn.Module:
    """Create model based on configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Created model
    """
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


def prepare_data(config: DictConfig):
    """Prepare data loaders.
    
    Args:
        config: Data configuration
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create dataset
    dataset = TemporalGraphDataset(
        root=config.data.root,
        name=config.data.name
    )
    
    # Create data loader
    data_loader = TemporalDataLoader(
        dataset=dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio
    )
    
    # Get loaders
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()
    test_loader = data_loader.get_test_loader()
    
    return train_loader, val_loader, test_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train temporal graph neural networks")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model", type=str, default="evolve_gcn",
                       help="Model name")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--eval_only", action="store_true",
                       help="Only evaluate, don't train")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model:
        config.model.name = args.model
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.device:
        config.system.device = args.device
    if args.seed:
        config.system.seed = args.seed
    
    # Set random seed
    set_seed(config.system.seed)
    
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    
    # Create directories
    os.makedirs(config.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(config.paths.assets_dir, exist_ok=True)
    
    # Prepare data
    print("Preparing data...")
    train_loader, val_loader, test_loader = prepare_data(config)
    
    # Create model
    print(f"Creating {config.model.name} model...")
    model = create_model(config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    # Create trainer
    trainer = TemporalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=OmegaConf.to_container(config),
        save_dir=config.paths.checkpoint_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train model
    if not args.eval_only:
        print("Starting training...")
        history = trainer.train(config.training.num_epochs)
        
        # Plot training history
        trainer.plot_training_history()
    
    # Evaluate model
    print("Evaluating model...")
    test_metrics = trainer.evaluate(config.model.name)
    
    # Print final results
    print("\nFinal Results:")
    print("=" * 50)
    for metric, value in test_metrics.items():
        if metric != "confusion_matrix":
            print(f"{metric}: {value:.4f}")
    
    print(f"\nTraining completed successfully!")
    print(f"Checkpoints saved to: {config.paths.checkpoint_dir}")
    print(f"Assets saved to: {config.paths.assets_dir}")


if __name__ == "__main__":
    main()
