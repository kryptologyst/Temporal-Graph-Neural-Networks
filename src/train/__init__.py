"""Training utilities for temporal graph neural networks."""

import os
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..utils import EarlyStopping, get_device, set_seed
from ..eval import TemporalEvaluator


class TemporalTrainer:
    """Trainer for temporal graph neural networks."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        config: Dict,
        save_dir: str = "checkpoints"
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Training configuration
            save_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.save_dir = save_dir
        
        # Set device
        self.device = get_device()
        self.model = self.model.to(self.device)
        
        # Set random seed
        set_seed(config.get("seed", 42))
        
        # Initialize optimizer and scheduler
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        
        # Initialize loss function
        self.criterion = self._get_criterion()
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get("patience", 10),
            min_delta=config.get("min_delta", 0.001),
            mode=config.get("early_stop_mode", "max")
        )
        
        # Initialize evaluator
        self.evaluator = TemporalEvaluator(
            num_classes=config.get("num_classes", 2),
            task=config.get("task", "node_classification"),
            save_path=os.path.join(save_dir, "evaluations")
        )
        
        # Initialize logging
        self.writer = SummaryWriter(os.path.join(save_dir, "logs"))
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = 0.0
        self.train_losses = []
        self.val_scores = []
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
    def _get_optimizer(self) -> optim.Optimizer:
        """Get optimizer based on config.
        
        Returns:
            Optimizer instance
        """
        optimizer_name = self.config.get("optimizer", "adam").lower()
        lr = self.config.get("learning_rate", 0.001)
        weight_decay = self.config.get("weight_decay", 0.0)
        
        if optimizer_name == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            momentum = self.config.get("momentum", 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _get_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Get learning rate scheduler based on config.
        
        Returns:
            Scheduler instance or None
        """
        scheduler_name = self.config.get("scheduler", "").lower()
        
        if scheduler_name == "step":
            step_size = self.config.get("step_size", 30)
            gamma = self.config.get("gamma", 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == "cosine":
            T_max = self.config.get("T_max", 100)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler_name == "plateau":
            patience = self.config.get("scheduler_patience", 10)
            factor = self.config.get("scheduler_factor", 0.5)
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=patience, factor=factor
            )
        else:
            return None
    
    def _get_criterion(self) -> nn.Module:
        """Get loss function based on config.
        
        Returns:
            Loss function
        """
        loss_name = self.config.get("loss", "cross_entropy").lower()
        
        if loss_name == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif loss_name == "nll":
            return nn.NLLLoss()
        elif loss_name == "mse":
            return nn.MSELoss()
        elif loss_name == "bce":
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
    
    def train_epoch(self) -> float:
        """Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            # Move batch to device
            batch = batch.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if hasattr(batch, 'timestamps'):
                outputs = self.model(
                    batch.x, 
                    batch.edge_index, 
                    batch.timestamps,
                    batch.edge_weight
                )
            else:
                outputs = self.model(batch.x, batch.edge_index, batch.edge_weight)
            
            # Compute loss
            loss = self.criterion(outputs, batch.y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config["grad_clip"]
                )
            
            # Update parameters
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self) -> float:
        """Validate for one epoch.
        
        Returns:
            Validation score
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                if hasattr(batch, 'timestamps'):
                    outputs = self.model(
                        batch.x, 
                        batch.edge_index, 
                        batch.timestamps,
                        batch.edge_weight
                    )
                else:
                    outputs = self.model(batch.x, batch.edge_index, batch.edge_weight)
                
                # Compute loss
                loss = self.criterion(outputs, batch.y)
                total_loss += loss.item()
                
                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += batch.y.size(0)
                correct += (predicted == batch.y).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        self.val_scores.append(accuracy)
        
        return accuracy
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Training history
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_score = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_score)
                else:
                    self.scheduler.step()
            
            # Log metrics
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Score/Val", val_score, epoch)
            self.writer.add_scalar("Learning_Rate", self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Score: {val_score:.4f}")
            
            # Early stopping
            if self.early_stopping(val_score, self.model):
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Save best model
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.save_checkpoint("best_model.pt")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        
        # Close writer
        self.writer.close()
        
        return {
            "train_losses": self.train_losses,
            "val_scores": self.val_scores
        }
    
    def evaluate(self, model_name: str = "model") -> Dict[str, float]:
        """Evaluate the model on test set.
        
        Args:
            model_name: Name for the model
            
        Returns:
            Test metrics
        """
        print("Evaluating model on test set...")
        
        metrics = self.evaluator.evaluate_model(
            self.model, 
            self.test_loader, 
            model_name,
            self.config
        )
        
        print("Test Results:")
        for metric, value in metrics.items():
            if metric != "confusion_matrix":
                print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_score": self.best_val_score,
            "config": self.config,
            "train_losses": self.train_losses,
            "val_scores": self.val_scores
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        filepath = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_score = checkpoint["best_val_score"]
        self.train_losses = checkpoint["train_losses"]
        self.val_scores = checkpoint["val_scores"]
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print(f"Loaded checkpoint from {filepath}")
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot training history.
        
        Args:
            save_path: Optional save path for plot
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot training loss
        ax1.plot(self.train_losses)
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True)
        
        # Plot validation score
        ax2.plot(self.val_scores)
        ax2.set_title("Validation Score")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score")
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, "training_history.png"), dpi=300, bbox_inches='tight')
        
        plt.close()
