"""Evaluation metrics and model leaderboard for temporal graph tasks."""

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from torchmetrics import (
    Accuracy, Precision, Recall, F1Score, AUROC, AveragePrecision,
    ConfusionMatrix
)

from ..utils import get_device


class TemporalMetrics:
    """Comprehensive metrics for temporal graph tasks."""
    
    def __init__(self, num_classes: int = 2, task: str = "node_classification"):
        """Initialize temporal metrics.
        
        Args:
            num_classes: Number of classes
            task: Task type ('node_classification', 'link_prediction', 'graph_classification')
        """
        self.num_classes = num_classes
        self.task = task
        self.device = get_device()
        
        # Initialize torchmetrics
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        self.precision = Precision(task="multiclass", num_classes=num_classes, average="macro").to(self.device)
        self.recall = Recall(task="multiclass", num_classes=num_classes, average="macro").to(self.device)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(self.device)
        self.auroc = AUROC(task="multiclass", num_classes=num_classes).to(self.device)
        self.avg_precision = AveragePrecision(task="multiclass", num_classes=num_classes).to(self.device)
        self.confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(self.device)
        
        # Store predictions and targets for temporal analysis
        self.predictions = []
        self.targets = []
        self.timestamps = []
    
    def update(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None
    ) -> None:
        """Update metrics with new predictions.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            timestamps: Optional timestamps
        """
        predictions = predictions.to(self.device)
        targets = targets.to(self.device)
        
        # Store for temporal analysis
        self.predictions.append(predictions.cpu())
        self.targets.append(targets.cpu())
        if timestamps is not None:
            self.timestamps.append(timestamps.cpu())
        
        # Update torchmetrics
        self.accuracy.update(predictions, targets)
        self.precision.update(predictions, targets)
        self.recall.update(predictions, targets)
        self.f1.update(predictions, targets)
        self.auroc.update(predictions, targets)
        self.avg_precision.update(predictions, targets)
        self.confusion_matrix.update(predictions, targets)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics.
        
        Returns:
            Dict of metric names and values
        """
        metrics = {
            "accuracy": self.accuracy.compute().item(),
            "precision": self.precision.compute().item(),
            "recall": self.recall.compute().item(),
            "f1": self.f1.compute().item(),
            "auroc": self.auroc.compute().item(),
            "avg_precision": self.avg_precision.compute().item(),
        }
        
        # Add confusion matrix
        cm = self.confusion_matrix.compute()
        metrics["confusion_matrix"] = cm.cpu().numpy().tolist()
        
        return metrics
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.auroc.reset()
        self.avg_precision.reset()
        self.confusion_matrix.reset()
        
        self.predictions = []
        self.targets = []
        self.timestamps = []
    
    def compute_temporal_metrics(self) -> Dict[str, float]:
        """Compute temporal-specific metrics.
        
        Returns:
            Dict of temporal metrics
        """
        if not self.predictions or not self.timestamps:
            return {}
        
        # Concatenate all predictions and timestamps
        all_predictions = torch.cat(self.predictions)
        all_targets = torch.cat(self.targets)
        all_timestamps = torch.cat(self.timestamps)
        
        # Temporal stability (consistency over time)
        temporal_stability = self._compute_temporal_stability(all_predictions, all_timestamps)
        
        # Temporal accuracy (accuracy over time)
        temporal_accuracy = self._compute_temporal_accuracy(all_predictions, all_targets, all_timestamps)
        
        return {
            "temporal_stability": temporal_stability,
            "temporal_accuracy": temporal_accuracy
        }
    
    def _compute_temporal_stability(
        self, 
        predictions: torch.Tensor, 
        timestamps: torch.Tensor
    ) -> float:
        """Compute temporal stability metric.
        
        Args:
            predictions: Model predictions
            timestamps: Timestamps
            
        Returns:
            Temporal stability score
        """
        # Group predictions by timestamp
        unique_timestamps = torch.unique(timestamps)
        stability_scores = []
        
        for t in unique_timestamps:
            mask = timestamps == t
            pred_at_t = predictions[mask]
            
            if len(pred_at_t) > 1:
                # Compute variance of predictions at this timestamp
                variance = torch.var(pred_at_t.float())
                stability_scores.append(variance.item())
        
        return np.mean(stability_scores) if stability_scores else 0.0
    
    def _compute_temporal_accuracy(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        timestamps: torch.Tensor
    ) -> float:
        """Compute temporal accuracy metric.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            timestamps: Timestamps
            
        Returns:
            Temporal accuracy score
        """
        # Group by timestamp and compute accuracy at each time
        unique_timestamps = torch.unique(timestamps)
        accuracies = []
        
        for t in unique_timestamps:
            mask = timestamps == t
            pred_at_t = predictions[mask]
            target_at_t = targets[mask]
            
            if len(pred_at_t) > 0:
                accuracy = (pred_at_t == target_at_t).float().mean().item()
                accuracies.append(accuracy)
        
        return np.mean(accuracies) if accuracies else 0.0


class ModelLeaderboard:
    """Leaderboard for comparing temporal GNN models."""
    
    def __init__(self, save_path: str = "assets/leaderboard.json"):
        """Initialize leaderboard.
        
        Args:
            save_path: Path to save leaderboard
        """
        self.save_path = save_path
        self.results = []
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Load existing results
        self.load()
    
    def add_result(
        self,
        model_name: str,
        metrics: Dict[str, float],
        config: Optional[Dict] = None,
        timestamp: Optional[str] = None
    ) -> None:
        """Add model result to leaderboard.
        
        Args:
            model_name: Name of the model
            metrics: Evaluation metrics
            config: Model configuration
            timestamp: Timestamp of evaluation
        """
        import datetime
        
        result = {
            "model_name": model_name,
            "metrics": metrics,
            "config": config or {},
            "timestamp": timestamp or datetime.datetime.now().isoformat(),
            "score": self._compute_score(metrics)
        }
        
        self.results.append(result)
        self.save()
    
    def _compute_score(self, metrics: Dict[str, float]) -> float:
        """Compute overall score from metrics.
        
        Args:
            metrics: Evaluation metrics
            
        Returns:
            Overall score
        """
        # Weighted combination of key metrics
        weights = {
            "accuracy": 0.3,
            "f1": 0.3,
            "auroc": 0.2,
            "avg_precision": 0.2
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                score += weight * metrics[metric]
        
        return score
    
    def get_leaderboard(self, top_k: int = 10) -> pd.DataFrame:
        """Get leaderboard as DataFrame.
        
        Args:
            top_k: Number of top results to return
            
        Returns:
            DataFrame with leaderboard results
        """
        if not self.results:
            return pd.DataFrame()
        
        # Sort by score
        sorted_results = sorted(self.results, key=lambda x: x["score"], reverse=True)
        
        # Convert to DataFrame
        df_data = []
        for i, result in enumerate(sorted_results[:top_k]):
            row = {
                "rank": i + 1,
                "model_name": result["model_name"],
                "score": result["score"],
                "timestamp": result["timestamp"]
            }
            
            # Add metrics
            for metric, value in result["metrics"].items():
                if metric != "confusion_matrix":
                    row[metric] = value
            
            df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    def save(self) -> None:
        """Save leaderboard to file."""
        with open(self.save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def load(self) -> None:
        """Load leaderboard from file."""
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                self.results = json.load(f)
        else:
            self.results = []
    
    def clear(self) -> None:
        """Clear all results."""
        self.results = []
        self.save()


class TemporalEvaluator:
    """Comprehensive evaluator for temporal graph models."""
    
    def __init__(
        self,
        num_classes: int = 2,
        task: str = "node_classification",
        save_path: str = "assets/evaluations"
    ):
        """Initialize evaluator.
        
        Args:
            num_classes: Number of classes
            task: Task type
            save_path: Path to save evaluation results
        """
        self.num_classes = num_classes
        self.task = task
        self.save_path = save_path
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Initialize metrics and leaderboard
        self.metrics = TemporalMetrics(num_classes, task)
        self.leaderboard = ModelLeaderboard(os.path.join(save_path, "leaderboard.json"))
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        data_loader,
        model_name: str,
        config: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Evaluate a model on the given data.
        
        Args:
            model: Model to evaluate
            data_loader: Data loader
            model_name: Name of the model
            config: Model configuration
            
        Returns:
            Dict of evaluation metrics
        """
        model.eval()
        self.metrics.reset()
        
        with torch.no_grad():
            for batch in data_loader:
                # Get predictions
                if hasattr(batch, 'timestamps'):
                    predictions = model(
                        batch.x, 
                        batch.edge_index, 
                        batch.timestamps,
                        batch.edge_weight
                    )
                else:
                    predictions = model(batch.x, batch.edge_index, batch.edge_weight)
                
                # Update metrics
                self.metrics.update(
                    predictions, 
                    batch.y,
                    getattr(batch, 'timestamps', None)
                )
        
        # Compute metrics
        metrics = self.metrics.compute()
        temporal_metrics = self.metrics.compute_temporal_metrics()
        metrics.update(temporal_metrics)
        
        # Add to leaderboard
        self.leaderboard.add_result(model_name, metrics, config)
        
        return metrics
    
    def compare_models(
        self,
        models: Dict[str, torch.nn.Module],
        data_loader,
        configs: Optional[Dict[str, Dict]] = None
    ) -> pd.DataFrame:
        """Compare multiple models.
        
        Args:
            models: Dict of model names and models
            data_loader: Data loader
            configs: Optional model configurations
            
        Returns:
            DataFrame with comparison results
        """
        for model_name, model in models.items():
            config = configs.get(model_name) if configs else None
            self.evaluate_model(model, data_loader, model_name, config)
        
        return self.leaderboard.get_leaderboard()
    
    def save_evaluation_report(
        self,
        metrics: Dict[str, float],
        model_name: str,
        filename: Optional[str] = None
    ) -> None:
        """Save detailed evaluation report.
        
        Args:
            metrics: Evaluation metrics
            model_name: Name of the model
            filename: Optional filename
        """
        if filename is None:
            filename = f"{model_name}_evaluation.json"
        
        report = {
            "model_name": model_name,
            "metrics": metrics,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        filepath = os.path.join(self.save_path, filename)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    def plot_metrics_comparison(
        self,
        metrics_list: List[Dict[str, float]],
        model_names: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """Plot metrics comparison.
        
        Args:
            metrics_list: List of metrics dictionaries
            model_names: List of model names
            save_path: Optional save path for plot
        """
        import matplotlib.pyplot as plt
        
        # Extract key metrics
        key_metrics = ["accuracy", "f1", "auroc", "avg_precision"]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(key_metrics):
            values = [m.get(metric, 0) for m in metrics_list]
            
            axes[i].bar(model_names, values)
            axes[i].set_title(f"{metric.upper()}")
            axes[i].set_ylabel("Score")
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_path, "metrics_comparison.png"), dpi=300, bbox_inches='tight')
        
        plt.close()
