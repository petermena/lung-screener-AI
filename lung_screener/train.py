"""Training pipeline for lung nodule detection model.

Handles the full training loop with:
- Mixed precision training
- Learning rate scheduling with warmup
- Early stopping
- Checkpoint saving
- Metrics logging and dashboard generation
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import LUNA16Dataset
from .metrics_dashboard import MetricsLogger, save_dashboard
from .model import build_model

logger = logging.getLogger(__name__)


class Trainer:
    """Training orchestrator for nodule detection models."""

    def __init__(self, config: dict, checkpoint_dir: str | Path = "./checkpoints"):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Build model
        self.model = build_model(config).to(self.device)
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {param_count:,}")

        # Training config
        train_config = config.get("training", {})
        self.epochs = train_config.get("epochs", 100)
        self.batch_size = train_config.get("batch_size", 32)
        self.lr = train_config.get("learning_rate", 0.001)
        self.weight_decay = train_config.get("weight_decay", 0.0001)
        self.patience = train_config.get("early_stopping_patience", 15)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Loss function with class weights (nodules are rare)
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 5.0]).to(self.device)
        )

        # Mixed precision
        self.scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

        # Scheduler
        sched_config = train_config.get("scheduler", {})
        warmup_epochs = sched_config.get("warmup_epochs", 5)
        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.epochs - warmup_epochs
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        # Metrics logger for dashboard
        self.metrics_logger = MetricsLogger(self.checkpoint_dir)

        # Tracking
        self.best_val_auc = 0.0
        self.epochs_without_improvement = 0

    def create_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders."""
        train_config = self.config.get("training", {})
        data_config = self.config.get("data", {})
        dataset_dir = data_config.get("dataset_dir", "./data/luna16")
        cache_dir = data_config.get("cache_dir", "./data/cache")
        val_split = data_config.get("val_split", 0.2)

        train_dataset = LUNA16Dataset(
            dataset_dir=dataset_dir,
            config=self.config,
            split="train",
            val_split=val_split,
            augment=True,
            cache_dir=cache_dir,
        )

        val_dataset = LUNA16Dataset(
            dataset_dir=dataset_dir,
            config=self.config,
            split="val",
            val_split=val_split,
            augment=False,
            cache_dir=cache_dir,
        )

        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")

        num_workers = train_config.get("num_workers", 4)
        pin_memory = torch.cuda.is_available()

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=num_workers > 0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )

        return train_loader, val_loader

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []

        pbar = tqdm(loader, desc="Training", leave=False)
        for batch in pbar:
            patches = batch["patch"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            with autocast("cuda", enabled=torch.cuda.is_available()):
                output = self.model(patches)
                loss = self.criterion(output["logits"], labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * patches.size(0)
            probs = torch.softmax(output["logits"], dim=1)[:, 1]
            preds = (probs > 0.5).long()
            correct += (preds == labels).sum().item()
            total += patches.size(0)

            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.4f}"})

        metrics = {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }

        # Compute AUC
        metrics["auc"] = self._compute_auc(all_labels, all_probs)

        return metrics

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []

        # Per-class tracking
        tp = fp = fn = tn = 0

        for batch in tqdm(loader, desc="Validation", leave=False):
            patches = batch["patch"].to(self.device)
            labels = batch["label"].to(self.device)

            with autocast("cuda", enabled=torch.cuda.is_available()):
                output = self.model(patches)
                loss = self.criterion(output["logits"], labels)

            total_loss += loss.item() * patches.size(0)
            probs = torch.softmax(output["logits"], dim=1)[:, 1]
            preds = (probs > 0.5).long()
            correct += (preds == labels).sum().item()
            total += patches.size(0)

            # Confusion matrix components
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics = {
            "loss": total_loss / total,
            "accuracy": correct / total,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "sensitivity": recall,  # Same as recall; critical for screening
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "auc": self._compute_auc(all_labels, all_probs),
        }

        return metrics

    def _compute_auc(self, labels: list, probs: list) -> float:
        """Compute AUC-ROC from labels and predicted probabilities."""
        labels = np.array(labels)
        probs = np.array(probs)

        if len(np.unique(labels)) < 2:
            return 0.0

        # Simple trapezoidal AUC computation
        sorted_indices = np.argsort(-probs)
        sorted_labels = labels[sorted_indices]

        num_pos = np.sum(labels == 1)
        num_neg = np.sum(labels == 0)

        if num_pos == 0 or num_neg == 0:
            return 0.0

        tp_rate = np.cumsum(sorted_labels) / num_pos
        fp_rate = np.cumsum(1 - sorted_labels) / num_neg

        # Prepend origin
        tp_rate = np.concatenate([[0], tp_rate])
        fp_rate = np.concatenate([[0], fp_rate])

        auc = np.trapezoid(tp_rate, fp_rate)
        return float(auc)

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }

        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / "latest.pth")

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
            logger.info(f"  Saved new best model (AUC: {metrics['auc']:.4f})")

    def load_checkpoint(self, path: str | Path) -> int:
        """Load a checkpoint and return the epoch number."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_auc = checkpoint.get("metrics", {}).get("auc", 0.0)
        return checkpoint.get("epoch", 0)

    def train(self, resume_from: str | Path | None = None):
        """Run the full training loop.

        Args:
            resume_from: Optional path to checkpoint to resume from.
        """
        train_loader, val_loader = self.create_dataloaders()
        start_epoch = 0

        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
            logger.info(f"Resumed from epoch {start_epoch}")

        logger.info(f"Starting training for {self.epochs} epochs")

        for epoch in range(start_epoch, self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader)
            logger.info(
                f"  Train - Loss: {train_metrics['loss']:.4f}, "
                f"Acc: {train_metrics['accuracy']:.4f}, "
                f"AUC: {train_metrics['auc']:.4f}"
            )

            # Validate
            val_metrics = self.validate(val_loader)
            logger.info(
                f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                f"Acc: {val_metrics['accuracy']:.4f}, "
                f"AUC: {val_metrics['auc']:.4f}, "
                f"Sens: {val_metrics['sensitivity']:.4f}, "
                f"Spec: {val_metrics['specificity']:.4f}"
            )

            # Step scheduler
            self.scheduler.step()

            # Log metrics to dashboard
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.metrics_logger.log_epoch(epoch, train_metrics, val_metrics, current_lr)

            # Check for improvement
            is_best = val_metrics["auc"] > self.best_val_auc
            if is_best:
                self.best_val_auc = val_metrics["auc"]
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(
                    f"Early stopping after {self.patience} epochs without improvement"
                )
                break

        # Generate final dashboard
        dashboard_path = save_dashboard(
            self.metrics_logger.metrics_file,
            self.checkpoint_dir / "dashboard.html",
        )
        logger.info(f"Training dashboard: {dashboard_path}")
        logger.info(f"Training complete. Best validation AUC: {self.best_val_auc:.4f}")
