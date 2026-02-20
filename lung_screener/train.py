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

from .dataset import CombinedLungDataset, LUNA16Dataset
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

        # Loss function — no class weights needed since dataset balancing
        # (pos_neg_ratio) already handles the imbalance. Combining both
        # caused exploding gradients and model collapse.
        self.criterion = nn.CrossEntropyLoss()

        # Mixed precision
        self.scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

        # Scheduler
        sched_config = train_config.get("scheduler", {})
        warmup_epochs = sched_config.get("warmup_epochs", 5)
        warmup_epochs = min(warmup_epochs, max(self.epochs - 1, 0))
        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, total_iters=max(warmup_epochs, 1)
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=max(self.epochs - warmup_epochs, 1)
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
        """Create training and validation data loaders.

        When ``data.datasets`` is configured the trainer combines all
        listed datasets (e.g. LUNA16 + LUNA25) via
        :class:`CombinedLungDataset`.  Otherwise falls back to a single
        LUNA16 dataset for backwards compatibility.
        """
        train_config = self.config.get("training", {})

        train_dataset = CombinedLungDataset.from_config(
            self.config, split="train", augment=True,
        )
        val_dataset = CombinedLungDataset.from_config(
            self.config, split="val", augment=False,
        )

        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")

        # Pre-populate disk cache so DataLoader workers find fast .npy
        # files instead of having to load and resample raw .mhd volumes
        # (~30-60s each).  This is a one-time cost; subsequent runs are
        # instant.
        train_dataset.warm_disk_cache()
        val_dataset.warm_disk_cache()

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
            prefetch_factor=3 if num_workers > 0 else None,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=3 if num_workers > 0 else None,
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
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
        loss_count = 0
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

            batch_loss = loss.item()
            if np.isfinite(batch_loss):
                total_loss += batch_loss * patches.size(0)
                loss_count += patches.size(0)
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
            "loss": total_loss / loss_count if loss_count > 0 else float("nan"),
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

    def save_checkpoint(
        self,
        epoch: int,
        metrics: dict,
        is_best: bool = False,
        phase: str = "complete",
    ):
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
            metrics: Metrics dict (train or val depending on phase).
            is_best: Whether this is the best model so far.
            phase: One of "train_done" (training phase finished, validation pending)
                   or "complete" (both phases finished).
        """
        checkpoint = {
            "epoch": epoch,
            "phase": phase,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "best_val_auc": self.best_val_auc,
            "epochs_without_improvement": self.epochs_without_improvement,
        }

        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / "latest.pth")

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
            logger.info(f"  Saved new best model (AUC: {metrics['auc']:.4f})")

    def load_checkpoint(self, path: str | Path) -> tuple[int, str]:
        """Load a checkpoint and return (epoch number, phase).

        Returns:
            Tuple of (epoch, phase) where phase is "train_done" if the
            training phase completed but validation hasn't run yet, or
            "complete" if the full epoch finished.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_auc = checkpoint.get(
            "best_val_auc",
            checkpoint.get("metrics", {}).get("auc", 0.0),
        )
        self.epochs_without_improvement = checkpoint.get(
            "epochs_without_improvement", 0
        )
        phase = checkpoint.get("phase", "complete")
        return checkpoint.get("epoch", 0), phase

    def train(self, resume_from: str | Path | None = None):
        """Run the full training loop.

        Args:
            resume_from: Optional path to checkpoint to resume from.
        """
        train_loader, val_loader = self.create_dataloaders()
        start_epoch = 0
        skip_training_phase = False

        resumed_train_metrics = None

        if resume_from:
            start_epoch, phase = self.load_checkpoint(resume_from)
            if phase == "train_done":
                # Training finished but validation didn't run — resume at validation
                skip_training_phase = True
                # Load train metrics saved in the intermediate checkpoint
                ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
                resumed_train_metrics = ckpt.get("metrics", {})
                logger.info(
                    f"Resumed epoch {start_epoch + 1} after training phase "
                    f"(skipping to validation)"
                )
            else:
                # Full epoch was complete, move to next epoch
                start_epoch += 1
                logger.info(f"Resumed from epoch {start_epoch + 1}")

        logger.info(f"Starting training for {self.epochs} epochs")

        for epoch in range(start_epoch, self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")

            # Train (skip if resuming mid-epoch after training was already done)
            if skip_training_phase:
                train_metrics = resumed_train_metrics
                logger.info("  Train - skipped (already completed before interruption)")
                skip_training_phase = False
            else:
                train_metrics = self.train_epoch(train_loader)
                logger.info(
                    f"  Train - Loss: {train_metrics['loss']:.4f}, "
                    f"Acc: {train_metrics['accuracy']:.4f}, "
                    f"AUC: {train_metrics['auc']:.4f}"
                )

                # Save intermediate checkpoint so validation can be resumed
                self.save_checkpoint(
                    epoch, train_metrics, is_best=False, phase="train_done"
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

            # Save checkpoint (full epoch complete)
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
