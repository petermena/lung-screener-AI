"""Incremental retraining from radiologist feedback.

Automates the feedback-to-model-improvement loop:
1. Export confirmed/rejected feedback as training annotations
2. Merge with existing training data (LUNA16 or prior prepared data)
3. Fine-tune the current best model on the combined dataset
4. Validate that the new model doesn't regress on the held-out set
5. Promote the new model only if it improves or maintains performance

All operations run fully offline — no cloud dependencies.
"""

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .dataset import LUNA16Dataset
from .feedback import FeedbackStore
from .metrics_dashboard import MetricsLogger, save_dashboard
from .model import build_model

logger = logging.getLogger(__name__)


@dataclass
class RetrainResult:
    """Result of an incremental retrain cycle."""

    success: bool
    promoted: bool  # True if new model replaced the old best
    reason: str
    feedback_records_used: int
    epochs_trained: int
    baseline_auc: float
    new_auc: float
    baseline_sensitivity: float
    new_sensitivity: float
    checkpoint_path: str = ""
    retrain_id: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "promoted": self.promoted,
            "reason": self.reason,
            "feedback_records_used": self.feedback_records_used,
            "epochs_trained": self.epochs_trained,
            "baseline_auc": round(self.baseline_auc, 4),
            "new_auc": round(self.new_auc, 4),
            "baseline_sensitivity": round(self.baseline_sensitivity, 4),
            "new_sensitivity": round(self.new_sensitivity, 4),
            "checkpoint_path": self.checkpoint_path,
            "retrain_id": self.retrain_id,
        }

    def summary(self) -> str:
        lines = [f"Retrain Result ({self.retrain_id})"]
        lines.append(f"  Status: {'SUCCESS' if self.success else 'FAILED'} — {self.reason}")
        lines.append(f"  Feedback records used: {self.feedback_records_used}")
        lines.append(f"  Epochs trained: {self.epochs_trained}")
        lines.append(f"  AUC:         {self.baseline_auc:.4f} → {self.new_auc:.4f}")
        lines.append(f"  Sensitivity: {self.baseline_sensitivity:.4f} → {self.new_sensitivity:.4f}")
        if self.promoted:
            lines.append(f"  New best model promoted to: {self.checkpoint_path}")
        else:
            lines.append("  Model NOT promoted (baseline was better or equal)")
        return "\n".join(lines)


class IncrementalRetrainer:
    """Orchestrates feedback-driven model fine-tuning."""

    def __init__(
        self,
        config: dict,
        base_checkpoint: str | Path,
        feedback_dir: str | Path,
        checkpoint_dir: str | Path = "./checkpoints",
    ):
        self.config = config
        self.base_checkpoint = Path(base_checkpoint)
        self.feedback_dir = Path(feedback_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        retrain_config = config.get("retrain", {})
        self.min_feedback = retrain_config.get("min_feedback_records", 20)
        self.epochs = retrain_config.get("epochs", 30)
        self.lr = retrain_config.get("learning_rate", 0.0001)
        self.batch_size = retrain_config.get("batch_size", 32)
        self.patience = retrain_config.get("early_stopping_patience", 10)
        self.regression_tolerance = retrain_config.get("regression_tolerance", 0.01)
        self.weight_decay = retrain_config.get("weight_decay", 0.0001)

        self.retrain_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history_dir = self.checkpoint_dir / "retrain_history"
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.feedback_store = FeedbackStore(self.feedback_dir)

    def retrain(self) -> RetrainResult:
        """Run a full incremental retrain cycle.

        Returns:
            RetrainResult describing the outcome.
        """
        # Step 1: Check feedback availability
        records = self.feedback_store.load_all()
        used_records = self._filter_unused_records(records)

        if len(used_records) < self.min_feedback:
            return RetrainResult(
                success=False,
                promoted=False,
                reason=(
                    f"Insufficient feedback: {len(used_records)} records "
                    f"(minimum {self.min_feedback} required)"
                ),
                feedback_records_used=len(used_records),
                epochs_trained=0,
                baseline_auc=0.0,
                new_auc=0.0,
                baseline_sensitivity=0.0,
                new_sensitivity=0.0,
                retrain_id=self.retrain_id,
            )

        logger.info(
            f"Starting retrain {self.retrain_id} with "
            f"{len(used_records)} feedback records"
        )

        # Step 2: Export feedback as training data
        feedback_data_dir = self.history_dir / self.retrain_id / "feedback_data"
        export_result = self.feedback_store.export_for_training(feedback_data_dir)
        logger.info(
            f"Exported {export_result['confirmed_count']} confirmed, "
            f"{export_result['rejected_count']} rejected"
        )

        # Step 3: Load base model and evaluate baseline
        model = build_model(self.config).to(self.device)
        checkpoint = torch.load(
            self.base_checkpoint, map_location=self.device, weights_only=False
        )
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        val_loader = self._create_val_loader()
        baseline_metrics = self._evaluate(model, val_loader)
        logger.info(
            f"Baseline — AUC: {baseline_metrics['auc']:.4f}, "
            f"Sensitivity: {baseline_metrics['sensitivity']:.4f}"
        )

        # Step 4: Merge feedback into training data and create loader
        train_loader = self._create_merged_train_loader(feedback_data_dir)

        # Step 5: Fine-tune
        model, train_epochs, best_val_metrics = self._fine_tune(
            model, train_loader, val_loader
        )

        # Step 6: Decide whether to promote
        new_auc = best_val_metrics.get("auc", 0.0)
        new_sensitivity = best_val_metrics.get("sensitivity", 0.0)

        promoted = False
        reason = ""

        # Guard against sensitivity regression (critical for screening)
        if new_sensitivity < baseline_metrics["sensitivity"] - self.regression_tolerance:
            reason = (
                f"Rejected: sensitivity regressed "
                f"({baseline_metrics['sensitivity']:.4f} → {new_sensitivity:.4f})"
            )
        elif new_auc < baseline_metrics["auc"] - self.regression_tolerance:
            reason = (
                f"Rejected: AUC regressed "
                f"({baseline_metrics['auc']:.4f} → {new_auc:.4f})"
            )
        else:
            promoted = True
            reason = "New model meets or exceeds baseline performance"

        # Step 7: Save or promote
        retrain_ckpt_path = (
            self.history_dir / self.retrain_id / "retrained.pth"
        )
        retrain_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "retrain_id": self.retrain_id,
                "feedback_records_used": len(used_records),
                "metrics": best_val_metrics,
                "config": self.config,
            },
            retrain_ckpt_path,
        )

        if promoted:
            best_path = self.checkpoint_dir / "best.pth"
            # Back up current best
            if best_path.exists():
                backup = self.checkpoint_dir / f"best_before_{self.retrain_id}.pth"
                shutil.copy2(best_path, backup)
                logger.info(f"Backed up previous best to {backup}")

            shutil.copy2(retrain_ckpt_path, best_path)
            logger.info(f"Promoted retrained model to {best_path}")

        # Step 8: Record history
        self._mark_records_used(used_records)

        result = RetrainResult(
            success=True,
            promoted=promoted,
            reason=reason,
            feedback_records_used=len(used_records),
            epochs_trained=train_epochs,
            baseline_auc=baseline_metrics["auc"],
            new_auc=new_auc,
            baseline_sensitivity=baseline_metrics["sensitivity"],
            new_sensitivity=new_sensitivity,
            checkpoint_path=str(retrain_ckpt_path if not promoted else best_path),
            retrain_id=self.retrain_id,
        )

        # Save result to history
        result_path = self.history_dir / self.retrain_id / "result.json"
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(result.summary())
        return result

    def _filter_unused_records(self, records: list) -> list:
        """Filter out feedback records that were already used in prior retrains."""
        used_file = self.history_dir / "used_feedback_ids.json"
        if used_file.exists():
            with open(used_file) as f:
                used_ids = set(json.load(f))
        else:
            used_ids = set()

        return [r for r in records if r.finding_id not in used_ids]

    def _mark_records_used(self, records: list):
        """Mark feedback records as consumed by this retrain cycle."""
        used_file = self.history_dir / "used_feedback_ids.json"
        if used_file.exists():
            with open(used_file) as f:
                used_ids = json.load(f)
        else:
            used_ids = []

        used_ids.extend(r.finding_id for r in records)

        with open(used_file, "w") as f:
            json.dump(used_ids, f)

    def _create_val_loader(self) -> DataLoader:
        """Create the validation data loader (unchanged from original data)."""
        data_config = self.config.get("data", {})
        dataset_dir = data_config.get("dataset_dir", "./data/luna16")
        cache_dir = data_config.get("cache_dir", "./data/cache")
        val_split = data_config.get("val_split", 0.2)

        val_dataset = LUNA16Dataset(
            dataset_dir=dataset_dir,
            config=self.config,
            split="val",
            val_split=val_split,
            augment=False,
            cache_dir=cache_dir,
        )

        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def _create_merged_train_loader(
        self, feedback_data_dir: Path
    ) -> DataLoader:
        """Create a training loader that combines original + feedback data.

        Merges the feedback annotations into the original dataset directory
        by copying the exported CSVs so the LUNA16Dataset can load them.
        """
        data_config = self.config.get("data", {})
        dataset_dir = data_config.get("dataset_dir", "./data/luna16")
        cache_dir = data_config.get("cache_dir", "./data/cache")
        val_split = data_config.get("val_split", 0.2)

        # Create a merged dataset directory for this retrain
        merged_dir = self.history_dir / self.retrain_id / "merged_data"
        merged_dir.mkdir(parents=True, exist_ok=True)

        # Copy original annotations
        orig_annotations = Path(dataset_dir) / "annotations.csv"
        orig_candidates = Path(dataset_dir) / "candidates_V2.csv"

        merged_annotations = merged_dir / "annotations.csv"
        merged_candidates = merged_dir / "candidates_V2.csv"

        # Start with originals
        if orig_annotations.exists():
            shutil.copy2(orig_annotations, merged_annotations)
        if orig_candidates.exists():
            shutil.copy2(orig_candidates, merged_candidates)

        # Append feedback positives to annotations
        feedback_annotations = feedback_data_dir / "feedback_annotations.csv"
        if feedback_annotations.exists():
            with open(feedback_annotations) as f:
                lines = f.readlines()
            if len(lines) > 1:  # Has data beyond header
                with open(merged_annotations, "a") as f:
                    for line in lines[1:]:  # Skip header
                        f.write(line)
                logger.info(f"Appended {len(lines) - 1} feedback annotations")

        # Append feedback negatives to candidates
        feedback_negatives = feedback_data_dir / "feedback_negatives.csv"
        if feedback_negatives.exists():
            with open(feedback_negatives) as f:
                lines = f.readlines()
            if len(lines) > 1:
                with open(merged_candidates, "a") as f:
                    for line in lines[1:]:
                        # Convert from "seriesuid,x,y,z,class" format
                        f.write(line)
                logger.info(f"Appended {len(lines) - 1} feedback negatives")

        # Symlink subset directories so volume files are findable
        orig_dataset = Path(dataset_dir)
        for subset in orig_dataset.glob("subset*"):
            link = merged_dir / subset.name
            if not link.exists():
                link.symlink_to(subset.resolve())

        train_dataset = LUNA16Dataset(
            dataset_dir=merged_dir,
            config=self.config,
            split="train",
            val_split=val_split,
            augment=True,
            cache_dir=cache_dir,
        )

        logger.info(f"Merged training set: {len(train_dataset)} samples")

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

    def _fine_tune(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> tuple[nn.Module, int, dict]:
        """Fine-tune the model with a lower learning rate.

        Returns:
            Tuple of (best model, epochs trained, best validation metrics).
        """
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 5.0]).to(self.device)
        )
        scaler = GradScaler()

        metrics_logger = MetricsLogger(
            self.history_dir / self.retrain_id
        )

        best_auc = 0.0
        best_state = None
        best_metrics = {}
        epochs_without_improvement = 0
        epochs_trained = 0

        for epoch in range(self.epochs):
            # Train
            model.train()
            total_loss = 0.0
            total = 0

            for batch in train_loader:
                patches = batch["patch"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                with autocast():
                    output = model(patches)
                    loss = criterion(output["logits"], labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item() * patches.size(0)
                total += patches.size(0)

            train_loss = total_loss / max(total, 1)
            scheduler.step()
            epochs_trained = epoch + 1

            # Validate
            val_metrics = self._evaluate(model, val_loader)
            current_lr = optimizer.param_groups[0]["lr"]

            metrics_logger.log_epoch(
                epoch,
                {"loss": train_loss},
                val_metrics,
                current_lr,
            )

            logger.info(
                f"  Retrain epoch {epoch + 1}/{self.epochs} — "
                f"loss: {train_loss:.4f}, val_auc: {val_metrics['auc']:.4f}, "
                f"val_sens: {val_metrics['sensitivity']:.4f}"
            )

            # Track best
            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_metrics = val_metrics.copy()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                logger.info(f"  Early stopping after {self.patience} epochs without improvement")
                break

        # Restore best state
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(self.device)

        # Save dashboard
        save_dashboard(
            metrics_logger.metrics_file,
            self.history_dir / self.retrain_id / "dashboard.html",
        )

        return model, epochs_trained, best_metrics

    @torch.no_grad()
    def _evaluate(self, model: nn.Module, loader: DataLoader) -> dict:
        """Evaluate model on a data loader."""
        model.eval()
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 5.0]).to(self.device)
        )

        total_loss = 0.0
        total = 0
        tp = fp = fn = tn = 0
        all_probs = []
        all_labels = []

        for batch in loader:
            patches = batch["patch"].to(self.device)
            labels = batch["label"].to(self.device)

            with autocast():
                output = model(patches)
                loss = criterion(output["logits"], labels)

            total_loss += loss.item() * patches.size(0)
            total += patches.size(0)

            probs = torch.softmax(output["logits"], dim=1)[:, 1]
            preds = (probs > 0.5).long()

            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "loss": total_loss / max(total, 1),
            "accuracy": (tp + tn) / max(total, 1),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "sensitivity": recall,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "auc": self._compute_auc(all_labels, all_probs),
        }

    def _compute_auc(self, labels: list, probs: list) -> float:
        """Compute AUC-ROC."""
        labels = np.array(labels)
        probs = np.array(probs)

        if len(np.unique(labels)) < 2:
            return 0.0

        sorted_indices = np.argsort(-probs)
        sorted_labels = labels[sorted_indices]

        num_pos = np.sum(labels == 1)
        num_neg = np.sum(labels == 0)

        if num_pos == 0 or num_neg == 0:
            return 0.0

        tp_rate = np.cumsum(sorted_labels) / num_pos
        fp_rate = np.cumsum(1 - sorted_labels) / num_neg

        tp_rate = np.concatenate([[0], tp_rate])
        fp_rate = np.concatenate([[0], fp_rate])

        return float(np.trapezoid(tp_rate, fp_rate))

    def get_retrain_history(self) -> list[dict]:
        """Load the history of all past retrain cycles."""
        history = []
        for result_file in sorted(self.history_dir.glob("*/result.json")):
            with open(result_file) as f:
                history.append(json.load(f))
        return history
