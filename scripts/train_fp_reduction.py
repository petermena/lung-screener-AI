#!/usr/bin/env python3
"""Train a second-stage false-positive reduction model.

End-to-end pipeline that:
1. Loads the first-stage best.pth and scores all training candidates
2. Collects false positives above threshold as hard negatives
3. Trains FPReductionNet with early stopping
4. Saves the checkpoint and evaluates the combined two-stage pipeline

Designed to run on AWS (g4dn.xlarge / p3.2xlarge) after first-stage
training is complete.

Usage:
    python scripts/train_fp_reduction.py
    python scripts/train_fp_reduction.py --config config/fp_reduction.yaml
    python scripts/train_fp_reduction.py --first-stage-ckpt checkpoints/best.pth
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# NumPy 2.0 renamed np.trapz -> np.trapezoid
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lung_screener.dataset import CombinedLungDataset
from lung_screener.fp_reduction import FPReductionNet
from lung_screener.model import build_model

logger = logging.getLogger(__name__)


# ======================================================================
# Stage 1: Score all training candidates with first-stage model
# ======================================================================


@torch.no_grad()
def score_candidates(
    first_stage_model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run the first-stage model over all candidates and collect scores.

    Only stores lightweight metadata (labels, probs, series) — NOT the
    patches themselves, which would require ~55 GB for 132K candidates.
    Patches are collected separately for the subset that passes filtering.

    Returns:
        Dict with keys:
            labels:  (N,) int array (ground-truth)
            probs:   (N,) float32 array (first-stage nodule probability)
            series:  (N,) list of seriesuid strings
    """
    first_stage_model.eval()
    all_labels = []
    all_probs = []
    all_series = []

    for batch in tqdm(dataloader, desc="Scoring candidates (stage 1)"):
        patches = batch["patch"].to(device)
        labels = batch["label"].numpy()
        series = batch["seriesuid"]

        with autocast("cuda", enabled=device.type == "cuda"):
            output = first_stage_model(patches)

        probs = torch.softmax(output["logits"], dim=1)[:, 1].cpu().numpy()

        all_labels.append(labels)
        all_probs.append(probs)
        all_series.extend(series)

        del patches, output
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "labels": np.concatenate(all_labels, axis=0),
        "probs": np.concatenate(all_probs, axis=0),
        "series": all_series,
    }


def collect_patches(
    dataset: Dataset,
    indices: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """Load patches for a specific subset of indices from the dataset.

    This avoids storing all 132K patches in RAM — only the subset needed
    for FP reduction training is loaded (typically a few thousand).
    """
    patches = []
    for i in tqdm(indices, desc="Collecting patches for FP reduction"):
        sample = dataset[int(i)]
        patches.append(sample["patch"].numpy())
    return np.stack(patches, axis=0)


# ======================================================================
# Stage 2: Build hard-negative mining dataset
# ======================================================================


class FPReductionDataset(Dataset):
    """Dataset for FP reduction training built from first-stage scores.

    Contains:
        - All true positives (label=1) that the first stage scored above
          the threshold (correctly detected nodules)
        - Hard false positives (label=0) that the first stage scored above
          the threshold (the mistakes we want to learn to reject)

    The FP reduction model learns to distinguish these two groups.
    """

    def __init__(
        self,
        patches: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
    ):
        self.patches = patches
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        patch = self.patches[idx].copy()  # (1, D, H, W)

        if self.augment:
            patch = self._augment(patch)

        return {
            "patch": torch.from_numpy(patch).float(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

    def _augment(self, patch: np.ndarray) -> np.ndarray:
        """Light augmentation for FP reduction (less aggressive than stage 1)."""
        # patch shape: (1, D, H, W) — augment spatial dims
        p = patch[0]  # (D, H, W)

        # Random flips
        for axis in range(3):
            if np.random.random() > 0.5:
                p = np.flip(p, axis=axis).copy()

        # Random 90-degree rotation
        if np.random.random() > 0.5:
            k = np.random.randint(1, 4)
            axes = [(0, 1), (0, 2), (1, 2)]
            ax = axes[np.random.randint(0, 3)]
            p = np.rot90(p, k=k, axes=ax).copy()

        # Gaussian noise (subtle)
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.005, p.shape).astype(np.float32)
            p = p + noise

        return np.ascontiguousarray(p[np.newaxis])


def build_fp_datasets(
    scored: dict[str, np.ndarray],
    source_dataset: Dataset,
    threshold: float,
    val_fraction: float = 0.2,
    hard_negative_ratio: float = 1.0,
) -> tuple[FPReductionDataset, FPReductionDataset]:
    """Split scored candidates into FP-reduction train/val sets.

    Args:
        scored: Output from score_candidates() (labels, probs, series only).
        source_dataset: The original dataset to load patches from.
        threshold: First-stage threshold — candidates above this are
            passed to the FP reduction stage.
        val_fraction: Fraction of data for validation.
        hard_negative_ratio: Max ratio of hard negatives to positives.
            1.0 means at most 1:1 balance. Use higher (e.g. 3.0) for
            more hard examples.

    Returns:
        (train_dataset, val_dataset)
    """
    labels = scored["labels"]
    probs = scored["probs"]
    series = scored["series"]

    # Candidates that passed the first-stage threshold
    above_threshold = probs >= threshold

    # True positives: correctly detected nodules
    tp_mask = above_threshold & (labels == 1)
    # Hard false positives: non-nodules the first stage thought were nodules
    hard_fp_mask = above_threshold & (labels == 0)
    # Also include false negatives (missed nodules) so the second stage
    # can learn their patterns
    fn_mask = (~above_threshold) & (labels == 1)

    tp_count = int(tp_mask.sum())
    fp_count = int(hard_fp_mask.sum())
    fn_count = int(fn_mask.sum())

    logger.info(
        "First-stage scoring at threshold %.3f:\n"
        "  True positives (detected nodules):  %d\n"
        "  Hard false positives (FP above threshold): %d\n"
        "  False negatives (missed nodules):   %d",
        threshold, tp_count, fp_count, fn_count,
    )

    # Build the FP reduction dataset:
    #   label=1: true nodules (TP + FN from stage 1)
    #   label=0: false positives from stage 1
    pos_indices = np.where(tp_mask | fn_mask)[0]
    neg_indices = np.where(hard_fp_mask)[0]

    # Cap hard negatives to avoid overwhelming positives
    max_negatives = int(len(pos_indices) * hard_negative_ratio)
    if len(neg_indices) > max_negatives:
        np.random.seed(42)
        neg_indices = np.random.choice(neg_indices, max_negatives, replace=False)

    all_indices = np.concatenate([pos_indices, neg_indices])
    np.random.seed(42)
    np.random.shuffle(all_indices)

    # Split by unique series UIDs for proper generalization
    idx_series = [series[i] for i in all_indices]
    unique_series = sorted(set(idx_series))
    np.random.seed(42)
    np.random.shuffle(unique_series)
    split_idx = int(len(unique_series) * (1 - val_fraction))
    train_series = set(unique_series[:split_idx])

    train_mask = np.array([s in train_series for s in idx_series])
    train_indices = all_indices[train_mask]
    val_indices = all_indices[~train_mask]

    train_labels = labels[train_indices]
    val_labels = labels[val_indices]

    logger.info(
        "FP reduction datasets:\n"
        "  Train: %d samples (%d pos, %d neg)\n"
        "  Val:   %d samples (%d pos, %d neg)",
        len(train_indices),
        int((train_labels == 1).sum()),
        int((train_labels == 0).sum()),
        len(val_indices),
        int((val_labels == 1).sum()),
        int((val_labels == 0).sum()),
    )

    # Collect only the patches we actually need (typically a few thousand
    # instead of all 132K) — this saves ~50 GB of RAM.
    logger.info("  Loading patches for selected candidates...")
    train_patches = collect_patches(source_dataset, train_indices)
    val_patches = collect_patches(source_dataset, val_indices)

    train_ds = FPReductionDataset(
        patches=train_patches,
        labels=train_labels,
        augment=True,
    )
    val_ds = FPReductionDataset(
        patches=val_patches,
        labels=val_labels,
        augment=False,
    )

    return train_ds, val_ds


# ======================================================================
# Stage 3: Train FPReductionNet
# ======================================================================


def compute_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute AUC-ROC via trapezoidal rule."""
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

    return float(_trapezoid(tp_rate, fp_rate))


class FPReductionTrainer:
    """Trainer for the FP reduction second-stage model."""

    def __init__(
        self,
        config: dict,
        checkpoint_dir: Path,
    ):
        self.config = config
        self.checkpoint_dir = checkpoint_dir / "fp_reduction"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("FP reduction training on device: %s", self.device)

        # FP reduction training config
        fp_train_cfg = config.get("fp_reduction_training", {})
        self.epochs = fp_train_cfg.get("epochs", 60)
        self.batch_size = fp_train_cfg.get("batch_size", 32)
        self.lr = fp_train_cfg.get("learning_rate", 0.0005)
        self.weight_decay = fp_train_cfg.get("weight_decay", 0.0001)
        self.patience = fp_train_cfg.get("early_stopping_patience", 12)
        self.num_workers = fp_train_cfg.get("num_workers", 2)

        # Build FP reduction model
        self.model = FPReductionNet(in_channels=1, base_filters=32).to(self.device)
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("FPReductionNet parameters: %s", f"{param_count:,}")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Loss — will be configured with class weights in train()
        self.criterion = None

        # Mixed precision
        self.scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

        # Scheduler: warmup + cosine
        warmup_epochs = min(3, max(self.epochs - 1, 0))
        warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=max(warmup_epochs, 1))
        cosine = CosineAnnealingLR(self.optimizer, T_max=max(self.epochs - warmup_epochs, 1))
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

        # Tracking
        self.best_val_auc = 0.0
        self.epochs_without_improvement = 0

    def train(
        self,
        train_dataset: FPReductionDataset,
        val_dataset: FPReductionDataset,
    ) -> Path:
        """Run the full FP reduction training loop.

        Returns:
            Path to best checkpoint.
        """
        # Compute class weights from training data to handle imbalance
        train_labels = train_dataset.labels
        num_neg = int((train_labels == 0).sum())
        num_pos = int((train_labels == 1).sum())
        if num_pos > 0 and num_neg > 0:
            # Inverse frequency weighting: give the minority class higher weight
            weight_neg = len(train_labels) / (2.0 * num_neg)
            weight_pos = len(train_labels) / (2.0 * num_pos)
            # Optional manual scaling from config
            fp_cfg = self.config.get("fp_reduction_training", {})
            scale = fp_cfg.get("class_weight_scale", 1.0)
            weight_pos *= scale
            class_weights = torch.tensor(
                [weight_neg, weight_pos], dtype=torch.float32
            ).to(self.device)
            logger.info(
                "Class weights — neg: %.3f  pos: %.3f  "
                "(train has %d neg / %d pos, scale=%.1f)",
                weight_neg, weight_pos, num_neg, num_pos, scale,
            )
        else:
            class_weights = None
            logger.warning("Could not compute class weights (pos=%d, neg=%d)", num_pos, num_neg)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        pin_memory = self.device.type == "cuda"

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )

        history = []

        for epoch in range(self.epochs):
            # Train
            train_metrics = self._train_epoch(train_loader)
            # Validate
            val_metrics = self._validate(val_loader)

            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                "Epoch %d/%d  lr=%.6f\n"
                "  Train — loss: %.4f  acc: %.4f  auc: %.4f\n"
                "  Val   — loss: %.4f  acc: %.4f  auc: %.4f  "
                "sens: %.4f  spec: %.4f",
                epoch + 1, self.epochs, lr,
                train_metrics["loss"], train_metrics["accuracy"], train_metrics["auc"],
                val_metrics["loss"], val_metrics["accuracy"], val_metrics["auc"],
                val_metrics["sensitivity"], val_metrics["specificity"],
            )

            history.append({
                "epoch": epoch + 1,
                "lr": lr,
                "train": train_metrics,
                "val": val_metrics,
            })

            # Check improvement
            is_best = val_metrics["auc"] > self.best_val_auc
            if is_best:
                self.best_val_auc = val_metrics["auc"]
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self._save_checkpoint(epoch, val_metrics, is_best)

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(
                    "Early stopping after %d epochs without improvement",
                    self.patience,
                )
                break

        # Save training history
        history_path = self.checkpoint_dir / "fp_training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info("FP reduction training complete. Best val AUC: %.4f", self.best_val_auc)
        return self.checkpoint_dir / "best.pth"

    def _train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []

        for batch in tqdm(loader, desc="FP Train", leave=False):
            patches = batch["patch"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            with autocast("cuda", enabled=self.device.type == "cuda"):
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

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
            "auc": compute_auc(np.array(all_labels), np.array(all_probs)),
        }

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        tp = fp = fn = tn = 0
        all_probs = []
        all_labels = []

        for batch in tqdm(loader, desc="FP Val", leave=False):
            patches = batch["patch"].to(self.device)
            labels = batch["label"].to(self.device)

            with autocast("cuda", enabled=self.device.type == "cuda"):
                output = self.model(patches)
                loss = self.criterion(output["logits"], labels)

            total_loss += loss.item() * patches.size(0)
            probs = torch.softmax(output["logits"], dim=1)[:, 1]
            preds = (probs > 0.5).long()
            correct += (preds == labels).sum().item()
            total += patches.size(0)

            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return {
            "loss": total_loss / total if total > 0 else float("nan"),
            "accuracy": correct / total if total > 0 else 0.0,
            "auc": compute_auc(np.array(all_labels), np.array(all_probs)),
            "precision": precision,
            "recall": recall,
            "sensitivity": recall,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "f1": 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0,
        }

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "best_val_auc": self.best_val_auc,
            "epochs_without_improvement": self.epochs_without_improvement,
            "config": self.config,
        }
        torch.save(checkpoint, self.checkpoint_dir / "latest.pth")
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
            logger.info("  Saved new best FP reduction model (AUC: %.4f)", metrics["auc"])


# ======================================================================
# Stage 4: Evaluate combined two-stage pipeline
# ======================================================================


@torch.no_grad()
def evaluate_two_stage(
    first_stage_model: nn.Module,
    fp_model: FPReductionNet,
    dataloader: DataLoader,
    device: torch.device,
    stage1_threshold: float,
    fp_threshold: float = 0.5,
) -> dict:
    """Evaluate the combined two-stage detection pipeline.

    Stage 1: NoduleResNet3D scores candidates, filters by stage1_threshold
    Stage 2: FPReductionNet re-scores survivors, filters by fp_threshold
    Final confidence: average of both stage scores

    Returns:
        Dict with single-stage and two-stage metrics for comparison.
    """
    first_stage_model.eval()
    fp_model.eval()

    all_labels = []
    # Stage 1 predictions
    s1_probs = []
    # Two-stage predictions
    combined_probs = []
    combined_preds = []

    for batch in tqdm(dataloader, desc="Evaluating two-stage pipeline"):
        patches = batch["patch"].to(device)
        labels = batch["label"].numpy()

        # Stage 1 forward
        with autocast("cuda", enabled=device.type == "cuda"):
            s1_output = first_stage_model(patches)
        s1_prob = torch.softmax(s1_output["logits"], dim=1)[:, 1].cpu().numpy()

        # Stage 2: only run on candidates that passed stage 1
        above = s1_prob >= stage1_threshold
        batch_combined = np.zeros_like(s1_prob)

        if above.any():
            fp_input = patches[above]
            with autocast("cuda", enabled=device.type == "cuda"):
                fp_output = fp_model(fp_input)
            fp_prob = torch.softmax(fp_output["logits"], dim=1)[:, 1].cpu().numpy()

            # Combined score: average of both stages
            combined_above = (s1_prob[above] + fp_prob) / 2.0
            # Apply FP threshold
            passed_fp = fp_prob >= fp_threshold
            combined_above[~passed_fp] = 0.0
            batch_combined[above] = combined_above

        all_labels.append(labels)
        s1_probs.append(s1_prob)
        combined_probs.append(batch_combined)

    all_labels = np.concatenate(all_labels)
    s1_probs = np.concatenate(s1_probs)
    combined_probs = np.concatenate(combined_probs)

    num_pos = int((all_labels == 1).sum())
    num_neg = int((all_labels == 0).sum())

    # Stage 1 metrics at stage1_threshold
    s1_preds = (s1_probs >= stage1_threshold).astype(int)
    s1_tp = int(((s1_preds == 1) & (all_labels == 1)).sum())
    s1_fp = int(((s1_preds == 1) & (all_labels == 0)).sum())
    s1_fn = int(((s1_preds == 0) & (all_labels == 1)).sum())
    s1_tn = int(((s1_preds == 0) & (all_labels == 0)).sum())

    # Two-stage metrics: a candidate is positive if combined_prob > 0
    # (meaning it passed both thresholds)
    ts_preds = (combined_probs > 0).astype(int)
    ts_tp = int(((ts_preds == 1) & (all_labels == 1)).sum())
    ts_fp = int(((ts_preds == 1) & (all_labels == 0)).sum())
    ts_fn = int(((ts_preds == 0) & (all_labels == 1)).sum())
    ts_tn = int(((ts_preds == 0) & (all_labels == 0)).sum())

    def _metrics(tp, fp, fn, tn, probs, labels):
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = sens
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        auc = compute_auc(labels, probs)
        return {
            "auc_roc": round(auc, 4),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    results = {
        "num_candidates": len(all_labels),
        "num_positives": num_pos,
        "num_negatives": num_neg,
        "stage1_threshold": stage1_threshold,
        "fp_threshold": fp_threshold,
        "stage1_only": _metrics(s1_tp, s1_fp, s1_fn, s1_tn, s1_probs, all_labels),
        "two_stage": _metrics(ts_tp, ts_fp, ts_fn, ts_tn, combined_probs, all_labels),
    }

    # Improvement summary
    s1_m = results["stage1_only"]
    ts_m = results["two_stage"]
    results["improvement"] = {
        "fp_reduction_count": s1_m["fp"] - ts_m["fp"],
        "fp_reduction_pct": round(
            (1 - ts_m["fp"] / s1_m["fp"]) * 100 if s1_m["fp"] > 0 else 0.0, 1
        ),
        "sensitivity_delta": round(ts_m["sensitivity"] - s1_m["sensitivity"], 4),
        "specificity_delta": round(ts_m["specificity"] - s1_m["specificity"], 4),
        "precision_delta": round(ts_m["precision"] - s1_m["precision"], 4),
    }

    return results


def format_two_stage_report(results: dict) -> str:
    """Format two-stage evaluation results as a readable report."""
    s1 = results["stage1_only"]
    ts = results["two_stage"]
    imp = results["improvement"]

    lines = [
        "=" * 65,
        "  TWO-STAGE PIPELINE EVALUATION",
        "=" * 65,
        "",
        f"  Candidates:       {results['num_candidates']} "
        f"({results['num_positives']} pos / {results['num_negatives']} neg)",
        f"  Stage-1 threshold: {results['stage1_threshold']}",
        f"  FP threshold:      {results['fp_threshold']}",
        "",
        "-" * 65,
        f"  {'Metric':<20} {'Stage 1 Only':>15} {'Two-Stage':>15} {'Delta':>10}",
        "-" * 65,
        f"  {'AUC-ROC':<20} {s1['auc_roc']:>15.4f} {ts['auc_roc']:>15.4f} "
        f"{ts['auc_roc'] - s1['auc_roc']:>+10.4f}",
        f"  {'Sensitivity':<20} {s1['sensitivity']:>15.4f} {ts['sensitivity']:>15.4f} "
        f"{imp['sensitivity_delta']:>+10.4f}",
        f"  {'Specificity':<20} {s1['specificity']:>15.4f} {ts['specificity']:>15.4f} "
        f"{imp['specificity_delta']:>+10.4f}",
        f"  {'Precision':<20} {s1['precision']:>15.4f} {ts['precision']:>15.4f} "
        f"{imp['precision_delta']:>+10.4f}",
        f"  {'F1 Score':<20} {s1['f1']:>15.4f} {ts['f1']:>15.4f} "
        f"{ts['f1'] - s1['f1']:>+10.4f}",
        "",
        "-" * 65,
        "  CONFUSION MATRIX COMPARISON",
        "-" * 65,
        f"  {'':>20} {'Stage 1':>20} {'Two-Stage':>20}",
        f"  {'True Positives':<20} {s1['tp']:>20} {ts['tp']:>20}",
        f"  {'False Positives':<20} {s1['fp']:>20} {ts['fp']:>20}",
        f"  {'False Negatives':<20} {s1['fn']:>20} {ts['fn']:>20}",
        f"  {'True Negatives':<20} {s1['tn']:>20} {ts['tn']:>20}",
        "",
        "-" * 65,
        "  FP REDUCTION IMPACT",
        "-" * 65,
        f"  False positives removed: {imp['fp_reduction_count']} "
        f"({imp['fp_reduction_pct']}% reduction)",
        f"  Sensitivity change:      {imp['sensitivity_delta']:+.4f}",
        f"  Specificity change:      {imp['specificity_delta']:+.4f}",
        "",
        "=" * 65,
    ]
    return "\n".join(lines)


# ======================================================================
# Main entry point
# ======================================================================


def load_config(config_path: str | None = None) -> dict:
    """Load config with defaults, optionally overridden by a YAML file."""
    import yaml

    default_path = PROJECT_ROOT / "config" / "default.yaml"
    config = {}
    if default_path.exists():
        with open(default_path) as f:
            config = yaml.safe_load(f)

    if config_path:
        with open(config_path) as f:
            override = yaml.safe_load(f)
            if override:
                _deep_merge(config, override)

    return config


def _deep_merge(base: dict, override: dict):
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def main():
    parser = argparse.ArgumentParser(
        description="Train FP reduction model (second-stage classifier)"
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Override config YAML (merged with default.yaml)",
    )
    parser.add_argument(
        "--first-stage-ckpt", type=str, default=None,
        help="Path to first-stage best.pth (default: checkpoints/best.pth)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints",
        help="Base checkpoint directory",
    )
    parser.add_argument(
        "--stage1-threshold", type=float, default=None,
        help="First-stage threshold for hard negative mining "
             "(default: from config inference.threshold)",
    )
    parser.add_argument(
        "--fp-threshold", type=float, default=None,
        help="FP reduction threshold (default: from config fp_reduction.threshold)",
    )
    parser.add_argument(
        "--hard-negative-ratio", type=float, default=3.0,
        help="Max ratio of hard negatives to positives (default: 3.0)",
    )
    parser.add_argument(
        "--score-batch-size", type=int, default=96,
        help="Batch size for first-stage scoring (default: 96)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)
    checkpoint_dir = Path(args.checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage1_threshold = args.stage1_threshold or config.get("inference", {}).get("threshold", 0.15)
    fp_threshold = args.fp_threshold or config.get("fp_reduction", {}).get("threshold", 0.5)
    first_stage_path = args.first_stage_ckpt or str(checkpoint_dir / "best.pth")

    logger.info("=" * 65)
    logger.info("  FP REDUCTION TRAINING PIPELINE")
    logger.info("=" * 65)
    logger.info("  First-stage checkpoint: %s", first_stage_path)
    logger.info("  Stage-1 threshold:      %.3f", stage1_threshold)
    logger.info("  FP reduction threshold: %.3f", fp_threshold)
    logger.info("  Hard negative ratio:    %.1f", args.hard_negative_ratio)
    logger.info("  Device:                 %s", device)
    logger.info("=" * 65)

    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load first-stage model
    # ------------------------------------------------------------------
    logger.info("\n[Step 1/4] Loading first-stage model...")
    first_stage_model = build_model(config).to(device)
    ckpt = torch.load(first_stage_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        first_stage_model.load_state_dict(ckpt["model_state_dict"])
    else:
        first_stage_model.load_state_dict(ckpt)
    first_stage_model.eval()
    logger.info("  Loaded first-stage model from %s (epoch %s)", first_stage_path, ckpt.get("epoch", "?"))

    # ------------------------------------------------------------------
    # Step 2: Score all training candidates
    # ------------------------------------------------------------------
    logger.info("\n[Step 2/4] Scoring all training candidates...")

    # Load the full training set WITHOUT balancing so we get all candidates
    # We temporarily override pos_neg_ratio to get everything
    scoring_config = json.loads(json.dumps(config))  # deep copy
    scoring_config.setdefault("training", {})["pos_neg_ratio"] = 0.01  # get all negatives

    train_dataset = CombinedLungDataset.from_config(scoring_config, split="train", augment=False)
    train_dataset.warm_disk_cache()

    num_workers = config.get("training", {}).get("num_workers", 2)
    scoring_loader = DataLoader(
        train_dataset,
        batch_size=args.score_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    logger.info("  Training candidates to score: %d", len(train_dataset))

    scored = score_candidates(first_stage_model, scoring_loader, device)
    del scoring_loader  # release worker processes
    logger.info(
        "  Scored %d candidates (%.1f%% positive)",
        len(scored["labels"]),
        100.0 * scored["labels"].mean(),
    )

    # Free first-stage model from GPU before loading patches
    del first_stage_model, ckpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 3: Build datasets and train FPReductionNet
    # ------------------------------------------------------------------
    logger.info("\n[Step 3/4] Training FP reduction model...")

    train_ds, val_ds = build_fp_datasets(
        scored,
        source_dataset=train_dataset,
        threshold=stage1_threshold,
        val_fraction=0.2,
        hard_negative_ratio=args.hard_negative_ratio,
    )

    if len(train_ds) == 0:
        logger.error(
            "No training samples for FP reduction! "
            "Check that first-stage model produces candidates above threshold %.3f",
            stage1_threshold,
        )
        sys.exit(1)

    trainer = FPReductionTrainer(config, checkpoint_dir)
    best_ckpt_path = trainer.train(train_ds, val_ds)

    # Free scored data from memory
    del scored, train_dataset, train_ds, val_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 4: Evaluate combined two-stage pipeline
    # ------------------------------------------------------------------
    logger.info("\n[Step 4/4] Evaluating combined two-stage pipeline...")

    # Reload first-stage model (was freed after scoring to save GPU memory)
    first_stage_model = build_model(config).to(device)
    ckpt = torch.load(first_stage_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        first_stage_model.load_state_dict(ckpt["model_state_dict"])
    else:
        first_stage_model.load_state_dict(ckpt)
    first_stage_model.eval()
    del ckpt

    # Load best FP reduction model
    fp_model = FPReductionNet(in_channels=1, base_filters=32).to(device)
    fp_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    fp_model.load_state_dict(fp_ckpt["model_state_dict"])
    fp_model.eval()

    # Evaluate on validation set
    val_dataset = CombinedLungDataset.from_config(config, split="val", augment=False)
    val_dataset.warm_disk_cache()

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.score_batch_size,
        shuffle=False,
        num_workers=config.get("training", {}).get("num_workers", 2),
        pin_memory=device.type == "cuda",
    )

    eval_results = evaluate_two_stage(
        first_stage_model=first_stage_model,
        fp_model=fp_model,
        dataloader=val_loader,
        device=device,
        stage1_threshold=stage1_threshold,
        fp_threshold=fp_threshold,
    )

    # Save results
    eval_path = checkpoint_dir / "fp_reduction" / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    # Print report
    report = format_two_stage_report(eval_results)
    print("\n" + report)
    logger.info("Evaluation results saved to %s", eval_path)

    # Update config to enable FP reduction with trained model
    fp_config_update = {
        "fp_reduction": {
            "enabled": True,
            "model_path": str(best_ckpt_path),
            "threshold": fp_threshold,
        }
    }
    fp_config_path = checkpoint_dir / "fp_reduction" / "fp_reduction_config.yaml"
    import yaml
    with open(fp_config_path, "w") as f:
        yaml.dump(fp_config_update, f, default_flow_style=False)
    logger.info("FP reduction config snippet saved to %s", fp_config_path)

    elapsed = time.time() - t_start
    logger.info(
        "\nFP reduction pipeline complete in %.1f minutes.\n"
        "  Best FP reduction AUC: %.4f\n"
        "  FP reduction: %d fewer false positives (%.1f%% reduction)\n"
        "  Sensitivity delta: %+.4f",
        elapsed / 60,
        fp_ckpt.get("best_val_auc", 0.0),
        eval_results["improvement"]["fp_reduction_count"],
        eval_results["improvement"]["fp_reduction_pct"],
        eval_results["improvement"]["sensitivity_delta"],
    )


if __name__ == "__main__":
    main()
