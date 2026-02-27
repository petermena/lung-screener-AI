"""Training pipeline for lung nodule detection model.

Handles the full training loop with:
- Mixed precision training
- Focal Loss for hard-example mining
- Stochastic Weight Averaging (SWA) for better generalisation
- Label smoothing for calibration
- Learning rate scheduling with warmup + cosine warm restarts
- Early stopping
- Checkpoint saving
- K-fold cross-validation
- Metrics logging and dashboard generation
"""

import copy
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, SWALR

# NumPy 2.0 renamed np.trapz → np.trapezoid
_trapezoid = getattr(np, "trapezoid", None) or np.trapz
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from .dataset import CombinedLungDataset, LUNA16Dataset
from .metrics_dashboard import MetricsLogger, save_dashboard
from .model import build_model

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced classification.

    Focuses training on hard-to-classify examples by down-weighting
    easy negatives.  Particularly effective for nodule detection where
    most candidates are non-nodules.

    ``FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)``

    Args:
        alpha: Per-class weight (scalar or list). Default balances
               positive/negative classes.
        gamma: Focusing parameter — larger values focus more on hard
               examples. gamma=0 reduces to standard cross-entropy.
        label_smoothing: Label smoothing factor (0 = no smoothing).
    """

    def __init__(
        self,
        alpha: float | list[float] | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
            else:
                self.register_buffer("alpha", torch.tensor([1 - alpha, alpha], dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        # Ensure float32 for numerical stability under mixed precision
        logits = logits.float()
        probs = torch.softmax(logits, dim=1)

        # One-hot with optional label smoothing
        one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
        if self.label_smoothing > 0:
            one_hot = one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes

        pt = (probs * one_hot).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma
        ce = -torch.log(pt.clamp(min=1e-8))

        loss = focal_weight * ce

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss

        return loss.mean()


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
        self.eval_threshold = train_config.get(
            "eval_threshold",
            config.get("inference", {}).get("threshold", 0.5),
        )

        # Checkpoint safeguards
        self.checkpoint_save_every = train_config.get("checkpoint_save_every", 10)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        logger.info(f"Metrics threshold: {self.eval_threshold:.3f}")

        # Loss function — select based on config
        loss_config = train_config.get("loss", {})
        loss_type = loss_config.get("type", "cross_entropy")
        label_smoothing = loss_config.get("label_smoothing", 0.0)

        if loss_type == "focal":
            focal_gamma = loss_config.get("focal_gamma", 2.0)
            focal_alpha = loss_config.get("focal_alpha", None)
            self.criterion = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                label_smoothing=label_smoothing,
            )
            logger.info(f"Using Focal Loss (gamma={focal_gamma}, alpha={focal_alpha}, "
                        f"label_smoothing={label_smoothing})")
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            if label_smoothing > 0:
                logger.info(f"Using CrossEntropyLoss with label_smoothing={label_smoothing}")

        # Mixed precision
        self.scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

        # Scheduler
        sched_config = train_config.get("scheduler", {})
        sched_type = sched_config.get("type", "cosine")
        warmup_epochs = sched_config.get("warmup_epochs", 5)
        warmup_epochs = min(warmup_epochs, max(self.epochs - 1, 0))

        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, total_iters=max(warmup_epochs, 1)
        )

        if sched_type == "cosine_warm_restarts":
            # Cosine annealing with warm restarts — resets LR periodically
            # to escape local minima and explore more of the loss landscape.
            t_0 = sched_config.get("t_0", 20)
            t_mult = sched_config.get("t_mult", 2)
            main_scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=t_0, T_mult=t_mult,
            )
            logger.info(f"Scheduler: Cosine warm restarts (T_0={t_0}, T_mult={t_mult})")
        else:
            main_scheduler = CosineAnnealingLR(
                self.optimizer, T_max=max(self.epochs - warmup_epochs, 1)
            )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )

        # SWA (Stochastic Weight Averaging) — averages weights from the
        # later stages of training to find a flatter minimum that
        # generalises better. Typically adds +0.5-1.5% AUC.
        swa_config = train_config.get("swa", {})
        self.swa_enabled = swa_config.get("enabled", False)
        self.swa_start_epoch = swa_config.get("start_epoch", int(self.epochs * 0.75))
        self.swa_lr = swa_config.get("lr", self.lr * 0.5)
        self.swa_model = None
        self.swa_scheduler = None
        if self.swa_enabled:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.swa_lr)
            logger.info(f"SWA enabled: starts epoch {self.swa_start_epoch}, lr={self.swa_lr}")

        # Acquire checkpoint lock (must be after all config attrs are set)
        self._acquire_checkpoint_lock()

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

        # Log class distribution so imbalance is visible
        for name, ds in [("Train", train_dataset), ("Val", val_dataset)]:
            all_samples = []
            for child in ds.datasets:
                # Unwrap Subset (created by max_val_candidates cap)
                inner = child
                indices = None
                if isinstance(inner, Subset):
                    indices = inner.indices
                    inner = inner.dataset
                # ConcatDataset wrapping: flatten one more level
                if isinstance(inner, ConcatDataset):
                    samples = []
                    for grandchild in inner.datasets:
                        samples.extend(grandchild.samples)
                else:
                    samples = inner.samples
                if indices is not None:
                    samples = [samples[i] for i in indices]
                all_samples.extend(samples)
            n_pos = sum(1 for s in all_samples if s["label"] == 1)
            n_neg = len(all_samples) - n_pos
            ratio = n_neg / n_pos if n_pos > 0 else float("inf")
            logger.info(f"  {name} distribution: {n_pos} pos / {n_neg} neg (1:{ratio:.1f})")

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
                # Compute loss in float32 to avoid fp16 overflow in softmax/log
                loss = self.criterion(output["logits"].float(), labels)

            # Skip batch if loss is NaN to avoid corrupting model weights
            if torch.isnan(loss):
                logger.warning("  NaN training loss — skipping batch")
                self.optimizer.zero_grad()
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Verify model weights are still valid after the optimizer step.
            # GradScaler catches most inf/NaN gradients, but edge cases
            # (e.g. NaN from weight update arithmetic) can slip through and
            # permanently corrupt the model.
            if any(torch.isnan(p).any() for p in self.model.parameters()):
                best_path = self.checkpoint_dir / "best.pth"
                if best_path.exists():
                    logger.error(
                        "  NaN detected in model parameters after optimizer step "
                        "— reloading best checkpoint"
                    )
                    self.load_checkpoint(best_path)
                    return {"loss": float("nan"), "accuracy": 0.0, "auc": 0.0}
                else:
                    logger.error(
                        "  NaN detected in model parameters and no best "
                        "checkpoint available — aborting epoch"
                    )
                    return {"loss": float("nan"), "accuracy": 0.0, "auc": 0.0}

            total_loss += loss.item() * patches.size(0)
            probs = torch.softmax(output["logits"], dim=1)[:, 1]
            preds = (probs >= self.eval_threshold).long()
            correct += (preds == labels).sum().item()
            total += patches.size(0)

            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.4f}"})

        metrics = {
            "loss": total_loss / total,
            "accuracy": correct / total if total > 0 else 0.0,
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

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation", leave=False):
                patches = batch["patch"].to(self.device)
                labels = batch["label"].to(self.device)

                with autocast("cuda", enabled=torch.cuda.is_available()):
                    output = self.model(patches)
                    # Compute loss in float32 to avoid fp16 overflow in softmax/log
                    loss = self.criterion(output["logits"].float(), labels)

                # Skip entire batch if logits contain NaN (e.g. fp16 overflow)
                if torch.isnan(output["logits"]).any():
                    logger.warning("  NaN detected in validation logits — skipping batch")
                    continue

                batch_loss = loss.item()
                if np.isfinite(batch_loss):
                    total_loss += batch_loss * patches.size(0)
                    loss_count += patches.size(0)
                probs = torch.softmax(output["logits"], dim=1)[:, 1]
                preds = (probs >= self.eval_threshold).long()
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
            "accuracy": correct / total if total > 0 else 0.0,
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

        auc = _trapezoid(tp_rate, fp_rate)
        return float(auc)

    @torch.no_grad()
    def _update_swa_bn(self, train_loader: DataLoader) -> bool:
        """Update SWA model BatchNorm stats with fp32 forward passes.

        Includes NaN protection: if a forward pass produces NaN in any
        BatchNorm running stats, the batch is skipped and the stats are
        restored from the previous good state.  If ALL batches produce NaN,
        the original BN stats are restored so the model remains usable.

        Returns True if BN stats were successfully updated, False otherwise.
        """
        bn_modules = {}
        for name, module in self.swa_model.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                bn_modules[name] = module

        if not bn_modules:
            return True

        # Snapshot the *current* BN stats so we can restore them if the
        # entire update fails (e.g. every batch produces NaN).
        prev_bn_state = {
            name: (m.running_mean.clone(), m.running_var.clone(),
                   m.num_batches_tracked.clone())
            for name, m in bn_modules.items()
        }

        # Zero stats for fresh cumulative moving average
        for m in bn_modules.values():
            m.running_mean.zero_()
            m.running_var.zero_()
            m.num_batches_tracked.zero_()

        # Save original momentum and switch to cumulative moving average
        momenta = {name: m.momentum for name, m in bn_modules.items()}
        was_training = self.swa_model.training
        self.swa_model.train()
        for m in bn_modules.values():
            m.momentum = None

        nan_batches = 0
        total_batches = 0
        for batch in tqdm(train_loader, desc="SWA BN update", leave=False):
            total_batches += 1
            # Snapshot BN stats before forward pass so we can roll back
            bn_snapshots = {
                name: (m.running_mean.clone(), m.running_var.clone(),
                       m.num_batches_tracked.clone())
                for name, m in bn_modules.items()
            }

            x = batch["patch"].to(self.device)
            self.swa_model(x)

            # Check BN stats for NaN — restore snapshot if corrupted
            corrupted = False
            for name, m in bn_modules.items():
                if torch.isnan(m.running_mean).any() or torch.isnan(m.running_var).any():
                    corrupted = True
                    break

            if corrupted:
                nan_batches += 1
                for name, m in bn_modules.items():
                    prev_mean, prev_var, prev_count = bn_snapshots[name]
                    m.running_mean.copy_(prev_mean)
                    m.running_var.copy_(prev_var)
                    m.num_batches_tracked.copy_(prev_count)
                if nan_batches <= 3:
                    logger.warning("  NaN in BN stats during SWA update — skipping batch")

        if nan_batches > 0:
            logger.warning(f"  SWA BN update: skipped {nan_batches}/{total_batches} NaN batch(es)")

        for name, m in bn_modules.items():
            m.momentum = momenta[name]
        self.swa_model.train(was_training)

        # If ALL batches failed, BN stats are still zeros — running_var=0
        # causes division-by-near-zero, amplifying activations and causing
        # fp16 overflow during validation.  Restore the previous BN stats.
        if nan_batches == total_batches:
            logger.warning(
                "  SWA BN update failed entirely — restoring previous BN stats"
            )
            for name, m in bn_modules.items():
                prev_mean, prev_var, prev_count = prev_bn_state[name]
                m.running_mean.copy_(prev_mean)
                m.running_var.copy_(prev_var)
                m.num_batches_tracked.copy_(prev_count)
            return False

        return True

    def _acquire_checkpoint_lock(self):
        """Write a lock file to prevent other runs from using this directory.

        If a lock file already exists from a *different* config, the trainer
        refuses to start — this catches the exact scenario that destroyed
        the epoch-76 checkpoint.
        """
        lock_path = self.checkpoint_dir / ".train_lock.json"
        run_signature = {
            "epochs": self.epochs,
            "architecture": self.config.get("model", {}).get("architecture"),
            "swa_enabled": self.swa_enabled,
            "lr": self.lr,
            "started_at": datetime.now().isoformat(),
            "pid": __import__("os").getpid(),
        }

        if lock_path.exists():
            try:
                existing = json.loads(lock_path.read_text())
                # Same config signature is OK (resumed run). Different config
                # means a rogue run is about to overwrite production checkpoints.
                existing_sig = (
                    existing.get("epochs"),
                    existing.get("architecture"),
                    existing.get("swa_enabled"),
                )
                new_sig = (
                    run_signature["epochs"],
                    run_signature["architecture"],
                    run_signature["swa_enabled"],
                )
                if existing_sig != new_sig:
                    logger.error(
                        "CHECKPOINT DIRECTORY CONFLICT: %s is locked by a "
                        "different training config (started %s, epochs=%s, "
                        "arch=%s). Use --checkpoint-dir to pick a separate "
                        "directory, or delete %s to override.",
                        self.checkpoint_dir,
                        existing.get("started_at", "?"),
                        existing.get("epochs", "?"),
                        existing.get("architecture", "?"),
                        lock_path,
                    )
                    raise RuntimeError(
                        f"Checkpoint directory {self.checkpoint_dir} is locked "
                        f"by a different training configuration. Use a separate "
                        f"--checkpoint-dir to avoid overwriting existing checkpoints."
                    )
            except json.JSONDecodeError:
                pass  # Corrupt lock file — overwrite it

        lock_path.write_text(json.dumps(run_signature, indent=2))

    def _verify_checkpoint(self, path: Path, expected_epoch: int) -> bool:
        """Reload a saved checkpoint and verify it wasn't corrupted on disk."""
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if ckpt.get("epoch") != expected_epoch:
                logger.error(
                    "Checkpoint verification FAILED for %s: expected epoch %d, "
                    "got %d",
                    path, expected_epoch, ckpt.get("epoch"),
                )
                return False
            if "model_state_dict" not in ckpt:
                logger.error("Checkpoint verification FAILED: no model_state_dict")
                return False
            return True
        except Exception as e:
            logger.error("Checkpoint verification FAILED for %s: %s", path, e)
            return False

    def save_checkpoint(
        self,
        epoch: int,
        metrics: dict,
        is_best: bool = False,
        phase: str = "complete",
    ):
        """Save model checkpoint with safeguards.

        Safeguards added after the epoch-76 (AUC 0.982) checkpoint was
        lost to an accidental overwrite:

        1. **Numbered epoch checkpoints** — saves ``epoch_{N}.pth`` every
           ``checkpoint_save_every`` epochs so there is always a fallback.
        2. **Best checkpoint backup** — before overwriting ``best.pth``,
           copies the existing one to ``best_backup_epoch{N}_auc{AUC}.pth``.
        3. **Write-then-rename** — writes to a temp file first, then renames
           atomically to avoid partial-write corruption.
        4. **Post-save verification** — reloads and sanity-checks the
           checkpoint after writing.

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

        # --- Write latest.pth via temp file for atomic save ---
        latest_path = self.checkpoint_dir / "latest.pth"
        tmp_path = self.checkpoint_dir / "latest.pth.tmp"
        torch.save(checkpoint, tmp_path)
        tmp_path.rename(latest_path)

        # --- Numbered epoch checkpoint (every N epochs) ---
        if phase == "complete" and (epoch + 1) % self.checkpoint_save_every == 0:
            epoch_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pth"
            shutil.copy2(latest_path, epoch_path)
            logger.info(f"  Saved epoch checkpoint: {epoch_path.name}")

        if is_best:
            best_path = self.checkpoint_dir / "best.pth"

            # Back up previous best before overwriting
            if best_path.exists():
                try:
                    prev = torch.load(best_path, map_location="cpu", weights_only=False)
                    prev_epoch = prev.get("epoch", "?")
                    prev_auc = prev.get("best_val_auc", prev.get("metrics", {}).get("auc", 0))
                    backup_name = f"best_backup_epoch{prev_epoch}_auc{prev_auc:.4f}.pth"
                    backup_path = self.checkpoint_dir / backup_name
                    shutil.copy2(best_path, backup_path)
                    logger.info(f"  Backed up previous best → {backup_name}")
                except Exception as e:
                    logger.warning(f"  Could not back up previous best.pth: {e}")

            # Atomic write for best.pth
            tmp_best = self.checkpoint_dir / "best.pth.tmp"
            torch.save(checkpoint, tmp_best)
            tmp_best.rename(best_path)
            logger.info(f"  Saved new best model (AUC: {metrics['auc']:.4f})")

            # Verify the written checkpoint
            if not self._verify_checkpoint(best_path, epoch):
                logger.error("  CRITICAL: best.pth failed verification after save!")

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
        nan_streak = 0

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

            # Re-warm OS page cache for val volumes before every validation pass.
            # Training evicts val pages (training data ~25 GB > 16 GB RAM), so
            # without this, val workers page-fault every volume from NVMe and
            # each validation batch takes 50-100× longer than necessary.
            val_loader.dataset.warm_disk_cache()

            # Validate — use SWA model for eval when active
            if self.swa_enabled and epoch >= self.swa_start_epoch and self.swa_model is not None:
                # Only update SWA model if base model weights are clean.
                # Averaging NaN weights into the SWA model would corrupt it
                # permanently, making all subsequent validation NaN.
                base_has_nan = any(
                    torch.isnan(p).any() for p in self.model.parameters()
                )
                if base_has_nan:
                    logger.warning(
                        "  Skipping SWA update — base model contains NaN weights"
                    )
                else:
                    self.swa_model.update_parameters(self.model)
                # BN update requires a forward pass over training data
                bn_ok = self._update_swa_bn(train_loader)
                # Validate with SWA-averaged model
                original_model = self.model
                self.model = self.swa_model
                val_metrics = self.validate(val_loader)
                self.model = original_model
                # If SWA validation produced NaN, fall back to base model
                # so training can continue instead of counting NaN streaks.
                if np.isnan(val_metrics["loss"]):
                    reason = "SWA BN update and validation" if not bn_ok else "SWA validation"
                    logger.warning(
                        f"  {reason} returned NaN — "
                        "falling back to base model for this epoch"
                    )
                    val_metrics = self.validate(val_loader)
            else:
                val_metrics = self.validate(val_loader)

            logger.info(
                f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                f"Acc: {val_metrics['accuracy']:.4f}, "
                f"AUC: {val_metrics['auc']:.4f}, "
                f"Sens: {val_metrics['sensitivity']:.4f}, "
                f"Spec: {val_metrics['specificity']:.4f}"
            )

            # Detect NaN validation loss — halt if it persists
            if np.isnan(val_metrics["loss"]):
                nan_streak += 1
                logger.warning(
                    f"  Validation loss is NaN ({nan_streak} consecutive epoch(s))"
                )
                if nan_streak >= 3:
                    logger.error(
                        f"Halting: validation loss NaN for {nan_streak} consecutive "
                        f"epochs. Resume from best checkpoint: "
                        f"{self.checkpoint_dir / 'best.pth'}"
                    )
                    break
            else:
                nan_streak = 0

            # Step scheduler — switch to SWA scheduler after swa_start_epoch
            if self.swa_enabled and epoch >= self.swa_start_epoch and self.swa_scheduler is not None:
                self.swa_scheduler.step()
            else:
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
            # When SWA is active, save the averaged model as best
            self.save_checkpoint(epoch, val_metrics, is_best)

            # Early stopping (disabled during SWA phase to let averaging converge)
            if self.swa_enabled and epoch >= self.swa_start_epoch:
                pass  # Don't early-stop during SWA
            elif self.epochs_without_improvement >= self.patience:
                if self.swa_enabled:
                    logger.info(
                        f"Early stopping triggered — switching to SWA phase "
                        f"for final {self.epochs - epoch - 1} epochs"
                    )
                    self.swa_start_epoch = epoch + 1
                else:
                    logger.info(
                        f"Early stopping after {self.patience} epochs without improvement"
                    )
                    break

        # Save final SWA model if active — but skip if training halted
        # due to NaN, as the SWA model is likely corrupted.
        if self.swa_enabled and self.swa_model is not None and nan_streak < 3:
            logger.info("Saving SWA-averaged model as swa_best.pth")
            bn_ok = self._update_swa_bn(train_loader)
            if not bn_ok:
                logger.warning(
                    "Skipping SWA model save — BN update failed entirely"
                )
            else:
                swa_checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.swa_model.module.state_dict(),
                    "config": self.config,
                    "best_val_auc": self.best_val_auc,
                    "swa": True,
                }
                swa_tmp = self.checkpoint_dir / "swa_best.pth.tmp"
                swa_path = self.checkpoint_dir / "swa_best.pth"
                torch.save(swa_checkpoint, swa_tmp)
                swa_tmp.rename(swa_path)
        elif self.swa_enabled and nan_streak >= 3:
            logger.warning(
                "Skipping SWA model save — training halted due to NaN. "
                "Resume from best checkpoint: %s",
                self.checkpoint_dir / "best.pth",
            )

        # Generate final dashboard
        dashboard_path = save_dashboard(
            self.metrics_logger.metrics_file,
            self.checkpoint_dir / "dashboard.html",
        )
        logger.info(f"Training dashboard: {dashboard_path}")
        logger.info(f"Training complete. Best validation AUC: {self.best_val_auc:.4f}")


def train_kfold(
    config: dict,
    n_folds: int = 5,
    checkpoint_dir: str | Path = "./checkpoints",
    resume: bool = False,
):
    """Run k-fold cross-validation training.

    Trains ``n_folds`` independent models, each validated on a
    different fold.  All fold checkpoints are saved so they can
    later be ensembled for inference.

    Args:
        config: Full configuration dict.
        n_folds: Number of folds.
        checkpoint_dir: Root checkpoint directory.
        resume: If True, skip completed folds and resume incomplete ones
            from their ``latest.pth`` checkpoint.

    Returns:
        Dict with per-fold and aggregated metrics.
    """
    checkpoint_dir = Path(checkpoint_dir)
    all_fold_metrics: list[dict] = []

    for fold in range(n_folds):
        fold_dir = checkpoint_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        best_path = fold_dir / "best.pth"
        latest_path = fold_dir / "latest.pth"

        # --- Resume logic ---------------------------------------------------
        if resume and best_path.exists():
            # Check if this fold ran to completion by inspecting the latest
            # checkpoint's epoch against the configured total.
            ckpt = torch.load(latest_path if latest_path.exists() else best_path,
                              map_location="cpu", weights_only=False)
            total_epochs = config.get("training", {}).get("epochs", 150)
            finished_epoch = ckpt.get("epoch", 0)

            if finished_epoch >= total_epochs - 1 and ckpt.get("phase") == "complete":
                # Fold fully finished — load its metrics and skip.
                auc = ckpt.get("best_val_auc",
                               ckpt.get("metrics", {}).get("auc", 0.0))
                logger.info(f"\n{'='*60}")
                logger.info(f"  K-FOLD: Fold {fold + 1}/{n_folds}  [SKIPPED — already complete, AUC={auc:.4f}]")
                logger.info(f"{'='*60}\n")
                all_fold_metrics.append({
                    "fold": fold,
                    "best_val_auc": auc,
                    "checkpoint": str(best_path),
                })
                continue

        resume_path: Path | None = None
        if resume and latest_path.exists():
            resume_path = latest_path
            ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
            logger.info(f"\n{'='*60}")
            logger.info(f"  K-FOLD: Fold {fold + 1}/{n_folds}  [RESUMING from epoch {ckpt.get('epoch', 0) + 1}]")
            logger.info(f"{'='*60}\n")
        else:
            logger.info(f"\n{'='*60}")
            logger.info(f"  K-FOLD: Fold {fold + 1}/{n_folds}")
            logger.info(f"{'='*60}\n")

        # Inject fold index into config so the dataset can split accordingly
        fold_config = copy.deepcopy(config)
        fold_config.setdefault("data", {})["kfold"] = {
            "enabled": True,
            "n_folds": n_folds,
            "fold_index": fold,
        }

        trainer = Trainer(fold_config, checkpoint_dir=fold_dir)
        trainer.train(resume_from=resume_path)

        fold_metrics = {
            "fold": fold,
            "best_val_auc": trainer.best_val_auc,
            "checkpoint": str(fold_dir / "best.pth"),
        }
        all_fold_metrics.append(fold_metrics)

        logger.info(f"Fold {fold + 1} best AUC: {trainer.best_val_auc:.4f}")

    # Aggregate
    aucs = [m["best_val_auc"] for m in all_fold_metrics]
    summary = {
        "n_folds": n_folds,
        "folds": all_fold_metrics,
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "min_auc": float(np.min(aucs)),
        "max_auc": float(np.max(aucs)),
    }

    summary_path = checkpoint_dir / "kfold_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nK-Fold Summary: AUC = {summary['mean_auc']:.4f} "
                f"± {summary['std_auc']:.4f} "
                f"(range {summary['min_auc']:.4f} - {summary['max_auc']:.4f})")
    logger.info(f"Results saved to {summary_path}")

    return summary
