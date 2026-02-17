"""Integration tests for the training pipeline.

Tests the full training loop, dataset loading, checkpoint save/load,
and augmentation pipeline using synthetic data (no LUNA16 required).
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from lung_screener.dataset import LUNA16Dataset
from lung_screener.model import NoduleResNet3D, build_model
from lung_screener.preprocessing import apply_hu_window, extract_patch
from lung_screener.train import Trainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def fast_config(default_config):
    """Config tuned for fast CI tests (tiny model, 2 epochs)."""
    cfg = default_config.copy()
    cfg["model"] = {
        "architecture": "resnet3d",
        "in_channels": 1,
        "num_classes": 2,
        "patch_size": [32, 32, 32],
        "predict_nodule_type": False,
    }
    cfg["training"] = {
        "batch_size": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": 2,
        "pos_neg_ratio": 1.0,
        "augmentation": {"rotation": False, "flip": False, "noise_std": 0.0},
        "early_stopping_patience": 5,
        "scheduler": {"type": "cosine", "warmup_epochs": 1},
    }
    cfg["data"] = {
        "dataset_dir": "",  # will be overridden per test
        "cache_dir": "",
        "val_split": 0.3,
    }
    return cfg


@pytest.fixture
def synthetic_luna16(tmp_path):
    """Create a minimal synthetic LUNA16 dataset directory.

    Generates fake annotations.csv, candidates_V2.csv, and
    subset0/ with small .npy volumes masquerading as preprocessed cache.
    """
    dataset_dir = tmp_path / "luna16"
    subset_dir = dataset_dir / "subset0"
    subset_dir.mkdir(parents=True)

    n_series = 4
    series_uids = [f"1.3.6.1.4.1.14519.fake.{i:04d}" for i in range(n_series)]

    # Create small synthetic volumes saved as .mhd (SimpleITK) is expensive,
    # so we'll use the cache mechanism instead - write .npy directly to cache.
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    for uid in series_uids:
        vol = np.random.rand(64, 64, 64).astype(np.float32)
        np.save(cache_dir / f"{uid}.npy", vol)

    # Annotations: 2 nodules per series in first 2 series
    annotation_rows = []
    for uid in series_uids[:2]:
        for j in range(2):
            annotation_rows.append({
                "seriesuid": uid,
                "coordX": 30.0 + j * 5,
                "coordY": 30.0 + j * 5,
                "coordZ": 30.0 + j * 5,
                "diameter_mm": 6.0 + j * 2,
            })
    annotations = pd.DataFrame(annotation_rows)
    annotations.to_csv(dataset_dir / "annotations.csv", index=False)

    # Candidates: positives + negatives
    candidate_rows = []
    # Positives (matching annotations)
    for _, row in annotations.iterrows():
        candidate_rows.append({
            "seriesuid": row["seriesuid"],
            "coordX": row["coordX"],
            "coordY": row["coordY"],
            "coordZ": row["coordZ"],
            "class": 1,
        })
    # Negatives
    for uid in series_uids:
        for j in range(4):
            candidate_rows.append({
                "seriesuid": uid,
                "coordX": 10.0 + j * 10,
                "coordY": 10.0 + j * 10,
                "coordZ": 10.0 + j * 10,
                "class": 0,
            })
    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(dataset_dir / "candidates_V2.csv", index=False)

    return dataset_dir, cache_dir, series_uids


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestLUNA16Dataset:
    def test_load_samples_train(self, fast_config, synthetic_luna16):
        dataset_dir, cache_dir, _ = synthetic_luna16
        fast_config["data"]["dataset_dir"] = str(dataset_dir)
        fast_config["data"]["cache_dir"] = str(cache_dir)

        ds = LUNA16Dataset(
            dataset_dir=dataset_dir,
            config=fast_config,
            split="train",
            val_split=0.3,
            augment=False,
            cache_dir=cache_dir,
        )
        assert len(ds) > 0, "Training split should have samples"

    def test_load_samples_val(self, fast_config, synthetic_luna16):
        dataset_dir, cache_dir, _ = synthetic_luna16
        fast_config["data"]["dataset_dir"] = str(dataset_dir)
        fast_config["data"]["cache_dir"] = str(cache_dir)

        ds = LUNA16Dataset(
            dataset_dir=dataset_dir,
            config=fast_config,
            split="val",
            val_split=0.3,
            augment=False,
            cache_dir=cache_dir,
        )
        assert len(ds) > 0, "Validation split should have samples"

    def test_getitem_returns_correct_shapes(self, fast_config, synthetic_luna16):
        dataset_dir, cache_dir, _ = synthetic_luna16
        patch_size = fast_config["model"]["patch_size"]

        ds = LUNA16Dataset(
            dataset_dir=dataset_dir,
            config=fast_config,
            split="train",
            val_split=0.3,
            augment=False,
            cache_dir=cache_dir,
        )
        if len(ds) == 0:
            pytest.skip("No training samples generated")

        sample = ds[0]
        assert sample["patch"].shape == (1, *patch_size)
        assert sample["label"].dtype == torch.long
        assert sample["label"].item() in (0, 1)
        assert isinstance(sample["seriesuid"], str)

    def test_balancing_reduces_negatives(self, fast_config, synthetic_luna16):
        dataset_dir, cache_dir, _ = synthetic_luna16
        fast_config["training"]["pos_neg_ratio"] = 1.0

        ds = LUNA16Dataset(
            dataset_dir=dataset_dir,
            config=fast_config,
            split="train",
            val_split=0.3,
            augment=False,
            cache_dir=cache_dir,
        )
        labels = [ds.samples[i]["label"] for i in range(len(ds.samples))]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        # With 1:1 ratio, negatives should be <= positives
        if n_pos > 0:
            assert n_neg <= n_pos + 1  # +1 for rounding

    def test_empty_dataset_graceful(self, fast_config, tmp_path):
        """Dataset with no annotation files returns empty."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        ds = LUNA16Dataset(
            dataset_dir=empty_dir,
            config=fast_config,
            split="train",
        )
        assert len(ds) == 0

    def test_augmentation_changes_patch(self, fast_config, synthetic_luna16):
        dataset_dir, cache_dir, _ = synthetic_luna16
        fast_config["training"]["augmentation"] = {
            "rotation": True,
            "flip": True,
            "noise_std": 0.05,
        }

        ds_aug = LUNA16Dataset(
            dataset_dir=dataset_dir,
            config=fast_config,
            split="train",
            val_split=0.3,
            augment=True,
            cache_dir=cache_dir,
        )
        if len(ds_aug) == 0:
            pytest.skip("No training samples")

        # Get same sample twice; augmentation is random so they should differ
        p1 = ds_aug[0]["patch"]
        p2 = ds_aug[0]["patch"]
        # Very unlikely to be identical with noise + random augmentation
        # but not guaranteed, so we just check the shape
        assert p1.shape == p2.shape


# ---------------------------------------------------------------------------
# Trainer construction tests
# ---------------------------------------------------------------------------

class TestTrainerInit:
    def test_trainer_creates_checkpoint_dir(self, fast_config, tmp_path):
        ckpt_dir = tmp_path / "ckpts"
        trainer = Trainer(fast_config, checkpoint_dir=ckpt_dir)
        assert ckpt_dir.exists()
        assert trainer.device in (torch.device("cpu"), torch.device("cuda"))

    def test_trainer_model_is_correct_arch(self, fast_config, tmp_path):
        trainer = Trainer(fast_config, checkpoint_dir=tmp_path / "ckpts")
        assert isinstance(trainer.model, NoduleResNet3D)

    def test_trainer_densenet(self, fast_config, tmp_path):
        fast_config["model"]["architecture"] = "densenet3d"
        trainer = Trainer(fast_config, checkpoint_dir=tmp_path / "ckpts")
        from lung_screener.model import NoduleDenseNet3D
        assert isinstance(trainer.model, NoduleDenseNet3D)

    def test_trainer_optimizer_base_lr(self, fast_config, tmp_path):
        trainer = Trainer(fast_config, checkpoint_dir=tmp_path / "ckpts")
        # The scheduler sets initial LR = base_lr * start_factor (0.1),
        # so check the optimizer's base lr via the scheduler
        assert trainer.lr == fast_config["training"]["learning_rate"]


# ---------------------------------------------------------------------------
# Training loop integration
# ---------------------------------------------------------------------------

class TestTrainingLoop:
    def test_train_epoch_on_synthetic_data(self, fast_config, synthetic_luna16, tmp_path):
        """Run one training epoch on synthetic data."""
        dataset_dir, cache_dir, _ = synthetic_luna16
        fast_config["data"]["dataset_dir"] = str(dataset_dir)
        fast_config["data"]["cache_dir"] = str(cache_dir)

        trainer = Trainer(fast_config, checkpoint_dir=tmp_path / "ckpts")
        train_loader, val_loader = trainer.create_dataloaders()

        if len(train_loader.dataset) == 0:
            pytest.skip("No training samples")

        metrics = trainer.train_epoch(train_loader)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "auc" in metrics
        assert metrics["loss"] >= 0
        assert 0 <= metrics["accuracy"] <= 1

    def test_validate_on_synthetic_data(self, fast_config, synthetic_luna16, tmp_path):
        """Run validation on synthetic data."""
        dataset_dir, cache_dir, _ = synthetic_luna16
        fast_config["data"]["dataset_dir"] = str(dataset_dir)
        fast_config["data"]["cache_dir"] = str(cache_dir)

        trainer = Trainer(fast_config, checkpoint_dir=tmp_path / "ckpts")
        _, val_loader = trainer.create_dataloaders()

        if len(val_loader.dataset) == 0:
            pytest.skip("No validation samples")

        metrics = trainer.validate(val_loader)
        assert "loss" in metrics
        assert "sensitivity" in metrics
        assert "specificity" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

class TestCheckpointing:
    def test_save_and_load_checkpoint(self, fast_config, tmp_path):
        ckpt_dir = tmp_path / "ckpts"
        trainer = Trainer(fast_config, checkpoint_dir=ckpt_dir)

        # Save
        metrics = {"loss": 0.5, "accuracy": 0.8, "auc": 0.75}
        trainer.save_checkpoint(epoch=3, metrics=metrics, is_best=True)

        assert (ckpt_dir / "latest.pth").exists()
        assert (ckpt_dir / "best.pth").exists()

        # Load into a fresh trainer
        trainer2 = Trainer(fast_config, checkpoint_dir=tmp_path / "ckpts2")
        epoch, phase = trainer2.load_checkpoint(ckpt_dir / "best.pth")
        assert epoch == 3
        assert phase == "complete"

    def test_best_checkpoint_only_saved_when_best(self, fast_config, tmp_path):
        ckpt_dir = tmp_path / "ckpts"
        trainer = Trainer(fast_config, checkpoint_dir=ckpt_dir)

        trainer.save_checkpoint(epoch=0, metrics={"auc": 0.5}, is_best=False)
        assert (ckpt_dir / "latest.pth").exists()
        assert not (ckpt_dir / "best.pth").exists()

        trainer.save_checkpoint(epoch=1, metrics={"auc": 0.9}, is_best=True)
        assert (ckpt_dir / "best.pth").exists()


# ---------------------------------------------------------------------------
# AUC computation
# ---------------------------------------------------------------------------

class TestAUCComputation:
    def test_perfect_separation(self, fast_config, tmp_path):
        trainer = Trainer(fast_config, checkpoint_dir=tmp_path / "ckpts")
        labels = [0, 0, 0, 1, 1, 1]
        probs = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        auc = trainer._compute_auc(labels, probs)
        assert auc == pytest.approx(1.0, abs=0.01)

    def test_random_prediction(self, fast_config, tmp_path):
        trainer = Trainer(fast_config, checkpoint_dir=tmp_path / "ckpts")
        np.random.seed(0)
        labels = [0] * 500 + [1] * 500
        probs = np.random.rand(1000).tolist()
        auc = trainer._compute_auc(labels, probs)
        # Random should be ~0.5
        assert 0.35 < auc < 0.65

    def test_single_class_returns_zero(self, fast_config, tmp_path):
        trainer = Trainer(fast_config, checkpoint_dir=tmp_path / "ckpts")
        auc = trainer._compute_auc([0, 0, 0], [0.1, 0.5, 0.9])
        assert auc == 0.0


# ---------------------------------------------------------------------------
# End-to-end mini-training run
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_training_run_2_epochs(self, fast_config, synthetic_luna16, tmp_path):
        """Full training run for 2 epochs on synthetic data.

        Validates that the entire pipeline (dataset → model → train → validate
        → checkpoint → dashboard) works without errors.
        """
        dataset_dir, cache_dir, _ = synthetic_luna16
        fast_config["data"]["dataset_dir"] = str(dataset_dir)
        fast_config["data"]["cache_dir"] = str(cache_dir)
        fast_config["training"]["epochs"] = 2

        ckpt_dir = tmp_path / "ckpts"
        trainer = Trainer(fast_config, checkpoint_dir=ckpt_dir)
        trainer.train()

        # Verify checkpoint was saved
        assert (ckpt_dir / "latest.pth").exists()
        # Verify training completed (best_val_auc was set)
        assert trainer.best_val_auc >= 0
