"""Tests for the evaluation pipeline."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from lung_screener.evaluate import (
    _bootstrap_auc_ci,
    _compute_auc,
    _froc_sensitivity,
    _operating_points,
    _youden_optimal,
    format_report,
)


class TestComputeAUC:
    def test_perfect_separation(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        auc = _compute_auc(labels, probs)
        assert auc == pytest.approx(1.0, abs=0.01)

    def test_random_predictions(self):
        rng = np.random.RandomState(42)
        labels = np.array([0] * 500 + [1] * 500)
        probs = rng.rand(1000)
        auc = _compute_auc(labels, probs)
        assert 0.35 < auc < 0.65

    def test_single_class(self):
        assert _compute_auc(np.array([0, 0, 0]), np.array([0.1, 0.5, 0.9])) == 0.0
        assert _compute_auc(np.array([1, 1, 1]), np.array([0.1, 0.5, 0.9])) == 0.0

    def test_worst_case(self):
        labels = np.array([1, 1, 1, 0, 0, 0])
        probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        auc = _compute_auc(labels, probs)
        assert auc == pytest.approx(0.0, abs=0.01)


class TestBootstrapCI:
    def test_returns_valid_range(self):
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        lo, hi = _bootstrap_auc_ci(labels, probs, n_bootstrap=100)
        assert 0.0 <= lo <= hi <= 1.0

    def test_perfect_has_narrow_ci(self):
        labels = np.array([0] * 50 + [1] * 50)
        probs = np.concatenate([np.linspace(0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
        lo, hi = _bootstrap_auc_ci(labels, probs, n_bootstrap=200)
        assert hi - lo < 0.15


class TestOperatingPoints:
    def test_returns_all_thresholds(self):
        labels = np.array([0, 0, 1, 1])
        probs = np.array([0.2, 0.4, 0.6, 0.8])
        points = _operating_points(labels, probs)
        assert len(points) == 9
        thresholds = [p["threshold"] for p in points]
        assert 0.5 in thresholds

    def test_metrics_valid(self):
        labels = np.array([0, 0, 1, 1])
        probs = np.array([0.2, 0.4, 0.6, 0.8])
        for p in _operating_points(labels, probs):
            assert 0.0 <= p["sensitivity"] <= 1.0
            assert 0.0 <= p["specificity"] <= 1.0


class TestFROCSensitivity:
    def test_returns_dict(self):
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        result = _froc_sensitivity(labels, probs)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_empty_class(self):
        labels = np.array([0, 0, 0])
        probs = np.array([0.1, 0.5, 0.9])
        assert _froc_sensitivity(labels, probs) == {}


class TestYoudenOptimal:
    def test_perfect_threshold(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        threshold, j = _youden_optimal(labels, probs)
        assert 0.3 < threshold < 0.7
        assert j == pytest.approx(1.0, abs=0.05)

    def test_single_class_returns_defaults(self):
        threshold, j = _youden_optimal(
            np.array([0, 0, 0]), np.array([0.1, 0.5, 0.9])
        )
        assert threshold == 0.5
        assert j == 0.0


class TestFormatReport:
    def test_produces_string(self):
        results = {
            "checkpoint": "test.pth",
            "epoch": 5,
            "num_samples": 100,
            "num_positives": 20,
            "num_negatives": 80,
            "threshold": 0.5,
            "confusion_matrix": {"tp": 15, "fp": 10, "fn": 5, "tn": 70},
            "metrics": {
                "auc_roc": 0.85,
                "auc_95ci_low": 0.78,
                "auc_95ci_high": 0.92,
                "sensitivity": 0.75,
                "specificity": 0.875,
                "precision": 0.6,
                "recall": 0.75,
                "f1_score": 0.667,
                "accuracy": 0.85,
                "npv": 0.933,
                "ece": 0.05,
            },
            "optimal_threshold": {"threshold": 0.45, "youden_j": 0.625},
            "operating_points": [
                {"threshold": 0.5, "sensitivity": 0.75, "specificity": 0.875,
                 "precision": 0.6, "tp": 15, "fp": 10, "fn": 5, "tn": 70},
            ],
            "froc_sensitivity": {"sens_at_fpr_0.1": 0.6},
        }
        report = format_report(results)
        assert "EVALUATION REPORT" in report
        assert "AUC-ROC" in report
        assert "0.8500" in report
