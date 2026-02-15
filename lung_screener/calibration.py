"""Confidence calibration for model outputs.

Applies temperature scaling or Platt scaling to convert raw model
outputs into well-calibrated probabilities, so that when the model
says 80% confidence, approximately 80% of those cases are true positives.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)


class TemperatureScaler:
    """Temperature scaling for confidence calibration.

    Learns a single scalar temperature parameter T on a held-out
    calibration set. The calibrated probability is:
        p_cal = softmax(logits / T)

    When T > 1, the model becomes less confident (more uniform).
    When T < 1, the model becomes more confident (more peaked).

    Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
    """

    def __init__(self):
        self.temperature = 1.0
        self._fitted = False

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100,
    ):
        """Learn the temperature parameter on calibration data.

        Uses L-BFGS or gradient descent to minimize negative log-likelihood
        of the true labels under the temperature-scaled softmax.

        Args:
            logits: Raw model logits, shape (N, num_classes).
            labels: True labels, shape (N,).
            lr: Learning rate for optimization.
            max_iter: Maximum optimization iterations.
        """
        logits_t = torch.from_numpy(logits).float()
        labels_t = torch.from_numpy(labels).long()

        # Temperature parameter (log-space for positivity)
        log_temp = nn.Parameter(torch.zeros(1))
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.LBFGS([log_temp], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            temp = log_temp.exp()
            scaled = logits_t / temp
            loss = criterion(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)

        self.temperature = float(log_temp.exp().item())
        self._fitted = True

        # Compute calibration metrics
        with torch.no_grad():
            scaled_logits = logits_t / self.temperature
            probs = torch.softmax(scaled_logits, dim=1)[:, 1].numpy()
            ece_before = _expected_calibration_error(
                torch.softmax(logits_t, dim=1)[:, 1].numpy(),
                labels,
            )
            ece_after = _expected_calibration_error(probs, labels)

        logger.info(
            f"Temperature scaling: T={self.temperature:.4f}, "
            f"ECE: {ece_before:.4f} -> {ece_after:.4f}"
        )

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits.

        Args:
            logits: Raw model logits, shape (N, num_classes).

        Returns:
            Calibrated probabilities for the positive class, shape (N,).
        """
        logits_t = torch.from_numpy(logits).float()
        scaled = logits_t / self.temperature
        probs = torch.softmax(scaled, dim=1)[:, 1]
        return probs.numpy()

    def calibrate_single(self, logits: np.ndarray) -> float:
        """Calibrate a single sample's logits.

        Args:
            logits: Shape (num_classes,).

        Returns:
            Calibrated probability for the positive class.
        """
        return float(self.calibrate(logits[np.newaxis])[0])

    def save(self, path: str | Path):
        """Save calibration parameters."""
        data = {
            "temperature": self.temperature,
            "fitted": self._fitted,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info(f"Saved calibration to {path}")

    def load(self, path: str | Path):
        """Load calibration parameters."""
        with open(path) as f:
            data = json.load(f)
        self.temperature = data["temperature"]
        self._fitted = data.get("fitted", True)
        logger.info(f"Loaded calibration from {path} (T={self.temperature:.4f})")


class PlattScaler:
    """Platt scaling (logistic regression) for binary calibration.

    Fits a logistic regression A*f + B to map raw model scores to
    calibrated probabilities.

    Reference: Platt, "Probabilistic Outputs for SVMs", 1999
    """

    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self._fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray, max_iter: int = 200):
        """Fit Platt scaling parameters.

        Args:
            scores: Raw model confidence scores, shape (N,).
            labels: True binary labels, shape (N,).
            max_iter: Maximum iterations for optimization.
        """
        scores_t = torch.from_numpy(scores).float().unsqueeze(1)
        labels_t = torch.from_numpy(labels).float()

        a_param = nn.Parameter(torch.ones(1))
        b_param = nn.Parameter(torch.zeros(1))

        optimizer = torch.optim.LBFGS([a_param, b_param], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            logits = a_param * scores_t.squeeze() + b_param
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)

        self.a = float(a_param.item())
        self.b = float(b_param.item())
        self._fitted = True

        # Compute calibration improvement
        raw_ece = _expected_calibration_error(scores, labels)
        calibrated = self.calibrate(scores)
        cal_ece = _expected_calibration_error(calibrated, labels)

        logger.info(
            f"Platt scaling: A={self.a:.4f}, B={self.b:.4f}, "
            f"ECE: {raw_ece:.4f} -> {cal_ece:.4f}"
        )

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        """Apply Platt scaling.

        Args:
            scores: Raw model confidence scores, shape (N,).

        Returns:
            Calibrated probabilities, shape (N,).
        """
        logits = self.a * scores + self.b
        return 1.0 / (1.0 + np.exp(-logits))

    def save(self, path: str | Path):
        data = {"a": self.a, "b": self.b, "fitted": self._fitted}
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str | Path):
        with open(path) as f:
            data = json.load(f)
        self.a = data["a"]
        self.b = data["b"]
        self._fitted = data.get("fitted", True)


def _expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Partitions predictions into equal-width bins by predicted probability
    and measures the gap between predicted confidence and actual accuracy.

    Args:
        probs: Predicted probabilities, shape (N,).
        labels: True binary labels, shape (N,).
        n_bins: Number of calibration bins.

    Returns:
        ECE value (0 = perfectly calibrated).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probs)

    if total == 0:
        return 0.0

    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]

        mask = (probs >= lo) & (probs < hi)
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)

        count = mask.sum()
        if count == 0:
            continue

        avg_confidence = probs[mask].mean()
        avg_accuracy = labels[mask].mean()
        ece += (count / total) * abs(avg_accuracy - avg_confidence)

    return ece


def compute_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute data for a reliability diagram.

    Args:
        probs: Predicted probabilities, shape (N,).
        labels: True binary labels, shape (N,).
        n_bins: Number of bins.

    Returns:
        Dict with bin_centers, bin_accuracies, bin_confidences, bin_counts.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    centers = []
    accuracies = []
    confidences = []
    counts = []

    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)

        count = mask.sum()
        counts.append(int(count))
        centers.append((lo + hi) / 2)

        if count > 0:
            accuracies.append(float(labels[mask].mean()))
            confidences.append(float(probs[mask].mean()))
        else:
            accuracies.append(0.0)
            confidences.append((lo + hi) / 2)

    return {
        "bin_centers": centers,
        "bin_accuracies": accuracies,
        "bin_confidences": confidences,
        "bin_counts": counts,
        "ece": float(_expected_calibration_error(probs, labels, n_bins)),
    }
