"""Active learning for intelligent training data selection.

Identifies the most informative unlabeled samples for annotation,
maximizing model improvement per labeled example. Uses uncertainty
sampling and diversity-based selection.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast

from .model import build_model
from .preprocessing import CTPreprocessor, extract_patch

logger = logging.getLogger(__name__)


@dataclass
class UncertainSample:
    """A candidate sample ranked by model uncertainty."""

    series_uid: str
    center_voxel: tuple[int, int, int]
    center_world: tuple[float, float, float]
    diameter_mm: float
    uncertainty: float  # Higher = more uncertain = more valuable to label
    predicted_prob: float
    uncertainty_type: str  # "entropy", "margin", "mc_dropout"

    def to_dict(self) -> dict:
        return {
            "series_uid": self.series_uid,
            "center_voxel": list(self.center_voxel),
            "center_world": [round(c, 1) for c in self.center_world],
            "diameter_mm": round(self.diameter_mm, 1),
            "uncertainty": round(self.uncertainty, 4),
            "predicted_prob": round(self.predicted_prob, 3),
            "uncertainty_type": self.uncertainty_type,
        }


class ActiveLearner:
    """Selects the most informative samples for annotation.

    Supports multiple uncertainty estimation strategies:
    - Entropy: Max entropy of predicted distribution
    - Margin: Smallest margin between top two class probabilities
    - MC Dropout: Monte Carlo dropout for Bayesian uncertainty estimation
    """

    def __init__(
        self,
        config: dict,
        model_path: str | Path,
        device: str | None = None,
    ):
        self.config = config
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.preprocessor = CTPreprocessor(config)
        self.patch_size = tuple(
            config.get("model", {}).get("patch_size", [48, 48, 48])
        )

        # Load model (auto-detect architecture from checkpoint weights)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model = build_model(config, state_dict=state_dict).to(self.device)
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def rank_by_entropy(
        self,
        patches: np.ndarray,
        candidates: list[dict],
        series_uid: str = "",
    ) -> list[UncertainSample]:
        """Rank candidates by prediction entropy (highest uncertainty first).

        Entropy = -sum(p * log(p)) is maximized when the model is most
        uncertain about the classification.

        Args:
            patches: Array of shape (N, 1, D, H, W).
            candidates: List of candidate dicts from preprocessing.
            series_uid: Series identifier.

        Returns:
            List of UncertainSample sorted by decreasing uncertainty.
        """
        self.model.eval()
        samples = []

        for i in range(0, len(patches), 64):
            batch = torch.from_numpy(patches[i:i + 64]).float().to(self.device)
            with autocast():
                output = self.model(batch)

            probs = torch.softmax(output["logits"], dim=1).cpu().numpy()

            for j, (prob, cand) in enumerate(
                zip(probs, candidates[i:i + 64])
            ):
                # Shannon entropy
                entropy = -np.sum(prob * np.log(prob + 1e-10))
                samples.append(UncertainSample(
                    series_uid=series_uid,
                    center_voxel=cand["center_voxel"],
                    center_world=cand.get("center_world", (0, 0, 0)),
                    diameter_mm=cand.get("diameter_mm", 0.0),
                    uncertainty=float(entropy),
                    predicted_prob=float(prob[1]),
                    uncertainty_type="entropy",
                ))

        samples.sort(key=lambda s: s.uncertainty, reverse=True)
        return samples

    @torch.no_grad()
    def rank_by_margin(
        self,
        patches: np.ndarray,
        candidates: list[dict],
        series_uid: str = "",
    ) -> list[UncertainSample]:
        """Rank candidates by prediction margin (smallest margin = most uncertain).

        Margin = P(class1) - P(class2). Small margins indicate the model
        is torn between classes.

        Args:
            patches: Array of shape (N, 1, D, H, W).
            candidates: List of candidate dicts.
            series_uid: Series identifier.

        Returns:
            List of UncertainSample sorted by decreasing uncertainty (smallest margin).
        """
        self.model.eval()
        samples = []

        for i in range(0, len(patches), 64):
            batch = torch.from_numpy(patches[i:i + 64]).float().to(self.device)
            with autocast():
                output = self.model(batch)

            probs = torch.softmax(output["logits"], dim=1).cpu().numpy()

            for j, (prob, cand) in enumerate(
                zip(probs, candidates[i:i + 64])
            ):
                sorted_p = np.sort(prob)[::-1]
                margin = sorted_p[0] - sorted_p[1]
                # Invert so smaller margin = higher uncertainty
                uncertainty = 1.0 - margin

                samples.append(UncertainSample(
                    series_uid=series_uid,
                    center_voxel=cand["center_voxel"],
                    center_world=cand.get("center_world", (0, 0, 0)),
                    diameter_mm=cand.get("diameter_mm", 0.0),
                    uncertainty=float(uncertainty),
                    predicted_prob=float(prob[1]),
                    uncertainty_type="margin",
                ))

        samples.sort(key=lambda s: s.uncertainty, reverse=True)
        return samples

    def rank_by_mc_dropout(
        self,
        patches: np.ndarray,
        candidates: list[dict],
        series_uid: str = "",
        n_forward: int = 10,
    ) -> list[UncertainSample]:
        """Rank candidates using Monte Carlo dropout uncertainty.

        Runs multiple forward passes with dropout enabled and measures
        the variance of predictions. High variance = high epistemic
        uncertainty.

        Args:
            patches: Array of shape (N, 1, D, H, W).
            candidates: List of candidate dicts.
            series_uid: Series identifier.
            n_forward: Number of stochastic forward passes.

        Returns:
            List of UncertainSample sorted by decreasing uncertainty.
        """
        # Enable dropout during inference
        self.model.train()  # Enables dropout

        all_probs = []
        with torch.no_grad():
            for _ in range(n_forward):
                run_probs = []
                for i in range(0, len(patches), 64):
                    batch = torch.from_numpy(
                        patches[i:i + 64]
                    ).float().to(self.device)
                    with autocast():
                        output = self.model(batch)
                    probs = torch.softmax(output["logits"], dim=1)[:, 1]
                    run_probs.extend(probs.cpu().numpy())
                all_probs.append(run_probs)

        self.model.eval()

        # Shape: (n_forward, N)
        all_probs = np.array(all_probs)

        # Predictive mean and variance
        mean_probs = all_probs.mean(axis=0)
        var_probs = all_probs.var(axis=0)

        samples = []
        for j, cand in enumerate(candidates):
            samples.append(UncertainSample(
                series_uid=series_uid,
                center_voxel=cand["center_voxel"],
                center_world=cand.get("center_world", (0, 0, 0)),
                diameter_mm=cand.get("diameter_mm", 0.0),
                uncertainty=float(var_probs[j]),
                predicted_prob=float(mean_probs[j]),
                uncertainty_type="mc_dropout",
            ))

        samples.sort(key=lambda s: s.uncertainty, reverse=True)
        return samples

    def select_for_annotation(
        self,
        samples: list[UncertainSample],
        n_select: int = 50,
        diversity_weight: float = 0.3,
    ) -> list[UncertainSample]:
        """Select a diverse set of uncertain samples for annotation.

        Combines uncertainty ranking with spatial diversity to avoid
        selecting many similar candidates from the same region.

        Args:
            samples: Uncertainty-ranked samples.
            n_select: Number of samples to select.
            diversity_weight: Weight for diversity vs uncertainty (0-1).

        Returns:
            Selected subset of samples.
        """
        if len(samples) <= n_select:
            return samples

        selected = [samples[0]]  # Start with most uncertain

        for _ in range(1, n_select):
            best_score = -float("inf")
            best_idx = -1

            for i, candidate in enumerate(samples):
                if candidate in selected:
                    continue

                # Uncertainty component (normalized)
                unc_score = candidate.uncertainty

                # Diversity: minimum distance to already selected
                min_dist = float("inf")
                for sel in selected:
                    dist = np.sqrt(
                        sum(
                            (a - b) ** 2
                            for a, b in zip(candidate.center_world, sel.center_world)
                        )
                    )
                    min_dist = min(min_dist, dist)

                # Normalize distance (assume ~500mm max lung span)
                div_score = min(min_dist / 500.0, 1.0)

                combined = (1 - diversity_weight) * unc_score + diversity_weight * div_score

                if combined > best_score:
                    best_score = combined
                    best_idx = i

            if best_idx >= 0:
                selected.append(samples[best_idx])

        return selected

    def save_candidates(
        self,
        samples: list[UncertainSample],
        output_path: str | Path,
    ):
        """Save selected candidates to JSON for annotation workflow."""
        data = {
            "num_samples": len(samples),
            "samples": [s.to_dict() for s in samples],
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(samples)} candidates to {output_path}")
