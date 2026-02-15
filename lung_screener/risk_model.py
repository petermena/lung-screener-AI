"""Patient demographics and cancer risk scoring.

Implements the Brock/PanCan model for estimating malignancy probability
based on patient demographics and nodule characteristics, and provides
risk-adjusted Lung-RADS categorization.
"""

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PatientDemographics:
    """Patient demographic and clinical history data."""

    age: int = 0
    sex: str = ""  # "M" or "F"
    smoking_status: str = ""  # "current", "former", "never"
    pack_years: float = 0.0
    years_since_quit: float = 0.0
    family_history_lung_cancer: bool = False
    prior_cancer_history: bool = False
    emphysema: bool = False

    def to_dict(self) -> dict:
        return {
            "age": self.age,
            "sex": self.sex,
            "smoking_status": self.smoking_status,
            "pack_years": self.pack_years,
            "years_since_quit": self.years_since_quit,
            "family_history_lung_cancer": self.family_history_lung_cancer,
            "prior_cancer_history": self.prior_cancer_history,
            "emphysema": self.emphysema,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PatientDemographics":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dicom(cls, ds) -> "PatientDemographics":
        """Extract available demographics from a DICOM dataset."""
        age = 0
        age_str = getattr(ds, "PatientAge", "")
        if age_str:
            try:
                age = int(age_str.rstrip("Y").rstrip("y"))
            except (ValueError, AttributeError):
                pass

        sex = getattr(ds, "PatientSex", "").upper()
        if sex not in ("M", "F"):
            sex = ""

        return cls(age=age, sex=sex)


@dataclass
class RiskScore:
    """Result of a cancer risk calculation."""

    model_name: str
    malignancy_probability: float
    risk_category: str  # "low", "moderate", "high", "very_high"
    factors: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "malignancy_probability": round(self.malignancy_probability, 4),
            "risk_category": self.risk_category,
            "factors": self.factors,
        }


def compute_brock_score(
    demographics: PatientDemographics,
    nodule_diameter_mm: float,
    nodule_type: str = "solid",
    nodule_count: int = 1,
    upper_lobe: bool = False,
    spiculation: bool = False,
) -> RiskScore:
    """Compute the Brock University / PanCan malignancy probability.

    Implements the full PanCan model (McWilliams et al., NEJM 2013)
    for estimating the probability that a pulmonary nodule is malignant.

    The model uses logistic regression with the following predictors:
    - Age, sex, family history, emphysema
    - Nodule size, type (solid/part-solid/ground-glass), location, count
    - Spiculation

    Args:
        demographics: Patient demographic data.
        nodule_diameter_mm: Nodule diameter in mm.
        nodule_type: "solid", "part_solid", or "ground_glass".
        nodule_count: Number of nodules detected.
        upper_lobe: Whether the nodule is in an upper lobe.
        spiculation: Whether the nodule has spiculated margins.

    Returns:
        RiskScore with malignancy probability and category.
    """
    # PanCan model coefficients (McWilliams et al., NEJM 2013)
    intercept = -6.7892

    # Age (per year)
    beta_age = 0.0391
    age_val = demographics.age if demographics.age > 0 else 62  # default to screening median

    # Sex (female = 1)
    beta_sex = 0.7838
    sex_val = 1.0 if demographics.sex == "F" else 0.0

    # Family history of lung cancer
    beta_family = 0.7668
    family_val = 1.0 if demographics.family_history_lung_cancer else 0.0

    # Emphysema
    beta_emphysema = 0.3654
    emphysema_val = 1.0 if demographics.emphysema else 0.0

    # Nodule size (diameter in mm)
    beta_size = 0.1283
    size_val = nodule_diameter_mm

    # Nodule type
    beta_part_solid = 0.3772
    beta_ground_glass = -0.2485  # GGN slightly lower risk than solid at same size
    type_part_solid = 1.0 if nodule_type == "part_solid" else 0.0
    type_ground_glass = 1.0 if nodule_type == "ground_glass" else 0.0

    # Nodule count (log-transformed)
    beta_count = -0.0824
    count_val = math.log(max(1, nodule_count))

    # Upper lobe location
    beta_upper = 0.6581
    upper_val = 1.0 if upper_lobe else 0.0

    # Spiculation
    beta_spiculation = 0.7729
    spiculation_val = 1.0 if spiculation else 0.0

    # Linear predictor
    logit = (
        intercept
        + beta_age * age_val
        + beta_sex * sex_val
        + beta_family * family_val
        + beta_emphysema * emphysema_val
        + beta_size * size_val
        + beta_part_solid * type_part_solid
        + beta_ground_glass * type_ground_glass
        + beta_count * count_val
        + beta_upper * upper_val
        + beta_spiculation * spiculation_val
    )

    # Convert to probability
    probability = 1.0 / (1.0 + math.exp(-logit))

    # Categorize risk
    if probability < 0.05:
        risk_category = "low"
    elif probability < 0.10:
        risk_category = "moderate"
    elif probability < 0.30:
        risk_category = "high"
    else:
        risk_category = "very_high"

    factors = {
        "age": age_val,
        "sex": demographics.sex or "unknown",
        "family_history": demographics.family_history_lung_cancer,
        "emphysema": demographics.emphysema,
        "nodule_diameter_mm": nodule_diameter_mm,
        "nodule_type": nodule_type,
        "nodule_count": nodule_count,
        "upper_lobe": upper_lobe,
        "spiculation": spiculation,
    }

    logger.debug(
        f"Brock score: probability={probability:.4f}, "
        f"category={risk_category}, logit={logit:.3f}"
    )

    return RiskScore(
        model_name="Brock/PanCan",
        malignancy_probability=probability,
        risk_category=risk_category,
        factors=factors,
    )


def compute_lung_rads_with_risk(
    nodule_diameter_mm: float,
    nodule_type: str = "solid",
    risk_score: RiskScore | None = None,
) -> str:
    """Compute Lung-RADS category accounting for nodule type.

    ACR Lung-RADS v2022 thresholds differ by nodule type:
    - Solid nodules: 2 (<6mm), 3 (6-8mm), 4A (8-15mm), 4B (>=15mm)
    - Part-solid: 2 (<6mm total), 3 (>=6mm, solid <6mm), 4A (solid 6-8mm), 4B (solid >=8mm)
    - Ground-glass: 2 (<30mm), 3 (>=30mm)

    Args:
        nodule_diameter_mm: Nodule diameter in mm.
        nodule_type: "solid", "part_solid", or "ground_glass".
        risk_score: Optional risk score for upgrade consideration.

    Returns:
        Lung-RADS category string.
    """
    d = nodule_diameter_mm

    if nodule_type == "ground_glass":
        if d < 30:
            category = "2"
        else:
            category = "3"
    elif nodule_type == "part_solid":
        # For part-solid, diameter here is total nodule size
        # The solid component size would need separate measurement
        # Using total size as conservative proxy
        if d < 6:
            category = "2"
        elif d < 8:
            category = "3"
        elif d < 15:
            category = "4A"
        else:
            category = "4B"
    else:
        # Solid nodule thresholds
        if d < 6:
            category = "2"
        elif d < 8:
            category = "3"
        elif d < 15:
            category = "4A"
        else:
            category = "4B"

    # Risk-based upgrade: if Brock probability >= 15% and category is 3, upgrade to 4A
    if risk_score and risk_score.malignancy_probability >= 0.15:
        if category == "3":
            category = "4A"
            logger.info(
                f"Lung-RADS upgraded from 3 to 4A based on Brock probability "
                f"({risk_score.malignancy_probability:.1%})"
            )

    return category


def screening_eligibility(demographics: PatientDemographics) -> dict:
    """Check USPSTF lung cancer screening eligibility criteria.

    USPSTF 2021 recommends annual LDCT for adults who:
    - Are aged 50-80
    - Have a 20+ pack-year smoking history
    - Currently smoke or quit within the past 15 years

    Args:
        demographics: Patient demographics.

    Returns:
        Dict with eligibility status and reasoning.
    """
    reasons = []
    eligible = True

    if demographics.age < 50 or demographics.age > 80:
        eligible = False
        reasons.append(
            f"Age {demographics.age} is outside the 50-80 range"
        )

    if demographics.pack_years < 20:
        eligible = False
        reasons.append(
            f"Pack-years ({demographics.pack_years}) below 20-year threshold"
        )

    if demographics.smoking_status == "former" and demographics.years_since_quit > 15:
        eligible = False
        reasons.append(
            f"Quit smoking {demographics.years_since_quit:.0f} years ago (>15 year limit)"
        )

    if demographics.smoking_status == "never":
        eligible = False
        reasons.append("No smoking history")

    return {
        "eligible": eligible,
        "criteria": "USPSTF 2021",
        "reasons": reasons if not eligible else ["Meets all criteria"],
    }
