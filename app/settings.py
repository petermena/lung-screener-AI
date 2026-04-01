from pathlib import Path
import os

DATA_ROOT = Path(os.getenv("LUNG_SCREENER_DATA_ROOT", "data/studies"))
MODEL_NAME = os.getenv("LUNG_SCREENER_MODEL_NAME", "lung_nodule_ct_detection")
MODEL_BUNDLE_DIR = Path(os.getenv("LUNG_SCREENER_MODEL_BUNDLE_DIR", "data/model_bundles"))
MODEL_AUTO_DOWNLOAD = os.getenv("LUNG_SCREENER_MODEL_AUTO_DOWNLOAD", "true").lower() in {"1", "true", "yes"}
DEVICE = os.getenv("LUNG_SCREENER_DEVICE", "cuda")
API_TITLE = "Lung CT Screener API"
