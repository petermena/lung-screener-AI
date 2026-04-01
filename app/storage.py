from __future__ import annotations

from pathlib import Path
from typing import Iterable
import uuid

from app.settings import DATA_ROOT


def ensure_data_root() -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return DATA_ROOT


def create_study_dir() -> Path:
    root = ensure_data_root()
    study_id = str(uuid.uuid4())
    study_dir = root / study_id
    study_dir.mkdir(parents=True, exist_ok=False)
    return study_dir


def list_studies() -> list[str]:
    root = ensure_data_root()
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def save_uploaded_files(study_dir: Path, files: Iterable[tuple[str, bytes]]) -> int:
    count = 0
    for name, payload in files:
        if not name.lower().endswith(".dcm"):
            continue
        (study_dir / name).write_bytes(payload)
        count += 1
    return count
