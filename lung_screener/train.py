from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from lung_screener.data import load_volume, preprocess_volume
from lung_screener.model import ModelConfig, build_model


class CTDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        volume = load_volume(Path(row.scan_path))
        x = preprocess_volume(volume)
        y = torch.tensor(int(row.label), dtype=torch.long)
        return x, y


def split_frame(frame: pd.DataFrame, val_size: float, no_stratify: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if val_size <= 0:
        print("Validation split disabled (--val-size <= 0). Using training set for validation metrics.")
        return frame, frame.copy()

    class_counts = frame["label"].value_counts()
    has_min_for_stratify = not class_counts.empty and class_counts.min() >= 2
    val_count = max(1, int(round(len(frame) * val_size)))
    enough_val_for_classes = val_count >= frame["label"].nunique()

    use_stratify = (not no_stratify) and has_min_for_stratify and enough_val_for_classes

    if use_stratify:
        return train_test_split(frame, test_size=val_size, random_state=42, stratify=frame["label"])

    print(
        "Falling back to non-stratified split. "
        "If you want stratification, ensure >=2 samples per class and enough validation samples."
    )
    return train_test_split(frame, test_size=val_size, random_state=42, stratify=None)


def run_training(
    csv_path: Path,
    output_dir: Path,
    backbone: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    val_size: float,
    no_stratify: bool,
) -> None:
    frame = pd.read_csv(csv_path)
    if len(frame) < 2:
        raise ValueError("Need at least 2 rows in the CSV to run training.")

    train_df, val_df = split_frame(frame, val_size=val_size, no_stratify=no_stratify)

    train_loader = DataLoader(CTDataset(train_df), batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(CTDataset(val_df), batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model(ModelConfig(backbone=backbone)).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "backbone": backbone,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
            }
            torch.save(checkpoint, output_dir / "best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lung cancer CT screening classifier")
    parser.add_argument("--csv", type=Path, required=True, help="CSV with columns: scan_path,label")
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--backbone", type=str, default="medicalnet_resnet18")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split ratio. Use 0 to disable split.")
    parser.add_argument("--no-stratify", action="store_true", help="Disable stratified splitting.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        csv_path=args.csv,
        output_dir=args.output,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        val_size=args.val_size,
        no_stratify=args.no_stratify,
    )
