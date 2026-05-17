import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# ── Dataset ───────────────────────────────────────────────────────────────────
class SkinDataset(Dataset):
    def __init__(self, df, paths, transform=None):
        self.df        = df.reset_index(drop=True)
        self.paths     = paths
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img   = Image.open(self.paths[idx]).convert("RGB")
        label = 1 if self.df.iloc[idx].dx == "mel" else 0
        if self.transform:
            img = self.transform(img)
        return img, label

# ── Transforms ────────────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── Scan & validate images ────────────────────────────────────────────────────
SUBFOLDERS = ["HAM10000_images_part_1", "HAM10000_images_part_2", ""]

def find_image(img_dir, image_id):
    for sf in SUBFOLDERS:
        p = os.path.join(img_dir, sf, f"{image_id}.jpg") if sf else os.path.join(img_dir, f"{image_id}.jpg")
        if os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 melanoma classifier on HAM10000")
    parser.add_argument("--img-dir",    default="data/images",
                        help="Root directory of HAM10000 images (default: data/images)")
    parser.add_argument("--csv",        default="data/images/HAM10000_metadata.csv",
                        help="Path to HAM10000_metadata.csv")
    parser.add_argument("--ckpt-dir",   default="checkpoints",
                        help="Directory to save checkpoints (default: checkpoints)")
    parser.add_argument("--epochs",     type=int,   default=10,   help="Max training epochs")
    parser.add_argument("--patience",   type=int,   default=3,    help="Early-stopping patience (val loss)")
    parser.add_argument("--batch-size", type=int,   default=32,   help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-4, help="Adam learning rate")
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Scanning dataset...")
    df_full = pd.read_csv(args.csv)
    valid_rows, valid_paths, skipped = [], [], 0

    for _, row in df_full.iterrows():
        path = find_image(args.img_dir, row.image_id)
        if path is None:
            skipped += 1
            continue
        try:
            with Image.open(path) as img:
                img.verify()
            valid_rows.append(row)
            valid_paths.append(path)
        except Exception:
            skipped += 1

    if skipped:
        print(f"Skipped {skipped} missing/corrupt images")

    df_valid = pd.DataFrame(valid_rows).reset_index(drop=True)
    labels   = [1 if r.dx == "mel" else 0 for _, r in df_valid.iterrows()]

    # ── Stratified 80/20 split ────────────────────────────────────────────────
    indices = list(range(len(df_valid)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )

    train_df    = df_valid.iloc[train_idx]
    train_paths = [valid_paths[i] for i in train_idx]
    val_df      = df_valid.iloc[val_idx]
    val_paths   = [valid_paths[i] for i in val_idx]

    print(f"Train: {len(train_df)} samples | Val: {len(val_df)} samples")

    # ── Class imbalance: WeightedRandomSampler ────────────────────────────────
    train_labels   = [1 if r.dx == "mel" else 0 for _, r in train_df.iterrows()]
    class_counts   = np.bincount(train_labels)
    class_weights  = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in train_labels]
    sampler        = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    mel_count     = sum(train_labels)
    non_mel_count = len(train_labels) - mel_count
    print(f"Class balance — Melanoma: {mel_count} | Non-Melanoma: {non_mel_count}")

    train_dataset = SkinDataset(train_df, train_paths, transform=train_transforms)
    val_dataset   = SkinDataset(val_df,   val_paths,   transform=val_transforms)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_loader    = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model    = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model    = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn   = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_ctr  = 0
    history       = []

    print("\n── Training ──────────────────────────────────────────────────────────")
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, lbs in train_loader:
            imgs, lbs = imgs.to(device), lbs.to(device)
            preds = model(imgs)
            loss  = loss_fn(preds, lbs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for imgs, lbs in val_loader:
                imgs, lbs = imgs.to(device), lbs.to(device)
                out      = model(imgs)
                val_loss += loss_fn(out, lbs).item() * imgs.size(0)
                probs    = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                all_preds.extend(out.argmax(dim=1).cpu().numpy())
                all_labels.extend(lbs.cpu().numpy())
                all_probs.extend(probs)
        val_loss /= len(val_loader.dataset)

        acc = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)

        print(f"Epoch {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}")
        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4), "acc": round(acc, 4), "auc": round(auc, 4)
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "resnet18_best.pth"))
            print(f"  ✓ Best checkpoint saved (val_loss={val_loss:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # ── Save final model & history ────────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "resnet18_final.pth"))
    with open(os.path.join(args.ckpt_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n── Validation Classification Report ──────────────────────────────────")
    print(classification_report(all_labels, all_preds, target_names=["Non-Melanoma", "Melanoma"]))
    print(f"AUC-ROC : {auc:.4f}")
    print(f"\nBest model  → {os.path.join(args.ckpt_dir, 'resnet18_best.pth')}")
    print(f"Final model → {os.path.join(args.ckpt_dir, 'resnet18_final.pth')}")
    print(f"History     → {os.path.join(args.ckpt_dir, 'training_history.json')}")


if __name__ == "__main__":
    main()
