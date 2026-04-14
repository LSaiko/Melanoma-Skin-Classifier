import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# ── Config ────────────────────────────────────────────────────────────────────
IMG_DIR    = "C:/Users/Admin/data/images"
CSV_PATH   = "C:/Users/Admin/data/images/HAM10000_metadata.csv"
CKPT_DIR   = "C:/Users/Admin/data/checkpoints"
SUBFOLDERS = ["HAM10000_images_part_1", "HAM10000_images_part_2", ""]
NUM_EPOCHS = 10
PATIENCE   = 3
BATCH_SIZE = 32
LR         = 1e-4

os.makedirs(CKPT_DIR, exist_ok=True)

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
def find_image(image_id):
    for sf in SUBFOLDERS:
        p = f"{IMG_DIR}/{sf}/{image_id}.jpg" if sf else f"{IMG_DIR}/{image_id}.jpg"
        if os.path.exists(p):
            return p
    return None

print("Scanning dataset...")
df_full = pd.read_csv(CSV_PATH)
valid_rows, valid_paths, skipped = [], [], 0

for _, row in df_full.iterrows():
    path = find_image(row.image_id)
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

# ── Stratified 80/20 split ────────────────────────────────────────────────────
indices          = list(range(len(df_valid)))
train_idx, val_idx = train_test_split(
    indices, test_size=0.2, stratify=labels, random_state=42
)

train_df    = df_valid.iloc[train_idx]
train_paths = [valid_paths[i] for i in train_idx]
val_df      = df_valid.iloc[val_idx]
val_paths   = [valid_paths[i] for i in val_idx]

print(f"Train: {len(train_df)} samples | Val: {len(val_df)} samples")

# ── Class imbalance: WeightedRandomSampler ────────────────────────────────────
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
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# ── Model ─────────────────────────────────────────────────────────────────────
model    = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: melanoma / non-melanoma

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn   = nn.CrossEntropyLoss()

# ── Training loop ─────────────────────────────────────────────────────────────
best_val_loss = float("inf")
patience_ctr  = 0
history       = []

print("\n── Training ──────────────────────────────────────────────────────────")
for epoch in range(NUM_EPOCHS):
    # Train
    model.train()
    train_loss = 0.0
    for imgs, lbs in train_loader:
        preds = model(imgs)
        loss  = loss_fn(preds, lbs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_loader.dataset)

    # Validate
    model.eval()
    val_loss                      = 0.0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, lbs in val_loader:
            out      = model(imgs)
            val_loss += loss_fn(out, lbs).item() * imgs.size(0)
            probs    = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_labels.extend(lbs.numpy())
            all_probs.extend(probs)
    val_loss /= len(val_loader.dataset)

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"Epoch {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}")
    history.append({
        "epoch": epoch, "train_loss": round(train_loss, 4),
        "val_loss": round(val_loss, 4), "acc": round(acc, 4), "auc": round(auc, 4)
    })

    # Best checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_ctr  = 0
        torch.save(model.state_dict(), f"{CKPT_DIR}/resnet18_best.pth")
        print(f"  ✓ Best checkpoint saved (val_loss={val_loss:.4f})")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

# ── Save final model & history ────────────────────────────────────────────────
torch.save(model.state_dict(), f"{CKPT_DIR}/resnet18_final.pth")
with open(f"{CKPT_DIR}/training_history.json", "w") as f:
    json.dump(history, f, indent=2)

# ── Final report ──────────────────────────────────────────────────────────────
print("\n── Validation Classification Report ──────────────────────────────────")
print(classification_report(all_labels, all_preds, target_names=["Non-Melanoma", "Melanoma"]))
print(f"AUC-ROC : {auc:.4f}")
print(f"\nBest model  → {CKPT_DIR}/resnet18_best.pth")
print(f"Final model → {CKPT_DIR}/resnet18_final.pth")
print(f"History     → {CKPT_DIR}/training_history.json")
