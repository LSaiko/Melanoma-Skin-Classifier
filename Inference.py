# -*- coding: utf-8 -*-
"""
Inference.py — Melanoma Skin Lesion Classifier
═══════════════════════════════════════════════
Accepts a single skin lesion image and outputs:
  • Melanoma / Non-Melanoma prediction with percentage confidence
  • Grad-CAM heatmap highlighting the region driving the prediction
  • Bounding box drawn around the detected lesion area
  • Full analysis report saved as a PNG

Usage:
    python Inference.py --image path/to/lesion.jpg
    python Inference.py --image path/to/lesion.jpg --checkpoint path/to/resnet18_best.pth
    python Inference.py --image path/to/lesion.jpg --output path/to/report.png
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

LABELS      = ["Non-Melanoma", "Melanoma"]
CKPT_PATH   = "C:/Users/Admin/data/checkpoints/resnet18_best.pth"
RESULTS_DIR = "C:/Users/Admin/data/results"

# ── Grad-CAM (ResNet18 — hooks into layer4) ───────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model  = model
        self._acts  = None
        self._grads = None
        target = model.layer4[-1]
        target.register_forward_hook(self._save_acts)
        target.register_full_backward_hook(self._save_grads)

    def _save_acts(self, module, inp, out):
        self._acts = out.detach()

    def _save_grads(self, module, grad_in, grad_out):
        self._grads = grad_out[0].detach()

    def generate(self, tensor, class_idx):
        """Return a (224, 224) float32 heatmap in [0, 1]."""
        self.model.zero_grad()
        out = self.model(tensor)
        out[0, class_idx].backward()

        weights = self._grads[0].mean(dim=(1, 2), keepdim=True)   # (C,1,1)
        cam     = F.relu((weights * self._acts[0]).sum(dim=0))     # (H,W)
        cam     = cam.cpu().numpy()
        cam     = cv2.resize(cam, (224, 224))
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam.astype(np.float32)


def get_bounding_box(cam, threshold=0.45):
    """Derive the tightest bounding box enclosing the hot region."""
    mask      = (cam >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return x, y, x + w, y + h


def build_report(orig_np, overlay_np, bbox_img_np,
                 probs, pred_class, bbox, image_name, output_path):
    """Compose and save the 4-panel analysis report."""
    mel_pct     = probs[1] * 100
    non_mel_pct = probs[0] * 100
    confidence  = max(probs) * 100
    conf_label  = ("High Confidence"     if confidence > 85 else
                   "Moderate Confidence" if confidence > 65 else
                   "Low Confidence")
    verdict_color = "#ef4444" if pred_class == 1 else "#22c55e"
    verdict_text  = "⚠  MELANOMA DETECTED" if pred_class == 1 else "✓  NON-MELANOMA"

    fig = plt.figure(figsize=(15, 5.5), facecolor="#111827")
    fig.suptitle(
        f"Skin Lesion Analysis  ·  {image_name}",
        fontsize=14, fontweight="bold", color="white", y=0.97
    )
    gs = GridSpec(1, 4, figure=fig, wspace=0.30, left=0.03, right=0.97)

    # ── Panels 0-2: image views ───────────────────────────────────────────────
    panel_data = [
        ("Original Image",    orig_np),
        ("Grad-CAM Heatmap",  overlay_np),
        ("Lesion Detection",  bbox_img_np),
    ]
    for col, (title, img) in enumerate(panel_data):
        ax = fig.add_subplot(gs[col])
        ax.imshow(img)
        ax.set_title(title, color="#d1d5db", fontsize=10, pad=6)
        ax.axis("off")
        ax.set_facecolor("#111827")
        for sp in ax.spines.values():
            sp.set_edgecolor("#374151")

    # ── Panel 3: diagnosis ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    ax4.set_facecolor("#111827")
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis("off")
    ax4.set_title("Diagnosis", color="#d1d5db", fontsize=10, pad=6)

    # Verdict
    ax4.text(0.5, 0.88, verdict_text, ha="center", va="center",
             fontsize=12, fontweight="bold", color=verdict_color)

    # Melanoma bar
    ax4.barh(0.72, 0.88, height=0.07, left=0.06, color="#374151")
    ax4.barh(0.72, 0.88 * mel_pct / 100, height=0.07, left=0.06, color="#ef4444")
    ax4.text(0.5, 0.79, f"Melanoma         {mel_pct:.1f}%",
             ha="center", color="#ef4444", fontsize=9)

    # Non-melanoma bar
    ax4.barh(0.58, 0.88, height=0.07, left=0.06, color="#374151")
    ax4.barh(0.58, 0.88 * non_mel_pct / 100, height=0.07, left=0.06, color="#22c55e")
    ax4.text(0.5, 0.65, f"Non-Melanoma  {non_mel_pct:.1f}%",
             ha="center", color="#22c55e", fontsize=9)

    # Confidence
    ax4.text(0.5, 0.43, f"Confidence: {confidence:.1f}%",
             ha="center", color="white", fontsize=10, fontweight="bold")
    ax4.text(0.5, 0.34, conf_label,
             ha="center", color="#9ca3af", fontsize=9)

    # Bounding box coords
    if bbox:
        x1, y1, x2, y2 = bbox
        ax4.text(0.5, 0.22, f"Lesion region:  [{x1},{y1}] → [{x2},{y2}]",
                 ha="center", color="#6b7280", fontsize=7.5)

    # Disclaimer
    ax4.text(0.5, 0.06, "⚕  For medical reference only",
             ha="center", color="#4b5563", fontsize=7.5)

    # Border accent
    for sp in ax4.spines.values():
        sp.set_edgecolor("#374151")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def run_inference(image_path, checkpoint_path=CKPT_PATH, output_path=None):
    # ── Load model ────────────────────────────────────────────────────────────
    model    = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()

    gradcam = GradCAM(model)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ── Load & preprocess image ───────────────────────────────────────────────
    orig        = Image.open(image_path).convert("RGB")
    orig_resized = orig.resize((224, 224))
    orig_np     = np.array(orig_resized)
    tensor      = preprocess(orig).unsqueeze(0).requires_grad_(True)

    # ── Predict ───────────────────────────────────────────────────────────────
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].numpy()
    pred_class = int(np.argmax(probs))

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    tensor_grad = preprocess(orig).unsqueeze(0).requires_grad_(True)
    cam  = gradcam.generate(tensor_grad, pred_class)
    bbox = get_bounding_box(cam, threshold=0.45)

    # ── Heatmap overlay ───────────────────────────────────────────────────────
    heatmap_bgr = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay_np  = cv2.addWeighted(orig_np, 0.55, heatmap_rgb, 0.45, 0)

    # ── Bounding box annotation ───────────────────────────────────────────────
    bbox_img = orig_np.copy()
    if bbox:
        x1, y1, x2, y2 = bbox
        color     = (239, 68, 68) if pred_class == 1 else (34, 197, 94)
        label_str = f"{LABELS[pred_class]}  {probs[pred_class]*100:.1f}%"
        cv2.rectangle(bbox_img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(bbox_img, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(bbox_img, label_str,
                    (x1 + 2, max(y1 - 4, th)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Build & save report ───────────────────────────────────────────────────
    image_name = os.path.basename(image_path)
    if output_path is None:
        base       = os.path.splitext(image_name)[0]
        output_path = f"{RESULTS_DIR}/{base}_analysis.png"

    build_report(orig_np, overlay_np, bbox_img,
                 probs, pred_class, bbox, image_name, output_path)

    # ── Console summary ───────────────────────────────────────────────────────
    mel_pct    = probs[1] * 100
    non_mel_pct = probs[0] * 100
    conf_label  = ("High Confidence"     if max(probs) > 0.85 else
                   "Moderate Confidence" if max(probs) > 0.65 else
                   "Low Confidence")
    print(f"\n{'═'*46}")
    print(f"  Skin Lesion Analysis Report")
    print(f"{'═'*46}")
    print(f"  Image        : {image_name}")
    print(f"  Prediction   : {LABELS[pred_class]}")
    print(f"  Melanoma     : {mel_pct:.1f}%")
    print(f"  Non-Melanoma : {non_mel_pct:.1f}%")
    print(f"  Confidence   : {max(probs)*100:.1f}%  ({conf_label})")
    if bbox:
        print(f"  Bounding Box : x1={bbox[0]} y1={bbox[1]} x2={bbox[2]} y2={bbox[3]}")
    print(f"  Report saved : {output_path}")
    print(f"{'═'*46}\n")
    print("⚕  Results are for medical reference only.")

    return {"prediction": LABELS[pred_class], "melanoma_pct": mel_pct,
            "non_melanoma_pct": non_mel_pct, "confidence": max(probs) * 100,
            "bounding_box": bbox, "report_path": output_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Melanoma skin lesion classifier with Grad-CAM & bounding box"
    )
    parser.add_argument("--image",      required=True,
                        help="Path to the skin lesion image (JPG/PNG)")
    parser.add_argument("--checkpoint", default=CKPT_PATH,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--output",     default=None,
                        help="Output path for the analysis report PNG")
    args = parser.parse_args()
    run_inference(args.image, args.checkpoint, args.output)
