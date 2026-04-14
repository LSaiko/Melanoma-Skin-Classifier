# Melanoma Skin Lesion Classifier

A fine-tuned **ResNet-18** classifier trained on the [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) dermoscopy dataset. Given a skin lesion image, the model predicts **Melanoma vs. Non-Melanoma** and produces a visual analysis report with confidence percentages, a Grad-CAM heatmap, and a lesion bounding box.

---

## Project Structure

```
data/
├── Training_Model.py       # Full training pipeline
├── Inference.py            # Single-image analysis with visual report
├── Augmentation.py         # Transform definitions
├── Load_Image.py           # Dataset class
├── Load_PreTrained.py      # Model loading utilities
├── checkpoints/
│   ├── resnet18_best.pth       # Best checkpoint (lowest val loss)
│   ├── resnet18_final.pth      # Final epoch checkpoint
│   └── training_history.json  # Per-epoch metrics
├── results/
│   └── <image>_analysis.png   # Inference reports
└── images/
    ├── HAM10000_metadata.csv
    ├── HAM10000_images_part_1/
    └── HAM10000_images_part_2/
```

---

## Training

### Features
| Feature | Detail |
|---|---|
| Base model | ResNet-18 (ImageNet pretrained) |
| Classes | Melanoma (`mel`) · Non-Melanoma (all other dx) |
| Dataset split | 80 % train / 20 % validation (stratified) |
| Class imbalance | `WeightedRandomSampler` (inverse class frequency) |
| Augmentation | Flip, rotation ±20°, colour jitter |
| Optimiser | Adam — lr `1e-4` |
| Loss | CrossEntropyLoss |
| Early stopping | Patience = 3 epochs (monitors val loss) |
| Metrics | Accuracy, AUC-ROC, Precision, Recall, F1 |

### Run Training
```bash
python Training_Model.py
```

### Training Results (Run 1 — full dataset, 10 epochs, no val split)
| Epoch | Loss |
|---|---|
| 0 | 0.1241 |
| 1 | 0.1793 |
| 2 | 0.1919 |
| 3 | 0.1498 |
| 4 | 0.1283 |
| 5 | 0.0774 |
| **6** | **0.0233** ← best |
| 7 | 0.0681 |
| 8 | 0.0804 |
| 9 | 0.1809 |

> **Note:** Run 1 used training loss only (no val split). The updated `Training_Model.py` now uses a stratified 80/20 split and reports both train loss and val loss + AUC per epoch.

---

## Inference — Submitting a Skin Lesion Image

```bash
python Inference.py --image path/to/lesion.jpg
```

Optional arguments:
```
--checkpoint   Path to .pth model file (default: checkpoints/resnet18_best.pth)
--output       Path to save the report PNG (default: results/<name>_analysis.png)
```

### What the Report Shows

The output is a **4-panel PNG report**:

| Panel | Description |
|---|---|
| **Original Image** | Input lesion resized to 224×224 |
| **Grad-CAM Heatmap** | Jet-coloured overlay — red/yellow regions most influenced the prediction |
| **Lesion Detection** | Original image with a **bounding box** drawn around the highest-activation region, labelled with prediction + % |
| **Diagnosis** | Prediction verdict · Melanoma % bar · Non-Melanoma % bar · Confidence score |

### Example Console Output
```
══════════════════════════════════════════════
  Skin Lesion Analysis Report
══════════════════════════════════════════════
  Image        : ISIC_0024306.jpg
  Prediction   : Melanoma
  Melanoma     : 87.3%
  Non-Melanoma : 12.7%
  Confidence   : 87.3%  (High Confidence)
  Bounding Box : x1=48 y1=52 x2=178 y2=191
  Report saved : results/ISIC_0024306_analysis.png
══════════════════════════════════════════════

⚕  Results are for medical reference only.
```

### Bounding Box
The bounding box is derived from the **Grad-CAM activation map**: the heatmap is thresholded at 45 %, the largest contour is found with OpenCV, and `cv2.boundingRect` returns `[x1, y1, x2, y2]` coordinates on the 224×224 image space.

---

## Dependencies

```bash
pip install torch torchvision pillow pandas numpy scikit-learn opencv-python matplotlib
```

---

## Dataset

**HAM10000** — Human Against Machine with 10000 training images  
Tschandl, P. et al. (2018). *The HAM10000 dataset.* Scientific Data.  
Download: https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection

---

> ⚕ This tool is intended for research and educational purposes only and does not constitute medical advice.
