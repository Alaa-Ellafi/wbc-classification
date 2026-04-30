# WBC Classification from Blood Smear Images

**13-class white blood cell classification** from microscopy images using deep learning.  
Course challenge — IMA205, Télécom Paris · 2026  
**Best macro F1-score: 0.78** on the full dataset (private leaderboard)

---

## Problem

White blood cell (WBC) morphology is used to diagnose diseases such as leukaemia, infections, and anaemia. Manual classification by haematologists is time-consuming and subjective. This project builds a CNN-based classifier to automate WBC classification across **13 cell types** from stained blood smear microscopy images.

**Classes:** SNE · LY · MO · EO · BA · VLY · BNE · MMY · MY · PMY · BL · PC · PLY

**Dataset:** 28,901 images, 70% train/val — 30% test (labels withheld). Evaluated on macro-averaged F1 to account for class imbalance.

---

## Approach

Three notebooks document an iterative model comparison, all using ImageNet-pretrained backbones (no medical pre-training allowed per competition rules):

| Notebook | Models | Notes |
|----------|--------|-------|
| `01_efficientnet_convnext.ipynb` | EfficientNet · ConvNeXt | Baseline comparison, data augmentation pipeline |
| `02_convnext_small.ipynb` | ConvNeXt-Small | Hyperparameter tuning, LR scheduling |
| `03_convnext_base.ipynb` | ConvNeXt-Base | Final model, confidence thresholding at inference |

**Key design choices:**
- Stain normalisation and aggressive augmentation to handle microscopy variability
- Macro F1 as the optimisation target throughout (not accuracy)
- Post-processing with confidence thresholding to handle rare classes (BL, PC, PLY)
- ConvNeXt-Base as final model — best performance vs. EfficientNet on this dataset

---


## Stack

- Python · PyTorch · torchvision
- ConvNeXt · EfficientNet (ImageNet pretrained)
- NumPy · Pandas · Scikit-learn · Matplotlib

---

## Structure

```
wbc-classification/
├── notebooks/
│   ├── 01_efficientnet_convnext.ipynb
│   ├── 02_convnext_small.ipynb
│   └── 03_convnext_base.ipynb
├── report.pdf
└── requirements.txt
```

---

## Report

A full write-up of the methodology, architecture choices, results analysis, and error analysis is available in [`report.pdf`](./report.pdf).

---

## Reference

Maxime Di Folco, Pietro Gori. *IMA205 Challenge 2026*, Kaggle, 2026.
