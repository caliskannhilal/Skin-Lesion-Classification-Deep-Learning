# Skin Lesion Classification — Deep Learning Project 
An end-to-end deep learning project for multi-class skin lesion image classification, developed as part of the Deep Learning (Fall 2025) course. Using a dataset of 27,934 dermoscopic images spanning 8 lesion categories, this project implements and evaluates a baseline CNN, several fine-tuned variants, and a transfer learning model (VGG16).

---

## Project Motivation

Skin cancer detection is clinically important and challenging due to imbalanced data, visually similar lesion types, and limited resolution. This project investigates:

- How CNNs behave on a strongly imbalanced dataset.
- How architecture and regularization affect model generalization.
- Whether transfer learning from ImageNet improves classification accuracy.
- How evaluation metrics such as ROC curves, AUC, and confusion matrices reveal model behavior.

The overall goal is to build a reproducible deep learning workflow and critically analyze baseline, fine-tuned, and transfer learning models.

---

## Workflow

### 1. Data Preparation & Visualization
- Kaggle-based dataset of 27,934 dermoscopic images.
- Images resized to 42×42.
- Noise injection added to training samples only.
- Final split: 64% training, 16% validation, 20% testing.
- Strong class imbalance (Nevus dominates; DF and VASC extremely rare).

### 2. Baseline CNN Architecture
A shallow baseline CNN:

- Conv(16) → MaxPool
- Conv(16) → MaxPool
- Flatten → Dense(128)
- Dense(8, softmax)

Trained for 15 epochs (Adam optimizer, lr=1e-3).

Performance:
- Validation accuracy ~64%
- Test accuracy ~63%
- Validation ROC AUC ~0.5 (near random)
- Test ROC AUC ~0.78–0.95 depending on class

### 3. Baseline Evaluation
Findings:
- Strong predictions for the majority class (NV).
- Very poor recall for minority classes (DF, SCC, VASC).
- Macro F1-score: 0.38
- Weighted F1-score: 0.61

The model heavily favors dominant classes.

### 4. Fine-Tuned CNN Variants
Multiple CNN configurations tested, adjusting:

- Filter depth (up to 256)
- Dense layers (128–512)
- Dropout (0.0–0.5)
- Label smoothing (0.1)
- Kernel sizes (3×3 and 5×5)
- Learning rate schedules
- Data augmentation (flips, rotation, zoom)

Best-performing model: **Variation 4B**
- Test accuracy: 70%
- Weighted F1-score: 0.69
- Macro-AUC: 0.91

Regularization + augmentation proved most effective.

### 5. Transfer Learning (VGG16)
- VGG16 convolutional base frozen.
- Custom classifier: GAP → Dense(256) → Dropout(0.5) → Softmax(8).
- Trained for 15 epochs.

Performance:
- Validation accuracy: 64.08%
- Test accuracy: 64%
- Average AUC: ~0.85

While more stable, VGG16 did not outperform the best tuned CNN due to frozen layers and domain gap.

---

## Results Summary

| Model                   | Test Accuracy | Weighted F1 | Macro AUC |
|------------------------|--------------|-------------|-----------|
| Baseline CNN           | ~63%         | ~0.61       | ~0.88     |
| Fine-Tuned (Var. 4B)   | **70%**      | **0.69**    | **0.91**  |
| VGG16 Transfer Learning| ~64%         | 0.60        | ~0.85     |

---

## Key Features

- Full deep learning experimentation pipeline: data loading, preprocessing, training, tuning, and evaluation.
- Multi-model comparison (baseline vs. fine-tuned vs. transfer learning).
- Robust evaluation metrics: confusion matrices, ROC curves, F1-scores.
- Documentation includes a detailed report covering risks, ethics, and literature review.
- Group collaboration with clearly defined roles.

---

## Project Structure


DeepL_Group12Assign.pdf                    # Full assignment report
notebooks_Group_12_Deep_Learning_Assignment.py   # Complete training/evaluation pipeline
README.md                                  # Project documentation

---

## Limitations and Future Work

- Class imbalance harms minority class recall significantly.
- Small 42×42 resolution makes fine-grained lesion patterns hard to capture.
- No class-weighted or focal loss applied; these could improve performance.
- Future improvements:
  - Fine-tune VGG16 instead of freezing all layers.
  - Experiment with ResNet/EfficientNet architectures.
  - Apply oversampling, SMOTE, or synthetic dermoscopic augmentations.
  - Use cross-validation to stabilize metrics.

---

## Authors

- Huijuan Lin — Data Preparation & Visualization  
- Dora Grozdanova — Baseline Model  
- Hilal Caliskan Egilli — Baseline Evaluation & Analysis  
- Gala Troncoso Tapia — Fine-Tuning  
- Jin Liu — Transfer Learning  
- Violeta Sandu — Literature Review & Ethics
