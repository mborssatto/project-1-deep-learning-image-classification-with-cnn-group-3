# 🧠 AI Bootcamp Project — Image Classification 

### Team G3 — Mariana • Cristina • Adrian • Kira  

---

## 📘 Executive Summary

This project explores **image classification** using the **CIFAR-10 dataset**, containing 60,000 color images (32×32 pixels) divided into 10 categories (e.g., airplane, car, cat, dog).  

After testing 19 model variations, the **winning model** used **Transfer Learning with VGG16** as a base model and achieved an **accuracy of 86%** on the test set.

---
## 📂 Project files


| File / Folder | Description |
|----------------|--------------|
| 📂 **history** | Contains unpolished files with different models we tried |
| **Group 3 week 3 presentation.pdf** | Final presentation summarizing project results |
| **README.md** | Project overview and documentation (this file) |
| **Team internal results.xlsx** | Unpolished file used internally to track our learnings |
| **main BEST_VGG16.ipynb** | Final notebook with our winning **Trainer Learning** model 👈 *start here* |
| **main inhouse CNN.ipynb** | Final notebook with our in-house **CNN model** |

---

## 🚀 Project Overview

| Model | Approach | Accuracy | Notes |
|--------|-----------|-----------|-------|
| **Baseline CNN** | Custom convolutional model | ~69% | Initial version |
| **Optimized CNN** | Added batch normalization, dropout, early stopping | ~75–80% | Reduced overfitting |
| **Transfer Learning (VGG16)** | Fine-tuned pretrained model | **86%** | Best overall performance |

**Trade-off:** Transfer learning required more preprocessing and tuning, but significantly improved performance and training stability.

---

## 🧩 Dataset & Preprocessing

**Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)  
- 60,000 images (50,000 train / 10,000 test)  
- 10 classes, RGB format  

**Steps:**
1. Normalized images to `[0, 1]` range.  
2. Converted labels into categorical (one-hot encoded).  
3. Split data into training, validation, and test sets (80/10/10).  
4. Avoided excessive data augmentation (caused distortions on small images).  

---

## 🤖 Models & Learning Process

### 1️⃣ Baseline CNN
- **Layers:** Convolution → MaxPooling → Dense  
- **Results:** Validation accuracy ≈ 69%  
- Served as the foundation for later experiments.

### 2️⃣ Improved CNN
- **Changes tested:**
  - Added **batch normalization** → improved convergence  
  - Added **dropout layers** → reduced overfitting  
  - Increased epochs (15 → 100) with **early stopping**  
- **Results:** More stable validation accuracy across datasets.

### 3️⃣ Transfer Learning (Winning Model)
- **Base model:** `VGG16` (pretrained on ImageNet)  
- **Additional layers:** Global Average Pooling, Batch Normalization, Dropout  
- **Training setup:**  
  - Fine-tuned top layers only  
  - Used Adam optimizer and early stopping  
- **Results:**  
  - Validation loss: 0.34  
  - Validation accuracy: **86.15%**

---

## 📊 Evaluation & Learnings

**What worked:**
- Batch normalization and dropout significantly improved performance.  
- Early stopping helped avoid overfitting.  
- Transfer learning drastically improved accuracy and reduced training time.  

**What didn’t:**
- Data augmentation decreased accuracy due to image distortion.  
- Not all pretrained models (MobileNetV2, EfficientNetB0) performed equally well.  

**Key Learnings:**
- Iterative experimentation—changing one parameter at a time—was essential.  
- Proper model selection (VGG16) can yield substantial gains even with small datasets.  

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:**  
  - TensorFlow / Keras  
  - NumPy  
  - Pandas  
  - Matplotlib / Seaborn  

---

## 👩‍💻 Team

**Mariana Borssatto**  
**Cristina Insignares**  
**Adrián A. H.**  
**Kira Redberg**
