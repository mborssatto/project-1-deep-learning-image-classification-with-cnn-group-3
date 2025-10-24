# CIFAR-10 Image Classification with VGG16 (Transfer Learning)

This notebook trains and evaluates an image classifier on **CIFAR-10** using **VGG16** with transfer learning. It includes data preprocessing, light augmentation, a frozen-backbone phase, and optional fine-tuning of the top VGG blocks. Plots and metrics (accuracy, confusion matrix) are generated for the report/presentation.

> If you prefer another backbone (e.g., EfficientNetB0 or MobileNetV2), see **“Swap the backbone”** below.

---

## ✅ What this notebook does

- Loads **CIFAR-10** (60k images, 10 classes).
- Splits into **train/val/test**.
- **Normalizes** pixels and applies **light augmentation**.
- Builds **VGG16** (`include_top=False`, ImageNet weights) + small classifier head.
- Trains in two phases:
  1) **Feature extraction** (backbone frozen).  
  2) **Fine-tuning** (unfreeze top layers with low LR).
- Evaluates with accuracy, per-class metrics, and a confusion matrix.
- Saves the best model (`.keras` format).

---

## 📦 Requirements

- Python 3.9+
- TensorFlow 2.11+ (or 2.14+ recommended)
- NumPy, Matplotlib, scikit-learn

```bash
pip install "tensorflow>=2.11" numpy matplotlib scikit-learn
```

> GPU highly recommended (CUDA/cuDNN set up if using NVIDIA).

---

## 🚀 Quick start

1. **Clone / open** the project and launch Jupyter / Colab.
2. Open **`BEST_VGG16_mainTL_test.ipynb`**.
3. Run all cells in order.

The notebook will:
- Download CIFAR-10 automatically.
- Train the model (feature extraction → fine-tuning).
- Print validation/test accuracy and show plots.

---

## 🧱 Notebook structure (what each section does)

1. **Imports & config**  
   Set random seeds, image size, batch size, and paths.

2. **Data loading & split**  
   Load CIFAR-10 → create train/val/test splits (e.g., 80/10/10).

3. **Preprocessing & augmentation**  
   - Resize **32×32 → 224×224** (VGG16 expects 224).  
   - Use `tf.keras.applications.vgg16.preprocess_input`.  
   - Apply light aug: `RandomFlip`, small `RandomRotation`, `RandomZoom`.

4. **Model: VGG16 backbone + head**  
   - `VGG16(include_top=False, weights="imagenet")`, `GlobalAveragePooling2D`, `Dropout`, and a `Dense(10, softmax)` head.

5. **Training phase 1 — Feature extraction**  
   - Freeze backbone, train head with `Adam(lr=1e-3)`.  
   - Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`.

6. **Training phase 2 — Fine-tuning**  
   - Unfreeze last **N** conv layers, recompile with **lower LR** (e.g., `1e-5`).  
   - Train a few more epochs.

7. **Evaluation & visualization**  
   - Validation & test accuracy.  
   - `classification_report` & confusion matrix.  
   - Learning curves (train vs val).

8. **Save model**  
   - Saves best checkpoint as `vgg16_cifar10.keras`.

---

## 📈 Expected results (fill with your run)

- **Val accuracy (feature extraction):** ~`XX%`
- **Val accuracy (fine-tuned):** ~`YY%`
- **Test accuracy (best):** ~`ZZ%`
- Common confusions: e.g., **cat ↔ dog**, **truck ↔ automobile**

> Replace `XX/YY/ZZ` with your actual numbers after running.

---

## 🔁 Swap the backbone (optional)

If you want native **32×32** support (no 224 resize) or a lighter model:

- **EfficientNetB0**
  ```python
  from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
  base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(32,32,3), pooling="avg")
  ```
- **MobileNetV2**
  ```python
  from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
  base = MobileNetV2(include_top=False, weights="imagenet", input_shape=(32,32,3), pooling="avg")
  ```

Keep the rest (head, callbacks, two-phase training) the same; just switch the `preprocess_input` and `input_shape`.

---

## 🔧 Hyperparameters you can tweak

- **Augmentation**: keep light for tiny images (rotation ≤ 0.1, zoom ≤ 0.1).
- **Dropout**: 0.3–0.5 after GAP helps generalization.
- **Unfreeze depth**: last 4–12 conv layers; lower LR (`1e-5 … 3e-6`).
- **Batch size**: compare 32 vs 64 (32 sometimes generalizes better).
- **Optimizer**: `Adam` or `AdamW` (if available).

---

## 🧪 Using a different dataset (e.g., Animals-10)

Point `image_dataset_from_directory` to your folder and keep the same pipeline. If class folders are in Italian (e.g., `cane`, `gatto`), training is fine; translate labels only for plots/reports.

---

## 🛠 Troubleshooting

- **Shape mismatch**: VGG16 needs **224×224×3** → ensure resize and `preprocess_input`.  
- **`compile()` error**: Re-compile after changing `trainable` flags.  
- **Overfitting**: Add/raise `Dropout`, use light aug, early stopping, or unfreeze fewer layers.  
- **Underfitting**: Train longer, unfreeze more top layers, or increase LR slightly.

---

## 📚 References

- **CIFAR-10**: Krizhevsky, 2009  
- **VGG16**: Simonyan & Zisserman, 2014  
- `tf.keras.applications.vgg16.preprocess_input`  
- Keras docs: callbacks (`EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`)

---

## 📄 License & Acknowledgments

- Dataset: **CIFAR-10** (University of Toronto).  
- Pretrained weights: **ImageNet**.  
- Code built on **TensorFlow / Keras**.  
- Use under your course/project license.
