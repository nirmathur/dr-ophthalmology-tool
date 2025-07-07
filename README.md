
# Diabetic Retinopathy Detection (Image-Based)

This repository uses **EfficientNetB3 + Grad-CAM** to classify retinal images from the APTOS 2019 blindness detection dataset.

## 🔧 Setup
Update these paths in `config.py`:
- TRAIN_IMG_DIR
- TEST_IMG_DIR
- TRAIN_CSV_PATH
- TEST_CSV_PATH

## 🚀 Run

```bash
python main.py
```

## 🧠 Highlights
- EfficientNetB3 with frozen base
- Mixed precision training
- Augmented image generators
- Grad-CAM explainability
