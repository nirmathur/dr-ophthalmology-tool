
# Diabetic Retinopathy Detection (Image-Based)

This repository uses **EfficientNetB3 + Grad-CAM** to classify retinal images from the APTOS 2019 blindness detection dataset.

## Dataset
Download the [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection/data) dataset from Kaggle.
Extract it so that the folder structure looks like:

```
data/
└── aptos2019-blindness-detection/
    ├── train.csv
    └── train_images/
        └── ...
```

Place the dataset under `data/aptos2019-blindness-detection` so the paths in `src/config.py` work out of the box.
The training script performs a sanity check at startup and will raise helpful
errors if `train.csv` or any of the referenced images are missing.

## Setup
Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All important paths are defined in `src/config.py`. Adjust them only if you store the dataset in a different location.
Training outputs will be written to `model_output/` which is created automatically.

## Run

```bash
python main.py
```

## Highlights
- EfficientNetB3 with frozen base and optional fine-tuning
- L2 regularization and dropout to reduce overfitting
- Mixed precision training
- Augmented image generators
- Automatic dataset sanity checks and class-weighted training
- TensorBoard logs under `logs/`
- Grad-CAM explainability

After the first training phase the last 20 layers of EfficientNetB3 are unfrozen and
the model trains a few extra epochs with a smaller learning rate. Training and
validation metrics are written to `logs/<timestamp>/training_log.csv` so you can
visualize them in TensorBoard.

To evaluate a saved model on a separate test set, run:

```bash
python -m src.eval_model
```

## Reproducibility
Random seeds for NumPy, Python and TensorFlow are set via `src/utils.set_seeds` which is called inside `main.py`.
