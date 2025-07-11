
# Diabetic Retinopathy Detection (Image-Based)

This project started as a competition submission, but it's grown into something I believe actually fills a gap in how we build AI for healthcare.

It’s a lightweight, modular pipeline for diabetic retinopathy detection — built not for the cloud, but for real clinicians, students, and researchers who want to understand what their model is doing. It trains fast, runs locally, and shows its work through Grad-CAM explainability.

Most DR models today are either black-box systems trained on massive datasets with enormous compute, or academic projects that never leave the paper. This tool is neither. It's designed to be used, tweaked, criticised, and trusted — especially by those in under-resourced settings who don’t have the infrastructure to fine-tune Google-scale models.

Millions of people in rural or underserved regions still lose their vision due to late DR detection. This repo is a bet that more transparent, portable, and clinician-guided tools could help change that. It's not trying to beat state-of-the-art — it's trying to democratise it.

Built with EfficientNetB3 and Grad-CAM, it’s ready to run on the APTOS 2019 dataset — or any local dataset you want to try it on.

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
Training outputs will be written to `model_output/`, which is created automatically.

## Run

```bash
python main.py
```

## Highlights
- EfficientNetB3 with frozen base and optional fine-tuning
- L2 regularisation and dropout to reduce overfitting
- Mixed precision training
- Augmented image generators
- Automatic dataset sanity checks and class-weighted training
- TensorBoard logs under `logs/`
- Grad-CAM explainability

After the first training phase, the last 20 layers of EfficientNetB3 are unfrozen, and
the model trains a few extra epochs with a smaller learning rate. Training and
validation metrics are written to `logs/<timestamp>/training_log.csv` so you can
visualise them in TensorBoard.

To evaluate a saved model on a separate test set, run:

```bash
python -m src.eval_model
```

### Grad-CAM Visualisation

Generate Grad-CAM overlays for arbitrary images or entire folders:

```bash
python -m src.generate_gradcam path/to/image1.png path/to/folder --output gradcam_output
```

### Report Generation

After running `eval_model`, you can compile an HTML (or PDF) report combining the evaluation metrics and Grad-CAM samples:

```bash
python -m src.report_generator evaluation/ gradcam_output --html report.html --pdf report.pdf
```

## Command Line Interface
Run training, evaluation or inference via a unified CLI:

```bash
python -m src.cli train --epochs 30
python -m src.cli evaluate --output-dir evaluation
python -m src.cli predict images/*.png --output predictions
```


## Reproducibility
Random seeds for NumPy, Python and TensorFlow are set via `src/utils.set_seeds`, which is called inside `main.py`.

## Docker
Build the image and run the API server:

```bash
docker build -t dr-app .
docker run -p 8000:8000 dr-app
```

## API Usage
The server exposes a `/predict` endpoint. Submit an eye image via `POST` and
receive the predicted grade along with a Grad-CAM overlay encoded as base64.
