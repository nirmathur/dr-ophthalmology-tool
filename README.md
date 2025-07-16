
# Diabetic Retinopathy Detection (Image-Based)

This project started as a competition submission, but it’s grown into something I believe fills a real gap in AI for healthcare.

It’s a lightweight, modular pipeline for diabetic retinopathy detection — designed for transparency, speed, and adaptability. It trains fast on cloud GPUs (I used Google Colab’s L100's), runs locally, and shows its reasoning via Grad-CAM.

Most DR models today are either black-box systems trained with massive compute or academic demos that never get deployed. This isn’t that. It’s for clinicians, students, and researchers who want a transparent foundation they can understand, customise, and trust, especially in environments where infrastructure is limited.

Millions of people in under-resourced regions still lose vision from late-stage DR. This repo is a bet that accessible, explainable, and clinician-guided tools can help shift that. It’s not trying to outdo Google, it’s about democratising what they proved was possible.

Built with EfficientNetB3 and Grad-CAM, it runs on the APTOS 2019 dataset out of the box, but can be adapted to any local dataset or workflow.

The notebook contains the original code I used when building and training the model on Google Colab.



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

### Custom dataset directory
The CLI looks for the dataset under `data/aptos2019-blindness-detection` by default. If your images and CSV files live elsewhere – for example on a mounted Google Drive path – use the `--data-dir` flag to override the location:

```bash
python -m src.cli train \
  --data-dir /content/drive/MyDrive/aptos2019-blindness-detection \
  --epochs 30
```

The same option is available for `evaluate` and `predict` so you can keep the dataset outside the project folder.


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
