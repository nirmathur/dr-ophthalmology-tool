
# config.py

# File paths
import os

# Base dataset directory. Allows overriding via the DATA_DIR environment variable
DATA_DIR = os.getenv(
    "DATA_DIR", os.path.join("data", "aptos2019-blindness-detection")
)

# Training data
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")

# Model output
MODEL_DIR = "model_output"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "best_model.h5")

# Ensure required directories exist
for directory in [TRAIN_IMG_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

# Image settings
IMAGE_SIZE = (380, 380)
BATCH_SIZE = 32
EPOCHS = 50

# Fine-tuning settings
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 1e-5

# Test data (used by evaluation script)
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TEST_CSV_PATH = os.path.join(DATA_DIR, "test.csv")
