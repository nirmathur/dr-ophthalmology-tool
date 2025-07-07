
# config.py

# File paths
import os

# Base dataset directory
DATA_DIR = os.path.join("data", "aptos2019-blindness-detection")

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images").replace("\\", "/")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images").replace("\\", "/")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv").replace("\\", "/")
TEST_CSV_PATH = os.path.join(DATA_DIR, "test.csv").replace("\\", "/")
MODEL_SAVE_PATH = os.path.join("model_output", "best_model.h5").replace("\\", "/")

# Image settings
IMAGE_SIZE = (380, 380)
BATCH_SIZE = 32
EPOCHS = 50
