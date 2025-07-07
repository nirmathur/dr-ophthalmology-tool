
# train.py

import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.model import build_model
from src.config import IMAGE_SIZE, MODEL_SAVE_PATH, EPOCHS

def train_model(train_gen, val_gen, class_weights=None):
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model = build_model(IMAGE_SIZE)

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    return model, history
