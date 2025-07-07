
# train.py

import os
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from src.model import build_model
from src.config import IMAGE_SIZE, MODEL_SAVE_PATH, EPOCHS, FINE_TUNE_EPOCHS, FINE_TUNE_LR

def train_model(train_gen, val_gen, class_weights=None):
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model = build_model(IMAGE_SIZE)

    log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
        TensorBoard(log_dir=log_dir),
        CSVLogger(os.path.join(log_dir, "training_log.csv"))
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    # Unfreeze some layers for fine-tuning
    base_model = model.get_layer('efficientnetb3')
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy', AUC(), Precision(), Recall()]
    )

    ft_history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    # Combine histories
    for k,v in ft_history.history.items():
        history.history.setdefault(k, []).extend(v)

    return model, history
