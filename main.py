
# main.py

from src.data_loader import get_data_generators
from src.train import train_model
from src.evaluate import get_last_conv_layer_name, make_gradcam_heatmap, display_gradcam
from src.config import MODEL_SAVE_PATH
from src.utils import set_seeds
import numpy as np
from tensorflow.keras.preprocessing import image
import datetime

def main():
    set_seeds()
    train_gen, val_gen = get_data_generators()
    model, history = train_model(train_gen, val_gen)

    print(f"\n✅ Model training complete. Saved to {MODEL_SAVE_PATH}")

    # Log best val accuracy
    best_val_acc = max(history.history['val_accuracy'])
    log_entry = f"{datetime.datetime.now()}: Val Accuracy = {best_val_acc:.4f} | Epochs = {len(history.history['loss'])}\n"
    with open("tuning_log.md", "a") as log_file:
        log_file.write(log_entry)

    # Run Grad-CAM on a sample image
    sample_img = train_gen.filepaths[0]
    img = image.load_img(sample_img, target_size=train_gen.target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    last_conv = get_last_conv_layer_name(model)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv)
    display_gradcam(sample_img, heatmap)

if __name__ == "__main__":
    main()
