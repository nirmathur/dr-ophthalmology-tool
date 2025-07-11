import argparse
import glob
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from src.config import MODEL_SAVE_PATH
from src.evaluate import make_gradcam_heatmap, save_gradcam, get_last_conv_layer_name


def collect_images(paths):
    images = []
    for p in paths:
        if os.path.isdir(p):
            images.extend(glob.glob(os.path.join(p, "*.png")))
            images.extend(glob.glob(os.path.join(p, "*.jpg")))
        else:
            images.append(p)
    return images


def run(image_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model = load_model(MODEL_SAVE_PATH)
    last_conv = get_last_conv_layer_name(model)

    for img_path in image_paths:
        img = image.load_img(img_path, target_size=model.input_shape[1:3])
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        pred_label = np.argmax(preds[0])

        heatmap = make_gradcam_heatmap(img_array, model, last_conv, pred_label)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, f"{base_name}_gradcam.png")
        save_gradcam(img_path, heatmap, out_path)
        print(f"Saved Grad-CAM for {img_path} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM images for arbitrary inputs")
    parser.add_argument("images", nargs="+", help="Paths to images or directories")
    parser.add_argument("--output", default="gradcam_output", help="Directory to save Grad-CAM images")
    args = parser.parse_args()

    img_list = collect_images(args.images)
    if not img_list:
        raise ValueError("No images found for Grad-CAM generation")

    run(img_list, args.output)


if __name__ == "__main__":
    main()

