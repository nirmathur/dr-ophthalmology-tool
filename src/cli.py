import argparse
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from src import config
from src.utils import set_seeds
from src.data_loader import get_data_generators
from src.train import train_model
from src.eval_model import evaluate_model
from src.evaluate import make_gradcam_heatmap, get_last_conv_layer_name, save_gradcam


def override_config(data_dir=None, model_path=None, epochs=None, fine_tune_epochs=None, batch_size=None):
    """Override values in config module based on CLI arguments."""
    if data_dir:
        config.DATA_DIR = data_dir
        config.TRAIN_IMG_DIR = os.path.join(data_dir, "train_images")
        config.TRAIN_CSV_PATH = os.path.join(data_dir, "train.csv")
        config.TEST_IMG_DIR = os.path.join(data_dir, "test_images")
        config.TEST_CSV_PATH = os.path.join(data_dir, "test.csv")
    if model_path:
        config.MODEL_SAVE_PATH = model_path
    if epochs:
        config.EPOCHS = epochs
    if fine_tune_epochs:
        config.FINE_TUNE_EPOCHS = fine_tune_epochs
    if batch_size:
        config.BATCH_SIZE = batch_size


def cmd_train(args):
    override_config(args.data_dir, args.model_path, args.epochs, args.ft_epochs, args.batch_size)
    set_seeds()
    train_gen, val_gen, class_weights = get_data_generators()
    model, history = train_model(train_gen, val_gen, class_weights)
    best_val_acc = max(history.history['val_accuracy'])
    print(f"Training complete. Best val acc: {best_val_acc:.4f}")


def cmd_evaluate(args):
    override_config(args.data_dir, args.model_path)
    metrics = evaluate_model(args.output_dir)
    print("Evaluation metrics saved to", args.output_dir)


def cmd_predict(args):
    override_config(args.data_dir, args.model_path, batch_size=1)
    model = load_model(config.MODEL_SAVE_PATH)
    last_conv = get_last_conv_layer_name(model)
    os.makedirs(args.output, exist_ok=True)
    for img_path in args.images:
        img = image.load_img(img_path, target_size=model.input_shape[1:3])
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        preds = model.predict(img_array)
        grade = int(np.argmax(preds[0]))
        heatmap = make_gradcam_heatmap(img_array, model, last_conv, grade)
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.output, f"{base}_gradcam.png")
        save_gradcam(img_path, heatmap, out_path)
        print(f"{img_path} -> grade {grade}, gradcam saved to {out_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="DR Ophthalmology CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--data-dir", help="Dataset directory")
    p_train.add_argument("--model-path", default=config.MODEL_SAVE_PATH, help="Path to save model")
    p_train.add_argument("--epochs", type=int, default=config.EPOCHS)
    p_train.add_argument("--ft-epochs", type=int, default=config.FINE_TUNE_EPOCHS)
    p_train.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate on test set")
    p_eval.add_argument("--data-dir", help="Dataset directory")
    p_eval.add_argument("--model-path", default=config.MODEL_SAVE_PATH)
    p_eval.add_argument("--output-dir", default="evaluation")
    p_eval.set_defaults(func=cmd_evaluate)

    p_pred = sub.add_parser("predict", help="Predict images with Grad-CAM")
    p_pred.add_argument("images", nargs='+', help="Image paths")
    p_pred.add_argument("--data-dir", help="Dataset directory (optional)")
    p_pred.add_argument("--model-path", default=config.MODEL_SAVE_PATH)
    p_pred.add_argument("--output", default="predictions")
    p_pred.set_defaults(func=cmd_predict)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
