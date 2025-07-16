import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from src import config
import itertools


def evaluate_model(output_dir="evaluation"):
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found: {config.MODEL_SAVE_PATH}")

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(config.TEST_CSV_PATH)
    df['filename'] = df['id_code'].astype(str) + '.png'

    datagen = ImageDataGenerator(rescale=1./255)
    test_gen = datagen.flow_from_dataframe(
        dataframe=df,
        directory=config.TEST_IMG_DIR,
        x_col='filename',
        y_col='diagnosis',
        target_size=config.IMAGE_SIZE,
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    model = load_model(config.MODEL_SAVE_PATH)
    preds = model.predict(test_gen)
    y_true = test_gen.classes
    y_pred = np.argmax(preds, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    print("Confusion Matrix")
    print(cm)
    print(classification_report(y_true, y_pred))

    metrics = {
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix plot
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(test_gen.class_indices))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # ROC curves for each class
    plt.figure()
    y_true_ohe = np.eye(preds.shape[1])[y_true]
    for i in range(preds.shape[1]):
        fpr, tpr, _ = roc_curve(y_true_ohe[:, i], preds[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"))
    plt.close()

    return metrics


if __name__ == "__main__":
    evaluate_model()
