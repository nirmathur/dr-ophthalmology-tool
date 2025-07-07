import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from src.config import IMAGE_SIZE, MODEL_SAVE_PATH, TEST_CSV_PATH, TEST_IMG_DIR


def evaluate_model():
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_SAVE_PATH}")

    df = pd.read_csv(TEST_CSV_PATH)
    df['filename'] = df['id_code'].astype(str) + '.png'

    datagen = ImageDataGenerator(rescale=1./255)
    test_gen = datagen.flow_from_dataframe(
        dataframe=df,
        directory=TEST_IMG_DIR,
        x_col='filename',
        y_col='diagnosis',
        target_size=IMAGE_SIZE,
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    model = load_model(MODEL_SAVE_PATH)
    preds = model.predict(test_gen)
    y_true = test_gen.classes
    y_pred = np.argmax(preds, axis=1)

    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    # ROC curves for each class
    y_true_ohe = np.eye(preds.shape[1])[y_true]
    for i in range(preds.shape[1]):
        fpr, tpr, _ = roc_curve(y_true_ohe[:, i], preds[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    evaluate_model()
