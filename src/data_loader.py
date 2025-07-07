
# data_loader.py

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import TRAIN_IMG_DIR, TRAIN_CSV_PATH, IMAGE_SIZE, BATCH_SIZE

def get_data_generators():
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    train_df['diagnosis'] = train_df['diagnosis'].astype(str)
    train_df['filename'] = train_df['id_code'] + '.png'

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    val_df['filename'] = val_df['id_code'] + '.png'
    val_df['diagnosis'] = val_df['diagnosis'].astype(str)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        brightness_range=[0.8, 1.2],
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='constant',
        cval=0
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=TRAIN_IMG_DIR,
        x_col='filename',
        y_col='diagnosis',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=TRAIN_IMG_DIR,
        x_col='filename',
        y_col='diagnosis',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    return train_generator, val_generator
