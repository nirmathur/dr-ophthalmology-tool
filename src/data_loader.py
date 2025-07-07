
# data_loader.py

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import TRAIN_IMG_DIR, TRAIN_CSV_PATH, IMAGE_SIZE, BATCH_SIZE
from src.utils import validate_dataset

def random_crop_and_color(img):
    crop_h = tf.cast(tf.shape(img)[0] * 0.9, tf.int32)
    crop_w = tf.cast(tf.shape(img)[1] * 0.9, tf.int32)
    img = tf.image.random_crop(img, size=[crop_h, crop_w, 3])
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.random_brightness(img, 0.05)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img


def get_data_generators():
    train_df = validate_dataset(TRAIN_CSV_PATH, TRAIN_IMG_DIR)
    train_df['diagnosis'] = train_df['diagnosis'].astype(str)

    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df['diagnosis'])
    val_df['filename'] = val_df['id_code'] + '.png'
    val_df['diagnosis'] = val_df['diagnosis'].astype(str)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        brightness_range=[0.8, 1.2],
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        channel_shift_range=10.0,
        horizontal_flip=True,
        fill_mode='constant',
        cval=0,
        preprocessing_function=random_crop_and_color,
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

    # Compute class weights to handle imbalance
    classes = np.unique(train_df['diagnosis'].astype(int))
    weights = compute_class_weight(
        class_weight='balanced', classes=classes,
        y=train_df['diagnosis'].astype(int))
    class_weights = dict(zip(classes, weights))

    return train_generator, val_generator, class_weights
