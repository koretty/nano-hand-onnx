from typing import Iterable
from tensorflow.keras import layers, models
import tensorflow as tf


def build_model() -> tf.keras.Model:
    model = models.Sequential([
        layers.InputLayer(input_shape=(64, 64, 1)),
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        layers.SeparableConv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        layers.SeparableConv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation='relu'),
        layers.Dense(4, activation='sigmoid')
    ])
    return model
