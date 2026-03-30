from typing import Iterable

import tensorflow as tf


def build_model(
    input_dim: int,
    num_classes: int,
    hidden_units: Iterable[int] = (64, 32),
    dropout_rate: float = 0.1,
) -> tf.keras.Model:
    """Build a simple MLP model.

    This is intentionally small so it can be replaced quickly.
    """
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = inputs

    for i, units in enumerate(hidden_units):
        x = tf.keras.layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        # TODO: Replace with LayerNorm/BatchNorm if your model needs it.
        x = tf.keras.layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probs")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="baseline_mlp")

    # TODO: Support architecture registry (e.g., transformer/cnn/rnn) here.
    return model
