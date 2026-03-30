import tensorflow as tf


def build_default_loss() -> tf.keras.losses.Loss:
    # TODO: Switch to focal loss / label smoothing when needed.
    return tf.keras.losses.SparseCategoricalCrossentropy()


def build_default_metrics() -> list[tf.keras.metrics.Metric]:
    # TODO: Add task-specific metrics (AUC/F1/IoU/etc.).
    return [
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    ]
