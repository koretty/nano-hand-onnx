from collections.abc import Callable

import tensorflow as tf

PreprocessFn = Callable[[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]


def cast_types(features: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Cast tensors into model-friendly dtypes."""
    # TODO: Add shape assertions if your input schema is strict.
    features = tf.cast(features, tf.float32)
    label = tf.cast(label, tf.int32)
    return features, label


def normalize_features(features: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Simple per-sample normalization for demonstration."""
    # TODO: Replace with dataset-level statistics for real training.
    mean = tf.reduce_mean(features)
    std = tf.math.reduce_std(features)
    features = (features - mean) / (std + 1e-6)
    return features, label


def compose_preprocess(functions: list[PreprocessFn]) -> PreprocessFn:
    """Compose multiple preprocessing steps into a single callable."""

    def _apply(features: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        for fn in functions:
            features, label = fn(features, label)
        return features, label

    return _apply


def build_default_preprocess() -> PreprocessFn:
    """Default pipeline that can be swapped from train.py."""
    # TODO: Insert augmentation functions here for image/audio/text use cases.
    steps = [cast_types, normalize_features]
    return compose_preprocess(steps)
