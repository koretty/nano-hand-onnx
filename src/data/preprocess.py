from collections.abc import Callable

import tensorflow as tf

PreprocessFn = Callable[[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]


def cast_types(features: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    features = tf.cast(features, tf.float32)
    label = tf.cast(label, tf.int32)
    return features, label


def normalize_features(features: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    mean = tf.reduce_mean(features)
    std = tf.math.reduce_std(features)
    features = (features - mean) / (std + 1e-6)
    return features, label


def compose_preprocess(functions: list[PreprocessFn]) -> PreprocessFn:
    def _apply(features: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        for fn in functions:
            features, label = fn(features, label)
        return features, label

    return _apply


def build_default_preprocess() -> PreprocessFn:
    steps = [cast_types, normalize_features]
    return compose_preprocess(steps)
