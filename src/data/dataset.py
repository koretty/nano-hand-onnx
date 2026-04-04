from dataclasses import dataclass
from typing import Optional

import tensorflow as tf

from data.preprocess import PreprocessFn


@dataclass
class DatasetBundle:
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    train_epoch_steps: int
    val_epoch_steps: int


def _make_synthetic_split(
    num_samples: int,
    input_dim: int,
    num_classes: int,
    seed: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Create a tiny synthetic classification dataset."""
    # TODO: Replace this with file loading (CSV/TFRecord/parquet/etc.).
    features = tf.random.normal(shape=(num_samples, input_dim), seed=seed)

    # A simple deterministic boundary so labels are not random noise only.
    logits = tf.reduce_sum(features[:, : min(input_dim, num_classes)], axis=1)
    labels = tf.cast(tf.math.floormod(tf.cast(tf.abs(logits) * 10.0, tf.int32), num_classes), tf.int32)
    return features, labels


def build_dataset(
    features: tf.Tensor,
    labels: tf.Tensor,
    preprocess_fn: Optional[PreprocessFn],
    batch_size: int,
    shuffle_buffer_size: int,
    training: bool,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((features, labels))

    if training:
        ds = ds.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)

    if preprocess_fn is not None:
        # TODO: Add caching strategy depending on preprocess cost and memory budget.
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    # TODO: Add prefetch_to_device for accelerator-specific optimization.
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_train_val_datasets(data_cfg, preprocess_fn: Optional[PreprocessFn] = None) -> DatasetBundle:
    """Build train/validation datasets from a config-like object."""
    train_x, train_y = _make_synthetic_split(
        num_samples=data_cfg.train_samples,
        input_dim=data_cfg.input_dim,
        num_classes=data_cfg.num_classes,
        seed=data_cfg.seed,
    )
    val_x, val_y = _make_synthetic_split(
        num_samples=data_cfg.val_samples,
        input_dim=data_cfg.input_dim,
        num_classes=data_cfg.num_classes,
        seed=data_cfg.seed + 1,
    )

    train_ds = build_dataset(
        features=train_x,
        labels=train_y,
        preprocess_fn=preprocess_fn,
        batch_size=data_cfg.batch_size,
        shuffle_buffer_size=data_cfg.shuffle_buffer_size,
        training=True,
    )
    val_ds = build_dataset(
        features=val_x,
        labels=val_y,
        preprocess_fn=preprocess_fn,
        batch_size=data_cfg.batch_size,
        shuffle_buffer_size=data_cfg.shuffle_buffer_size,
        training=False,
    )

    train_steps = max(1, data_cfg.train_samples // data_cfg.batch_size)
    val_steps = max(1, data_cfg.val_samples // data_cfg.batch_size)
    return DatasetBundle(train_ds=train_ds, val_ds=val_ds, train_steps=train_steps, val_steps=val_steps)
