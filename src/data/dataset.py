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
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_train_val_datasets(
        train_x: tf.Tensor,
        train_y: tf.Tensor,
        val_x: tf.Tensor,
        val_y: tf.Tensor,
        batch_size: int,
        shuffle_buffer_size: int,   
        preprocess_fn: Optional[PreprocessFn] = None
    ) -> DatasetBundle:

    train_ds = build_dataset(
        features=train_x,
        labels=train_y,
        preprocess_fn=preprocess_fn,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        training=True,
    )

    val_ds = build_dataset(
        features=val_x,
        labels=val_y,
        preprocess_fn=preprocess_fn,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        training=False,
    )

    train_steps = max(1, train_x.shape[0] // batch_size)
    val_steps = max(1, val_x.shape[0] // batch_size)
    return DatasetBundle(train_ds=train_ds, val_ds=val_ds, train_epoch_steps=train_steps, val_epoch_steps=val_steps)
