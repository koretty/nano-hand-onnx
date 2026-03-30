from dataclasses import dataclass
from typing import Optional

import tensorflow as tf


@dataclass
class TrainResult:
    history: tf.keras.callbacks.History


class Trainer:
    """Thin training orchestrator around Keras fit.

    Keep this class small so switching to a custom loop is easy.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        loss_fn: tf.keras.losses.Loss,
        metrics: list[tf.keras.metrics.Metric],
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics

    def compile(self) -> None:
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=self.metrics,
        )

    def fit(
        self,
        train_ds: tf.data.Dataset,
        val_ds: Optional[tf.data.Dataset],
        epochs: int,
        verbose: int = 1,
        callbacks: Optional[list[tf.keras.callbacks.Callback]] = None,
    ) -> TrainResult:
        # TODO: Add checkpointing and early stopping as default callbacks.
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
        )
        return TrainResult(history=history)


def build_default_optimizer(learning_rate: float) -> tf.keras.optimizers.Optimizer:
    # TODO: Expose optimizer type from config (AdamW/SGD/RMSprop).
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)
