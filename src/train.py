import tensorflow as tf

from config import get_config
from data.dataset import build_train_val_datasets
from data.preprocess import build_default_preprocess
from models.model import build_model
from utils.metrics import build_default_loss, build_default_metrics
from utils.trainer import Trainer, build_default_optimizer


def main() -> None:
    # 1) Load experiment settings.
    cfg = get_config()

    # 2) Build pluggable preprocess pipeline.
    preprocess_fn = build_default_preprocess()

    # 3) Build tf.data datasets (already includes preprocess_fn).
    datasets = build_train_val_datasets(cfg.data, preprocess_fn=preprocess_fn)

    # 4) Build model from config.
    model = build_model(
        input_dim=cfg.data.input_dim,
        num_classes=cfg.data.num_classes,
        hidden_units=cfg.model.hidden_units,
        dropout_rate=cfg.model.dropout_rate,
    )

    # 5) Build trainer dependencies.
    optimizer = build_default_optimizer(learning_rate=cfg.train.learning_rate)
    loss_fn = build_default_loss()
    metrics = build_default_metrics()

    # 6) Compile + train.
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metrics=metrics,
    )
    trainer.compile()

    # TODO: Move callbacks to config or callback factory.
    callbacks: list[tf.keras.callbacks.Callback] = []

    result = trainer.fit(
        train_ds=datasets.train_ds,
        val_ds=datasets.val_ds,
        epochs=cfg.train.epochs,
        verbose=cfg.train.verbose,
        callbacks=callbacks,
    )

    # 7) Minimal output to verify the run finished.
    final_metrics = {k: v[-1] for k, v in result.history.history.items()}
    print("Final metrics:", final_metrics)

    # TODO: Save model/artifacts and track experiments (MLflow/W&B/etc.).


if __name__ == "__main__":
    # TODO: Add argparse to override config values from CLI.
    tf.random.set_seed(get_config().data.seed)
    main()
