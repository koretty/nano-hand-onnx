from dataclasses import dataclass, field


@dataclass
class DataConfig:
    batch_size: int = 32
    input_dim: int = 16
    num_classes: int = 3
    train_samples: int = 1024
    val_samples: int = 256
    shuffle_buffer_size: int = 1024
    seed: int = 42


@dataclass
class ModelConfig:
    hidden_units: tuple[int, ...] = (64, 32)
    dropout_rate: float = 0.1


@dataclass
class TrainConfig:
    learning_rate: float = 1e-3
    epochs: int = 5
    verbose: int = 1


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


DEFAULT_CONFIG = ExperimentConfig()


def get_config() -> ExperimentConfig:
    """Return a mutable config object for the current run."""
    # TODO: Load from YAML/JSON/CLI args for environment-specific runs.
    return DEFAULT_CONFIG
