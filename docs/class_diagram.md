# クラス図（詳細）

TensorFlow学習テンプレートの主要構成要素を、現在の実装シグネチャに合わせて示します。

```mermaid
classDiagram
    class DataConfig {
        +batch_size: int
        +input_dim: int
        +num_classes: int
        +train_samples: int
        +val_samples: int
        +shuffle_buffer_size: int
        +seed: int
    }

    class ModelConfig {
        +hidden_units: tuple
        +dropout_rate: float
    }

    class TrainConfig {
        +learning_rate: float
        +epochs: int
        +verbose: int
    }

    class ExperimentConfig {
        +data: DataConfig
        +model: ModelConfig
        +train: TrainConfig
    }

    class DatasetBundle {
        +train_ds: tf.data.Dataset
        +val_ds: tf.data.Dataset
        +train_epoch_steps: int
        +val_epoch_steps: int
    }

    class DatasetFactory {
        +build_dataset(features, labels, preprocess_fn, batch_size, shuffle_buffer_size, training) tf.data.Dataset
        +build_train_val_datasets(train_x, train_y, val_x, val_y, batch_size, shuffle_buffer_size, preprocess_fn) DatasetBundle
    }

    class Trainer {
        -model: tf.keras.Model
        -optimizer: tf.keras.optimizers.Optimizer
        -loss_fn: tf.keras.losses.Loss
        -metrics: list
        +compile() None
        +fit(train_ds, val_ds, epochs, verbose, callbacks) TrainResult
    }

    class TrainResult {
        +history: tf.keras.callbacks.History
    }

    class ModelFactory {
        +build_model() tf.keras.Model
    }

    class MetricsFactory {
        +build_default_loss() Loss
        +build_default_metrics() list
    }

    class PreprocessPipeline {
        +cast_types(features, label)
        +normalize_features(features, label)
        +compose_preprocess(functions)
        +build_default_preprocess()
    }

    ExperimentConfig --> DataConfig
    ExperimentConfig --> ModelConfig
    ExperimentConfig --> TrainConfig
    DatasetFactory --> DatasetBundle
    Trainer --> TrainResult
    Trainer --> MetricsFactory
    Trainer --> ModelFactory
    DatasetBundle --> PreprocessPipeline
```

現状のコードでは train.py 側の呼び出しシグネチャと、DatasetFactory / ModelFactory のシグネチャが一致していないため、結合作業が必要です。

モデル構成グラフ（models/model.py 実装準拠）:

```mermaid
graph LR
    I[Input 64x64x1] --> C1[Conv2D 16, k3 s2, ReLU]
    C1 --> C2[SeparableConv2D 32, k3 s2, ReLU]
    C2 --> C3[SeparableConv2D 64, k3 s2, ReLU]
    C3 --> G[GlobalAveragePooling2D]
    G --> D1[Dense 32, ReLU]
    D1 --> O[Dense 4, Sigmoid]
```