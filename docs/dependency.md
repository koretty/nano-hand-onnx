# モジュール依存関係

プロジェクト内部のモジュール依存を、現在の import 関係に合わせて示します。

```mermaid
graph TD
    train_py["src/train.py"] --> config_py["src/config.py"]
    train_py --> dataset_py["src/data/dataset.py"]
    train_py --> preprocess_py["src/data/preprocess.py"]
    train_py --> model_py["src/models/model.py"]
    train_py --> metrics_py["src/utils/metrics.py"]
    train_py --> trainer_py["src/utils/trainer.py"]
    train_py --> tf_train["tensorflow"]

    dataset_py --> preprocess_py
    dataset_py --> tf_data["tensorflow / tf.data"]
    trainer_py --> tf_keras["TensorFlow Keras"]
    model_py --> tf_keras
    metrics_py --> tf_keras

    config_py --> py_stdlib["Python dataclasses"]
```

- 循環依存はない。主経路は train.py から各モジュールへ一方向に依存する。
- data層は preprocess の型契約（PreprocessFn）に依存し、model/trainer へ依存しない。
- model / metrics / trainer は TensorFlow Keras に収束する。
- 依存関係は整理されているが、train.py と dataset.py / model.py 間の呼び出し契約には差分がある。

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