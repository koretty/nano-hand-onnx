# Data Flow

データの流れ（設定 -> 前処理 -> データセット -> 学習 -> 出力）を示します。

- 入力: src/config.py の ExperimentConfig
- 前処理: src/data/preprocess.py の関数合成
- 実行主体: src/train.py
- データ供給: src/data/dataset.py（tf.data）
- 学習: src/utils/trainer.py
- 出力: 学習履歴と最終メトリクス表示

```mermaid
graph TD
    A[train.py main] --> B[get_config]
    B --> C[build_default_preprocess]

    C --> D[build_train_val_datasets]
    B --> D

    D --> E[train_ds and val_ds]

    B --> F[build_model]
    B --> G[build_default_optimizer]
    H[build_default_loss] --> I[Trainer]
    J[build_default_metrics] --> I
    F --> I
    G --> I

    E --> I
    I --> K[model.compile]
    K --> L[model.fit]
    L --> M[TrainResult.history]
    M --> N[final metrics print]
```

実運用では、前処理差し替え・モデル差し替え・データ読み込み差し替えをこの主経路に沿って行います。