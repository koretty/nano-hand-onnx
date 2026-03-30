# nano-hand-onnx 
>TensorFlow/Kerasを用いてhand-tracking-mouseの軽量モデルを学習するためのリポジトリです。

---

## Overview

このリポジトリは、TensorFlow/Kerasを用いて hand-tracking-mouse 向けの軽量モデルを学習するためのテンプレートプロジェクトです。
データ処理・モデル定義・学習ループを分離し、用途に応じて差し替えながら開発できる構成にしています。



### Features



---

## Quick Start


### Requirements


### Installation

```bash
pip install tensorflow
```

必要に応じて仮想環境を作成してからインストールしてください。

### Run

```bash
python src/train.py
```

---

## Usage

### Example

src/train.py の実行で、以下の流れが一気に動作します。

1. config読み込み
2. preprocessパイプライン構築
3. tf.dataのtrain/val構築
4. モデル構築
5. optimizer/loss/metrics構築
6. Trainerで学習
7. 最終メトリクス表示



### Configuration

src/config.py で主なハイパーパラメータを管理します。

- batch_size
- learning_rate
- epochs
- hidden_units
- dropout_rate
- train_samples / val_samples

実運用時は、この設定にデータパス・モデル保存先・コールバック設定などを追加してください。

---

## Tech Stack

| Category | Technology | Reason |
| :-- | :-- | :-- |
| Runtime | Python | 学習実験の反復がしやすい |
| ML Framework | TensorFlow / Keras | 学習ループとモデル開発を標準化しやすい |
| Data Pipeline | tf.data | 前処理の組み込みと高速化拡張がしやすい |
| Config | dataclasses | 可読性が高く最小構成で管理できる |
| Target Domain | hand-tracking-mouse | 手特徴量ベースの軽量推論モデル開発に合わせやすい |

---

## Project Structure

```text
src/
├── train.py                # 学習実行エントリーポイント
├── config.py               # ハイパーパラメータ管理
├── data/
│   ├── dataset.py          # データ読み込み
│   └── preprocess.py       # 前処理（差し替え可能）
├── models/
│   └── model.py            # モデル定義
└── utils/
    ├── trainer.py          # 学習ループ
    └── metrics.py          # 評価指標
```

---

## Roadmap

- [x] 最小実行できる学習テンプレート
- [x] 前処理差し替え可能な構成
- [x] モデル差し替え可能な構成
- [ ] hand-tracking-mouse用の実データローダー実装
- [ ] 推論に必要な特徴量設計（ランドマーク正規化など）
- [ ] チェックポイント保存とEarlyStopping標準化
- [ ] 外部設定ファイル（YAML/JSON）対応
- [ ] 評価指標拡張（精度以外の指標追加）

---

## License

MIT License