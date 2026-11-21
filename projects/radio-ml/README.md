# Radio ML (`radio-astronomy` 連携 Edge / Cloud ハイブリッド推論)

ベランダ電波観測所リポジトリ（[`tozastation/radio-astronomy`](https://github.com/tozastation/radio-astronomy)）で取得・アーカイブされた電波観測データ（気象衛星 LRPT 画像・スペクトログラム・銀河 21cm 水素線分光データ）に対して、機械学習モデルの学習と推論を適用するためのワークスペースです。

## 2階層の推論配置

`radio-astronomy` 側の環境（KubeEdge エッジノード + Garage S3 + Temporal + 母艦 Mac）に合わせて、計算負荷と即時性からモデルを配置します。

### 1. Edge 推論（KubeEdge / GPD Pocket3 CPU / ONNX Runtime）

観測現場でのリアルタイム判定と S3アップロード前のノイズフィルタリングを狙いとしています。`ml-astronomy` で PyTorch 学習した軽量モデル（1D-CNN / 小型 2D-CNN）を ONNX 形式へエクスポートし、Edge Pod（Rust `ort` クレートまたは Python `onnxruntime`）上で CPU 推論を実行します。

対象タスクの候補は次の項目です。

- 銀河 21cm 水素線分光 (`03 / GALACTIC 21CM`): 1420.406 MHz ドリフトスキャンにおける都市ノイズ（RFI）混入スイープの自動検知と品質スコアリング
- 衛星追尾ウォーターフォール: パス中の信号ロック・混信パターンのリアルタイム検知

### 2. Cloud / 母艦推論 & 学習（Mac Apple Silicon MPS / PyTorch）

GPU (MPS) 帯域を活かした高解像度画像解析・マルチモーダル診断・モデル再学習を狙いとしています。Garage S3 (`04 / ARCHIVE`) に保存された観測データを取得し、Mac 上の PyTorch (MPS) でバッチ推論を実行して結果を `satellite-viewer` へ還元します。

対象タスクの候補は次の項目です。

- METEOR-M2 AVHRR 画像 (1568×792): 雲・低気圧の渦・積雪・海氷のセグメンテーション
- マルチモーダル受信品質診断: スペクトログラム画像 × ドップラー偏移・仰角時系列からの復調成否・劣化要因分析

## 関連 Issue

- [Issue #7: `radio-astronomy` (KubeEdge CPU / Mac MPS) とのハイブリッド推論連携パイプラインの設計と実装](https://github.com/tozastation/ml-astronomy/issues/7)
