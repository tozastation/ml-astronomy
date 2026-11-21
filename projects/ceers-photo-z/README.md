# JWST CEERS データ解析・モデル学習・推論API (`ceers-photo-z`)

ジェイムズ・ウェッブ宇宙望遠鏡（JWST）の CEERS（Cosmic Evolution Early Release Science）サーベイ等の HDF5 観測データを用いた可視化・PyTorch 学習・推論サービングのプロトタイプディレクトリです（旧 `auto_area_estimation/`）。

## ファイル構成

| ファイル | 役割 |
|---|---|
| `ceers_visualize.ipynb` | HDF5 データセットの構造確認および特徴量の可視化 |
| `ceers_training.ipynb` | PyTorch（Apple Silicon MPS 対応）および Ray Tune を用いたモデル学習実験 |
| `custom_dataset.py` | HDF5 ファイルを読み込む PyTorch 用カスタムデータセット (`HDF5Dataset`) |
| `model_serve.py` | 学習済みモデル (`best_linear_regression_model.pth`) を用いた Flask 推論 API サーバー |
| `mps_info.py` | Apple Silicon 環境での PyTorch MPS バックエンド利用可否チェック |
| `main.ipynb` / `multimodal_universe_getting_start.ipynb` | MultimodalUniverse データセット（Image / Spectra / Time-Series）の動作確認ノートブック |
| `ceers_columns.csv` / `ceers_params.json` | CEERS カタログのカラム定義およびパラメータ情報 |
| `pyproject.toml` / `uv.lock` | 当時の独立した実験用環境定義（Ray Tune / Flask 等を含む） |

## 今後の課題

- `model_serve.py` 内で推論時に `StandardScaler().fit_transform()` を暫定実行しているため、実運用時は学習時のスケーラーを `joblib` 等で保存・ロードする構成に改善予定です。
- `pyproject.toml` は将来的にルート環境との整理・統合を検討します。
