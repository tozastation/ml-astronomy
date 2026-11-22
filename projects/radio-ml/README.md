# Radio ML (`radio-astronomy` 連携 METEOR-M2 画像補完 & Garage S3 パイプライン)

ステータス: **Cloud / Mac MPS 補完パイプライン実装済み**

ベランダ電波観測所リポジトリ（[`tozastation/radio-astronomy`](https://github.com/tozastation/radio-astronomy)）で受信・Garage S3 (`results/`) へアーカイブされた **METEOR-M2 気象衛星 LRPT (MSU-MR) 画像** に対して、Mac (Apple Silicon MPS) 上で欠損補完モデルを実行し、補完済み画像を Garage S3 へ転送して `satellite-viewer` 上に表示するための ML パイプラインです。

## 構成ファイル

- [`src/inpainting.py`](src/inpainting.py) — Notebook `09`/`10`/`11` で検証済みの 8ch Single-Pass (`CoastlineGuidedSharpResSEUNet`)・11ch Temporal Prev-Guided (`PrevImageGuidedResSEUNet`) モデル、自動軌道位置合わせ、ひまわり較正地表アトラス＋Leave-One-Out 過去パス逆投影、SatDump トーン較正、マスクオーバーレイ生成
- [`src/s3_loader.py`](src/s3_loader.py) — Garage S3 (SigV4 `GET` / `HEAD` / `PUT`) クライアントおよびローカルキャッシュ同期ユーティリティ
- [`src/model_cache.py`](src/model_cache.py) — `config_hash` 付きモデルキャッシュユーティリティ
- [`scripts/run_inpaint_pipeline.py`](scripts/run_inpaint_pipeline.py) — Mac から手動実行する補完推論＆Garage S3 転送 CLI
- [`tests/test_inpaint_pipeline.py`](tests/test_inpaint_pipeline.py) — SigV4 PUT/HEAD・モデルロード/推論・差分判定の単体テスト
- `models/` — 学習済みチェックポイント（`nb09_coastline_patchgan_bundle.pt`, `nb11_temporal_prev_bundle.pt`）
- `notebooks/` — 実験・モデル構築ノートブック `01`〜`11`（`01`〜`08`: 多波長同期・ひまわり事前学習・Coarse-to-Fine・海岸線 Prior 実験、`09`: PatchGAN＋自動軌道位置合わせ、`10`: 全実測パス横断評価、`11`: 時系列前イメージガイド 11ch 補完）

## 補完例（`METEOR-M2 3_20261026_003009`）

| 補完前（Raw AVHRR 221） | 補完後（11ch `PrevImageGuidedResSEUNet`） |
| :---: | :---: |
| ![補完前 Raw AVHRR 221](assets/20261026_raw_avhrr_221.png) | ![補完後 AVHRR 221](assets/20261026_inpainted_avhrr_221.png) |

## 使い方（Mac からの手動実行）

事前に GPD Pocket3 へ SSH 接続できる環境（または環境変数 `GARAGE_ACCESS_KEY` / `GARAGE_SECRET_KEY` 設定済み）で実行します。

### 0. 学習済みモデル（`.pt`）の取得

学習済み重みファイル（`nb09_coastline_patchgan_bundle.pt`, `nb11_temporal_prev_bundle.pt`）は Git リポジトリ本体には含めず、GitHub Release `radio-ml-models-v1` で配布しています。ローカルに無い場合は以下で取得してください（`east_asia_cal_base_atlas.npz` は初回推論時に `data/himawari_multiband_cache.npz` から自動生成・キャッシュされます）。

```bash
gh release download radio-ml-models-v1 --repo tozastation/ml-astronomy -D projects/radio-ml/models
```

### 1. 補完対象パスと S3 ステータスの確認

```bash
uv run python projects/radio-ml/scripts/run_inpaint_pipeline.py --list
```

### 2. ローカル検証のみ（Garage S3 へは書き込まない）

```bash
uv run python projects/radio-ml/scripts/run_inpaint_pipeline.py --pass-id "METEOR-M2 3_20261122_000222" --dry-run
```

### 3. 全未補完パスの推論 ＆ Garage S3 への一括転送

```bash
uv run python projects/radio-ml/scripts/run_inpaint_pipeline.py
```

### 主なオプション

- `--pass-id <ID>`: 特定のパスのみ補完（複数指定可）
- `--latest <N>`: 最新の復号済みパス N 件のみ処理
- `--mode {auto,temporal,single}`: 東アジア視野内昼間パスは 11ch 前イメージガイド (`temporal`)、視野外・夜間赤外専用パスは 8ch 単体 (`single`) を `auto` で自動切替
- `--quality {standard,high}`: `high` で 4-way TTA と細粒度ストライドを有効化
- `--force`: 既に同一 `config_hash` で補完済みでも再計算・再アップロード

## Garage S3 データ契約（`ml_inpainted/`）

元の復号画像（`MSU-MR-*.png` 等）を一切変更せず、各パスの S3 ディレクトリ配下に `ml_inpainted/` サブディレクトリを追加します。

- `results/<satellite>/<pass_id>/ml_inpainted/avhrr_221_inpainted.png`: AI 補完済み AVHRR 221 False Color
- `results/<satellite>/<pass_id>/ml_inpainted/msu_mr_124_inpainted.png`: AI 補完済み MSU-MR 124 False Color
- `results/<satellite>/<pass_id>/ml_inpainted/raw_avhrr_221_aligned.png`: Before/After 比較スライダー用の補完前 Raw AVHRR 221
- `results/<satellite>/<pass_id>/ml_inpainted/inpaint_mask_overlay.png`: 実測・部分救済(`#D4A039`)・AI補完(`#E07A38`) 境界オーバーレイ
- `results/<satellite>/<pass_id>/ml_inpainted/metadata.json`: 使用モデル・ハッシュ・位置合わせ精度・品質 Tier（PNG 4 枚アップロード完了後に最後に書き込まれる完了マーカー）

## テスト

```bash
uv run --with pytest python -m pytest projects/radio-ml/tests/test_inpaint_pipeline.py -v
```
