# Gemini アシスタント設定

このファイルには、このプロジェクトでGeminiがより良く支援するための指示とコンテキストが含まれています。

## プロジェクト概要
`ml-astronomy`（旧 `try-astronomy`）は、公開天文サーベイデータのマルチモーダル機械学習および電波観測所（`tozastation/radio-astronomy`）連携の推論パイプラインを扱うプロジェクトで、以下の内容を扱います。
- MultimodalUniverse（SDSS × HSC AGN分類、Galaxy10 DECaLS、Gaia）を用いた機械学習・可視化
- JWST（CEERS）データ解析・モデル学習および推論APIプロトタイプ
- `radio-astronomy` 向け Edge (ONNX CPU) / Cloud (Mac MPS) 推論パイプライン
- Jupyter Notebookでの可視化を通した、天文学の演習問題と解説
- 天文学可視化プレイグラウンドおよび物理シミュレーション

## 開発コマンド
- Python環境: 依存関係管理に`uv`を使用（pyproject.tomlを参照）
- データ分析と可視化にJupyterノートブック
- モデル訓練とサービング機能

## 主な可視化の例
Jupyter Notebookでは以下のような天文学的なデータの可視化を扱います。
- HR図（ヘルツシュプルング・ラッセル図）: 恒星の絶対等級と色指数をプロットし、恒星の進化段階を分析します。
- 光度曲線: 太陽系外惑星が主星の前を通過（トランジット）する際の光度の変化をプロットし、惑星の存在や半径を推定します。
- スペクトル分析: 恒星や銀河の光を波長ごとに分解し、その組成や赤方偏移を調査します。
- 天体の軌道シミュレーション: 惑星や彗星の軌道を計算し、その動きをアニメーションで可視化します。
- 宇宙の大規模構造: 銀河の三次元分布をプロットし、宇宙の泡構造を可視化します。

## 演習問題の作成方針
Jupyter Notebookを用いた演習問題と解説は、以下の観点を重視して作成します。
- 理論と実践の融合: 各章の冒頭で天文学的な背景理論を簡潔に解説し、続く演習でPythonコードを用いてその理論を実際に計算・可視化することで、知識の定着を図ります。
- 可視化による直感的理解: 数式や文章だけでは理解が難しい概念も、グラフやアニメーションとして可視化することで、直感的な理解を促します。学習者は自身のコードで天体現象を再現する体験を得られます。
- 段階的なステップ: 各演習問題は分析プロセスを体験できるよう、1. 理論と目的の確認、2. データ準備と計算、3. 可視化と考察のセクションで構成します。このステップ分割により、初学者でも無理なく取り組めます。
- 対話的な探求: 学習者がコードのパラメータを自由に変更し、結果がどのように変わるかを探求できるような構成を目指します。能動的な試行錯誤を通じて、より深い学びを促します。
- 実データへの接続: 可能な限り、実際の観測データやシミュレーションデータを使用し、天文学研究の現場で行われているデータ解析の初歩を体験できるようにします。

### フォーマットの統一とセル使用指針

#### Markdownセルの使用
- 理論的背景と物理概念の解説
- 数式と計算過程の詳細（LaTeX記法使用）
- データの分析結果と数値的まとめ（表形式推奨）
- 各段階での考察と物理的意味の解釈
- 学習成果の総括と次のステップへの橋渡し

#### Pythonコードセルの使用
- 可視化専用: グラフ、図表、アニメーションの生成のみ
- ライブラリのインポートと初期設定
- 日本語フォント設定（各可視化セルの冒頭で必須）

### 日本語フォント設定の標準化

#### 各可視化セルの冒頭で必ず実行
```python
# 日本語フォント設定（セル冒頭で必須）
import matplotlib.pyplot as plt
import japanize_matplotlib

plt.rcParams['font.family'] = ['IPAexGothic', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
```

#### 推奨フォント階層
1. `IPAexGothic` - 主要日本語フォント
2. `DejaVu Sans` - フォールバック英語フォント
3. システムデフォルト

#### 文字化け防止チェックリスト
- 各可視化関数の冒頭でフォント設定を確認
- プロット後の日本語表示テスト
- 異なる環境での動作確認

### セル構成の実践例

#### 標準的な問題構成
```
1. Markdownセル: 問題文と学習目的
2. Markdownセル: 理論的背景と学習ポイント
3. Markdownセル: 1. 理論と目的の確認
4. Markdownセル: 2. データ準備と計算（数式・表・結果）
5. Pythonセル: 3. 可視化と考察（フォント設定+グラフ生成）
6. Markdownセル: 学習成果と考察（物理的意味の解釈）
```

## コンテキスト対策とファイル分割戦略

### ノートブック分割の基本方針

#### 大規模コンテンツのモジュール化
- 長大な単一ノートブックは問題別に分割する
- 各問題は独立したノートブックとして完結させる
- セクション全体の概要ファイルで全体像を把握できるようにする

#### 命名規則の統一
```
section_number_overview.ipynb          # セクション概要
problem_section_number_topic.ipynb     # 個別問題
```

#### 実装例（Section 1の場合）
```
chapter1/
├── section1_overview.ipynb            # Section 1全体概要
├── problem_1_1_fullerene.ipynb        # 問1.1: フラーレン
├── problem_1_2_atomic_structure.ipynb # 問1.2: 原子構造
├── problem_1_3_hydrogen_density.ipynb # 問1.3: 水素密度
├── problem_1_4_interstellar_medium.ipynb # 問1.4: 星間物質
├── problem_1_5_time_distance_scales.ipynb # 問1.5: 時間距離
└── problem_1_6_planck_angular_units.ipynb # 問1.6: プランク単位
```

### 分割されたノートブックの設計指針

#### 各問題ノートブックの構成
1. 自完結性: 他のノートブックに依存しない独立した内容
2. 一貫したフォーマット: 全問題で同じ構造（理論→計算→可視化）
3. 適切なボリューム: 1問題あたり6-10セル程度の適切なサイズ
4. 相互参照: 概要ファイルからの適切なリンク構造

#### 概要ノートブックの役割
- 全体の学習目標と進行順序の明示
- 各問題の相互関係と段階的理解の説明
- 学習チェックリストと進捗管理機能
- 統合的な可視化による全体像の把握

### コンテキスト管理の手順と運用

#### 作業時の分割手順
1. 統合ノートブックで内容を完成させる
2. 問題ごとの自然な境界で分割する
3. 各分割ファイルの独立性を確認する
4. 概要ファイルで全体の整合性を保つ

#### 継続的なメンテナンス
- 新しい問題追加時は分割原則を維持する
- ファイル間の重複を避け、必要に応じて共通部分をライブラリ化する
- 定期的な全体構造の見直しと最適化

#### 計算の扱い方
- 単純な数値計算: Markdownセル内に数式と結果を記載
- 複雑な計算過程: Markdownセル内で段階的に説明
- データ整理: Markdownセル内の表形式推奨
- 配列操作: 可視化に直結する場合のみPythonセル使用

#### 可視化コードの構造
```python
def visualize_problem_name():
    """問題名の包括的可視化"""
    
    # 日本語フォント設定（必須）
    import matplotlib.pyplot as plt
    import japanize_matplotlib
    
    plt.rcParams['font.family'] = ['IPAexGothic', 'DejaVu Sans']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False
    
    # データ定義
    # （Markdownで計算済みの値を使用）
    
    # 可視化処理
    fig, axes = plt.subplots(...)
    
    # グラフ作成
    
    plt.tight_layout()
    plt.show()

# 実行
visualize_problem_name()
```

## プロジェクト構造
- `projects/multimodal-universe/` - MultimodalUniverse（SDSS×HSC AGN分類、Galaxy10 DECaLS、Gaia）
- `projects/ceers-photo-z/` - JWST CEERSデータ解析・PyTorch (MPS) 学習・Flask推論API
- `projects/radio-ml/` - `radio-astronomy` 連携 ML パイプライン（Mac MPS による METEOR-M2 画像欠損補完は稼働中、KubeEdge CPU / ONNX Edge 推論は構想中）
- `fundamentals/extreme_unraveling-the-universe/` - Jupyterノートブック形式の天文学教育コンテンツと演習問題
- `fundamentals/playground/` - 天体物理・軌道計算スクリプト
- `scripts/` - 開発用ユーティリティスクリプト

## 姉妹リポジトリ連携（radio-astronomy / ローカル: ../radio-astronomy）
- 観測所システム構成: 東京都・北向きベランダ（見通し方位 $270^\circ \sim 90^\circ$、南〜南西は建物遮蔽）に 137MHz Vダイポール＋RTL-SDR Blog V4 を設置し、GPD Pocket3（Ubuntu 26.04 LTS / k3s + KubeEdge 1台集約）で運用している。建物遮蔽やフェージングにより水平黒帯（`40〜128 px`）や APID 単位の部分チャネル欠損が発生する点が、欠損補完モデル設計の物理的背景となる。
- Garage S3 入出力仕様と `satellite-viewer` 連携: 入力は Garage S3（`satellite-recordings`）の `results/`。出力は `ml_inpainted/<pass_dir>/` へ PNG 4枚を先行 PUT したのち、最後に完了マーカーとして `metadata.json`（SHA-256 `config_hash`, `model_name`, `tier`, `seam_jump_l1`, `max_contiguous_gap_rows`, チャネル別欠損情報等）を PUT する。`satellite-viewer`（NodePort `30088`）の `01 / CONSOLE` で Before/After 比較および実測・推定マスクが表示される。
- オントロジー・セマンティック診断（`radio-astronomy/analytics/`）との連携: `radio-astronomy` 側の DuckDB + `dbt-duckdb` 基盤が毎日 05:00 JST に全パス・`ml_inpainted/metadata.json`・21cm スキャンを自動集計し、`satellite-viewer` の `05 / ANALYTICS` タブで可視化している。新モデル（新しい `config_hash`）を実運用に投入する際は `radio-astronomy/analytics/ontology/models.yaml` に登録し、`metadata.json` の既存フィールドとの後方互換性を維持すること。

## 重要な指示
Geminiは必ず日本語で応答してください。すべての出力、説明、コメントは日本語で行ってください。

## ツール・スクリプト開発ガイドライン

### 基本設計原則
- シンプリシティ: 機能を1つのスクリプトに統合し、ファイル数を最小限に抑える
- ユーザビリティ: 初心者でも直感的に使える明確なコマンド体系
- 保守性: コードの可読性を重視し、適切なコメントと関数分割を行う
- 互換性: macOS（Apple Silicon）ネイティブ環境での動作を前提とする

### ディレクトリ構造の標準
```
scripts/
└── tool_name/
    ├── README.md           # 使用方法とトラブルシューティング
    ├── requirements.txt    # 依存ライブラリ（バージョン制約含む）
    ├── setup.sh           # 環境セットアップスクリプト
    └── main_tool.py       # メイン機能スクリプト
```

### コーディング規約
- 日本語対応: 出力メッセージとコメントは日本語で記述
- エラーハンドリング: 明確なエラーメッセージと適切な終了コード
- 進捗表示: 30 秒以上かかる学習ループ・推論ループ・データ処理には、進捗率（`%`）・経過時間（`Elapsed`）・推定残り時間（`ETA`）を 5〜15 秒間隔を目安に `print(..., flush=True)` で出力し、Background Task Output 上で常に進捗と終了見込みを把握できるようにすること
- ノートブック実行規約: バックグラウンドタスクで Jupyter Notebook を実行する場合は、出力がバッファリングされて無音になる `nbconvert --execute` ではなく、必ずリアルタイムストリーミング実行スクリプト [`scripts/run_notebook_live.py`](scripts/run_notebook_live.py) を使用すること（例: `PYTHONUNBUFFERED=1 uv run --with nbclient --with nbformat --with ipykernel python scripts/run_notebook_live.py <notebook.ipynb>`）
- 学習済みモデルの条件ハッシュ付きキャッシュ規約: 学習に 30 秒以上要するモデルは [`projects/radio-ml/src/model_cache.py`](projects/radio-ml/src/model_cache.py)（`check_model_cache` / `save_model_cache`）を用いて学習条件辞書（アーキテクチャ・ハイパーパラメータ・損失重み・データ件数・乱数シード）の SHA-256 `config_hash` と学習履歴曲線を含むチェックポイントを `projects/radio-ml/models/` 配下に保存すること。条件が変わらない再実行時は学習をスキップして即時ロードし、条件変更時（または `FORCE_RETRAIN=True` 指定時）のみ自動的に再学習してキャッシュを更新すること
- ヘルプ機能: `--help`オプションで詳細な使用方法を表示

### 依存関係管理
- バージョン指定: requirements.txtで互換性のあるバージョン範囲を指定
- システム依存: OSレベルの依存関係を setup.sh で自動インストール
- フォールバック: インストール失敗時の代替手段を提供

### セキュリティ考慮事項
- 権限最小化: 必要最小限の権限で動作するよう設計
- 入力検証: ユーザー入力の妥当性チェックを徹底
- パス処理: Path オブジェクトを使用した安全なファイル操作

### ドキュメント化
- README.md: 目的、使用方法、トラブルシューティングを含む
- インラインコメント: 複雑な処理にはコメントで説明を追加
- 使用例: 実際の使用場面に基づいた具体例を提示

### テスト・検証
- 機能テスト: 基本機能の動作確認を setup.sh に組み込む
- 環境確認: 必要な依存関係の存在チェック
- エラーケース: 一般的なエラー条件での動作検証

## 注意事項
- 機械学習モデルにPyTorch（Apple Silicon MPS 対応）を使用
- インタラクティブ開発にJupyterノートブック
- macOS ネイティブ環境（`mise` + `uv`）での開発を前提とする
- 天文学の概念とデータ分析に教育的焦点

## 共通運用ルール（`radio-astronomy` と統一）

### コミュニケーション・ドキュメント解説方針
- 公式情報・一次情報の明記: 外部ツール、フレームワーク、論文、公開データセットを参照する際は、必ず公式サイトや GitHub リポジトリ、論文（arXiv 等）の一次情報リンクを添える。
- 数式・公式のドキュメント記述ルール: 数式や公式をドキュメントやノートブックに記載・解説する際は、数式単体で終わらせず、必ず以下の 3 要素をセットで解説する。
  1. 記号一覧（テーブル等）: 式を構成する各変数・定数・演算子の物理的・数学的意味と単位の明記。
  2. 日本語での読み解き: 式が物理的・工学的に何を主張しているのかを平易な日本語で要約。
  3. 展開ステップ・直感イメージ: 高校数学〜大学初年級の数学や基礎物理で追える途中式の展開や幾何学的・物理的直感イメージの補足。
- 作業内容・意図の解説と確認: コマンド実行やコード変更を行う際は「今何をしているのか・なぜ必要か」を説明し、大きな方針決定や区切りではユーザーの承認・確認を挟みながら進める。

### コミットルール
- 日本語で記述する
- 絵文字は禁止
- Conventional Commits の形式に従う（例: `feat: ...`, `fix: ...`, `docs: ...`）
- 「〜を追加」「〜を修正」のように簡潔な文章で記述する

## PR 作成・改善報告ルール（可視化データによる根拠明示）
- 改善・実装の PR を作成する際は、必ず**可視化されたデータ**（定量比較表、学習曲線・評価プロット、Before/After 比較画像、誤差・不確実性分布、`radio-astronomy/analytics` の実測補完・欠損分布指標等）を PR 本文の根拠として掲載すること。
- 比較対象（ベースライン / 前モデル / 実測パス）と評価条件（サンプル数・データ範囲・`config_hash` 等）を併記し、改善幅とトレードオフを視覚的に検証可能にすること。
