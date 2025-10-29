# Pokémon Cry Stats Predictor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-red.svg)](https://streamlit.io/)

ポケモンの鳴き声から6つのステータス（HP、攻撃、防御、素早さ、特攻、特防）を予測する機械学習プロジェクト。

⚠️ **このプロジェクトは学術・研究・教育目的のみを意図しています。ポケモンは任天堂・クリーチャーズ・ゲームフリークの登録商標です。**

## 📊 プロジェクト概要

このプロジェクトは、音響特徴量抽出と機械学習を組み合わせて、ポケモンの鳴き声からゲーム内ステータスを予測します。複数のモデル（Random Forest、XGBoost、Neural Network）を比較評価し、最適なモデルを選択します。

### 主な機能

- **音響特徴量抽出**: 59次元の高度な特徴量（MFCC、Chroma、Spectral Contrast、Tonnetzなど）
- **複数モデル比較**: Random Forest、XGBoost、Neural Networkの性能比較
- **クロスバリデーション**: 5-Fold CVによるモデル評価
- **予測システム**: 新しい音声からステータスを予測
- **分析ノートブック**: Jupyter Notebookでの詳細分析と可視化

## 🎯 モデル性能

| モデル | 特徴量次元 | Overall R² | 改善率 |
|--------|-----------|-----------|--------|
| Baseline (RF) | 29 | 0.067 | - |
| Random Forest | 59 | 0.063 | -6% |
| **Neural Network** | **59** | **0.116** | **+73%** |
| XGBoost | 59 | -0.116 | - |

**Best Model**: Neural Network (R² = 0.116)

### ステータスごとの予測精度（Neural Network）

- **良好**: SP_ATTACK (R²=0.316), SP_DEFENSE (R²=0.227)
- **中程度**: HP (R²=0.156), DEFENSE (R²=0.127)
- **困難**: SPEED (R²=-0.475), ATTACK (R²=-0.193)

## 📁 プロジェクト構成

```
PokémonCryML/
├── dashboard/
│   ├── app.py                       # Streamlitダッシュボードのメインアプリ
│   └── tabs/
│       ├── data_management.py       # データ管理タブ
│       ├── model_training.py        # モデル訓練タブ（実装予定）
│       └── prediction.py            # 予測タブ（実装予定）
├── data/
│   ├── cries/                       # 鳴き声音声ファイル（.ogg）
│   ├── raw_stats.csv                # ポケモンステータスデータ
│   ├── audio_features.csv           # 基本音響特徴量（29次元）
│   ├── audio_features_advanced.csv  # 拡張音響特徴量（59次元）
│   ├── processed_features.csv       # 統合データ（29次元）
│   └── processed_features_advanced.csv # 統合データ（59次元）
├── models/
│   ├── pokemon_stats_predictor.joblib  # ベースラインモデル（RF 29D）
│   ├── pokemon_stats_rf_advanced.joblib # Random Forest（59D）
│   ├── pokemon_stats_nn.keras          # Neural Network（59D）★最良
│   └── scaler.joblib                   # 特徴量スケーラー
├── results/
│   └── model_comparison_*.json      # モデル比較結果
├── scripts/
│   ├── fetch_stats.py              # ステータスデータ収集
│   ├── download_cries.py           # 鳴き声ダウンロード（PokeAPI）
│   ├── extract_audio_features.py   # 基本特徴量抽出（29D）
│   ├── extract_audio_features_advanced.py # 拡張特徴量抽出（59D）
│   ├── merge_dataset.py            # データ統合（29D）
│   ├── merge_dataset_advanced.py   # データ統合（59D）
│   ├── train_model.py              # ベースラインモデル学習
│   ├── train_model_advanced.py     # 高度なモデル学習・比較
│   ├── predict.py                  # 予測（ベースライン）
│   └── predict_advanced.py         # 予測（Neural Network）
├── notebooks/
│   └── analysis.ipynb              # 分析・可視化ノートブック
├── run_full_pipeline.py            # エンドツーエンド実行（Python）
├── run_pipeline.sh                 # エンドツーエンド実行（Shell）
├── requirements.txt                # 依存関係
└── README.md
```

## 🚀 セットアップ

### 1. 仮想環境の有効化

```bash
source .venv/bin/activate  # Linux/macOS
# または
.venv\Scripts\activate     # Windows
```

### 2. 依存関係のインストール

```bash
pip install -r requirements.txt
```

**主な依存関係:**
- pandas, numpy: データ処理
- librosa: 音響特徴量抽出
- scikit-learn: 機械学習（Random Forest）
- xgboost: 勾配ブースティング
- tensorflow/keras: ニューラルネットワーク
- matplotlib, seaborn: 可視化
- jupyter: ノートブック

## 🎬 クイックスタート

### エンドツーエンド実行

全ての処理を一括で実行する場合：

#### 方法1: Pythonスクリプト（推奨）

```bash
# 基本的な使い方（100匹のポケモンで実行）
python run_full_pipeline.py

# より多くのポケモンで実行
python run_full_pipeline.py --max-pokemon 500

# 既にダウンロード済みの場合（ダウンロードをスキップ）
python run_full_pipeline.py --skip-download

# 特定のポケモンで予測テスト
python run_full_pipeline.py --test-pokemon pikachu mewtwo charizard

# ヘルプ表示
python run_full_pipeline.py --help
```

#### 方法2: シェルスクリプト（Linux/macOS）

```bash
# 基本的な使い方
./run_pipeline.sh

# ダウンロードをスキップ
export SKIP_DOWNLOAD=1
./run_pipeline.sh

# モデル学習をスキップ
export SKIP_TRAINING=1
./run_pipeline.sh
```

これらのスクリプトは以下を自動的に実行します：
1. ステータスデータの取得
2. 鳴き声音声のダウンロード（PokeAPI）
3. 音響特徴量の抽出
4. データの統合
5. モデルの学習
6. 予測のテスト

## 📖 詳細な使い方

### データ準備フェーズ

#### 1. ステータスデータを取得

```bash
python scripts/fetch_stats.py
```
→ `data/raw_stats.csv` が作成されます（veekun/pokedexから取得）

#### 2. 鳴き声音声をダウンロード

```bash
python scripts/download_cries.py
```
→ `data/cries/` に音声ファイル（.ogg）がダウンロードされます

#### 3. 音響特徴量を抽出

**基本特徴量（29次元）:**
```bash
python scripts/extract_audio_features.py
```
→ `data/audio_features.csv` が作成されます

**拡張特徴量（59次元）:**
```bash
python scripts/extract_audio_features_advanced.py
```
→ `data/audio_features_advanced.csv` が作成されます

**抽出される特徴量:**
- **基本（29次元）**: MFCC (13 mean + 13 std), ZCR, Spectral Centroid, Rolloff
- **拡張（+30次元）**: Chroma (12), Spectral Contrast (7), Tonnetz (6), Tempo, Bandwidth, Flatness, RMS

#### 4. データを統合

**基本データ:**
```bash
python scripts/merge_dataset.py
```
→ `data/processed_features.csv` が作成されます

**拡張データ:**
```bash
python scripts/merge_dataset_advanced.py
```
→ `data/processed_features_advanced.csv` が作成されます

### モデル学習フェーズ

#### ベースラインモデル（Random Forest, 29次元）

```bash
python scripts/train_model.py
```
→ `models/pokemon_stats_predictor.joblib` が作成されます

**自動データ生成機能:**
必要なCSVファイルが存在しない場合、自動的にデータパイプラインを実行して生成します。
- `data/raw_stats.csv` がなければ `fetch_stats.py` を実行
- `data/cries/*.ogg` がなければ `download_cries.py` を実行
- `data/audio_features.csv` がなければ `extract_audio_features.py` を実行
- `data/processed_features.csv` がなければ `merge_dataset.py` を実行

初回実行時や、データファイルを削除した後でも、トレーニングスクリプトを実行するだけで全て自動的に準備されます

**出力例:**
```
=== Model Evaluation ===
HP:      RMSE: 18.35, R²: 0.169
ATTACK:  RMSE: 19.87, R²: 0.203
...
Overall R² score: 0.067
```

#### 高度なモデル（3モデル比較、59次元）

```bash
# 全モデルを訓練（デフォルト）
python scripts/train_model_advanced.py

# または特定のモデルのみ訓練
python scripts/train_model_advanced.py --model-type nn   # Neural Networkのみ
python scripts/train_model_advanced.py --model-type rf   # Random Forestのみ
python scripts/train_model_advanced.py --model-type xgb  # XGBoostのみ
python scripts/train_model_advanced.py --model-type all  # 全モデル

# パラメータのカスタマイズ
python scripts/train_model_advanced.py --test-size 0.3 --random-state 123
```

**自動データ生成機能:**
必要なCSVファイルが存在しない場合、自動的にデータパイプラインを実行して生成します。
- `data/raw_stats.csv` がなければ `fetch_stats.py` を実行
- `data/cries/*.ogg` がなければ `download_cries.py` を実行
- `data/audio_features_advanced.csv` がなければ `extract_audio_features_advanced.py` を実行
- `data/processed_features_advanced.csv` がなければ `merge_dataset_advanced.py` を実行

初回実行時や、データファイルを削除した後でも、トレーニングスクリプトを実行するだけで全て自動的に準備されます

**オプション:**
- `--model-type`: モデルタイプ (rf/xgb/nn/all、デフォルト: all)
- `--test-size`: テストセットのサイズ (0.0-1.0、デフォルト: 0.2)
- `--random-state`: ランダムシード (デフォルト: 42)
- `--params`: モデルパラメータ（JSON形式）

**実行内容（--model-type allの場合）:**
1. Random Forestモデルの学習
2. XGBoostモデルの学習（6つの個別回帰器）
3. Neural Networkモデルの学習（Keras）
4. 5-Fold Cross Validation（Random Forest）
5. モデル性能の比較
6. 最良モデルの保存

**出力例（--model-type nnの場合）:**
```
============================================================
Model Comparison Summary
============================================================
neural_network      : Overall R² = 0.116

🏆 Best Model: Neural Network (R² = 0.116)

✅ Results saved to results/model_comparison_20251028_062404.json
✅ Neural Network model saved to models/pokemon_stats_nn.keras
✅ Scaler saved to models/scaler.joblib
```

### 予測フェーズ

#### ベースラインモデルで予測

```bash
python scripts/predict.py data/cries/pikachu.ogg
```

#### 高度なモデル（Neural Network）で予測

```bash
python scripts/predict_advanced.py data/cries/pikachu.ogg
```

**出力例:**
```
Loading audio: data/cries/pikachu.ogg
Extracted 59 advanced features

=== Predicted Stats (Neural Network - Advanced) ===
HP             :   55.8
ATTACK         :   54.1
DEFENSE        :   44.3
SPEED          :   64.6
SP_ATTACK      :   52.1
SP_DEFENSE     :   44.5
```

**実際の値（Pikachu）:**
- HP: 35, ATTACK: 55, DEFENSE: 40, SPEED: 90, SP_ATTACK: 50, SP_DEFENSE: 50

### 分析ノートブック

Jupyter Notebookで詳細な分析と可視化を行う場合：

```bash
jupyter notebook notebooks/analysis.ipynb
```

**ノートブックの内容:**
- 日本語フォント設定（文字化け対策）
- ベースラインモデルの評価
- 高度なモデル（3モデル）の比較
- ステータスごとの詳細評価
- クロスバリデーション結果
- 予測例の表示
- 各種グラフ（棒グラフ、ヒートマップ、散布図）

## 🔬 技術詳細

### 音響特徴量

#### 基本特徴量（29次元）
- **MFCC (Mel-Frequency Cepstral Coefficients)**: 音色の特徴（13次元 × 2）
- **ZCR (Zero Crossing Rate)**: 音の高さの指標
- **Spectral Centroid**: スペクトルの重心
- **Spectral Rolloff**: スペクトルの減衰点

#### 拡張特徴量（+30次元）
- **Chroma Features**: 音高クラスの分布（12次元）
- **Spectral Contrast**: 周波数帯域間のコントラスト（7次元）
- **Tonnetz**: 調性の特徴（6次元）
- **Tempo**: テンポ（1次元）
- **Spectral Bandwidth**: 帯域幅（1次元）
- **Spectral Flatness**: 平坦度（1次元）
- **RMS Energy**: エネルギー（2次元）

### モデル

#### Random Forest
- アンサンブル学習手法
- MultiOutputRegressorでラップ
- n_estimators=100

#### XGBoost
- 勾配ブースティング
- 各ステータスごとに個別の回帰器

#### Neural Network (Keras)
- 構造: Dense(128) → Dropout(0.3) → Dense(64) → Dropout(0.2) → Dense(32) → Dense(6)
- 活性化関数: ReLU
- 最適化: Adam
- Early Stopping + ReduceLROnPlateau

### 評価指標

- **R² (決定係数)**: モデルの説明力（-∞～1、1が理想）
- **RMSE (Root Mean Squared Error)**: 予測誤差の平均
- **5-Fold Cross Validation**: 汎化性能の評価

## 📈 実験結果

### モデル性能比較

| モデル | Overall R² | HP | ATTACK | DEFENSE | SPEED | SP_ATTACK | SP_DEFENSE |
|--------|-----------|-----|--------|---------|--------|-----------|------------|
| Baseline (RF 29D) | 0.067 | 0.169 | 0.203 | -0.132 | -0.028 | 0.087 | 0.103 |
| Random Forest 59D | 0.063 | 0.249 | 0.193 | -0.032 | 0.038 | -0.064 | 0.073 |
| **Neural Network 59D** | **0.116** | **0.156** | **-0.193** | **0.127** | **-0.475** | **0.316** | **0.227** |
| XGBoost 59D | -0.116 | 0.126 | 0.193 | -0.498 | 0.052 | 0.037 | -0.105 |

### Cross Validation（Random Forest）

- Mean CV Score: -0.030
- Std Dev: 0.083
- 5つのFoldでばらつきが大きい

### 主な発見

1. **Neural Networkが最良**: ベースラインから73%の性能向上
2. **ステータスごとに精度が大きく異なる**:
   - SP_ATTACKは比較的予測しやすい（R²=0.316）
   - SPEEDは最も予測困難（R²=-0.475）
3. **特徴量の重要性**: 59次元への拡張で性能向上（ただしモデル選択が重要）
4. **XGBoostの過学習**: 今回のデータでは負のR²（改善が必要）

### 考察

**R²スコアが低い根本的な理由:**
- ポケモンの鳴き声とゲーム内ステータスには本質的に強い相関がない
- 鳴き声はキャラクター性の表現であり、戦闘能力とは独立して設計されている
- 100匹のデータでは複雑なパターン学習には不十分

**改善の可能性:**
- データ数の増加（全ポケモン1000匹以上）
- メタ情報の追加（タイプ、世代、進化段階など）
- より高度な音響特徴量（メロディパターン、音色分析）
- アンサンブル手法
- 転移学習（Wav2Vec2など）

## 🛠️ トラブルシューティング

### librosaの警告（n_fft警告）

```
UserWarning: n_fft=1024 is too large for input signal
```

**原因**: 一部のポケモンの鳴き声が非常に短い（< 1024サンプル）
**対処**: 警告は無視して問題ありません。librosaが自動的に調整します。

### TensorFlowのCUDA警告

```
Could not find cuda drivers on your machine, GPU will not be used.
```

**原因**: GPUドライバが見つからない
**対処**: CPUで実行されます。学習時間は長くなりますが動作します。

### 日本語フォントの文字化け

ノートブックのグラフで日本語が文字化けする場合：

```python
# ノートブックの最初のセルで実行
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'  # 英数字のみ表示
```

または日本語フォントをインストール：
```bash
# Ubuntu/Debian
sudo apt-get install fonts-noto-cjk

# macOS
brew install font-noto-sans-cjk-jp
```

## 📚 参考資料

### データソース
- **PokeAPI**: https://pokeapi.co/ - ポケモンの鳴き声音声
- **veekun/pokedex**: https://github.com/veekun/pokedex - ポケモンステータスデータ

### ライブラリ
- **librosa**: https://librosa.org/ - 音響特徴量抽出
- **scikit-learn**: https://scikit-learn.org/ - 機械学習
- **XGBoost**: https://xgboost.readthedocs.io/ - 勾配ブースティング
- **TensorFlow/Keras**: https://www.tensorflow.org/ - ディープラーニング

## 📄 ライセンス

このプロジェクトのソースコードはMITライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルを参照してください。

## ⚠️ 重要な注意事項

### ポケモン関連の著作権について

**このプロジェクトは学術・研究・教育目的のみを意図しています。**

- **ポケモン**は任天堂・クリーチャーズ・ゲームフリークの登録商標です
- ポケモンの名称、画像、音声データの著作権は株式会社ポケモン及び関連会社に帰属します
- このプロジェクトで使用するポケモンデータは以下のオープンソースプロジェクトから取得しています：
  - **PokeAPI** (https://pokeapi.co/) - 音声・画像データ（Fair Useに基づく非営利使用）
  - **veekun/pokedex** (https://github.com/veekun/pokedex) - ステータスデータ

### 利用制限

✅ **許可される使用:**
- 個人的な学習・研究目的
- 教育目的での使用
- 非営利的なデータ分析・機械学習の研究
- オープンソースプロジェクトへの貢献

❌ **禁止される使用:**
- 商用利用（販売、有料サービスへの組み込みなど）
- ポケモンデータの再配布・二次配布
- 著作権者の権利を侵害する行為
- PokeAPIの利用規約に違反する行為

### 免責事項

**このプロジェクトの利用により生じたいかなる損害についても、開発者は一切の責任を負いません。**

- データの正確性は保証されません
- ポケモン関連データの使用については、各自の責任で著作権法を遵守してください
- 商用利用を検討する場合は、必ず株式会社ポケモンの正式な許諾を得てください

### データソースの利用規約

- **PokeAPI利用規約**: https://pokeapi.co/docs/v2
- **veekun/pokedex**: MIT License (https://github.com/veekun/pokedex/blob/master/LICENSE)

**音声・画像データを含むデータセットをこのリポジトリで配布することはありません。** ユーザーが各自でPokeAPIから取得する必要があります

## 🤝 貢献

このプロジェクトへの貢献を歓迎します：
- バグ報告・Issue作成
- 機能追加の提案
- コードの改善（Pull Request）
- ドキュメントの改善
- モデル性能の改善

**貢献する際の注意:**
- ポケモン関連データ（音声、画像）をコミットに含めないでください
- `.gitignore`で除外されているファイルは共有しないでください
- コードのみを貢献してください

## 📞 サポート

問題が発生した場合は、以下を確認してください：
1. requirements.txtの依存関係が正しくインストールされているか
2. 仮想環境が有効化されているか
3. Pythonのバージョンが3.8以上か
4. データファイルが正しく生成されているか

## 🖥️ Streamlitダッシュボード

インタラクティブなWebアプリケーションでデータ管理と可視化を行えます。

### 起動方法

**ローカル環境（推奨）:**
```bash
source .venv/bin/activate
streamlit run dashboard/app.py
```

ブラウザで `http://localhost:8501` にアクセスしてください。

**Streamlit Cloud:**

このダッシュボードはStreamlit Cloudにデプロイ可能ですが、以下の制約があります：

⚠️ **重要: Streamlit Cloud Free Tierの制約**

- **CPUリソースが限定的**: トレーニングに非常に時間がかかります
  - ローカル環境: 5-8分
  - Streamlit Cloud: 30分以上（タイムアウトのリスクあり）

- **推奨事項**:
  1. **GitHub Actionsでトレーニング（推奨）**: GitHub Actionsワークフローで自動訓練（2-5分で完了）
  2. **ローカルでトレーニング**: ローカル環境でモデルを訓練し、pre-trained modelsをデプロイ
  3. **個別モデルを訓練**: "All Models"ではなく、1つずつモデルを訓練（2-15分/モデル）
  4. **予測とEvaluation用途**: Streamlit Cloudは予測とモデル評価に使用

### GitHub Actions統合（推奨）

Streamlit Cloud Free Tierの制約を回避するため、GitHub Actionsを使用してモデルを訓練できます。

**利点:**
- ⚡ **高速**: 2-5分で完了（Streamlit Cloudの10倍以上高速）
- 🔄 **自動化**: 訓練済みモデルを自動的にリポジトリにコミット
- 🎯 **信頼性**: タイムアウトやメモリ不足の問題なし
- 📊 **進捗確認**: GitHub ActionsのUIで実行状況を確認可能

**セットアップ手順:**

1. **GitHub Personal Access Tokenの作成**
   - GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
   - "Generate new token (classic)" をクリック
   - Scope: `repo` と `workflow` を選択
   - トークンを生成してコピー（一度しか表示されません）

2. **Streamlit Secretsに追加**
   - Streamlit Cloud: App Settings → Secrets
   - 以下を追加:
     ```toml
     GITHUB_TOKEN = "ghp_your_token_here"
     ```

3. **Streamlitダッシュボードから使用**
   - Model Training タブを開く
   - "Training Method" で "🤖 GitHub Actions (Recommended)" を選択
   - モデルタイプとパラメータを設定
   - "🏃 Start Training" をクリック
   - GitHub Actionsワークフローが自動的に開始されます

**手動実行（GitHub UIから）:**
   - リポジトリの "Actions" タブを開く
   - "Train Pokémon ML Models" ワークフローを選択
   - "Run workflow" をクリック
   - パラメータを入力して実行

**トレーニング時間:**
- Neural Network単体: 2-3分
- Random Forest単体: 1-2分
- XGBoost単体: 1-2分
- All Models: 2-5分

### 機能

#### 📁 Data Management Tab
トレーニングデータセットの管理機能を提供します。

**主な機能:**
- **データ閲覧**: ポケモンの画像、ステータス、音響特徴量を一覧表示
- **検索機能**: ポケモン名で検索
- **ページネーション**: 30件ごとの表示（100匹以上でも快適）
- **個別削除**: モーダルダイアログで確認後に削除
- **ポケモン追加**: 名前またはIDで1匹ずつ追加
- **一括追加**: 連番またはランダムで複数匹を自動追加
- **データセットリセット**: 全データの削除と再初期化

**パフォーマンス最適化:**
- データキャッシング（60秒TTL）で高速読み込み
- 画像URLキャッシング（1時間TTL）でAPI負荷軽減
- 最適化された画像サイズ（120px）で描画速度向上
- 削除・追加後の自動キャッシュクリアで即座にUI更新

**UX改善:**
- ✅ 削除確認はモーダルダイアログで表示（ページ位置に関係なく中央表示）
- ✅ 削除・追加後のポケモンは即座に画面から消える/表示される
- ✅ 不要なアニメーション（風船）を削除してレスポンス向上
- ✅ `use_container_width`の非推奨警告を解消（`width`パラメータに移行）

**技術仕様:**
- フレームワーク: Streamlit 1.50.0
- データソース: `data/raw_stats.csv`（8列）、`data/audio_features_advanced.csv`（59特徴量）
- 画像取得: PokeAPI
- キャッシュ: `@st.cache_data`デコレータ

#### 📊 Model Training Tab
モデルの訓練と評価を行います。

**主な機能:**
- **トレーニング方法の選択**:
  - 🤖 **GitHub Actions (推奨)**: GitHub Actionsで自動訓練（2-5分）
  - ☁️ **Direct Training**: Streamlit Cloud上で直接訓練（30分以上、タイムアウトリスク）
- **モデル選択**: Neural Network、Random Forest、XGBoost、All Models
- **パラメータ設定**: Test size、Random seed、モデル固有のハイパーパラメータ
- **リアルタイム進捗監視** (GitHub Actions):
  - ワークフローの状態を自動的に監視（⏳ Queued → 🔄 In Progress → ✅ Completed）
  - 10秒ごとに自動更新
  - GitHub ActionsのUIへの直接リンク
  - 完了時に成功/失敗/キャンセルを表示
  - トレーニング完了後にModel Evaluationタブで結果確認を促す
- **結果表示**: トレーニングログと警告の表示

**推奨設定:**
- GitHub Actions方式を使用（高速で信頼性が高い）
- Neural Networkモデル（Overall R² = 0.116で最良）
- Test size = 0.2（デフォルト）
- Random seed = 42（再現性のため）

#### 🔮 Prediction Tab
新しい音声ファイルからステータスを予測します。

**主な機能:**
- **モデル選択**: 訓練済みモデルから選択可能
  - 最良のモデルがデフォルトで選択されます（R²スコアに基づく）
  - 各モデルのR²スコアを表示して比較可能
  - ラジオボタンで簡単に切り替え
- **音声入力**: 
  - ファイルアップロード（OGG, WAV, MP3対応）
  - マイクから直接録音（audio-recorder-streamlitが必要）
- **予測結果表示**: 6つのステータス（HP、ATTACK、DEFENSE、SPEED、SP_ATTACK、SP_DEFENSE）
- **類似ポケモン表示**: 音響特徴量に基づく最も類似したポケモンTOP 3
  - ポケモンの画像表示（PokeAPI）
  - 実際のステータスとの比較
  - Euclidean距離による類似度

**モデル選択機能:**
- 🏆 最良のモデルにトロフィーアイコン表示
- R²スコアをリアルタイムで表示
- モデル変更時に自動的に再ロード
- ユーザーフレンドリーなUI

### 最近の更新 (2025-10-29)

今回の更新で以下の機能が追加されました：

**📊 GitHub Actions進捗監視（Train Tab）:**
- トレーニングの進捗をStreamlit内でリアルタイム表示
- 10秒ごとに自動更新（⏳ Queued → 🔄 In Progress → ✅ Completed）
- GitHub Actionsへの直接リンク
- 完了時に成功/失敗を明確に表示
- 「Stop Monitoring」ボタンで監視を停止可能

**🤖 最良モデル自動選択（Predict Tab）:**
- Model Evaluationの結果に基づいて最良のモデルを自動選択
- デフォルトで最高R²スコアのモデルをロード
- ユーザーがモデルを選択可能なUI
- 各モデルのR²スコアをリアルタイム表示
- 🏆 トロフィーアイコンで最良モデルを明示

**🔧 技術的改善:**
- GitHub Actionsワークフローに`contents: write`権限を追加（403エラー解消）
- インポートパスの修正（Streamlit Cloud対応）
- モデル選択時の自動リロード機能

### ダッシュボードのトラブルシューティング

**ポート8501が既に使用されている:**
```bash
# Streamlitプロセスを停止
pkill -f streamlit

# または特定のポートのプロセスを確認・停止
lsof -ti:8501 | xargs kill -9
```

**キャッシュが古い:**
- ブラウザで "Clear cache" ボタンをクリック（Streamlitの設定メニュー）
- または `Ctrl+R` でページをリロード

**データが更新されない:**
- データ追加・削除後は自動的にキャッシュがクリアされます
- 手動でCSVファイルを編集した場合は60秒待つか、ダッシュボードを再起動してください

## 🎓 今後の展望

1. **ダッシュボードの拡張**: Model Training、Predictionタブの実装
2. **リアルタイム音声入力**: マイクから直接予測
3. **画像表示の強化**: 予測結果と最も近いポケモンの画像
4. **データ拡張**: 全ポケモン（1000匹以上）への対応
5. **メタ情報の活用**: タイプ、世代などの追加特徴量
6. **モデルの改善**: ハイパーパラメータ最適化、アンサンブル手法

---

**Project Status**: Active Development  
**Last Updated**: 2025-10-28  
**License**: MIT (see [LICENSE](LICENSE) for full details)  
**Author**: shiyow5

## 📢 免責事項

このプロジェクトの利用により生じたいかなる損害についても、開発者は一切の責任を負いません。ポケモン関連データの使用については、各自の責任で著作権法を遵守してください。詳細は[ライセンスセクション](#-ライセンス)を参照してください
