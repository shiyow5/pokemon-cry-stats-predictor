# プロジェクト概要

## 目的
PokémonCryMLは、ポケモンの鳴き声（音声データ）から各種ステータス（HP、攻撃、防御、素早さ、特攻、特防）を機械学習で予測するプロジェクトです。

音響特徴量（MFCC、ゼロ交差率、スペクトル重心など）を抽出し、回帰モデルで各ステータス値を予測します。

## 技術スタック
- **言語**: Python 3.x
- **データ処理**: pandas, numpy
- **音響処理**: librosa
- **機械学習**: scikit-learn (RandomForestRegressor)
- **開発環境**: Jupyter Notebook, 仮想環境 (.venv)

## プロジェクト構造
```
PokémonCryML/
├── data/
│   ├── cries/                 # 音声ファイル（wav/ogg）
│   ├── raw_stats.csv          # ポケモンステータスデータ
│   ├── audio_features.csv     # 抽出した音響特徴量
│   └── processed_features.csv # 統合データセット
├── scripts/
│   ├── fetch_stats.py         # ステータスデータ取得
│   ├── extract_audio_features.py  # 音響特徴量抽出
│   └── merge_dataset.py       # データ統合
├── notebooks/
│   └── analysis.ipynb         # モデル構築・評価用notebook
├── .agent/
│   └── PLANS.md              # ExecPlan仕様書
├── .venv/                    # Python仮想環境
├── README.md
└── CLAUDE.md                 # Claudeへの指示
```

## 開発フロー
1. 音声データを `data/cries/` に配置
2. `scripts/fetch_stats.py` でステータスデータ取得
3. `scripts/extract_audio_features.py` で音響特徴量抽出
4. `scripts/merge_dataset.py` でデータ統合
5. `notebooks/analysis.ipynb` でモデル構築・評価

## 現在の実装状況
- ✅ データ収集スクリプト (fetch_stats.py)
- ✅ 特徴量抽出スクリプト (extract_audio_features.py)
- ✅ データ統合スクリプト (merge_dataset.py)
- ✅ 基本的なモデル構築例 (analysis.ipynb - speedのみ予測)
- ❌ requirements.txt（依存関係定義）
- ❌ 複数ステータスの同時予測
- ❌ モデルの保存/ロード機能
- ❌ テストコード
- ❌ linting/formatting設定