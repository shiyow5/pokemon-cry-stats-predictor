# 推奨コマンド

## 仮想環境

### 仮想環境の有効化
```bash
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 依存関係のインストール（requirements.txtが存在する場合）
```bash
pip install -r requirements.txt
```

### 現在使用されているライブラリ
- pandas
- librosa
- numpy
- scikit-learn
- jupyter

インストール例：
```bash
pip install pandas librosa numpy scikit-learn jupyter
```

## データ処理パイプライン

### 1. ステータスデータの取得
```bash
python scripts/fetch_stats.py
```
出力: `data/raw_stats.csv`

### 2. 音響特徴量の抽出
```bash
python scripts/extract_audio_features.py
```
前提: `data/cries/` に音声ファイル（.wav または .ogg）が存在
出力: `data/audio_features.csv`

### 3. データの統合
```bash
python scripts/merge_dataset.py
```
出力: `data/processed_features.csv`

## モデル構築

### Jupyter Notebookの起動
```bash
jupyter notebook notebooks/analysis.ipynb
```

または

```bash
jupyter lab
```

## 開発コマンド

### Pythonスクリプトの実行
```bash
python scripts/<script_name>.py
```

### Pythonインタラクティブシェル
```bash
python
```

## Git コマンド（プロジェクトがGitで管理されている場合）
```bash
git status
git add .
git commit -m "commit message"
git push
```

## システムコマンド（Linux）
```bash
ls                # ファイル一覧
cd <directory>    # ディレクトリ移動
pwd               # 現在のディレクトリ
cat <file>        # ファイル内容表示
grep <pattern>    # パターン検索
find <path>       # ファイル検索
```

## まだ設定されていないコマンド
- テスト実行コマンド (pytest など)
- Linting コマンド (flake8, pylint など)
- Formatting コマンド (black, autopep8 など)
- ビルド・デプロイコマンド