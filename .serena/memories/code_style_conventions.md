# コードスタイルと規約

## Python コーディング規約

### 基本スタイル
既存のコードから推測される規約：
- PEP 8に準拠（スペース4つインデント）
- 関数名: snake_case
- 変数名: snake_case
- クラス名: PascalCase（まだクラスは実装されていない）

### Docstrings
- 関数には docstring を記述（三重引用符 `"""`）
- 簡潔な説明を1行で記述
- 複雑な関数の場合は、引数と戻り値を記述

例：
```python
def fetch_pokemon_stats():
    """
    Pokémonのステータスデータを取得し、整形して保存する。
    KaggleやPokémonDBなどから取得したCSVを加工する想定。
    """
```

### インポート順序
1. 標準ライブラリ
2. サードパーティライブラリ
3. ローカルモジュール

例（extract_audio_features.pyから）:
```python
import os
import librosa
import pandas as pd
import numpy as np
```

### 定数
- 大文字スネークケース（UPPER_SNAKE_CASE）
- ファイルの先頭で定義

例：
```python
DATA_DIR = "data/cries"
OUTPUT_PATH = "data/audio_features.csv"
```

### 出力メッセージ
- 成功メッセージには絵文字 ✅ を使用
- 処理の進捗を print で表示

例：
```python
print(f"✅ Saved features: {OUTPUT_PATH}")
print(f"Processed: {name}")
```

### ファイルパス
- 相対パスを使用
- `os.path.join()` や f-string で構築

## Jupyter Notebook
- analysis.ipynb で実験的なモデル構築
- セル単位でステップを分割
- 結果を出力して確認

## まだ定義されていない規約
- 型ヒント（Type Hints）の使用
- linter (flake8, pylint) の設定
- formatter (black, autopep8) の設定
- テストの記述方法