import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# パス定義
DATA_PATH = "data/processed_features.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "pokemon_stats_predictor.joblib")

# ターゲット変数（予測するステータス）
TARGET_STATS = ["hp", "attack", "defense", "speed", "sp_attack", "sp_defense"]


def check_and_generate_data():
    """
    Check if required data files exist and generate them if they don't.
    
    Dependency chain:
    1. raw_stats.csv (from fetch_stats.py)
    2. cries/*.ogg (from download_cries.py) 
    3. audio_features.csv (from extract_audio_features.py)
    4. processed_features.csv (from merge_dataset.py)
    """
    import subprocess
    import sys
    from pathlib import Path
    
    print("Checking required data files...")
    
    # 1. Check raw_stats.csv
    raw_stats_path = Path("data/raw_stats.csv")
    if not raw_stats_path.exists():
        print(f"⚠️  {raw_stats_path} not found. Generating...")
        result = subprocess.run([sys.executable, "scripts/fetch_stats.py"])
        if result.returncode != 0:
            print(f"❌ Failed to generate {raw_stats_path}")
            sys.exit(1)
        print(f"✅ Generated {raw_stats_path}")
    else:
        print(f"✅ {raw_stats_path} exists")
    
    # 2. Check cries directory has audio files
    cries_dir = Path("data/cries")
    cries_dir.mkdir(parents=True, exist_ok=True)
    audio_files = list(cries_dir.glob("*.ogg"))
    if len(audio_files) == 0:
        print(f"⚠️  No audio files found in {cries_dir}/. Downloading...")
        result = subprocess.run([sys.executable, "scripts/download_cries.py"])
        if result.returncode != 0:
            print(f"❌ Failed to download audio files")
            sys.exit(1)
        audio_files = list(cries_dir.glob("*.ogg"))
        print(f"✅ Downloaded {len(audio_files)} audio files")
    else:
        print(f"✅ {cries_dir}/ has {len(audio_files)} audio files")
    
    # 3. Check audio_features.csv
    audio_features_path = Path("data/audio_features.csv")
    if not audio_features_path.exists():
        print(f"⚠️  {audio_features_path} not found. Extracting features...")
        result = subprocess.run([sys.executable, "scripts/extract_audio_features.py"])
        if result.returncode != 0:
            print(f"❌ Failed to extract audio features")
            sys.exit(1)
        print(f"✅ Generated {audio_features_path}")
    else:
        print(f"✅ {audio_features_path} exists")
    
    # 4. Check processed_features.csv
    processed_path = Path(DATA_PATH)
    if not processed_path.exists():
        print(f"⚠️  {processed_path} not found. Merging datasets...")
        result = subprocess.run([sys.executable, "scripts/merge_dataset.py"])
        if result.returncode != 0:
            print(f"❌ Failed to merge datasets")
            sys.exit(1)
        print(f"✅ Generated {processed_path}")
    else:
        print(f"✅ {processed_path} exists")
    
    print("✅ All required data files are ready!\n")


def load_data():
    """データを読み込んで特徴量とターゲットに分割"""
    df = pd.read_csv(DATA_PATH)
    
    # 音響特徴量を抽出（mfcc_mean、mfcc_std、zcr、spectral_centroid、rolloff）
    feature_cols = []
    feature_cols.extend([col for col in df.columns if col.startswith("mfcc_mean_")])
    feature_cols.extend([col for col in df.columns if col.startswith("mfcc_std_")])
    feature_cols.extend(["zcr", "spectral_centroid", "rolloff"])
    
    X = df[feature_cols]
    y = df[TARGET_STATS]
    
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    
    return X, y, df


def train_model(X_train, y_train):
    """Multi-output回帰モデルを学習"""
    print("\nTraining model...")
    
    # RandomForestRegressorをMultiOutputRegressorでラップ
    base_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model = MultiOutputRegressor(base_model)
    
    model.fit(X_train, y_train)
    print("✅ Model training completed")
    
    return model


def evaluate_model(model, X_test, y_test):
    """モデルを評価"""
    y_pred = model.predict(X_test)
    
    print("\n=== Model Evaluation ===")
    print(f"Test set size: {len(X_test)}")
    
    # 各ステータスごとに評価
    for i, stat in enumerate(TARGET_STATS):
        y_true = y_test.iloc[:, i]
        y_p = y_pred[:, i]
        
        rmse = np.sqrt(mean_squared_error(y_true, y_p))
        r2 = r2_score(y_true, y_p)
        
        print(f"\n{stat.upper()}:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²:   {r2:.3f}")
    
    # 全体の平均R²スコア
    overall_r2 = r2_score(y_test, y_pred)
    print(f"\nOverall R² score: {overall_r2:.3f}")


def save_model(model):
    """モデルを保存"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")


def main():
    print("=== Pokémon Stats Predictor Training ===\n")
    
    # Check and generate required data files if they don't exist
    check_and_generate_data()
    
    # データ読み込み
    X, y, df = load_data()
    
    # トレーニング/テスト分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")
    
    # モデル学習
    model = train_model(X_train, y_train)
    
    # 評価
    evaluate_model(model, X_test, y_test)
    
    # 保存
    save_model(model)


if __name__ == "__main__":
    main()
