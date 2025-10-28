import sys
import os
import numpy as np
import librosa
import joblib

# パス定義
MODEL_PATH = "models/pokemon_stats_predictor.joblib"
TARGET_STATS = ["hp", "attack", "defense", "speed", "sp_attack", "sp_defense"]


def extract_features(audio_path):
    """音声ファイルから特徴量を抽出"""
    y, sr = librosa.load(audio_path, sr=22050)
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # その他の特徴
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # 特徴量を結合（train_model.pyと同じ順序で）
    features = list(mfcc_mean) + list(mfcc_std) + [zcr, spectral_centroid, rolloff]
    
    return np.array(features).reshape(1, -1)


def load_model():
    """学習済みモデルを読み込む"""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please train the model first by running: python scripts/train_model.py")
        sys.exit(1)
    
    model = joblib.load(MODEL_PATH)
    return model


def predict_stats(audio_path):
    """音声ファイルからステータスを予測"""
    print(f"Loading audio: {audio_path}")
    
    # 特徴量抽出
    features = extract_features(audio_path)
    print(f"Extracted {features.shape[1]} features")
    
    # モデル読み込み
    model = load_model()
    
    # 予測
    predictions = model.predict(features)[0]
    
    # 結果表示
    print("\n=== Predicted Stats ===")
    for stat, value in zip(TARGET_STATS, predictions):
        print(f"{stat.upper():15s}: {value:6.1f}")
    
    return dict(zip(TARGET_STATS, predictions))


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict.py <audio_file_path>")
        print("Example: python scripts/predict.py data/cries/pikachu.wav")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    predict_stats(audio_path)


if __name__ == "__main__":
    main()
