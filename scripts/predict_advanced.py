import sys
import os
import numpy as np
import librosa
import joblib
from tensorflow import keras

# パス定義
MODEL_PATH = "models/pokemon_stats_nn.keras"
SCALER_PATH = "models/scaler.joblib"
TARGET_STATS = ["hp", "attack", "defense", "speed", "sp_attack", "sp_defense"]


def extract_features_advanced(audio_path):
    """音声ファイルから拡張特徴量を抽出（59次元）"""
    y, sr = librosa.load(audio_path, sr=22050)
    
    # 基本MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # 基本特徴
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # 拡張特徴
    # Chroma features (12次元)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Spectral contrast (7次元)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    
    # Tonnetz (6次元)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    
    # Tempo (1次元)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = tempo.item() if isinstance(tempo, np.ndarray) else float(tempo)
    
    # 追加のスペクトル特徴 (4次元)
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    rms = np.mean(librosa.feature.rms(y=y))
    rms_std = np.std(librosa.feature.rms(y=y))
    
    # 特徴量を結合（59次元）
    features = list(mfcc_mean) + list(mfcc_std) + [zcr, spectral_centroid, rolloff] + \
               list(chroma_mean) + list(contrast_mean) + list(tonnetz_mean) + \
               [tempo_val, spectral_bandwidth, spectral_flatness, rms, rms_std]
    
    return np.array(features).reshape(1, -1)


def load_model():
    """学習済みモデルとスケーラーを読み込む"""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please train the model first by running: python scripts/train_model_advanced.py")
        sys.exit(1)
    
    if not os.path.exists(SCALER_PATH):
        print(f"Error: Scaler file not found at {SCALER_PATH}")
        print("Please train the model first by running: python scripts/train_model_advanced.py")
        sys.exit(1)
    
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_stats(audio_path):
    """音声ファイルからステータスを予測"""
    print(f"Loading audio: {audio_path}")
    
    # 特徴量抽出
    features = extract_features_advanced(audio_path)
    print(f"Extracted {features.shape[1]} advanced features")
    
    # モデル読み込み
    model, scaler = load_model()
    
    # スケーリング
    features_scaled = scaler.transform(features)
    
    # 予測
    predictions = model.predict(features_scaled, verbose=0)[0]
    
    # 結果表示
    print("\n=== Predicted Stats (Neural Network - Advanced) ===")
    for stat, value in zip(TARGET_STATS, predictions):
        print(f"{stat.upper():15s}: {value:6.1f}")
    
    return dict(zip(TARGET_STATS, predictions))


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict_advanced.py <audio_file_path>")
        print("Example: python scripts/predict_advanced.py data/cries/pikachu.ogg")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    predict_stats(audio_path)


if __name__ == "__main__":
    main()
