import os
import librosa
import pandas as pd
import numpy as np

DATA_DIR = "data/cries"
OUTPUT_PATH = "data/audio_features.csv"

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    
    # MFCC (メル周波数ケプストラム係数)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # その他の特徴
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    features = {
        **{"mfcc_mean_" + str(i+1): mfcc_mean[i] for i in range(13)},
        **{"mfcc_std_" + str(i+1): mfcc_std[i] for i in range(13)},
        "zcr": zcr,
        "spectral_centroid": spectral_centroid,
        "rolloff": rolloff,
    }
    return features

def process_all_audio():
    # ディレクトリが存在しない場合のエラーチェック
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: {DATA_DIR} directory not found!")
        print(f"   Please run 'python scripts/download_cries.py' first to download audio files.")
        raise FileNotFoundError(f"{DATA_DIR} directory does not exist")
    
    data = []
    audio_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav") or f.endswith(".ogg")]
    
    if len(audio_files) == 0:
        print(f"⚠️  Warning: No audio files found in {DATA_DIR}")
        print(f"   Please run 'python scripts/download_cries.py' first to download audio files.")
        raise FileNotFoundError(f"No audio files found in {DATA_DIR}")
    
    print(f"Found {len(audio_files)} audio files to process")
    
    for file in audio_files:
        name = os.path.splitext(file)[0]
        path = os.path.join(DATA_DIR, file)
        feats = extract_features(path)
        feats["name"] = name
        data.append(feats)
        print(f"Processed: {name}")
    
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved features: {OUTPUT_PATH}")

if __name__ == "__main__":
    process_all_audio()
