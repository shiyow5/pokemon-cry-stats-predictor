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
    data = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".wav") or file.endswith(".ogg"):
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
