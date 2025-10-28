import os
import librosa
import pandas as pd
import numpy as np

DATA_DIR = "data/cries"
OUTPUT_PATH = "data/audio_features_advanced.csv"

def extract_features(audio_path):
    """
    音声ファイルから拡張音響特徴量を抽出
    
    基本特徴量（29次元）に加えて以下を追加：
    - Chroma features (12次元)
    - Spectral contrast (7次元)
    - Tonnetz (6次元)
    - Tempo (1次元)
    - 追加のスペクトル特徴 (4次元)
    
    合計: 29 + 12 + 7 + 6 + 1 + 4 = 59次元
    """
    y, sr = librosa.load(audio_path, sr=22050)
    
    # 基本特徴量
    # MFCC (メル周波数ケプストラム係数)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # ゼロ交差率
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # スペクトル特徴
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # 追加特徴量
    # Chroma features (音程・和音の情報)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Spectral contrast (スペクトルの明暗)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    
    # Tonnetz (音調空間特徴)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    
    # Tempo (テンポ)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # 追加のスペクトル特徴
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    rms = np.mean(librosa.feature.rms(y=y))
    
    # RMS エネルギーの標準偏差
    rms_std = np.std(librosa.feature.rms(y=y))
    
    # 特徴量を結合
    features = {
        **{"mfcc_mean_" + str(i+1): mfcc_mean[i] for i in range(13)},
        **{"mfcc_std_" + str(i+1): mfcc_std[i] for i in range(13)},
        "zcr": zcr,
        "spectral_centroid": spectral_centroid,
        "rolloff": rolloff,
        **{"chroma_" + str(i+1): chroma_mean[i] for i in range(12)},
        **{"contrast_" + str(i+1): contrast_mean[i] for i in range(7)},
        **{"tonnetz_" + str(i+1): tonnetz_mean[i] for i in range(6)},
        "tempo": tempo.item() if isinstance(tempo, np.ndarray) else float(tempo),
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_flatness": spectral_flatness,
        "rms": rms,
        "rms_std": rms_std,
    }
    
    return features

def process_all_audio():
    """全ての音声ファイルを処理"""
    data = []
    total_files = len([f for f in os.listdir(DATA_DIR) if f.endswith((".wav", ".ogg"))])
    
    print(f"Processing {total_files} audio files...")
    
    for i, file in enumerate(os.listdir(DATA_DIR), 1):
        if file.endswith(".wav") or file.endswith(".ogg"):
            name = os.path.splitext(file)[0]
            path = os.path.join(DATA_DIR, file)
            
            try:
                feats = extract_features(path)
                feats["name"] = name
                data.append(feats)
                
                if i % 10 == 0:
                    print(f"  [{i}/{total_files}] Processed")
            except Exception as e:
                print(f"  Error processing {name}: {e}")
    
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n✅ Saved advanced features: {OUTPUT_PATH}")
    print(f"   Total features: {len(df.columns) - 1} dimensions")  # -1 for 'name' column
    print(f"   Total samples: {len(df)}")

if __name__ == "__main__":
    process_all_audio()
