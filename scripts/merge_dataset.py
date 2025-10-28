import pandas as pd

stats_path = "data/raw_stats.csv"
audio_path = "data/audio_features.csv"
output_path = "data/processed_features.csv"

def merge_data():
    import os
    
    # 必要なファイルの存在チェック
    missing_files = []
    if not os.path.exists(stats_path):
        missing_files.append(stats_path)
    if not os.path.exists(audio_path):
        missing_files.append(audio_path)
    
    if missing_files:
        print(f"❌ Error: Required files not found:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease run the pipeline in order:")
        print("   1. python scripts/fetch_stats.py")
        print("   2. python scripts/download_cries.py")
        print("   3. python scripts/extract_audio_features.py")
        raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")
    
    stats = pd.read_csv(stats_path)
    audio = pd.read_csv(audio_path)
    
    merged = pd.merge(audio, stats, on="name", how="inner")
    merged.to_csv(output_path, index=False)
    print(f"✅ Merged dataset saved to {output_path}")
    print(f"Total Pokémon: {len(merged)}")

if __name__ == "__main__":
    merge_data()
