import pandas as pd

stats_path = "data/raw_stats.csv"
audio_path = "data/audio_features.csv"
output_path = "data/processed_features.csv"

def merge_data():
    stats = pd.read_csv(stats_path)
    audio = pd.read_csv(audio_path)
    
    merged = pd.merge(audio, stats, on="name", how="inner")
    merged.to_csv(output_path, index=False)
    print(f"✅ Merged dataset saved to {output_path}")
    print(f"Total Pokémon: {len(merged)}")

if __name__ == "__main__":
    merge_data()
