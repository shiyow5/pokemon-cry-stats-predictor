import pandas as pd

stats_path = "data/raw_stats.csv"
audio_path = "data/audio_features_advanced.csv"
output_path = "data/processed_features_advanced.csv"

def merge_data():
    stats = pd.read_csv(stats_path)
    audio = pd.read_csv(audio_path)
    
    merged = pd.merge(audio, stats, on="name", how="inner")
    # Remove species_id column if it exists
    if 'species_id' in merged.columns:
        merged = merged.drop('species_id', axis=1)
    merged.to_csv(output_path, index=False)
    print(f"✅ Merged advanced dataset saved to {output_path}")
    print(f"Total Pokémon: {len(merged)}")
    print(f"Features: {len([col for col in merged.columns if col not in ['name', 'hp', 'attack', 'defense', 'speed', 'sp_attack', 'sp_defense']])}")

if __name__ == "__main__":
    merge_data()
