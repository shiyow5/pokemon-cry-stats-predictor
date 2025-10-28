import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import os


def find_similar_pokemon(input_features, top_k=3):
    """
    Find most similar Pokémon based on Euclidean distance of feature vectors.
    
    Args:
        input_features: Feature vector of shape (1, 59) or (59,)
        top_k: Number of similar Pokémon to return
    
    Returns:
        List of tuples: (pokemon_name, distance, pokemon_features)
        Sorted by distance (ascending)
    """
    # Flatten input features if needed
    if input_features.ndim > 1:
        input_features = input_features.flatten()
    
    # Load feature data
    feature_file = os.path.join('data', 'audio_features_advanced.csv')
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    
    df = pd.read_csv(feature_file)
    
    distances = []
    for idx, row in df.iterrows():
        pokemon_name = row['name']
        # Extract features (all columns except 'name')
        pokemon_features = row.drop('name').values.astype(float)
        
        # Calculate Euclidean distance
        dist = euclidean(input_features, pokemon_features)
        distances.append((pokemon_name, dist, pokemon_features))
    
    # Sort by distance (ascending)
    distances.sort(key=lambda x: x[1])
    
    return distances[:top_k]


def get_pokemon_stats(pokemon_name):
    """
    Get actual stats for a Pokémon from the raw stats CSV.
    
    Args:
        pokemon_name: Name of the Pokémon
    
    Returns:
        Dictionary with stat names as keys and values, or None if not found
    """
    stats_file = os.path.join('data', 'raw_stats.csv')
    if not os.path.exists(stats_file):
        return None
    
    df = pd.read_csv(stats_file)
    pokemon_data = df[df['name'] == pokemon_name]
    
    if pokemon_data.empty:
        return None
    
    stats = pokemon_data.iloc[0]
    return {
        'hp': int(stats['hp']),
        'attack': int(stats['attack']),
        'defense': int(stats['defense']),
        'speed': int(stats['speed']),
        'sp_attack': int(stats['sp_attack']),
        'sp_defense': int(stats['sp_defense'])
    }
