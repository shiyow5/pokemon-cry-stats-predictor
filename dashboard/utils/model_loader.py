import json
import glob
import joblib
from tensorflow import keras
import os


def load_latest_results():
    """
    Load the latest model comparison results from JSON file.
    
    Returns:
        Dictionary with model comparison results, or None if not found
    """
    result_files = sorted(glob.glob('results/model_comparison_*.json'))
    
    if not result_files:
        return None
    
    with open(result_files[-1], 'r') as f:
        results = json.load(f)
    
    return results


def load_neural_network_model():
    """
    Load the best Neural Network model and scaler.
    
    Returns:
        Tuple of (model, scaler) or (None, None) if not found
    """
    model_path = 'models/pokemon_stats_nn.keras'
    scaler_path = 'models/scaler.joblib'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler


def load_model_by_name(model_name):
    """
    Load a specific model by name.
    
    Args:
        model_name: Name of the model file (without extension)
    
    Returns:
        Loaded model, or None if not found
    """
    # Try .keras extension first
    keras_path = f'models/{model_name}.keras'
    if os.path.exists(keras_path):
        return keras.models.load_model(keras_path)
    
    # Try .joblib extension
    joblib_path = f'models/{model_name}.joblib'
    if os.path.exists(joblib_path):
        return joblib.load(joblib_path)
    
    return None


def get_available_models():
    """
    Get list of available trained models.
    
    Returns:
        List of model names (without extensions)
    """
    models = []
    
    # Find all .keras files
    keras_files = glob.glob('models/*.keras')
    models.extend([os.path.basename(f).replace('.keras', '') for f in keras_files])
    
    # Find all .joblib files
    joblib_files = glob.glob('models/*.joblib')
    # Exclude scaler files
    models.extend([os.path.basename(f).replace('.joblib', '') 
                   for f in joblib_files if 'scaler' not in f])
    
    return sorted(set(models))
