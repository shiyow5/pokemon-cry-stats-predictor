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
    result_files = glob.glob('results/model_comparison_*.json')

    if not result_files:
        return None

    # Sort by modification time (newest first) instead of filename
    latest_file = max(result_files, key=os.path.getmtime)

    with open(latest_file, 'r') as f:
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


def load_random_forest_model():
    """
    Load the Random Forest model and scaler.
    
    Returns:
        Tuple of (model, scaler) or (None, None) if not found
    """
    model_path = 'models/pokemon_stats_rf_advanced.joblib'
    scaler_path = 'models/scaler.joblib'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler


def load_best_model():
    """
    Load the best performing model based on latest results.
    
    Returns:
        Tuple of (model, scaler, model_name) or (None, None, None) if not found
    """
    # Load latest results
    results = load_latest_results()
    
    if results is None:
        # Fallback to Neural Network if no results found
        model, scaler = load_neural_network_model()
        return model, scaler, "Neural Network"
    
    # Find best model by overall R²
    best_model_key = None
    best_r2 = float('-inf')
    
    for model_key, model_data in results.items():
        if 'overall_r2' in model_data:
            r2 = model_data['overall_r2']
            if r2 > best_r2:
                best_r2 = r2
                best_model_key = model_key
    
    if best_model_key is None:
        # Fallback to Neural Network
        model, scaler = load_neural_network_model()
        return model, scaler, "Neural Network"
    
    # Get model name
    model_name = results[best_model_key].get('model', 'Unknown')
    
    # Load appropriate model based on the key
    if 'random_forest' in best_model_key.lower() or 'rf' in best_model_key.lower():
        model, scaler = load_random_forest_model()
        if model is None:
            # Fallback to Neural Network
            model, scaler = load_neural_network_model()
            model_name = "Neural Network (fallback)"
    elif 'neural' in best_model_key.lower() or 'nn' in best_model_key.lower():
        model, scaler = load_neural_network_model()
    else:
        # Unknown model type, fallback to Neural Network
        model, scaler = load_neural_network_model()
        model_name = "Neural Network (fallback)"
    
    return model, scaler, model_name


def get_available_models_info():
    """
    Get information about available models with their performance metrics.
    
    Returns:
        List of dictionaries with model information, sorted by R² score (best first)
        Each dict contains: {
            'key': model key from results,
            'name': display name,
            'type': model type (rf/nn),
            'r2': overall R² score,
            'available': whether model file exists
        }
    """
    results = load_latest_results()
    
    if results is None:
        return []
    
    models_info = []
    
    for model_key, model_data in results.items():
        if 'overall_r2' in model_data:
            model_name = model_data.get('model', 'Unknown')
            r2_score = model_data['overall_r2']
            
            # Determine model type and file availability
            if 'random_forest' in model_key.lower() or 'rf' in model_key.lower():
                model_type = 'rf'
                model_file = 'models/pokemon_stats_rf_advanced.joblib'
            elif 'neural' in model_key.lower() or 'nn' in model_key.lower():
                model_type = 'nn'
                model_file = 'models/pokemon_stats_nn.keras'
            else:
                model_type = 'unknown'
                model_file = None
            
            available = model_file is not None and os.path.exists(model_file)
            
            models_info.append({
                'key': model_key,
                'name': model_name,
                'type': model_type,
                'r2': r2_score,
                'available': available
            })
    
    # Sort by R² score (descending)
    models_info.sort(key=lambda x: x['r2'], reverse=True)
    
    return models_info


def load_model_by_type(model_type):
    """
    Load a model by its type.
    
    Args:
        model_type: 'rf' for Random Forest, 'nn' for Neural Network
    
    Returns:
        Tuple of (model, scaler, model_name) or (None, None, None) if not found
    """
    if model_type == 'rf':
        model, scaler = load_random_forest_model()
        return model, scaler, "Random Forest"
    elif model_type == 'nn':
        model, scaler = load_neural_network_model()
        return model, scaler, "Neural Network"
    else:
        return None, None, None


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
