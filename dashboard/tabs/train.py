"""
Train tab for Pokémon Cry Stats Predictor Dashboard
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
import json
from datetime import datetime
import subprocess

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_data_availability():
    """
    Check if required data files are available
    
    Returns:
        Dictionary with availability status
    """
    required_files = {
        "audio_features": "data/audio_features_advanced.csv",
        "stats": "data/raw_stats.csv"
    }
    
    availability = {}
    for key, path in required_files.items():
        full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), path)
        availability[key] = os.path.exists(full_path)
    
    return availability


def get_dataset_info():
    """
    Get information about the dataset
    
    Returns:
        Dictionary with dataset statistics
    """
    try:
        features_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data/audio_features_advanced.csv"
        )
        stats_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data/raw_stats.csv"
        )
        
        features_df = pd.read_csv(features_path)
        stats_df = pd.read_csv(stats_path)
        
        return {
            "num_samples": len(features_df),
            "num_features": len(features_df.columns) - 1,  # Exclude name column
            "pokemon_list": features_df["name"].tolist()
        }
    except Exception as e:
        st.error(f"Error loading dataset info: {str(e)}")
        return None


def run_training(model_type, params, test_size, random_state):
    """
    Run model training by directly importing and calling the training script
    
    Args:
        model_type: Type of model to train ('rf', 'xgb', 'nn', 'all')
        params: Dictionary of model parameters
        test_size: Test set size (0.0 to 1.0)
        random_state: Random seed for reproducibility
    
    Returns:
        Training results or error message
    """
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    try:
        # Import the training script
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "scripts"
        )
        sys.path.insert(0, script_path)
        
        # Import training function
        from train_model_advanced import main as train_main
        
        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Run training with output capture
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            try:
                train_main(
                    model_type=model_type,
                    test_size=test_size,
                    random_state=random_state,
                    params=params
                )
                success = True
            except Exception as e:
                success = False
                stderr_capture.write(f"Training error: {str(e)}\n")
        
        # Get captured output
        stdout_text = stdout_capture.getvalue()
        stderr_text = stderr_capture.getvalue()
        
        if success:
            return {
                "success": True,
                "stdout": stdout_text,
                "stderr": stderr_text
            }
        else:
            return {
                "success": False,
                "error": stderr_text or stdout_text or "Unknown error occurred"
            }
            
    except ImportError as e:
        return {"success": False, "error": f"Failed to import training script: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}


def render():
    """Render the Train tab"""
    st.header("🏋️ Train Models")
    st.write(
        "Configure and train machine learning models to predict Pokémon stats from audio features. "
        "Select a model type, adjust parameters, and monitor training progress."
    )
    
    # Check data availability
    data_status = check_data_availability()
    
    if not all(data_status.values()):
        st.error("❌ **Required data files are missing!**")
        st.warning("The following files are missing:")
        for key, available in data_status.items():
            if not available:
                st.write(f"- {key}")
        st.info("""
**How to fix:**

**Option 1 (Recommended):** Use the UI
- Go to the **📁 Data Management** tab
- Click the '🚀 Initialize Dataset Now' button
- Wait for the initialization to complete (5-10 minutes)

**Option 2:** Use command line scripts
```bash
# Run these commands in order:
python scripts/fetch_stats.py
python scripts/download_cries.py
python scripts/extract_audio_features_advanced.py
python scripts/merge_dataset_advanced.py
```
        """)
        return
    
    st.success("✅ All required data files are available")
    
    # Display dataset information
    dataset_info = get_dataset_info()
    if dataset_info:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", dataset_info["num_samples"])
        with col2:
            st.metric("Feature Dimensions", dataset_info["num_features"])
        with col3:
            st.metric("Pokémon Species", len(set(dataset_info["pokemon_list"])))
    
    st.divider()
    
    # Model selection
    st.subheader("🤖 Model Configuration")
    
    model_type = st.selectbox(
        "Select Model Type",
        options=["Neural Network", "Random Forest", "XGBoost", "All Models"],
        help="Choose the machine learning model to train (or select 'All Models' to train all three models)"
    )
    
    # Model-specific parameters
    st.subheader("⚙️ Model Parameters")

    params = {}

    if model_type == "All Models":
        st.info("ℹ️ When training all models, default parameters will be used for each model.")

    elif model_type == "Neural Network":
        col1, col2 = st.columns(2)
        
        with col1:
            params["hidden_layers"] = st.text_input(
                "Hidden Layers",
                value="128,64,32",
                help="Comma-separated layer sizes (e.g., '128,64,32')"
            )
            params["epochs"] = st.slider("Epochs", 50, 500, 200, 10)
            params["batch_size"] = st.slider("Batch Size", 8, 64, 16, 8)
        
        with col2:
            params["learning_rate"] = st.number_input(
                "Learning Rate",
                min_value=0.0001,
                max_value=0.1,
                value=0.001,
                step=0.0001,
                format="%.4f"
            )
            params["dropout"] = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
            params["early_stopping_patience"] = st.slider("Early Stopping Patience", 10, 50, 20, 5)
    
    elif model_type == "Random Forest":
        col1, col2 = st.columns(2)
        
        with col1:
            params["n_estimators"] = st.slider("Number of Trees", 50, 500, 100, 50)
            params["max_depth"] = st.slider("Max Depth", 5, 50, 20, 5)
        
        with col2:
            params["min_samples_split"] = st.slider("Min Samples Split", 2, 20, 2, 1)
            params["min_samples_leaf"] = st.slider("Min Samples Leaf", 1, 10, 1, 1)
    
    elif model_type == "XGBoost":
        col1, col2 = st.columns(2)
        
        with col1:
            params["n_estimators"] = st.slider("Number of Boosting Rounds", 50, 500, 100, 50)
            params["max_depth"] = st.slider("Max Depth", 3, 15, 6, 1)
            params["learning_rate"] = st.number_input(
                "Learning Rate",
                min_value=0.01,
                max_value=0.3,
                value=0.1,
                step=0.01,
                format="%.2f"
            )
        
        with col2:
            params["subsample"] = st.slider("Subsample Ratio", 0.5, 1.0, 0.8, 0.1)
            params["colsample_bytree"] = st.slider("Feature Sampling Ratio", 0.5, 1.0, 0.8, 0.1)
    
    st.divider()
    
    # Training configuration
    st.subheader("📊 Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider(
            "Test Set Size",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            help="Proportion of dataset to use for testing"
        )
    
    with col2:
        random_state = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=999999,
            value=42,
            help="Seed for reproducibility"
        )
    
    st.divider()
    
    # Training execution
    st.subheader("🚀 Training Execution")
    
    # Display warning about training time
    if model_type == "All Models":
        st.info("⏱️ Training all models typically takes 5-8 minutes.")
    elif model_type == "Neural Network":
        st.info("⏱️ Neural Network training typically takes 2-5 minutes depending on parameters.")
    else:
        st.info("⏱️ Training typically takes 1-2 minutes.")
    
    # Training button
    if st.button("🏃 Start Training", type="primary", use_container_width=True):
        
        # Show warning if model already exists
        model_files = {
            "Neural Network": "models/pokemon_stats_nn.keras",
            "Random Forest": "models/pokemon_stats_rf.joblib",
            "XGBoost": "models/pokemon_stats_xgb.joblib"
        }
        
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            model_files.get(model_type, "")
        )
        
        if os.path.exists(model_path):
            st.warning(f"⚠️ A {model_type} model already exists and will be overwritten.")
        
        with st.spinner(f"Training {model_type} model... This may take several minutes."):
            
            # Convert model type to code
            model_code_map = {
                "Neural Network": "nn",
                "Random Forest": "rf",
                "XGBoost": "xgb",
                "All Models": "all"
            }
            
            # Run training
            result = run_training(
                model_type=model_code_map[model_type],
                params=params,
                test_size=test_size,
                random_state=random_state
            )
            
            if result.get("success"):
                st.success("✅ Training completed successfully!")
                
                # Display training output
                with st.expander("📋 Training Log", expanded=True):
                    st.code(result["stdout"], language="text")
                
                if result.get("stderr"):
                    with st.expander("⚠️ Warnings"):
                        st.code(result["stderr"], language="text")
                
                # Suggest going to evaluation tab
                st.info("👉 Go to the **Model Evaluation** tab to view the training results and metrics.")
                
            else:
                st.error("❌ Training failed!")
                st.code(result.get("error", "Unknown error"), language="text")
    
    # Additional information
    st.divider()
    
    with st.expander("ℹ️ Model Information"):
        st.write("""
        ### Available Models
        
        **Neural Network**
        - Multi-layer perceptron with configurable architecture
        - Best for capturing complex non-linear relationships
        - Supports dropout regularization and early stopping
        - Current best performer (Overall R² ≈ 0.116)
        
        **Random Forest**
        - Ensemble of decision trees
        - Good baseline model with interpretable feature importance
        - Robust to overfitting with proper configuration
        - Performance: Overall R² ≈ 0.076
        
        **XGBoost**
        - Gradient boosting with advanced regularization
        - Efficient and often high-performing
        - Good for structured/tabular data
        - Performance: Overall R² ≈ -0.033 (needs tuning)
        
        ### Training Process
        
        1. **Data Loading**: Audio features (59D) and stats are loaded
        2. **Train/Test Split**: Data is split according to your configuration
        3. **Feature Scaling**: Features are normalized using StandardScaler
        4. **Model Training**: Model is trained with specified parameters
        5. **Evaluation**: Model is evaluated on test set with R² and RMSE metrics
        6. **Model Saving**: Trained model and scaler are saved to `models/` directory
        
        ### Tips for Better Performance
        
        - **Neural Network**: Increase epochs but use early stopping to prevent overfitting
        - **Random Forest**: Increase number of trees, but be aware of training time
        - **XGBoost**: Tune learning rate and max_depth carefully
        - **General**: Ensure test_size is large enough for reliable evaluation (0.2 is recommended)
        """)
    
    with st.expander("🔧 Advanced Usage"):
        st.write("""
        ### Manual Training
        
        You can also train models manually using the command line:
        
        ```bash
        # Activate virtual environment
        source .venv/bin/activate
        
        # Train with default parameters
        python scripts/train_model_advanced.py
        ```
        
        ### Custom Training Scripts
        
        For more control, you can modify `scripts/train_model_advanced.py` directly or create
        your own training scripts using the provided data files:
        
        - Features: `data/audio_features_advanced.csv`
        - Stats: `data/raw_stats.csv`
        
        ### Model Files
        
        Trained models are saved in the `models/` directory:
        
        - Neural Network: `pokemon_stats_nn.keras` + `scaler.joblib`
        - Random Forest: `pokemon_stats_rf.joblib` + `scaler_rf.joblib`
        - XGBoost: `pokemon_stats_xgb.joblib` + `scaler_xgb.joblib`
        
        Results are saved in `results/model_comparison_TIMESTAMP.json`
        """)
