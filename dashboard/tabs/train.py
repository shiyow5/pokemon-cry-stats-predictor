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
import requests

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


def trigger_github_actions_training(model_type, test_size, random_state):
    """
    Trigger GitHub Actions workflow to train models

    Args:
        model_type: Type of model to train ('rf', 'xgb', 'nn', 'all')
        test_size: Test set size (0.0 to 1.0)
        random_state: Random seed for reproducibility

    Returns:
        dict: Success status, message, and workflow URL
    """
    # Get GitHub token from secrets
    github_token = st.secrets.get("GITHUB_TOKEN", "")

    if not github_token:
        return {
            "success": False,
            "error": "GitHub token not configured. Please add GITHUB_TOKEN to Streamlit secrets.\n\n"
                    "To set up:\n"
                    "1. Create a Personal Access Token on GitHub with 'repo' and 'workflow' scopes\n"
                    "2. Add it to Streamlit Cloud: App Settings → Secrets → GITHUB_TOKEN"
        }

    # GitHub API configuration
    repo_owner = "shiyow5"
    repo_name = "pokemon-cry-stats-predictor"
    workflow_id = "train-models.yml"

    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/workflows/{workflow_id}/dispatches"

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_token}",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    data = {
        "ref": "main",
        "inputs": {
            "model_type": model_type,
            "test_size": str(test_size),
            "random_state": str(random_state)
        }
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)

        if response.status_code == 204:
            # Success (204 No Content)
            return {
                "success": True,
                "message": "Training started successfully on GitHub Actions!",
                "url": f"https://github.com/{repo_owner}/{repo_name}/actions",
                "repo_owner": repo_owner,
                "repo_name": repo_name,
                "workflow_id": workflow_id
            }
        else:
            return {
                "success": False,
                "error": f"GitHub API error: {response.status_code}\n{response.text}"
            }

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Failed to trigger GitHub Actions: {str(e)}"
        }


def get_latest_workflow_run(repo_owner, repo_name, workflow_id):
    """
    Get the latest workflow run for a specific workflow

    Args:
        repo_owner: GitHub repository owner
        repo_name: GitHub repository name
        workflow_id: Workflow file name

    Returns:
        dict: Workflow run information or None
    """
    github_token = st.secrets.get("GITHUB_TOKEN", "")
    if not github_token:
        return None

    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/workflows/{workflow_id}/runs"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_token}",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    try:
        response = requests.get(url, headers=headers, params={"per_page": 1}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("workflow_runs"):
                return data["workflow_runs"][0]
        return None
    except requests.exceptions.RequestException:
        return None


def check_workflow_status(repo_owner, repo_name, run_id):
    """
    Check the status of a specific workflow run

    Args:
        repo_owner: GitHub repository owner
        repo_name: GitHub repository name
        run_id: Workflow run ID

    Returns:
        dict: Status information
    """
    github_token = st.secrets.get("GITHUB_TOKEN", "")
    if not github_token:
        return None

    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/runs/{run_id}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {github_token}",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "status": data.get("status"),  # queued, in_progress, completed
                "conclusion": data.get("conclusion"),  # success, failure, cancelled, etc.
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "html_url": data.get("html_url"),
                "name": data.get("name")
            }
        return None
    except requests.exceptions.RequestException:
        return None


def run_training(model_type, params, test_size, random_state):
    """
    Run model training using subprocess with timeout

    Args:
        model_type: Type of model to train ('rf', 'xgb', 'nn', 'all')
        params: Dictionary of model parameters
        test_size: Test set size (0.0 to 1.0)
        random_state: Random seed for reproducibility

    Returns:
        Training results or error message
    """
    try:
        # Prepare training script path
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "scripts",
            "train_model_advanced.py"
        )

        # Build command
        cmd = [
            sys.executable, script_path,
            "--model-type", model_type,
            "--test-size", str(test_size),
            "--random-state", str(random_state)
        ]

        # Add model-specific parameters as JSON
        if params:
            import json
            cmd.extend(["--params", json.dumps(params)])

        # Run subprocess with timeout (30 minutes = 1800 seconds)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )

        # Wait for completion with timeout
        try:
            stdout, _ = process.communicate(timeout=1800)  # 30 minute timeout
            returncode = process.returncode

            if returncode == 0:
                return {
                    "success": True,
                    "stdout": stdout,
                    "stderr": ""
                }
            else:
                return {
                    "success": False,
                    "error": f"Training script exited with code {returncode}\n\n{stdout}"
                }

        except subprocess.TimeoutExpired:
            process.kill()
            stdout, _ = process.communicate()
            return {
                "success": False,
                "error": f"Training timed out after 30 minutes. On Streamlit Cloud free tier, training can take significantly longer than locally. Consider training individual models instead of 'All Models'.\n\nPartial output:\n{stdout[:1000]}"
            }

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

    # Training method selection
    training_method = st.radio(
        "**Select Training Method:**",
        options=["🤖 GitHub Actions (Recommended)", "☁️ Direct Training (Streamlit Cloud)"],
        help="GitHub Actions runs training in the background (2-5 minutes). "
             "Direct training runs on Streamlit Cloud (may take 30+ minutes).",
        horizontal=True
    )

    # Display method-specific information
    if training_method == "🤖 GitHub Actions (Recommended)":
        st.success("✨ **GitHub Actions Training**\n\n"
                  "- ⚡ Fast: Completes in 2-5 minutes\n"
                  "- 🔄 Automatic: Models are automatically committed to repository\n"
                  "- 🎯 Reliable: No timeout issues\n"
                  "- 📊 Progress: View workflow progress on GitHub")

        # Display warning about training time
        if model_type == "All Models":
            st.info("⏱️ Expected time: **2-5 minutes** for all three models")
        elif model_type == "Neural Network":
            st.info("⏱️ Expected time: **2-3 minutes**")
        else:
            st.info("⏱️ Expected time: **1-2 minutes**")
    else:
        # Direct training warnings
        if model_type == "All Models":
            st.warning("⚠️ **Streamlit Cloud Performance Warning**\n\n"
                       "Training all models on Streamlit Cloud free tier may take **30+ minutes** due to CPU limitations "
                       "(vs 2-5 minutes with GitHub Actions). The training will timeout after 30 minutes.\n\n"
                       "**Recommendations:**\n"
                       "- Use **GitHub Actions** instead (2-5 minutes)\n"
                       "- Train models **individually** (10-15 minutes each)\n"
                       "- Or train locally and deploy pre-trained models")
            st.info("⏱️ Expected time: 30+ minutes on Streamlit Cloud (timeout risk)")
        elif model_type == "Neural Network":
            st.info("⏱️ Expected time: 10-20 minutes on Streamlit Cloud")
        else:
            st.info("⏱️ Expected time: 5-15 minutes on Streamlit Cloud")

    # Training button
    if st.button("🏃 Start Training", type="primary", use_container_width=True):
        
        # Convert model type to code
        model_code_map = {
            "Neural Network": "nn",
            "Random Forest": "rf",
            "XGBoost": "xgb",
            "All Models": "all"
        }
        model_code = model_code_map[model_type]
        
        # Check which training method is selected
        if training_method == "🤖 GitHub Actions (Recommended)":
            # GitHub Actions training
            with st.spinner("🚀 Triggering GitHub Actions workflow..."):
                result = trigger_github_actions_training(
                    model_type=model_code,
                    test_size=test_size,
                    random_state=random_state
                )
            
            if result.get("success"):
                st.success("✅ " + result["message"])

                # Store workflow information in session state for progress monitoring
                st.session_state.workflow_monitoring = {
                    "repo_owner": result["repo_owner"],
                    "repo_name": result["repo_name"],
                    "workflow_id": result["workflow_id"],
                    "url": result["url"],
                    "start_time": datetime.now()
                }

                st.info(
                    f"🔗 **View on GitHub:** [{result['url']}]({result['url']})\n\n"
                    "Progress will be displayed below. The page will auto-refresh every 10 seconds."
                )

                # Trigger rerun to start monitoring
                st.rerun()
            else:
                st.error("❌ Failed to trigger GitHub Actions")
                st.code(result.get("error", "Unknown error"), language="text")
        
        else:
            # Direct training on Streamlit Cloud
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
                # Run training
                result = run_training(
                    model_type=model_code,
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

    # GitHub Actions progress monitoring
    if "workflow_monitoring" in st.session_state:
        st.divider()
        st.subheader("📊 GitHub Actions Progress")

        monitoring_info = st.session_state.workflow_monitoring

        # Get the latest workflow run
        with st.spinner("Checking workflow status..."):
            import time
            time.sleep(2)  # Wait a bit for the workflow to be created

            latest_run = get_latest_workflow_run(
                monitoring_info["repo_owner"],
                monitoring_info["repo_name"],
                monitoring_info["workflow_id"]
            )

        if latest_run:
            run_id = latest_run["id"]
            status_info = check_workflow_status(
                monitoring_info["repo_owner"],
                monitoring_info["repo_name"],
                run_id
            )

            if status_info:
                status = status_info["status"]
                conclusion = status_info["conclusion"]

                # Display status with appropriate icon and color
                if status == "queued":
                    st.info("⏳ **Status:** Queued - Waiting to start")
                elif status == "in_progress":
                    st.info("🔄 **Status:** In Progress - Training models...")
                elif status == "completed":
                    if conclusion == "success":
                        st.success("✅ **Status:** Completed Successfully!")
                        st.balloons()
                        st.info("🎉 Training completed! Refresh the **Model Evaluation** tab to see the updated results.")
                        # Clear monitoring state
                        del st.session_state.workflow_monitoring
                    elif conclusion == "failure":
                        st.error("❌ **Status:** Failed")
                        st.warning("Please check the workflow logs on GitHub for details.")
                        # Clear monitoring state
                        del st.session_state.workflow_monitoring
                    elif conclusion == "cancelled":
                        st.warning("⚠️ **Status:** Cancelled")
                        # Clear monitoring state
                        del st.session_state.workflow_monitoring
                    else:
                        st.info(f"**Status:** {status} ({conclusion})")

                # Display workflow link
                if status_info.get("html_url"):
                    st.markdown(f"🔗 [View detailed logs on GitHub]({status_info['html_url']})")

                # Auto-refresh if workflow is still running
                if status in ["queued", "in_progress"]:
                    st.info("🔄 Auto-refreshing in 10 seconds...")
                    time.sleep(10)
                    st.rerun()
            else:
                st.warning("Unable to fetch workflow status. Please check GitHub manually.")
                st.markdown(f"🔗 [View on GitHub]({monitoring_info['url']})")
        else:
            st.warning("Workflow run not found yet. It may take a few seconds to appear.")
            st.markdown(f"🔗 [View on GitHub]({monitoring_info['url']})")

            # Auto-refresh to check again
            st.info("🔄 Refreshing in 5 seconds...")
            time.sleep(5)
            st.rerun()

        # Add a button to stop monitoring
        if st.button("⏹️ Stop Monitoring"):
            del st.session_state.workflow_monitoring
            st.rerun()

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
