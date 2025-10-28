"""
Data initialization utilities for the dashboard.

This module provides functions to check data availability and initialize
the dataset when files are missing (especially useful for Streamlit Cloud deployments).
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path object pointing to the project root
    """
    # dashboard/utils/data_initializer.py -> go up 2 levels to project root
    return Path(__file__).parent.parent.parent


def check_data_files() -> Dict[str, bool]:
    """
    Check if required data files exist.
    
    Returns:
        Dictionary mapping file names to their existence status
    """
    project_root = get_project_root()
    
    required_files = {
        "raw_stats.csv": project_root / "data" / "raw_stats.csv",
        "audio_features_advanced.csv": project_root / "data" / "audio_features_advanced.csv",
        "processed_features_advanced.csv": project_root / "data" / "processed_features_advanced.csv",
        "cries_directory": project_root / "data" / "cries",
    }
    
    status = {}
    for name, path in required_files.items():
        if name == "cries_directory":
            # Check if directory exists and has audio files
            status[name] = path.exists() and len(list(path.glob("*.ogg"))) > 0
        else:
            status[name] = path.exists()
    
    return status


def get_missing_files() -> list:
    """
    Get list of missing data files.
    
    Returns:
        List of missing file names
    """
    status = check_data_files()
    return [name for name, exists in status.items() if not exists]


def initialize_dataset(max_pokemon: int = 100, progress_callback=None) -> Tuple[bool, str]:
    """
    Initialize the dataset by running the necessary scripts.
    
    Args:
        max_pokemon: Maximum number of Pokémon to download (default: 100)
        progress_callback: Optional callback function to report progress
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    project_root = get_project_root()
    
    # Create data directory if it doesn't exist
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    steps = [
        ("Fetching Pokémon stats", "scripts/fetch_stats.py", []),
        ("Downloading Pokémon cries", "scripts/download_cries.py", []),
        ("Extracting audio features", "scripts/extract_audio_features_advanced.py", []),
        ("Merging datasets", "scripts/merge_dataset_advanced.py", []),
    ]
    
    messages = []
    
    for step_name, script_name, args in steps:
        if progress_callback:
            progress_callback(f"⏳ {step_name}...")
        
        script_path = project_root / script_name
        
        if not script_path.exists():
            error_msg = f"❌ Error: {script_name} not found at {script_path}"
            return False, error_msg
        
        try:
            # Run the script
            cmd = [sys.executable, str(script_path)] + args
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(project_root)
            )
            
            if result.returncode != 0:
                error_msg = f"❌ Failed at step: {step_name}\n{result.stderr}"
                return False, error_msg
            
            messages.append(f"✅ {step_name} completed")
            
        except subprocess.TimeoutExpired:
            error_msg = f"❌ Timeout: {step_name} took too long (>5 minutes)"
            return False, error_msg
        except Exception as e:
            error_msg = f"❌ Error during {step_name}: {str(e)}"
            return False, error_msg
    
    success_msg = "\n".join(messages) + "\n\n🎉 Dataset initialization complete!"
    return True, success_msg


def get_dataset_info() -> Dict[str, any]:
    """
    Get information about the current dataset.
    
    Returns:
        Dictionary with dataset statistics
    """
    project_root = get_project_root()
    
    info = {
        "total_pokemon": 0,
        "audio_files": 0,
        "has_features": False,
        "has_processed": False,
    }
    
    try:
        # Count audio files
        cries_dir = project_root / "data" / "cries"
        if cries_dir.exists():
            info["audio_files"] = len(list(cries_dir.glob("*.ogg")))
        
        # Check features files
        features_file = project_root / "data" / "audio_features_advanced.csv"
        if features_file.exists():
            info["has_features"] = True
            import pandas as pd
            df = pd.read_csv(features_file)
            info["total_pokemon"] = len(df)
        
        # Check processed file
        processed_file = project_root / "data" / "processed_features_advanced.csv"
        info["has_processed"] = processed_file.exists()
        
    except Exception as e:
        info["error"] = str(e)
    
    return info
