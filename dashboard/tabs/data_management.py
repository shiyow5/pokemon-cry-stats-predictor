"""
Data Management tab for Pokémon Cry Stats Predictor Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import requests
import librosa
import random
from typing import Optional, Dict, Tuple, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_pokemon_data() -> pd.DataFrame:
    """
    Load and merge Pokémon data from CSV files.
    Only returns Pokémon that exist in BOTH raw_stats.csv AND audio_features_advanced.csv
    (i.e., Pokémon that are actually in the training dataset).

    Returns:
        DataFrame with columns: name, species_id, hp, attack, defense,
                               speed, sp_attack, sp_defense
    """
    try:
        # Load stats data
        stats_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data/raw_stats.csv"
        )
        features_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data/audio_features_advanced.csv"
        )

        stats_df = pd.read_csv(stats_path)
        features_df = pd.read_csv(features_path)

        # Handle empty datasets
        if stats_df.empty or features_df.empty:
            return pd.DataFrame(columns=['name', 'species_id', 'hp', 'attack', 'defense', 'speed', 'sp_attack', 'sp_defense'])

        # Only keep Pokémon that exist in BOTH files
        common_names = set(stats_df['name']) & set(features_df['name'])
        stats_df = stats_df[stats_df['name'].isin(common_names)]

        # Sort by species_id for better display
        if not stats_df.empty:
            stats_df = stats_df.sort_values('species_id').reset_index(drop=True)

        return stats_df
    except Exception as e:
        st.error(f"Error loading Pokémon data: {str(e)}")
        return pd.DataFrame(columns=['name', 'species_id', 'hp', 'attack', 'defense', 'speed', 'sp_attack', 'sp_defense'])


@st.cache_data(ttl=3600)  # Cache for 1 hour (images don't change)
def get_pokemon_image_url(pokemon_name: str) -> Optional[str]:
    """
    Get Pokémon image URL from PokeAPI

    Args:
        pokemon_name: Name of the Pokémon

    Returns:
        Image URL or None if not found
    """
    try:
        # Convert name to lowercase and handle special cases
        api_name = pokemon_name.lower().replace(" ", "-")

        # Fetch from PokeAPI
        response = requests.get(f"https://pokeapi.co/api/v2/pokemon/{api_name}", timeout=5)

        if response.status_code == 200:
            data = response.json()
            # Try to get official artwork first, fallback to front_default
            image_url = (
                data.get("sprites", {}).get("other", {}).get("official-artwork", {}).get("front_default")
                or data.get("sprites", {}).get("front_default")
            )
            return image_url
        else:
            return None
    except Exception as e:
        return None


def extract_audio_features(audio_path: str) -> Dict:
    """
    Extract 59-dimensional audio features from audio file.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Dictionary with 59 audio features
    """
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Basic features
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # Zero crossing rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # Spectral features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # Advanced features
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    
    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Additional spectral features
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    rms = np.mean(librosa.feature.rms(y=y))
    rms_std = np.std(librosa.feature.rms(y=y))
    
    # Combine features
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


def fetch_pokemon_from_api(identifier: str) -> Optional[Dict]:
    """
    Fetch Pokémon data from PokeAPI.
    
    Args:
        identifier: Pokémon name or Pokédex number
    
    Returns:
        Dict with pokemon data or None if not found
    """
    try:
        # Clean identifier
        identifier = str(identifier).lower().strip().replace(" ", "-")
        
        # Fetch from PokeAPI
        response = requests.get(f"https://pokeapi.co/api/v2/pokemon/{identifier}", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract relevant information
            pokemon_data = {
                'name': data['name'],
                'species_id': data['id'],
                'hp': next((s['base_stat'] for s in data['stats'] if s['stat']['name'] == 'hp'), 0),
                'attack': next((s['base_stat'] for s in data['stats'] if s['stat']['name'] == 'attack'), 0),
                'defense': next((s['base_stat'] for s in data['stats'] if s['stat']['name'] == 'defense'), 0),
                'speed': next((s['base_stat'] for s in data['stats'] if s['stat']['name'] == 'speed'), 0),
                'sp_attack': next((s['base_stat'] for s in data['stats'] if s['stat']['name'] == 'special-attack'), 0),
                'sp_defense': next((s['base_stat'] for s in data['stats'] if s['stat']['name'] == 'special-defense'), 0),
                'cry_url': data.get('cries', {}).get('latest', None)
            }
            
            return pokemon_data
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching Pokémon from API: {str(e)}")
        return None


def add_pokemon(identifier: str) -> Tuple[bool, str]:
    """
    Add a new Pokémon to the dataset.
    
    Args:
        identifier: Pokémon name or Pokédex number
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Fetch Pokemon data from API
        pokemon_data = fetch_pokemon_from_api(identifier)
        
        if not pokemon_data:
            return False, f"Could not find Pokémon '{identifier}' in PokeAPI."
        
        pokemon_name = pokemon_data['name']
        
        # Paths to data files
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        stats_path = os.path.join(base_path, "data/raw_stats.csv")
        features_path = os.path.join(base_path, "data/audio_features_advanced.csv")
        audio_dir = os.path.join(base_path, "data/cries")
        audio_path = os.path.join(audio_dir, f"{pokemon_name}.ogg")
        
        # Load existing data
        stats_df = pd.read_csv(stats_path)
        
        # Check if Pokemon already exists
        if pokemon_name in stats_df['name'].values:
            return False, f"Pokémon '{pokemon_name}' already exists in the dataset."
        
        # Download cry audio if available
        audio_downloaded = False
        if pokemon_data['cry_url']:
            try:
                audio_response = requests.get(pokemon_data['cry_url'], timeout=10)
                if audio_response.status_code == 200:
                    os.makedirs(audio_dir, exist_ok=True)
                    with open(audio_path, 'wb') as f:
                        f.write(audio_response.content)
                    audio_downloaded = True
            except Exception as e:
                pass  # Continue even if audio download fails
        
        # Add to raw_stats.csv
        new_stats_row = pd.DataFrame([{
            'name': pokemon_name,
            'species_id': pokemon_data['species_id'],
            'hp': pokemon_data['hp'],
            'attack': pokemon_data['attack'],
            'defense': pokemon_data['defense'],
            'speed': pokemon_data['speed'],
            'sp_attack': pokemon_data['sp_attack'],
            'sp_defense': pokemon_data['sp_defense']
        }])
        
        stats_df = pd.concat([stats_df, new_stats_row], ignore_index=True)
        stats_df = stats_df.sort_values('species_id').reset_index(drop=True)
        stats_df.to_csv(stats_path, index=False)
        
        # For audio features, we would need to extract them
        # For now, add a placeholder row with zeros or skip
        message = f"✅ Successfully added '{pokemon_name.title()}' (#{pokemon_data['species_id']}) to raw_stats.csv."
        
        if audio_downloaded:
            message += f" Audio file downloaded."
            
            # Extract audio features automatically
            try:
                features = extract_audio_features(audio_path)
                features['name'] = pokemon_name
                
                # Load and update audio_features_advanced.csv
                features_df = pd.read_csv(features_path)
                new_features_row = pd.DataFrame([features])
                features_df = pd.concat([features_df, new_features_row], ignore_index=True)
                features_df.to_csv(features_path, index=False)
                
                message += " ✅ Audio features extracted automatically."
            except Exception as e:
                message += f" ⚠️ Audio features extraction failed: {str(e)}"
        else:
            message += " ⚠️ Audio file not available. Cannot extract features."
        
        return True, message
        
    except Exception as e:
        return False, f"Error adding Pokémon: {str(e)}"


def delete_pokemon(pokemon_name: str) -> Tuple[bool, str]:
    """
    Delete a Pokémon from all data files.
    
    Args:
        pokemon_name: Pokémon name
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Paths to data files
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        stats_path = os.path.join(base_path, "data/raw_stats.csv")
        features_path = os.path.join(base_path, "data/audio_features_advanced.csv")
        audio_path = os.path.join(base_path, f"data/cries/{pokemon_name}.ogg")
        
        # Load CSV files
        stats_df = pd.read_csv(stats_path)
        features_df = pd.read_csv(features_path)
        
        # Check if Pokemon exists
        if pokemon_name not in stats_df['name'].values:
            return False, f"Pokémon '{pokemon_name}' not found in dataset."
        
        # Remove from both dataframes
        stats_df = stats_df[stats_df['name'] != pokemon_name]
        features_df = features_df[features_df['name'] != pokemon_name]
        
        # Save updated CSV files
        stats_df.to_csv(stats_path, index=False)
        features_df.to_csv(features_path, index=False)
        
        # Try to delete audio file (optional - don't fail if missing)
        if os.path.exists(audio_path):
            os.remove(audio_path)
            audio_msg = "Audio file deleted."
        else:
            audio_msg = "Audio file not found (skipped)."
        
        return True, f"Successfully deleted '{pokemon_name}'. {audio_msg}"
        
    except Exception as e:
        return False, f"Error deleting Pokémon: {str(e)}"


@st.dialog("Delete Pokémon")
def confirm_delete_dialog():
    """
    Modal dialog for confirming Pokémon deletion.
    """
    pokemon_name = st.session_state.get('delete_pokemon_name', '')
    pokemon_id = st.session_state.get('delete_pokemon_id', 0)

    st.warning(f"⚠️ Are you sure you want to delete **{pokemon_name.title()}** (#{pokemon_id:03d})?")
    st.write("This action will:")
    st.write("- Remove from `raw_stats.csv`")
    st.write("- Remove from `audio_features_advanced.csv`")
    st.write("- Delete the audio file from `data/cries/`")
    st.write("")
    st.error("**This action cannot be undone!**")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Confirm Delete", type="primary", width='stretch', key="dialog_confirm"):
            with st.spinner(f"Deleting {pokemon_name}..."):
                success, message = delete_pokemon(pokemon_name)
                if success:
                    st.success(message)
                    st.session_state.delete_success = True
                    st.session_state.pop('delete_pokemon_name', None)
                    st.session_state.pop('delete_pokemon_id', None)
                    # Clear cache to force data reload
                    load_pokemon_data.clear()
                else:
                    st.error(message)
                    st.session_state.delete_success = False
            st.rerun()

    with col2:
        if st.button("❌ Cancel", width='stretch', key="dialog_cancel"):
            st.session_state.pop('delete_pokemon_name', None)
            st.session_state.pop('delete_pokemon_id', None)
            st.rerun()


def display_pokemon_card(pokemon_data: pd.Series, col) -> None:
    """
    Display a single Pokémon card in the given Streamlit column.

    Args:
        pokemon_data: Series with Pokémon data
        col: Streamlit column object to render into
    """
    with col:
        with st.container():
            # Card border and styling
            st.markdown("""
            <style>
            .pokemon-card {
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 10px;
                margin: 10px 0;
                background-color: #f9f9f9;
            }
            </style>
            """, unsafe_allow_html=True)

            # Get Pokemon image
            image_url = get_pokemon_image_url(pokemon_data['name'])

            if image_url:
                # Use smaller width for faster loading
                st.image(image_url, width=120)
            else:
                st.write("🎮")  # Placeholder if image not available

            # Pokemon name and ID
            st.markdown(f"### #{int(pokemon_data['species_id']):03d} - {pokemon_data['name'].title()}")

            # Stats display in two columns
            stat_col1, stat_col2 = st.columns(2)

            with stat_col1:
                st.metric("HP", int(pokemon_data['hp']))
                st.metric("Attack", int(pokemon_data['attack']))
                st.metric("Defense", int(pokemon_data['defense']))

            with stat_col2:
                st.metric("Speed", int(pokemon_data['speed']))
                st.metric("Sp. Attack", int(pokemon_data['sp_attack']))
                st.metric("Sp. Defense", int(pokemon_data['sp_defense']))

            # Delete button - opens modal dialog
            if st.button(f"🗑️ Delete", key=f"delete_{pokemon_data['name']}", width='stretch'):
                st.session_state.delete_pokemon_name = pokemon_data['name']
                st.session_state.delete_pokemon_id = int(pokemon_data['species_id'])
                st.session_state.show_delete_dialog = True
                st.rerun()


def bulk_add_pokemon(count: int, mode: str = "sequential") -> Tuple[int, List[str]]:
    """
    Add multiple Pokémon to the dataset automatically.
    
    Args:
        count: Number of Pokémon to add
        mode: "sequential" (by Pokédex number) or "random"
    
    Returns:
        Tuple of (successful_count, list of error messages)
    """
    try:
        # Load existing data to find what's already in dataset
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        stats_path = os.path.join(base_path, "data/raw_stats.csv")
        stats_df = pd.read_csv(stats_path)
        existing_ids = set(stats_df['species_id'].values)
        
        # PokeAPI has approximately 1025 Pokémon (as of Generation 9)
        # We'll try IDs from 1 to 1025
        max_pokemon_id = 1025
        available_ids = [i for i in range(1, max_pokemon_id + 1) if i not in existing_ids]
        
        if not available_ids:
            return 0, ["No new Pokémon available to add."]
        
        # Select IDs based on mode
        if mode == "random":
            selected_ids = random.sample(available_ids, min(count, len(available_ids)))
        else:  # sequential
            selected_ids = available_ids[:min(count, len(available_ids))]
        
        # Add each Pokémon
        success_count = 0
        errors = []
        
        for pokemon_id in selected_ids:
            success, message = add_pokemon(str(pokemon_id))
            if success:
                success_count += 1
            else:
                errors.append(f"ID {pokemon_id}: {message}")
        
        return success_count, errors
        
    except Exception as e:
        return 0, [f"Error in bulk add: {str(e)}"]


def reset_dataset() -> Tuple[bool, str]:
    """
    Reset the entire dataset by deleting all entries from both CSV files and all audio files.
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        stats_path = os.path.join(base_path, "data/raw_stats.csv")
        features_path = os.path.join(base_path, "data/audio_features_advanced.csv")
        audio_dir = os.path.join(base_path, "data/cries")
        
        # Read current data to get count
        try:
            stats_df = pd.read_csv(stats_path)
            original_count = len(stats_df)
        except:
            original_count = 0
        
        # Define proper column headers
        stats_columns = ['name', 'species_id', 'hp', 'attack', 'defense', 'speed', 'sp_attack', 'sp_defense']
        
        # Audio features columns (59 features)
        features_columns = ['name']
        features_columns += [f'mfcc_mean_{i+1}' for i in range(13)]
        features_columns += [f'mfcc_std_{i+1}' for i in range(13)]
        features_columns += ['zcr', 'spectral_centroid', 'rolloff']
        features_columns += [f'chroma_{i+1}' for i in range(12)]
        features_columns += [f'contrast_{i+1}' for i in range(7)]
        features_columns += [f'tonnetz_{i+1}' for i in range(6)]
        features_columns += ['tempo', 'spectral_bandwidth', 'spectral_flatness', 'rms', 'rms_std']
        
        # Create empty dataframes with proper columns
        empty_stats_df = pd.DataFrame(columns=stats_columns)
        empty_features_df = pd.DataFrame(columns=features_columns)
        
        # Save empty CSVs
        empty_stats_df.to_csv(stats_path, index=False)
        empty_features_df.to_csv(features_path, index=False)
        
        # Delete all audio files
        deleted_audio_count = 0
        if os.path.exists(audio_dir):
            for filename in os.listdir(audio_dir):
                if filename.endswith('.ogg'):
                    try:
                        os.remove(os.path.join(audio_dir, filename))
                        deleted_audio_count += 1
                    except Exception:
                        pass
        
        return True, f"✅ Dataset reset successfully! Removed {original_count} Pokémon and {deleted_audio_count} audio files."
        
    except Exception as e:
        return False, f"❌ Error resetting dataset: {str(e)}"


def render():
    """Render the Data Management tab"""
    st.header("📁 Training Data Management")
    st.write(
        "Manage the Pokémon training dataset. View, search, delete, or add Pokémon to the training data."
    )
    
    # Import data initializer
    from dashboard.utils.data_initializer import check_data_files, get_missing_files, initialize_dataset
    
    # Check if data files exist
    missing_files = get_missing_files()
    
    if missing_files:
        st.error(f"⚠️ **Missing Data Files Detected!**")
        st.warning(f"""
The following required data files are missing:
{chr(10).join([f'- {file}' for file in missing_files])}

This usually happens on first deployment or when data files are not in the repository.
        """)
        
        st.info("""
**Solution:** Click the button below to automatically initialize the dataset.
This will:
1. Fetch Pokémon stats from external API
2. Download Pokémon cry audio files
3. Extract audio features
4. Create all necessary data files

This process may take 5-10 minutes depending on the number of Pokémon.
        """)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Initialize Dataset Now", type="primary", use_container_width=True):
                progress_placeholder = st.empty()
                
                def progress_callback(message):
                    progress_placeholder.info(message)
                
                with st.spinner("Initializing dataset..."):
                    success, message = initialize_dataset(max_pokemon=100, progress_callback=progress_callback)
                    
                    if success:
                        st.success(message)
                        st.balloons()
                        st.info("🔄 Reloading page to show the new dataset...")
                        # Clear cache
                        load_pokemon_data.clear()
                        st.rerun()
                    else:
                        st.error(message)
                        st.error("Please check the logs or try running the initialization scripts manually.")
        
        st.divider()
        st.warning("⚠️ The Data Management features below require the dataset to be initialized first.")
        return  # Don't show the rest of the UI until data is initialized

    # Check if delete dialog should be shown
    if st.session_state.get('show_delete_dialog', False):
        st.session_state.pop('show_delete_dialog', None)
        confirm_delete_dialog()

    # Load data
    pokemon_df = load_pokemon_data()

    # Show info if dataset is empty, but don't return - allow adding Pokémon
    if pokemon_df.empty:
        st.info("📭 Dataset is currently empty. You can add Pokémon using the options below!")

    # Show success message if deletion was successful
    if 'delete_success' in st.session_state:
        if st.session_state.delete_success:
            st.success("✅ Pokémon deleted successfully!")
        del st.session_state.delete_success

    st.divider()
    
    # Only show search and pagination if there's data
    if not pokemon_df.empty:
        # Search functionality
        st.subheader("🔍 Search Pokémon")
        search_query = st.text_input(
            "Search by name or Pokédex number",
            placeholder="e.g., pikachu or 25",
            help="Enter a Pokémon name or Pokédex number to filter the list"
        )
        
        # Filter data based on search
        if search_query:
            filtered_df = pokemon_df[
                pokemon_df['name'].str.contains(search_query.lower(), case=False, na=False) |
                pokemon_df['species_id'].astype(str).str.contains(search_query, na=False)
            ]
        else:
            filtered_df = pokemon_df
    else:
        # Empty dataset - no search needed
        filtered_df = pokemon_df
        search_query = ""
    
    # Initialize pagination session state
    if 'data_mgmt_page' not in st.session_state:
        st.session_state.data_mgmt_page = 0
    if 'data_mgmt_items_per_page' not in st.session_state:
        st.session_state.data_mgmt_items_per_page = 30  # 30 items = 10 rows of 3 columns
    
    # Reset page when search query changes
    if 'last_search_query' not in st.session_state:
        st.session_state.last_search_query = ""
    if st.session_state.last_search_query != search_query:
        st.session_state.data_mgmt_page = 0
        st.session_state.last_search_query = search_query
    
    # Calculate pagination
    items_per_page = st.session_state.data_mgmt_items_per_page
    total_items = len(filtered_df)
    total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
    current_page = min(st.session_state.data_mgmt_page, total_pages - 1)
    st.session_state.data_mgmt_page = current_page
    
    # Calculate slice indices
    start_idx = current_page * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    
    # Display count with pagination info
    if total_items > 0:
        st.info(f"📊 Showing **{start_idx + 1}-{end_idx}** of **{total_items}** Pokémon (Page {current_page + 1} of {total_pages})")
    else:
        st.info(f"📊 Showing **0** of **{len(pokemon_df)}** Pokémon")
    
    # Pagination controls
    if total_pages > 1:
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        with col1:
            if st.button("⏮️ First", disabled=(current_page == 0), width='stretch'):
                st.session_state.data_mgmt_page = 0
                st.rerun()
        
        with col2:
            if st.button("◀️ Previous", disabled=(current_page == 0), width='stretch'):
                st.session_state.data_mgmt_page = current_page - 1
                st.rerun()
        
        with col3:
            st.markdown(f"<div style='text-align: center; padding: 8px;'>Page {current_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
        
        with col4:
            if st.button("Next ▶️", disabled=(current_page >= total_pages - 1), width='stretch'):
                st.session_state.data_mgmt_page = current_page + 1
                st.rerun()
        
        with col5:
            if st.button("Last ⏭️", disabled=(current_page >= total_pages - 1), width='stretch'):
                st.session_state.data_mgmt_page = total_pages - 1
                st.rerun()
    
    st.divider()
    
    # Display Pokemon in grid (paginated)
    if not filtered_df.empty:
        # Slice data for current page
        page_df = filtered_df.iloc[start_idx:end_idx]
        
        # Create grid with 3 columns
        cols_per_row = 3
        rows = (len(page_df) + cols_per_row - 1) // cols_per_row
        
        for row in range(rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                pokemon_idx = row * cols_per_row + col_idx
                if pokemon_idx < len(page_df):
                    pokemon = page_df.iloc[pokemon_idx]
                    display_pokemon_card(pokemon, cols[col_idx])
    else:
        st.warning("No Pokémon found matching your search criteria.")
    
    st.divider()
    
    # Add Pokemon section
    st.subheader("➕ Add New Pokémon")
    
    with st.expander("Add a Pokémon to the dataset"):
        st.write("Enter a Pokémon name or Pokédex number to add it to the training dataset.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            add_input = st.text_input(
                "Pokémon Name or Pokédex Number",
                placeholder="e.g., mewtwo or 150",
                key="add_pokemon_input"
            )
        
        with col2:
            add_button = st.button("➕ Add", type="primary", width='stretch')
        
        if add_button and add_input:
            with st.spinner(f"Adding Pokémon: {add_input}..."):
                success, message = add_pokemon(add_input)
                if success:
                    st.success(message)
                    # Clear cache to force data reload
                    load_pokemon_data.clear()
                    st.rerun()
                else:
                    st.error(message)
    
    # Bulk add Pokemon section
    # Initialize expander state in session
    if 'bulk_add_expanded' not in st.session_state:
        st.session_state.bulk_add_expanded = False

    with st.expander("Add Multiple Pokémon Automatically", expanded=st.session_state.bulk_add_expanded):
        st.write("Automatically add multiple Pokémon to the training dataset.")

        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            bulk_count = st.slider(
                "Number of Pokémon to Add",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
                key="bulk_count_slider",
                help="Slide to select how many Pokémon you want to add automatically",
                on_change=lambda: setattr(st.session_state, 'bulk_add_expanded', True)
            )

        with col2:
            bulk_mode = st.radio(
                "Selection Mode",
                options=["sequential", "random"],
                key="bulk_mode_radio",
                format_func=lambda x: "📋 Sequential (by Pokédex #)" if x == "sequential" else "🎲 Random",
                help="Sequential: Add Pokémon in order by Pokédex number. Random: Add randomly selected Pokémon.",
                on_change=lambda: setattr(st.session_state, 'bulk_add_expanded', True)
            )

        with col3:
            st.write("")  # Spacing
            st.write("")  # Spacing
            bulk_add_button = st.button("🚀 Bulk Add", type="primary", key="bulk_add_button")

        # Show preview of IDs that will be added
        st.divider()
        st.write("**Preview:**")

        # Calculate preview IDs
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        stats_path = os.path.join(base_path, "data/raw_stats.csv")

        try:
            stats_df = pd.read_csv(stats_path)
            existing_ids = set(stats_df['species_id'].values)
            max_pokemon_id = 1025
            available_ids = sorted([i for i in range(1, max_pokemon_id + 1) if i not in existing_ids])

            if not available_ids:
                st.warning("⚠️ No new Pokémon available to add!")
            else:
                # Preview IDs based on current selection
                if bulk_mode == "random":
                    preview_text = f"Will add {min(bulk_count, len(available_ids))} random Pokémon from {len(available_ids)} available"
                else:  # sequential
                    preview_ids = available_ids[:min(bulk_count, len(available_ids))]
                    if len(preview_ids) <= 10:
                        preview_text = f"Will add: #{', #'.join(map(str, preview_ids))}"
                    else:
                        preview_text = f"Will add: #{preview_ids[0]} - #{preview_ids[-1]} ({len(preview_ids)} Pokémon)"

                st.info(f"📋 {preview_text}")
                st.caption(f"Current dataset: {len(existing_ids)} Pokémon | Available: {len(available_ids)} Pokémon")
        except Exception as e:
            st.error(f"Error calculating preview: {str(e)}")

        if bulk_add_button:
            st.session_state.bulk_add_expanded = True
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Use a simpler approach: call bulk add and show results
            with st.spinner(f"Adding {bulk_count} Pokémon in {bulk_mode} mode..."):
                # Get available IDs
                base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                stats_path = os.path.join(base_path, "data/raw_stats.csv")
                stats_df = pd.read_csv(stats_path)
                existing_ids = set(stats_df['species_id'].values)

                # Support all Pokemon generations (ID 1-1025)
                max_pokemon_id = 1025
                available_ids = sorted([i for i in range(1, max_pokemon_id + 1) if i not in existing_ids])
                
                if not available_ids:
                    st.error("❌ No new Pokémon available to add!")
                else:
                    # Select IDs
                    if bulk_mode == "random":
                        selected_ids = random.sample(available_ids, min(bulk_count, len(available_ids)))
                    else:  # sequential
                        selected_ids = sorted(available_ids[:min(bulk_count, len(available_ids))])
                    
                    # Add each Pokémon with progress tracking
                    success_count = 0
                    errors = []
                    successfully_added_ids = []

                    for idx, pokemon_id in enumerate(selected_ids):
                        progress = (idx + 1) / len(selected_ids)
                        progress_bar.progress(progress)
                        status_text.text(f"Adding Pokémon #{pokemon_id}... ({idx + 1}/{len(selected_ids)})")

                        success, message = add_pokemon(str(pokemon_id))
                        if success:
                            success_count += 1
                            successfully_added_ids.append(pokemon_id)
                        else:
                            errors.append(f"ID {pokemon_id}: {message}")

                    progress_bar.progress(1.0)
                    status_text.text(f"Completed! Added {success_count} Pokémon.")

                    # Show results
                    if success_count > 0:
                        # Show ID range of successfully added Pokemon
                        if len(successfully_added_ids) <= 10:
                            ids_display = f"#{', #'.join(map(str, successfully_added_ids))}"
                        else:
                            ids_display = f"#{successfully_added_ids[0]} - #{successfully_added_ids[-1]}"
                        st.success(f"✅ Successfully added {success_count} Pokémon to the dataset!\n\n**Added IDs:** {ids_display}")

                    if errors:
                        st.warning(f"⚠️ {len(errors)} Pokémon could not be added:")
                        with st.expander("View errors"):
                            for error in errors[:10]:  # Show first 10 errors
                                st.text(error)
                            if len(errors) > 10:
                                st.text(f"... and {len(errors) - 10} more errors")

                    if success_count > 0:
                        st.info("🔄 Reloading page to show new Pokémon...")
                        # Clear cache to force data reload
                        load_pokemon_data.clear()
                        st.rerun()
    
    # Bulk delete / reset section
    with st.expander("⚠️ Danger Zone - Reset Dataset"):
        st.warning("**Warning**: This will delete ALL Pokémon from the training dataset!")
        st.write("""
        This action will:
        - Remove all entries from `raw_stats.csv`
        - Remove all entries from `audio_features_advanced.csv`
        - Delete all audio files from `data/cries/`
        
        **This action cannot be undone!**
        """)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            confirm_text = st.text_input(
                "Type 'DELETE ALL' to confirm:",
                key="reset_confirm_text",
                placeholder="DELETE ALL"
            )
        
        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            reset_button = st.button(
                "🗑️ Reset Dataset",
                type="secondary",
                width='stretch',
                disabled=(confirm_text != "DELETE ALL")
            )
        
        if reset_button and confirm_text == "DELETE ALL":
            with st.spinner("Resetting dataset..."):
                success, message = reset_dataset()
                if success:
                    st.success(message)
                    st.info("🔄 Reloading page...")
                    # Clear the confirmation text
                    if 'reset_confirm_text' in st.session_state:
                        del st.session_state['reset_confirm_text']
                    # Clear cache to force data reload
                    load_pokemon_data.clear()
                    st.rerun()
                else:
                    st.error(message)
    
    # Information section
    with st.expander("ℹ️ About Training Data"):
        st.write("""
        **Training Dataset Information:**
        
        The training dataset consists of Pokémon data stored in CSV files:
        - `data/raw_stats.csv`: Contains Pokémon stats (HP, Attack, Defense, Speed, Sp. Attack, Sp. Defense)
        - `data/audio_features_advanced.csv`: Contains 59-dimensional audio features extracted from cries
        - `data/cries/*.ogg`: Audio files of Pokémon cries
        
        **Managing Data:**
        - **Search**: Filter Pokémon by name or Pokédex number
        - **Delete**: Remove Pokémon from the training dataset (updates both CSV files)
        - **Add**: Add new Pokémon by fetching data from PokeAPI
        
        **Important Notes:**
        - Changes to the dataset require re-training models to take effect
        - Deleting a Pokémon removes it from all data files
        - Adding a Pokémon fetches stats from PokeAPI and downloads the cry audio
        """)
