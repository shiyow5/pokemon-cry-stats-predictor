"""
Predict tab for Pokémon Cry Stats Predictor Dashboard
"""

import streamlit as st
import numpy as np
import requests
from io import BytesIO
from dashboard.utils.audio_processor import extract_features_from_audio
from dashboard.utils.model_loader import load_best_model, get_available_models_info, load_model_by_type
from dashboard.utils.similarity import find_similar_pokemon, get_pokemon_stats

# Import audio recorder
try:
    from audio_recorder_streamlit import audio_recorder
except ImportError:
    audio_recorder = None


TARGET_STATS = ["hp", "attack", "defense", "speed", "sp_attack", "sp_defense"]


def get_pokemon_image_url(pokemon_name):
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
        st.warning(f"Could not fetch image for {pokemon_name}: {str(e)}")
        return None


def display_predictions(predictions, similar_pokemon):
    """
    Display prediction results and similar Pokémon
    
    Args:
        predictions: Dictionary of predicted stats
        similar_pokemon: List of (name, distance, features) tuples
    """
    st.subheader("🎯 Predicted Stats")
    
    # Create columns for stats display
    cols = st.columns(3)
    for i, (stat, value) in enumerate(predictions.items()):
        with cols[i % 3]:
            st.metric(label=stat.upper().replace("_", " "), value=f"{value:.1f}")
    
    st.divider()
    
    # Display similar Pokémon
    st.subheader("🔍 Most Similar Pokémon")
    st.write("Based on audio feature similarity (Euclidean distance):")
    
    for rank, (name, distance, features) in enumerate(similar_pokemon[:3], 1):
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 2])
            
            with col1:
                # Display Pokémon image
                image_url = get_pokemon_image_url(name)
                if image_url:
                    st.image(image_url, width=150)
                else:
                    st.write("🎮")
            
            with col2:
                st.markdown(f"### #{rank} {name.title()}")
                st.write(f"**Distance:** {distance:.2f}")
            
            with col3:
                # Get actual stats for this Pokémon
                actual_stats = get_pokemon_stats(name)
                if actual_stats:
                    st.write("**Actual Stats:**")
                    for stat in TARGET_STATS:
                        st.write(f"{stat.upper()}: {actual_stats.get(stat, 'N/A')}")
                else:
                    st.write("Stats not available")
            
            st.divider()


def render():
    """Render the Predict tab"""
    st.header("🎤 Predict Pokémon Stats from Audio")
    st.write(
        "Upload an audio file or record audio using your microphone to predict Pokémon stats. "
        "The system will also show the most similar Pokémon based on audio features."
    )
    
    # Check if data files exist
    from dashboard.utils.data_initializer import get_missing_files
    missing_files = get_missing_files()
    
    if missing_files:
        st.error("⚠️ **Missing Data Files!**")
        st.warning(f"""
Required data files are missing: {', '.join(missing_files)}

Please visit the **📁 Data Management** tab to initialize the dataset before using predictions.
        """)
        st.info("💡 Go to **📁 Data Management** → Click '🚀 Initialize Dataset Now'")
        return
    
    # Model selection section
    st.subheader("🤖 Model Selection")
    
    # Get available models
    models_info = get_available_models_info()
    
    if not models_info:
        st.warning("No trained models found. Please train models first.")
        return
    
    # Filter only available models
    available_models = [m for m in models_info if m['available']]
    
    if not available_models:
        st.error("No model files found. Please train models first.")
        return
    
    # Create model selection options
    model_options = {}
    for model_info in available_models:
        label = f"{model_info['name']} (R² = {model_info['r2']:.3f})"
        if model_info == available_models[0]:
            label += " 🏆"  # Best model indicator
        model_options[label] = model_info['type']
    
    # Get default selection (best model)
    default_label = list(model_options.keys())[0]
    
    # Initialize selected model in session state if not exists
    if "selected_model_type" not in st.session_state:
        st.session_state["selected_model_type"] = model_options[default_label]
        st.session_state["selected_model_label"] = default_label
    
    # Model selection radio buttons
    selected_label = st.radio(
        "Select a model for prediction:",
        options=list(model_options.keys()),
        index=list(model_options.keys()).index(st.session_state.get("selected_model_label", default_label)),
        help="The best model (highest R² score) is selected by default. You can choose a different model if desired."
    )
    
    selected_type = model_options[selected_label]
    
    # Check if model selection changed
    model_changed = (selected_type != st.session_state.get("selected_model_type"))
    
    # Update session state
    st.session_state["selected_model_type"] = selected_type
    st.session_state["selected_model_label"] = selected_label
    
    # Load model if not loaded or if selection changed
    if "model" not in st.session_state or "scaler" not in st.session_state or model_changed:
        with st.spinner(f"Loading {selected_label.split(' (')[0]} model..."):
            try:
                model, scaler, model_name = load_model_by_type(selected_type)
                if model is None or scaler is None:
                    st.error("Failed to load model: Model files not found")
                    st.info("Please ensure the model has been trained first.")
                    return
                st.session_state["model"] = model
                st.session_state["scaler"] = scaler
                st.session_state["model_name"] = model_name
                st.success(f"✅ {model_name} model loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")
                st.info("Please ensure the model has been trained first.")
                return

    model = st.session_state["model"]
    scaler = st.session_state["scaler"]
    model_name = st.session_state.get("model_name", "Neural Network")
    
    st.divider()

    # Create two columns for input methods
    col1, col2 = st.columns(2)
    
    audio_data = None
    audio_source = None
    
    with col1:
        st.subheader("📁 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["ogg", "wav", "mp3"],
            help="Upload a Pokémon cry audio file"
        )
        
        if uploaded_file is not None:
            audio_data = uploaded_file.read()
            audio_source = "file"
            st.audio(audio_data, format=f"audio/{uploaded_file.name.split('.')[-1]}")
    
    with col2:
        st.subheader("🎙️ Record Audio")
        
        if audio_recorder is None:
            st.warning("Audio recorder not available. Please install: pip install audio-recorder-streamlit")
        else:
            recorded_audio = audio_recorder(
                text="Click to record",
                recording_color="#e74c3c",
                neutral_color="#3498db",
                icon_name="microphone",
                icon_size="2x"
            )
            
            if recorded_audio is not None:
                audio_data = recorded_audio
                audio_source = "recording"
                if recorded_audio != st.session_state.get("last_recording"):
                    st.session_state["last_recording"] = recorded_audio
                st.audio(audio_data, format="audio/wav")
    
    # Process audio if available
    if audio_data is not None:
        st.divider()
        
        # Add a predict button
        if st.button("🔮 Predict Stats", type="primary", use_container_width=True):
            with st.spinner("Extracting features and making predictions..."):
                try:
                    # Extract features
                    features = extract_features_from_audio(audio_data)
                    st.success(f"Extracted {features.shape[1]} features from audio")
                    
                    # Scale features
                    features_scaled = scaler.transform(features)
                    
                    # Make predictions
                    predictions_array = model.predict(features_scaled, verbose=0)[0]
                    predictions = dict(zip(TARGET_STATS, predictions_array))
                    
                    # Find similar Pokémon
                    similar_pokemon = find_similar_pokemon(features, top_k=5)
                    
                    # Display results
                    st.success("Prediction complete!")
                    st.info(f"🤖 **Using Model:** {model_name}")
                    display_predictions(predictions, similar_pokemon)
                    
                    # Add comparison with most similar Pokémon
                    st.subheader("📊 Comparison with Most Similar Pokémon")
                    if similar_pokemon:
                        most_similar_name = similar_pokemon[0][0]
                        actual_stats = get_pokemon_stats(most_similar_name)
                        
                        if actual_stats:
                            # Create comparison table
                            comparison_data = {
                                "Stat": [s.upper().replace("_", " ") for s in TARGET_STATS],
                                "Predicted": [f"{predictions[s]:.1f}" for s in TARGET_STATS],
                                f"{most_similar_name.title()} (Actual)": [f"{actual_stats.get(s, 'N/A')}" for s in TARGET_STATS]
                            }
                            
                            st.table(comparison_data)
                            
                            # Calculate difference
                            if all(s in actual_stats for s in TARGET_STATS):
                                differences = {s: abs(predictions[s] - actual_stats[s]) for s in TARGET_STATS}
                                avg_diff = np.mean(list(differences.values()))
                                st.info(f"Average difference from most similar Pokémon: {avg_diff:.1f}")
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.exception(e)
    
    else:
        st.info("👆 Please upload an audio file or record audio to start prediction")
    
    # Add information section
    with st.expander("ℹ️ How it works"):
        st.write("""
        **Prediction Process:**
        
        1. **Feature Extraction**: The audio is analyzed to extract 59-dimensional features including:
           - MFCC (Mel-frequency cepstral coefficients): 26 dimensions
           - Spectral features: 3 dimensions (centroid, rolloff, zero-crossing rate)
           - Chroma features: 12 dimensions
           - Spectral contrast: 7 dimensions
           - Tonnetz: 6 dimensions
           - Tempo: 1 dimension
           - Additional spectral features: 4 dimensions
        
        2. **Normalization**: Features are scaled using the same scaler used during training
        
        3. **Prediction**: The neural network model predicts all 6 stats simultaneously
        
        4. **Similarity Search**: The system finds the most similar Pokémon by comparing 
           the extracted features with features from all Pokémon in the training dataset
        
        **Note**: The model was trained on Pokémon cries, so predictions will be most 
        accurate for audio that resembles Pokémon vocalizations.
        """)
