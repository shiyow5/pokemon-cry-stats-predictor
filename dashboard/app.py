import streamlit as st
import sys
import os

# Add parent directory to path to import pages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dashboard.tabs import evaluation, predict, train, data_management

# Page configuration
st.set_page_config(
    page_title="Pokémon Cry Stats Predictor",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Title and description
st.title("🎮 Pokémon Cry Stats Predictor Dashboard")

st.markdown("""
Welcome to the **Pokémon Cry Stats Predictor**! This interactive dashboard allows you to:

- 📊 **Model Evaluation**: View performance metrics and visualizations comparing different ML models
- 🎤 **Predict**: Upload audio or record from your microphone to predict Pokémon stats
- 🏋️ **Train**: Configure and train new models with custom parameters
- 📁 **Data Management**: View, search, delete, or add Pokémon to the training dataset

---
""")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Evaluation", "🎤 Predict", "🏋️ Train", "📁 Data Management"])

with tab1:
    evaluation.render()

with tab2:
    predict.render()

with tab3:
    train.render()

with tab4:
    data_management.render()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    Pokémon Cry Stats Predictor | Built with Streamlit | 
    <a href='https://github.com/anthropics/claude-code' target='_blank'>Claude Code</a>
</div>
""", unsafe_allow_html=True)
