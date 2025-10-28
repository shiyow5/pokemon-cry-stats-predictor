import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dashboard.utils.model_loader import load_latest_results


def render():
    """Render the Model Evaluation tab"""
    st.header("📊 Model Performance Evaluation")
    
    st.markdown("""
    This tab displays comprehensive evaluation metrics for different machine learning models
    trained to predict Pokémon stats from their cries.
    """)
    
    # Load results
    results = load_latest_results()
    
    if results is None:
        st.error("⚠️ **No model results found!**")
        st.warning("""
No trained models or evaluation results were found.

This usually means:
- Models haven't been trained yet
- Data files are missing
- Training was interrupted or failed
        """)
        st.info("""
**How to fix:**
1. Go to the **🏋️ Train** tab to train models using the UI, OR
2. Run training via command line: `python scripts/train_model_advanced.py`

**Note:** Training requires the dataset to be initialized first. Visit **📁 Data Management** if needed.
        """)
        return
    
    # Overall comparison section
    st.subheader("🏆 Overall R² Score Comparison")
    
    comparison_data = []
    for model_name, model_data in results.items():
        if 'overall_r2' in model_data:
            comparison_data.append({
                'Model': model_data['model'],
                'Overall R²': model_data['overall_r2']
            })
    
    if not comparison_data:
        st.warning("No model comparison data available")
        return
    
    df = pd.DataFrame(comparison_data).sort_values('Overall R²', ascending=False)
    
    # Display best model
    best_model = df.iloc[0]
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🥇 Best Model", best_model['Model'])
    with col2:
        st.metric("Best R² Score", f"{best_model['Overall R²']:.3f}")
    with col3:
        improvement = ((best_model['Overall R²'] - df.iloc[-1]['Overall R²']) / abs(df.iloc[-1]['Overall R²']) * 100) if len(df) > 1 else 0
        st.metric("Improvement over Worst", f"{improvement:.1f}%")
    
    # Bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=df['Model'],
            y=df['Overall R²'],
            marker_color=['green' if i == 0 else 'skyblue' for i in range(len(df))],
            text=[f"{v:.3f}" for v in df['Overall R²']],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Model R² Comparison",
        xaxis_title="Model",
        yaxis_title="R² Score",
        showlegend=False,
        height=400
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="red", 
                  annotation_text="Baseline (R²=0)", annotation_position="right")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed stats section
    st.subheader("📈 Detailed Performance by Stat")
    
    stats_data = []
    for model_name, model_data in results.items():
        if 'stats' in model_data:
            for stat, metrics in model_data['stats'].items():
                stats_data.append({
                    'Model': model_data['model'],
                    'Stat': stat.upper(),
                    'R²': metrics['r2'],
                    'RMSE': metrics['rmse']
                })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        
        # Pivot for heatmap
        heatmap_df = stats_df.pivot(index='Model', columns='Stat', values='R²')
        
        fig_heat = px.imshow(
            heatmap_df,
            labels=dict(x="Pokémon Stat", y="Model", color="R² Score"),
            title="R² Score Heatmap: Models vs Stats",
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            aspect="auto",
            text_auto='.3f'
        )
        
        st.plotly_chart(fig_heat, use_container_width=True)
        
        # Detailed table
        with st.expander("📋 View Detailed Metrics Table"):
            st.dataframe(
                stats_df.sort_values(['Model', 'Stat']),
                use_container_width=True,
                hide_index=True
            )
    
    # Cross-validation section
    if 'cross_validation_rf' in results:
        st.subheader("🔄 Cross-Validation Results (Random Forest)")
        
        cv = results['cross_validation_rf']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean CV Score", f"{cv['mean_cv_score']:.3f}")
        with col2:
            st.metric("Std Dev", f"{cv['std_cv_score']:.3f}")
        
        # CV plot
        folds = list(range(1, len(cv['cv_scores']) + 1))
        
        fig_cv = go.Figure()
        
        fig_cv.add_trace(go.Scatter(
            x=folds,
            y=cv['cv_scores'],
            mode='lines+markers',
            name='CV Score',
            line=dict(color='blue', width=2),
            marker=dict(size=10)
        ))
        
        fig_cv.add_hline(
            y=cv['mean_cv_score'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {cv['mean_cv_score']:.3f}",
            annotation_position="right"
        )
        
        fig_cv.update_layout(
            title="5-Fold Cross Validation Results",
            xaxis_title="Fold",
            yaxis_title="R² Score",
            height=400
        )
        
        st.plotly_chart(fig_cv, use_container_width=True)
    
    # Model insights
    st.subheader("💡 Key Insights")
    
    st.markdown("""
    - **Low R² scores** are expected: Pokémon cries and stats have weak correlation by design
    - **Neural Networks** tend to perform better with more features (59D vs 29D)
    - **SP_ATTACK and SP_DEFENSE** are generally more predictable than SPEED
    - **Cross-validation** shows model stability across different data splits
    """)
    
    # Download results
    st.subheader("📥 Download Results")
    
    results_json = pd.Series(results).to_json(indent=2)
    st.download_button(
        label="Download Full Results (JSON)",
        data=results_json,
        file_name="model_results.json",
        mime="application/json"
    )
