import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import load_css
from core.simulation import MonteCarloSimulation
from visualization.plots import (
    plot_power_curves, plot_type1_heatmap, plot_relative_efficiency, 
    plot_decision_stability
)

st.set_page_config(page_title="Monte Carlo Simulation Dashboard", layout="wide")
load_css()

st.title("🖥️ Monte Carlo Simulation Dashboard")
st.markdown("Run rigorous statistical simulations to evaluate original vs. modified neutrosophic test performance.")

col_config, col_results = st.columns([3, 9])

with col_config:
    st.markdown("### Simulation Configuration")
    test_type = st.selectbox("Test to Simulate", ["kruskal_wallis", "mann_whitney", "moods_median"])
    
    sample_sizes = st.multiselect("Sample Sizes (n)", [20, 50, 100, 200], default=[20, 50])
    indet_levels = st.multiselect("Indeterminacy Levels (δ)", [0.0, 0.10, 0.25, 0.40], default=[0.0, 0.10, 0.25])
    distributions = st.multiselect("Distributions", ["normal", "skewed", "heavy_tailed", "uniform", "bimodal"], default=["normal", "skewed"])
    effect_sizes = st.multiselect("Effect Sizes (Cohen's d max)", [0.0, 0.2, 0.5, 0.8], default=[0.0, 0.5])
    
    n_sims = st.session_state.get('n_sims', 1000)
    st.info(f"Using {n_sims} simulations per condition (from Global Settings).")
    
    run_sim = st.button("Run Simulation", type="primary", use_container_width=True, disabled=not (sample_sizes and indet_levels and distributions and effect_sizes))

with col_results:
    if run_sim:
        sim_engine = MonteCarloSimulation(n_simulations=n_sims, random_seed=st.session_state.get('random_seed', 42))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def cb_progress(current, total):
            pct = min(1.0, current / max(1, total))
            progress_bar.progress(pct)
            status_text.text(f"Running simulation: iteration {current} of {total}...")
            
        with st.spinner("Executing Monte Carlo loops..."):
            res_df = sim_engine.run(test_type, sample_sizes, indet_levels, distributions, effect_sizes, alpha=st.session_state.get('alpha', 0.05), progress_callback=cb_progress)
            
        st.success("Simulation Complete!")
        st.session_state[f'sim_results_{test_type}'] = res_df
        
    # Display if exists in state
    if f'sim_results_{test_type}' in st.session_state:
        df = st.session_state[f'sim_results_{test_type}']
        
        t1, t2 = st.tabs(["📊 Interactive Charts", "📋 Comprehensive Data Table"])
        
        with t1:
            r1c1, r1c2 = st.columns(2)
            
            with r1c1:
                st.plotly_chart(plot_power_curves(df), use_container_width=True)
                st.plotly_chart(plot_relative_efficiency(df), use_container_width=True)
                
            with r1c2:
                # Type I error is when effect_size = 0.0
                type1_df = df[df['effect_size'] == 0.0].copy()
                st.plotly_chart(plot_type1_heatmap(type1_df), use_container_width=True)
                st.plotly_chart(plot_decision_stability(df), use_container_width=True)
                
        with t2:
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Simulation Results CSV", data=csv, file_name=f"sim_results_{test_type}.csv", mime="text/csv")
    else:
        st.info("Configure parameters and click 'Run Simulation' to see results.")
