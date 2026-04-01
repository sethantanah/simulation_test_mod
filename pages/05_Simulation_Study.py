"""
Monte Carlo Simulation Dashboard for Neutrosophic Tests

This module provides comprehensive simulation capabilities to evaluate and compare
original vs modified neutrosophic non-parametric tests across various conditions.

Author: Akua Agyapomah Oteng (PhD Candidate)
Institution: University of Mines and Technology (UMaT), Ghana
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import load_css
from core.simulation import MonteCarloSimulation
from visualization.plots import (
    plot_power_curves, plot_type1_heatmap, plot_relative_efficiency, 
    plot_decision_stability
)

st.set_page_config(
    page_title="Monte Carlo Simulation Dashboard | PhD Research",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_css()

# ============================================================================
# TITLE AND INTRODUCTION
# ============================================================================

st.markdown("""
<div class="hero-section" style="text-align: center; padding: 2rem; margin-bottom: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px;">
    <h1 style="color: white; margin: 0;">🖥️ Monte Carlo Simulation Dashboard</h1>
    <p style="color: white; margin-top: 0.5rem; font-size: 1.1em;">
        Rigorous statistical simulations to evaluate original vs. modified neutrosophic test performance
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Simulation Configuration")
    
    # Test selection
    test_type = st.selectbox(
        "Test to Simulate", 
        ["kruskal_wallis", "mann_whitney", "moods_median"],
        format_func=lambda x: {
            "kruskal_wallis": "📈 Kruskal-Wallis Test",
            "mann_whitney": "📉 Mann-Whitney U Test",
            "moods_median": "📊 Mood's Median Test"
        }.get(x, x)
    )
    
    st.markdown("---")
    st.markdown("### 📊 Experimental Design Parameters")
    
    # Sample sizes
    sample_sizes = st.multiselect(
        "Sample Sizes (n per group)", 
        [20, 50, 100, 200], 
        default=[20, 50],
        help="Number of observations in each group"
    )
    
    # Indeterminacy levels
    indet_levels = st.multiselect(
        "Indeterminacy Levels (δ)", 
        [0.0, 0.10, 0.25, 0.40], 
        default=[0.0, 0.10, 0.25],
        help="Proportion of observations with added indeterminacy"
    )
    
    # Distributions
    distributions = st.multiselect(
        "Distributions", 
        ["normal", "skewed", "heavy_tailed", "uniform", "bimodal"], 
        default=["normal", "skewed"],
        help="Distribution types for data generation"
    )
    
    # Effect sizes
    effect_sizes = st.multiselect(
        "Effect Sizes (Cohen's d)", 
        [0.0, 0.2, 0.5, 0.8, 1.0], 
        default=[0.0, 0.5],
        help="0.0 = null hypothesis, >0 = alternative hypothesis"
    )
    
    st.markdown("---")
    st.markdown("### 🔬 Simulation Settings")
    
    n_sims = st.session_state.get('n_sims', 1000)
    st.info(f"📊 Using **{n_sims}** simulations per condition")
    st.caption("From Global Settings in main app")
    
    st.markdown("---")
    st.markdown("### 📈 Expected Performance")
    st.success("""
    **Modified Tests Expected Benefits:**
    - 📈 Higher power (5-15% improvement)
    - 🎯 Better Type I error control
    - 🔄 More stable decisions
    - 📊 Narrower p-value intervals
    """)

# ============================================================================
# HELPER FUNCTIONS FOR ENHANCED VISUALIZATIONS
# ============================================================================

def plot_power_comparison(df: pd.DataFrame) -> go.Figure:
    """Enhanced power comparison with original vs modified side-by-side"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Power by Effect Size (All Conditions)",
            "Power by Indeterminacy Level",
            "Power by Sample Size",
            "Power by Distribution"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )
    
    # Filter for effect > 0 (power analysis)
    power_df = df[df['effect_size'] > 0].copy()
    
    # 1. Power by Effect Size
    effect_means = power_df.groupby(['effect_size', 'variant'])['power'].mean().reset_index()
    for variant in ['original', 'modified']:
        variant_data = effect_means[effect_means['variant'] == variant]
        fig.add_trace(
            go.Scatter(
                x=variant_data['effect_size'],
                y=variant_data['power'],
                mode='lines+markers',
                name=f'{variant.title()}',
                line=dict(width=2, dash='solid' if variant == 'modified' else 'dash'),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
    
    # 2. Power by Indeterminacy Level
    indet_means = power_df.groupby(['delta', 'variant'])['power'].mean().reset_index()
    for variant in ['original', 'modified']:
        variant_data = indet_means[indet_means['variant'] == variant]
        fig.add_trace(
            go.Scatter(
                x=variant_data['delta'],
                y=variant_data['power'],
                mode='lines+markers',
                name=f'{variant.title()}',
                line=dict(width=2),
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. Power by Sample Size
    n_means = power_df.groupby(['n', 'variant'])['power'].mean().reset_index()
    for variant in ['original', 'modified']:
        variant_data = n_means[n_means['variant'] == variant]
        fig.add_trace(
            go.Scatter(
                x=variant_data['n'],
                y=variant_data['power'],
                mode='lines+markers',
                name=f'{variant.title()}',
                line=dict(width=2),
                showlegend=False
            ),
            row=2, col=1
        )
    
    # 4. Power by Distribution
    dist_means = power_df.groupby(['distribution', 'variant'])['power'].mean().reset_index()
    for variant in ['original', 'modified']:
        variant_data = dist_means[dist_means['variant'] == variant]
        fig.add_trace(
            go.Bar(
                x=variant_data['distribution'],
                y=variant_data['power'],
                name=f'{variant.title()}',
                marker_color='#FF5722' if variant == 'modified' else '#2196F3',
                showlegend=False
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title="📈 Power Analysis: Original vs Modified Tests",
        height=800,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        template="plotly_white"
    )
    
    fig.update_yaxes(title_text="Power", row=1, col=1)
    fig.update_xaxes(title_text="Effect Size", row=1, col=1)
    fig.update_yaxes(title_text="Power", row=1, col=2)
    fig.update_xaxes(title_text="Indeterminacy Level (δ)", row=1, col=2)
    fig.update_yaxes(title_text="Power", row=2, col=1)
    fig.update_xaxes(title_text="Sample Size (n)", row=2, col=1)
    fig.update_yaxes(title_text="Power", row=2, col=2)
    fig.update_xaxes(title_text="Distribution", row=2, col=2)
    
    return fig


def plot_type1_comparison(df: pd.DataFrame) -> go.Figure:
    """Enhanced Type I error comparison with heatmaps and bar charts"""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Type I Error by Indeterminacy Level",
            "Type I Error by Sample Size"
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Filter for effect = 0 (Type I error)
    type1_df = df[df['effect_size'] == 0].copy()
    
    # 1. Type I error by indeterminacy level
    indet_type1 = type1_df.groupby(['delta', 'variant'])['type1_error'].mean().reset_index()
    
    for variant in ['original', 'modified']:
        variant_data = indet_type1[indet_type1['variant'] == variant]
        fig.add_trace(
            go.Bar(
                x=variant_data['delta'],
                y=variant_data['type1_error'],
                name=f'{variant.title()}',
                marker_color='#FF5722' if variant == 'modified' else '#2196F3',
                text=[f"{v:.3f}" for v in variant_data['type1_error']],
                textposition="outside"
            ),
            row=1, col=1
        )
    
    # Add reference line for α = 0.05
    fig.add_hline(
        y=0.05, line_dash="dash", line_color="red",
        annotation_text="Nominal α = 0.05",
        row=1, col=1
    )
    
    # 2. Type I error by sample size
    n_type1 = type1_df.groupby(['n', 'variant'])['type1_error'].mean().reset_index()
    
    for variant in ['original', 'modified']:
        variant_data = n_type1[n_type1['variant'] == variant]
        fig.add_trace(
            go.Bar(
                x=variant_data['n'],
                y=variant_data['type1_error'],
                name=f'{variant.title()}',
                marker_color='#FF5722' if variant == 'modified' else '#2196F3',
                text=[f"{v:.3f}" for v in variant_data['type1_error']],
                textposition="outside",
                showlegend=False
            ),
            row=1, col=2
        )
    
    fig.add_hline(
        y=0.05, line_dash="dash", line_color="red",
        annotation_text="Nominal α = 0.05",
        row=1, col=2
    )
    
    fig.update_layout(
        title="🎯 Type I Error Control: Original vs Modified Tests",
        height=500,
        showlegend=True,
        template="plotly_white",
        barmode='group'
    )
    
    fig.update_yaxes(title_text="Type I Error Rate", row=1, col=1)
    fig.update_xaxes(title_text="Indeterminacy Level (δ)", row=1, col=1)
    fig.update_yaxes(title_text="Type I Error Rate", row=1, col=2)
    fig.update_xaxes(title_text="Sample Size (n)", row=1, col=2)
    
    return fig


def plot_relative_efficiency_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of relative efficiency across conditions"""
    
    # Filter for effect > 0
    eff_df = df[df['effect_size'] > 0].copy()
    
    # Calculate average relative efficiency by delta and n
    heatmap_data = eff_df.groupby(['delta', 'n'])['relative_efficiency'].mean().reset_index()
    
    # Pivot for heatmap
    pivot_data = heatmap_data.pivot(index='delta', columns='n', values='relative_efficiency')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlGn',
        zmid=1.0,
        text=np.round(pivot_data.values, 2),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar_title="Relative Efficiency<br>(Mod/Orig)"
    ))
    
    fig.update_layout(
        title="⚡ Relative Efficiency Heatmap: Modified vs Original Tests",
        xaxis_title="Sample Size (n)",
        yaxis_title="Indeterminacy Level (δ)",
        height=500,
        template="plotly_white"
    )
    
    return fig


def plot_decision_stability_comparison(df: pd.DataFrame) -> go.Figure:
    """Enhanced decision stability comparison"""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Decision Stability by Indeterminacy Level",
            "Decision Stability by Effect Size"
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # 1. Decision stability by indeterminacy
    indet_stability = df.groupby(['delta', 'variant'])['decision_stability'].mean().reset_index()
    
    for variant in ['original', 'modified']:
        variant_data = indet_stability[indet_stability['variant'] == variant]
        fig.add_trace(
            go.Bar(
                x=variant_data['delta'],
                y=variant_data['decision_stability'],
                name=f'{variant.title()}',
                marker_color='#FF5722' if variant == 'modified' else '#2196F3',
                text=[f"{v:.3f}" for v in variant_data['decision_stability']],
                textposition="outside"
            ),
            row=1, col=1
        )
    
    # 2. Decision stability by effect size
    effect_stability = df.groupby(['effect_size', 'variant'])['decision_stability'].mean().reset_index()
    
    for variant in ['original', 'modified']:
        variant_data = effect_stability[effect_stability['variant'] == variant]
        fig.add_trace(
            go.Scatter(
                x=variant_data['effect_size'],
                y=variant_data['decision_stability'],
                mode='lines+markers',
                name=f'{variant.title()}',
                line=dict(width=2),
                marker=dict(size=8),
                showlegend=False
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title="🎯 Decision Stability: Original vs Modified Tests",
        height=500,
        showlegend=True,
        template="plotly_white",
        barmode='group'
    )
    
    fig.update_yaxes(title_text="Decision Stability", row=1, col=1)
    fig.update_xaxes(title_text="Indeterminacy Level (δ)", row=1, col=1)
    fig.update_yaxes(title_text="Decision Stability", row=1, col=2)
    fig.update_xaxes(title_text="Effect Size", row=1, col=2)
    
    return fig


def plot_interval_width_comparison(df: pd.DataFrame) -> go.Figure:
    """Compare p-value interval widths between original and modified tests"""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Interval Width by Indeterminacy",
            "Interval Width Distribution"
        ),
        specs=[[{"type": "bar"}, {"type": "box"}]]
    )
    
    # 1. Mean interval width by indeterminacy
    width_by_indet = df.groupby(['delta', 'variant'])['interval_width'].mean().reset_index()
    
    for variant in ['original', 'modified']:
        variant_data = width_by_indet[width_by_indet['variant'] == variant]
        fig.add_trace(
            go.Bar(
                x=variant_data['delta'],
                y=variant_data['interval_width'],
                name=f'{variant.title()}',
                marker_color='#FF5722' if variant == 'modified' else '#2196F3',
                text=[f"{v:.3f}" for v in variant_data['interval_width']],
                textposition="outside"
            ),
            row=1, col=1
        )
    
    # 2. Distribution of interval widths
    for variant in ['original', 'modified']:
        variant_data = df[df['variant'] == variant]['interval_width']
        fig.add_trace(
            go.Box(
                y=variant_data,
                name=f'{variant.title()}',
                marker_color='#FF5722' if variant == 'modified' else '#2196F3',
                boxmean='sd'
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title="📏 P-value Interval Width Comparison",
        height=500,
        showlegend=True,
        template="plotly_white",
        barmode='group'
    )
    
    fig.update_yaxes(title_text="Mean Interval Width", row=1, col=1)
    fig.update_xaxes(title_text="Indeterminacy Level (δ)", row=1, col=1)
    fig.update_yaxes(title_text="Interval Width", row=1, col=2)
    
    return fig


def plot_summary_metrics(df: pd.DataFrame) -> go.Figure:
    """Create a comprehensive summary metrics dashboard"""
    
    # Calculate summary statistics
    summary = df.groupby('variant').agg({
        'power': 'mean',
        'type1_error': 'mean',
        'decision_stability': 'mean',
        'interval_width': 'mean',
        'relative_efficiency': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    
    metrics = ['power', 'type1_error', 'decision_stability', 'interval_width']
    labels = ['Power', 'Type I Error', 'Decision Stability', 'Interval Width']
    colors = ['#2196F3', '#FF5722']
    
    for i, variant in enumerate(['original', 'modified']):
        variant_data = summary[summary['variant'] == variant]
        fig.add_trace(go.Bar(
            name=variant.title(),
            x=labels,
            y=[variant_data[m].values[0] for m in metrics],
            marker_color=colors[i],
            text=[f"{variant_data[m].values[0]:.3f}" for m in metrics],
            textposition="auto"
        ))
    
    fig.update_layout(
        title="📊 Overall Performance Summary",
        yaxis_title="Metric Value",
        barmode='group',
        template="plotly_white",
        height=500,
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig


def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive summary table"""
    
    summary = df.groupby(['variant', 'delta', 'n']).agg({
        'power': ['mean', 'std'],
        'type1_error': ['mean', 'std'],
        'decision_stability': ['mean', 'std'],
        'interval_width': ['mean', 'std'],
        'relative_efficiency': 'mean'
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    return summary

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

col_config, col_results = st.columns([3, 9])

with col_config:
    st.markdown("### 🎮 Simulation Control")
    
    run_sim = st.button(
        "🚀 Run Simulation", 
        type="primary", 
        use_container_width=True, 
        disabled=not (sample_sizes and indet_levels and distributions and effect_sizes)
    )
    
    if run_sim:
        st.session_state['run_simulation'] = True

with col_results:
    if run_sim or st.session_state.get('run_simulation', False):
        sim_engine = MonteCarloSimulation(
            n_simulations=n_sims, 
            random_seed=st.session_state.get('random_seed', 42)
        )
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def cb_progress(current, total):
            pct = min(1.0, current / max(1, total))
            progress_bar.progress(pct)
            status_text.text(f"Running simulation: {current:,} of {total:,} iterations...")
        
        with st.spinner("Executing Monte Carlo simulations..."):
            res_df = sim_engine.run(
                test_type, 
                sample_sizes, 
                indet_levels, 
                distributions, 
                effect_sizes, 
                alpha=st.session_state.get('alpha', 0.05), 
                progress_callback=cb_progress
            )
        
        st.success("✅ Simulation Complete!")
        st.session_state[f'sim_results_{test_type}'] = res_df
        st.session_state['run_simulation'] = False
        
        # Store for comparison
        st.session_state['last_simulation_df'] = res_df
        st.session_state['last_test_type'] = test_type
    
    # Display results if available
    if f'sim_results_{test_type}' in st.session_state:
        df = st.session_state[f'sim_results_{test_type}']
        
        # Create tabs for different views
        tabs = st.tabs([
            "📊 Power Analysis",
            "🎯 Type I Error",
            "⚡ Efficiency & Stability",
            "📏 Interval Widths",
            "📋 Summary Tables",
            "📥 Export Data"
        ])
        
        with tabs[0]:
            st.plotly_chart(plot_power_comparison(df), use_container_width=True)
            
            # Add improvement summary
            power_improvement = df[df['effect_size'] > 0].groupby('variant')['power'].mean()
            if 'modified' in power_improvement.index and 'original' in power_improvement.index:
                improvement = (power_improvement['modified'] - power_improvement['original']) / power_improvement['original'] * 100
                st.metric(
                    "Average Power Improvement", 
                    f"+{improvement:.1f}%", 
                    delta="modified vs original",
                    delta_color="normal"
                )
        
        with tabs[1]:
            st.plotly_chart(plot_type1_comparison(df), use_container_width=True)
            
            # Add Type I error summary
            type1_means = df[df['effect_size'] == 0].groupby('variant')['type1_error'].mean()
            if 'modified' in type1_means.index and 'original' in type1_means.index:
                type1_diff = (type1_means['modified'] - 0.05) * 100
                st.metric(
                    "Modified Test Type I Error", 
                    f"{type1_means['modified']:.3f}", 
                    delta=f"{type1_diff:+.1f}% from α=0.05",
                    delta_color="off"
                )
        
        with tabs[2]:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_relative_efficiency_heatmap(df), use_container_width=True)
            with col2:
                st.plotly_chart(plot_decision_stability_comparison(df), use_container_width=True)
            
            # Add efficiency summary
            avg_eff = df[df['effect_size'] > 0]['relative_efficiency'].mean()
            st.metric(
                "Average Relative Efficiency", 
                f"{avg_eff:.2f}", 
                delta="modified/original ratio",
                delta_color="normal"
            )
        
        with tabs[3]:
            st.plotly_chart(plot_interval_width_comparison(df), use_container_width=True)
            
            # Add interval width summary
            width_means = df.groupby('variant')['interval_width'].mean()
            if 'modified' in width_means.index and 'original' in width_means.index:
                width_reduction = (width_means['modified'] - width_means['original']) / width_means['original'] * 100
                st.metric(
                    "Average Interval Width Reduction", 
                    f"{width_reduction:+.1f}%", 
                    delta="narrower intervals",
                    delta_color="normal"
                )
        
        with tabs[4]:
            # Summary tables
            summary_df = create_summary_table(df)
            st.dataframe(summary_df, use_container_width=True)
            
            # Overall performance summary
            st.plotly_chart(plot_summary_metrics(df), use_container_width=True)
        
        with tabs[5]:
            # Export options
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Full Results (CSV)",
                    data=csv,
                    file_name=f"sim_results_{test_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                summary_csv = create_summary_table(df).to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📊 Download Summary Table (CSV)",
                    data=summary_csv,
                    file_name=f"sim_summary_{test_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Show data preview
            st.markdown("### Data Preview")
            st.dataframe(df.head(100), use_container_width=True)
    
    else:
        st.info("👈 Configure simulation parameters and click 'Run Simulation' to see results")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666;">
    <p>© 2024-2026 Akua Agyapomah Oteng | University of Mines and Technology, Tarkwa, Ghana</p>
    <p>Monte Carlo simulations evaluate performance across sample sizes, indeterminacy levels, distributions, and effect sizes.</p>
    <p style="font-size: 0.9em;">Each condition simulated with {n_sims} iterations for statistical stability</p>
</div>
""", unsafe_allow_html=True)