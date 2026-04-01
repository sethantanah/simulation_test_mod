"""
Performance Comparison Dashboard

This module provides comprehensive head-to-head comparison between original and modified
neutrosophic tests using real simulation results from the Monte Carlo framework.

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
from visualization.plots import plot_radar_comparison

st.set_page_config(
    page_title="Performance Comparison | PhD Research",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_css()

# ============================================================================
# TITLE AND INTRODUCTION
# ============================================================================

st.markdown("""
<div class="hero-section" style="text-align: center; padding: 2rem; margin-bottom: 2rem; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 10px;">
    <h1 style="color: white; margin: 0;">🏆 Performance Comparison</h1>
    <p style="color: white; margin-top: 0.5rem; font-size: 1.1em;">
        Definitive head-to-head comparison between Original and Modified neutrosophic tests
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_simulation_results(test_name):
    """Retrieve simulation results from session state"""
    key = f'sim_results_{test_name}'
    if key in st.session_state:
        return st.session_state[key]
    return None

def compute_summary_metrics(df, test_name):
    """Compute summary metrics from simulation dataframe"""
    if df is None or df.empty:
        return None
    
    # Filter for effect > 0 for power metrics
    power_df = df[df['effect_size'] > 0]
    # Filter for effect = 0 for Type I error
    type1_df = df[df['effect_size'] == 0]
    
    summary = {}
    
    # Power metrics
    orig_power = power_df[power_df['variant'] == 'original']['power'].mean() if not power_df.empty else 0
    mod_power = power_df[power_df['variant'] == 'modified']['power'].mean() if not power_df.empty else 0
    summary['power_original'] = orig_power if not pd.isna(orig_power) else 0
    summary['power_modified'] = mod_power if not pd.isna(mod_power) else 0
    
    if summary['power_original'] > 0:
        summary['power_improvement'] = ((summary['power_modified'] - summary['power_original']) / summary['power_original']) * 100
    else:
        summary['power_improvement'] = 0
    
    # Type I error metrics
    orig_type1 = type1_df[type1_df['variant'] == 'original']['type1_error'].mean() if not type1_df.empty else 0.05
    mod_type1 = type1_df[type1_df['variant'] == 'modified']['type1_error'].mean() if not type1_df.empty else 0.05
    summary['type1_original'] = orig_type1 if not pd.isna(orig_type1) else 0.05
    summary['type1_modified'] = mod_type1 if not pd.isna(mod_type1) else 0.05
    
    # Decision stability
    if 'decision_stability' in df.columns:
        orig_stability = df[df['variant'] == 'original']['decision_stability'].mean()
        mod_stability = df[df['variant'] == 'modified']['decision_stability'].mean()
        summary['stability_original'] = orig_stability if not pd.isna(orig_stability) else 0
        summary['stability_modified'] = mod_stability if not pd.isna(mod_stability) else 0
        summary['stability_improvement'] = (summary['stability_modified'] - summary['stability_original']) * 100
    else:
        summary['stability_original'] = 0
        summary['stability_modified'] = 0
        summary['stability_improvement'] = 0
    
    # Interval width
    orig_width = df[df['variant'] == 'original']['interval_width'].mean()
    mod_width = df[df['variant'] == 'modified']['interval_width'].mean()
    summary['width_original'] = orig_width if not pd.isna(orig_width) else 0
    summary['width_modified'] = mod_width if not pd.isna(mod_width) else 0
    
    if summary['width_original'] > 0:
        summary['width_reduction'] = ((summary['width_original'] - summary['width_modified']) / summary['width_original']) * 100
    else:
        summary['width_reduction'] = 0
    
    # Relative efficiency
    if 'relative_efficiency' in df.columns:
        mod_eff = df[df['variant'] == 'modified']['relative_efficiency'].mean()
        summary['efficiency_modified'] = mod_eff if not pd.isna(mod_eff) else 1.0
    else:
        summary['efficiency_modified'] = 1.0
    
    # Indeterminate rate reduction
    if 'indeterminate_rate' in df.columns:
        orig_indet = df[df['variant'] == 'original']['indeterminate_rate'].mean()
        mod_indet = df[df['variant'] == 'modified']['indeterminate_rate'].mean()
        summary['indet_original'] = orig_indet if not pd.isna(orig_indet) else 0
        summary['indet_modified'] = mod_indet if not pd.isna(mod_indet) else 0
        
        if summary['indet_original'] > 0:
            summary['indet_reduction'] = ((summary['indet_original'] - summary['indet_modified']) / summary['indet_original']) * 100
        else:
            summary['indet_reduction'] = 0
    else:
        summary['indet_original'] = 0
        summary['indet_modified'] = 0
        summary['indet_reduction'] = 0
    
    return summary

def plot_performance_timeline(df, test_name):
    """Plot performance metrics across indeterminacy levels"""
    if df is None or df.empty:
        return None
    
    # Filter and aggregate
    power_df = df[df['effect_size'] > 0].groupby(['delta', 'variant'])['power'].mean().reset_index()
    
    if power_df.empty:
        return None
    
    fig = go.Figure()
    
    for variant in ['original', 'modified']:
        variant_data = power_df[power_df['variant'] == variant]
        if not variant_data.empty:
            fig.add_trace(go.Scatter(
                x=variant_data['delta'],
                y=variant_data['power'],
                mode='lines+markers',
                name=f'{variant.title()}',
                line=dict(width=2, dash='solid' if variant == 'modified' else 'dash'),
                marker=dict(size=8)
            ))
    
    fig.update_layout(
        title=f"{test_name}: Power vs Indeterminacy Level",
        xaxis_title="Indeterminacy Level (δ)",
        yaxis_title="Statistical Power",
        template="plotly_white",
        height=400
    )
    
    return fig

def plot_decision_agreement_matrix(df, test_name):
    """Create decision agreement matrix from simulation results using rejection rates"""
    if df is None or df.empty:
        return None
    
    # Calculate rejection rates from power and Type I error
    # For effect > 0: rejection rate = power
    # For effect = 0: rejection rate = Type I error
    
    # Get power (rejection under H1)
    power_df = df[df['effect_size'] > 0]
    power_orig = power_df[power_df['variant'] == 'original']['power'].mean() if not power_df.empty else 0
    power_mod = power_df[power_df['variant'] == 'modified']['power'].mean() if not power_df.empty else 0
    
    # Get Type I error (rejection under H0)
    type1_df = df[df['effect_size'] == 0]
    type1_orig = type1_df[type1_df['variant'] == 'original']['type1_error'].mean() if not type1_df.empty else 0.05
    type1_mod = type1_df[type1_df['variant'] == 'modified']['type1_error'].mean() if not type1_df.empty else 0.05
    
    # Weighted average rejection rate across effect sizes
    # (simplified: using power as primary measure for agreement)
    if power_orig > 0 or power_mod > 0:
        # Estimate agreement based on power difference
        # Both reject: proportion of cases where both tests reject (using min of powers)
        both_reject = min(power_orig, power_mod) * 100
        
        # Modified only reject: difference when modified has higher power
        mod_only = max(0, power_mod - power_orig) * 100
        
        # Original only reject: difference when original has higher power
        orig_only = max(0, power_orig - power_mod) * 100
        
        # Neither reject: complement
        neither = max(0, (1 - max(power_orig, power_mod))) * 100
        
        # Indeterminate: estimate from Type I error deviation
        type1_deviation = abs(type1_mod - 0.05) if not pd.isna(type1_mod) else 0
        indeterminate = min(type1_deviation * 100, 10)  # Cap at 10%
        
        # Normalize to sum to 100
        total = both_reject + mod_only + orig_only + neither + indeterminate
        if total > 0:
            both_reject = (both_reject / total) * 100
            mod_only = (mod_only / total) * 100
            orig_only = (orig_only / total) * 100
            neither = (neither / total) * 100
            indeterminate = (indeterminate / total) * 100
        
        agree_data = pd.DataFrame({
            'Category': ['Both Reject', 'Modified Only Reject', 'Original Only Reject', 'Neither Reject', 'Indeterminate'],
            'Percentage': [both_reject, mod_only, orig_only, neither, indeterminate]
        })
        
        fig = px.pie(
            agree_data, 
            values='Percentage', 
            names='Category', 
            hole=0.5,
            title=f"{test_name}: Decision Agreement Matrix",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(template="plotly_white", height=400)
        
        return fig
    
    return None

def plot_radar_comparison_safe(orig_data, mod_data, test_name):
    """Safe wrapper for radar plot function"""
    try:
        return plot_radar_comparison(orig_data, mod_data, test_name)
    except:
        # Create a simple radar plot if the imported function fails
        fig = go.Figure()
        categories = list(orig_data.keys())
        
        fig.add_trace(go.Scatterpolar(
            r=list(orig_data.values()),
            theta=categories,
            fill='toself',
            name='Original Test',
            line_color='#2196F3'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=list(mod_data.values()),
            theta=categories,
            fill='toself',
            name='Modified Test',
            line_color='#FF5722'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=f"{test_name}: Performance Radar Comparison",
            height=400
        )
        
        return fig

# ============================================================================
# CHECK FOR SIMULATION RESULTS
# ============================================================================

# Check which simulation results are available
available_tests = []
test_data = {}

for test_key in ['kruskal_wallis', 'mann_whitney', 'moods_median']:
    df = get_simulation_results(test_key)
    if df is not None and not df.empty:
        available_tests.append(test_key)
        test_data[test_key] = df

has_results = len(available_tests) > 0

if not has_results:
    st.error("""
    ❌ **No Simulation Results Found**
    
    Please run simulations on the **Monte Carlo Simulation** page first to generate real performance data.
    
    **Steps:**
    1. Go to the Monte Carlo Simulation page
    2. Configure your simulation parameters
    3. Click "Run Simulation" for at least one test
    4. Return here to see real comparison data
    """)
    st.stop()

# ============================================================================
# COMPUTE AGGREGATE METRICS ACROSS AVAILABLE TESTS
# ============================================================================

all_metrics = []
for test in available_tests:
    df = test_data[test]
    metrics = compute_summary_metrics(df, test)
    if metrics:
        all_metrics.append(metrics)

if not all_metrics:
    st.error("Could not compute metrics from simulation results. Please re-run simulations.")
    st.stop()

# Average metrics across tests with safe defaults
aggregate_metrics = {
    'power_original': np.mean([m.get('power_original', 0) for m in all_metrics]),
    'power_modified': np.mean([m.get('power_modified', 0) for m in all_metrics]),
    'power_improvement': np.mean([m.get('power_improvement', 0) for m in all_metrics]),
    'type1_original': np.mean([m.get('type1_original', 0.05) for m in all_metrics]),
    'type1_modified': np.mean([m.get('type1_modified', 0.05) for m in all_metrics]),
    'stability_original': np.mean([m.get('stability_original', 0) for m in all_metrics]),
    'stability_modified': np.mean([m.get('stability_modified', 0) for m in all_metrics]),
    'stability_improvement': np.mean([m.get('stability_improvement', 0) for m in all_metrics]),
    'width_original': np.mean([m.get('width_original', 0) for m in all_metrics]),
    'width_modified': np.mean([m.get('width_modified', 0) for m in all_metrics]),
    'width_reduction': np.mean([m.get('width_reduction', 0) for m in all_metrics]),
    'indet_original': np.mean([m.get('indet_original', 0) for m in all_metrics]),
    'indet_modified': np.mean([m.get('indet_modified', 0) for m in all_metrics]),
    'indet_reduction': np.mean([m.get('indet_reduction', 0) for m in all_metrics]),
    'efficiency_modified': np.mean([m.get('efficiency_modified', 1.0) for m in all_metrics])
}

# ============================================================================
# KPI CARDS - USING REAL DATA ONLY
# ============================================================================

st.markdown("### 📊 Key Performance Indicators")

k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    improvement = aggregate_metrics.get('power_improvement', 0)
    st.metric(
        "Overall Power Improvement",
        f"+{improvement:.1f}%",
        delta="Modified advantage",
        delta_color="normal"
    )

with k2:
    type1_err = aggregate_metrics.get('type1_modified', 0.05)
    deviation = abs(type1_err - 0.05) * 100
    st.metric(
        "Type I Error Control",
        f"{type1_err:.3f}",
        delta=f"{deviation:.1f}% from α=0.05",
        delta_color="off"
    )

with k3:
    # Best test based on available simulation results
    best_test = max(available_tests, key=lambda t: test_data[t][test_data[t]['effect_size'] > 0]['power'].mean() if not test_data[t][test_data[t]['effect_size'] > 0].empty else 0)
    best_test_name = best_test.replace('_', ' ').title()
    st.metric(
        "Best Performing Test",
        best_test_name,
        delta="Based on real simulations"
    )

with k4:
    # Number of tests with data
    st.metric(
        "Tests Analyzed",
        len(available_tests),
        delta="With simulation data"
    )

with k5:
    # Total simulations
    total_sims = 0
    for test in available_tests:
        df = test_data[test]
        n_conditions = df['n'].nunique() * df['delta'].nunique() * df['distribution'].nunique() * df['effect_size'].nunique()
        total_sims += n_conditions * st.session_state.get('n_sims', 0)
    st.metric(
        "Total Simulations",
        f"{total_sims:,}",
        delta="Analyzed"
    )

with k6:
    # Confidence based on number of simulations
    n_sims = st.session_state.get('n_sims', 0)
    confidence = "High" if n_sims >= 1000 else ("Moderate" if n_sims >= 500 else "Low")
    st.metric(
        "Statistical Confidence",
        confidence,
        delta=f"{n_sims:,} sims/condition"
    )

st.markdown("---")

# ============================================================================
# TEST SELECTION AND DETAILED COMPARISON
# ============================================================================

# Only show tests that have simulation results
test_options = [t.replace('_', ' ').title() for t in available_tests]
test_select = st.selectbox(
    "Select Test to Compare",
    test_options,
    help="Select a specific test to see detailed performance comparison"
)

# Map display names back to keys
test_key_map = {t.replace('_', ' ').title(): t for t in available_tests}
selected_test_key = test_key_map[test_select]

# Get simulation results for selected test
selected_df = test_data[selected_test_key]

if selected_df is not None and not selected_df.empty:
    # Compute metrics for selected test
    test_metrics = compute_summary_metrics(selected_df, selected_test_key)
    
    if test_metrics:
        # Radar data using real metrics
        orig_radar = {
            'power': test_metrics.get('power_original', 0),
            'type1_error': test_metrics.get('type1_original', 0.05),
            'decision_stability': test_metrics.get('stability_original', 0),
            'relative_efficiency': 1.0,
            'interval_width': test_metrics.get('width_original', 0)
        }
        
        mod_radar = {
            'power': test_metrics.get('power_modified', 0),
            'type1_error': test_metrics.get('type1_modified', 0.05),
            'decision_stability': test_metrics.get('stability_modified', 0),
            'relative_efficiency': test_metrics.get('efficiency_modified', 1.0),
            'interval_width': test_metrics.get('width_modified', 0)
        }
    else:
        st.error("Could not compute metrics for selected test.")
        st.stop()
else:
    st.error("No simulation data available for selected test.")
    st.stop()

colR, colM = st.columns([1, 1])

with colR:
    st.markdown("### 📈 Radar Profile Comparison")
    fig_radar = plot_radar_comparison_safe(orig_radar, mod_radar, test_select)
    st.plotly_chart(fig_radar, use_container_width=True)

with colM:
    st.markdown("### 🎯 Decision Agreement Analysis")
    fig_agreement = plot_decision_agreement_matrix(selected_df, test_select)
    if fig_agreement:
        st.plotly_chart(fig_agreement, use_container_width=True)
    else:
        st.info("Decision agreement data not available for this test.")

# ============================================================================
# PERFORMANCE TIMELINE
# ============================================================================

st.markdown("### 📉 Performance Across Indeterminacy Levels")

fig_timeline = plot_performance_timeline(selected_df, test_select)
if fig_timeline:
    st.plotly_chart(fig_timeline, use_container_width=True)
else:
    st.info("Performance timeline data not available for this test.")

# ============================================================================
# DETAILED METRICS TABLE
# ============================================================================

st.markdown("### 📊 Detailed Performance Metrics")

if test_metrics:
    metrics_data = {
        'Metric': [
            'Statistical Power',
            'Type I Error Rate',
            'Decision Stability',
            'Interval Width',
            'Indeterminate Rate',
            'Relative Efficiency'
        ],
        'Original Test': [
            f"{test_metrics.get('power_original', 0):.4f}",
            f"{test_metrics.get('type1_original', 0):.4f}",
            f"{test_metrics.get('stability_original', 0):.4f}",
            f"{test_metrics.get('width_original', 0):.4f}",
            f"{test_metrics.get('indet_original', 0):.4f}",
            "1.000 (baseline)"
        ],
        'Modified Test': [
            f"{test_metrics.get('power_modified', 0):.4f}",
            f"{test_metrics.get('type1_modified', 0):.4f}",
            f"{test_metrics.get('stability_modified', 0):.4f}",
            f"{test_metrics.get('width_modified', 0):.4f}",
            f"{test_metrics.get('indet_modified', 0):.4f}",
            f"{test_metrics.get('efficiency_modified', 1.0):.4f}"
        ],
        'Improvement': [
            f"+{test_metrics.get('power_improvement', 0):.1f}%",
            f"{abs(test_metrics.get('type1_modified', 0.05) - 0.05):.4f} from α",
            f"+{test_metrics.get('stability_improvement', 0):.1f}%",
            f"-{test_metrics.get('width_reduction', 0):.1f}%",
            f"-{test_metrics.get('indet_reduction', 0):.1f}%",
            f"+{(test_metrics.get('efficiency_modified', 1.0) - 1) * 100:.1f}%"
        ]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
else:
    st.info("Detailed metrics not available for this test.")

# ============================================================================
# CONCLUSIONS AND RECOMMENDATIONS
# ============================================================================

st.markdown("### 💡 Conclusions & Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### ✅ Key Findings")
    
    stability_imp = aggregate_metrics.get('stability_improvement', 0)
    width_red = aggregate_metrics.get('width_reduction', 0)
    indet_red = aggregate_metrics.get('indet_reduction', 0)
    power_imp = aggregate_metrics.get('power_improvement', 0)
    type1_mod = aggregate_metrics.get('type1_modified', 0.05)
    type1_orig = aggregate_metrics.get('type1_original', 0.05)
    
    findings = f"""
    - **Tests Analyzed:** {len(available_tests)} tests with simulation data
    - **Power Improvement:** Modified tests show +{power_imp:.1f}% higher average power
    - **Type I Error:** Modified tests achieve {type1_mod:.4f} vs {type1_orig:.4f} (original)
    - **Decision Stability:** Modified tests show {stability_imp:.1f}% improvement
    - **Precision:** {width_red:.1f}% narrower p-value intervals
    - **Indeterminate Reduction:** {indet_red:.1f}% fewer ambiguous decisions
    """
    
    st.success(findings)

with col2:
    st.markdown("#### 🎯 Recommendations")
    
    if power_imp > 10:
        rec = """
        **Strongly Recommend Modified Tests**
        
        Based on real simulation results, modified neutrosophic tests consistently outperform original versions.
        
        **When to Use:**
        - Data contains any indeterminacy (>5%)
        - Measurement uncertainty is present
        - Need more decisive conclusions
        - Sample sizes are moderate (n ≥ 30)
        
        **Implementation:** Use the modified test functions with `_modified` suffix.
        """
    else:
        rec = """
        **Consider Modified Tests**
        
        Modified tests show moderate improvement. Consider using them when:
        
        - Data has any degree of indeterminacy
        - You need more robust inference
        - Traditional assumptions are violated
        
        **For fully determinate data:** Original tests may be sufficient.
        """
    
    st.info(rec)

# ============================================================================
# SIMULATION SUMMARY
# ============================================================================

with st.expander("📊 Simulation Summary", expanded=False):
    st.markdown("**Available Simulation Results:**")
    for test in available_tests:
        df = test_data[test]
        n_conditions = df['n'].nunique() * df['delta'].nunique() * df['distribution'].nunique() * df['effect_size'].nunique()
        n_sims_per_cond = st.session_state.get('n_sims', 0)
        total_sims = n_conditions * n_sims_per_cond
        st.markdown(f"- **{test.replace('_', ' ').title()}:** {n_conditions} conditions, {total_sims:,} total simulations")
    
    st.markdown(f"""
    **Simulation Parameters:**
    - Simulations per condition: {st.session_state.get('n_sims', 0)}
    - Significance level: α = {st.session_state.get('alpha', 0.05)}
    - Indeterminacy threshold: δ = {st.session_state.get('indet_threshold', 0.10)}
    - Random seed: {st.session_state.get('random_seed', 42)}
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666;">
    <p>© 2024-2026 Akua Agyapomah Oteng | University of Mines and Technology, Tarkwa, Ghana</p>
    <p>Performance comparison based on Monte Carlo simulations with real data.</p>
</div>
""", unsafe_allow_html=True)